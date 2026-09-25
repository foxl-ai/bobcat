"""Bounded supervised decision training initialized from real Bobcat MLM weights.

The first objective is context-weighted cross entropy over exactly the offered
candidates. No calibration/final rows, teacher confidences, hidden sentinels or
Jev outputs are used. DDP normalizes by the global context weight, including
variable numbers of questions per context and gradient accumulation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from bobcat.batching import EncodedDataset
from bobcat.checkpoints import (
    S3CheckpointStore,
    prune_uploaded_generations,
    verify_checkpoint,
    write_checkpoint,
)
from bobcat.corpus import atomic_json
from bobcat.metrics import evaluate_rows
from bobcat.model import DecisionModel, ModelConfig
from bobcat.schema import file_hash, read_examples
from bobcat.supervision import MEAN_TARGET, evaluate_supervised
from bobcat.supervision import losses as supervised_losses
from bobcat.tokenization import ScratchTokenizer

FORMAT = "bobcat-real-decisions-v1"
BACKBONE_PREFIXES = ("embedding.", "encoder.", "encoder_norm.")


def load_language_parent(path: Path, tokenizer: ScratchTokenizer, minimum_tokens: int = 1):
    record = verify_checkpoint(path, expected_format="bobcat-real-mlm-v1")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    provenance = saved["provenance"]
    if (saved.get("format") != "bobcat-real-mlm-v1"
            or provenance.get("initialization") != "random"
            or provenance.get("tokenizer_sha256") != tokenizer.digest
            or provenance.get("tokenizer_encoding_profile") != tokenizer.encoding_profile
            or saved["counters"]["tokens"] < minimum_tokens
            or saved["step"] < 1 or saved["config"]["vocab_size"] != tokenizer.vocab_size):
        raise ValueError("Use a trained, verified Bobcat language parent and its tokenizer.")
    model = DecisionModel(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["model"], strict=True)
    return model, {
        "checkpoint_sha256": record["sha256"], "step": saved["step"],
        "counters": saved["counters"], "provenance": provenance,
    }


def parameter_groups(model: DecisionModel, *, backbone_lr: float, reader_lr: float,
                     freeze_backbone: bool):
    backbone, reader = [], []
    for name, parameter in model.named_parameters():
        language_only = name == "mlm_bias" or name.startswith("mlm_transform.")
        # Both biases add the same scalar to every valid candidate logit.
        # Softmax cannot identify them; Adam would amplify cancellation noise.
        common_offset = name in {"score.bias", "decision_norm.bias"}
        is_backbone = name.startswith(BACKBONE_PREFIXES)
        parameter.requires_grad_(
            not language_only and not common_offset and not (is_backbone and freeze_backbone)
        )
        if parameter.requires_grad:
            (backbone if is_backbone else reader).append(parameter)
    groups = []
    for name, parameters, lr in (
        ("backbone", backbone, backbone_lr), ("reader", reader, reader_lr),
    ):
        if parameters:
            groups.append({"params": parameters, "lr": lr, "base_lr": lr, "name": name})
    if not groups or not reader:
        raise ValueError("Decision training needs trainable decision parameters.")
    return groups


def load_partitions(root: Path, tokenizer: ScratchTokenizer, config: ModelConfig):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    public = manifest.get("schema") == "bobcat-public-decisions-v1"
    if manifest.get("schema") not in ("bobcat-korean-decisions-v1", "bobcat-public-decisions-v1"):
        raise ValueError("Use a checksummed real decision manifest.")
    datasets = {}
    for split in ("train", "dev_train"):
        filename = f"{split}.jsonl"
        expected = manifest["files"][filename]
        path = root / filename
        if path.is_symlink() or file_hash(path) != expected["sha256"]:
            raise ValueError(f"Decision source checksum mismatch: {split}")
        if public:
            from bobcat.public_decisions import load_partition, student_example
            records, _ = load_partition(root, split)
            if any(r["source_split"] != "train" for r in records):
                raise ValueError("Original public validation/test cannot enter model fitting.")
            examples = [student_example(row) for row in records]
        else:
            examples = read_examples(path)
        if (not examples or any(e.split != split for e in examples)
                or any(e.supervision == "hard_label" and e.target not in {
                    c.id for c in e.choices
                } for e in examples)
                or len({e.id for e in examples}) != len(examples)):
            raise ValueError("Use unique, correctly partitioned questions with offered targets.")
        datasets[split] = EncodedDataset(examples, tokenizer, config, include_sentinels=False)
    train, dev = datasets["train"], datasets["dev_train"]
    for field in ("group_id", "context_id"):
        if {getattr(e, field) for e in train.examples} & {
            getattr(e, field) for e in dev.examples
        }:
            raise ValueError(f"Training/development {field} overlap.")
    if {e.input_fingerprint() for e in train.examples} & {
        e.input_fingerprint() for e in dev.examples
    }:
        raise ValueError("Training/development input overlap.")
    return train, dev, {
        "manifest_sha256": file_hash(manifest_path),
        "train_sha256": manifest["files"]["train.jsonl"]["sha256"],
        "development_sha256": manifest["files"]["dev_train.jsonl"]["sha256"],
        "train_questions": len(train.examples), "train_contexts": len(train.groups),
        "train_independent_groups": len(train.world_groups),
        "development_questions": len(dev.examples),
        "license": manifest.get("dataset_license", manifest.get("licenses")),
        "train_mean_questions": sum(e.supervision == "score_mean" for e in train.examples),
        "development_mean_questions": sum(e.supervision == "score_mean" for e in dev.examples),
        "calibration_and_public_validation_used_for_optimization": False,
    }


def context_weights(batch, dataset: EncodedDataset) -> torch.Tensor:
    # A three-view NLI context has weight one, as does a one-question news title.
    return torch.tensor([
        1.0 / len(dataset.groups[e.context_id]) for e in batch.examples
    ], dtype=torch.float64)


def reduce_sum(value: torch.Tensor, world: int):
    if world > 1:
        dist.all_reduce(value)
    return value


@torch.inference_mode()
def development(model, dataset, device, rank, world, batch_size):
    was_training = model.training
    model.eval()
    rows = []
    for index, batch in enumerate(dataset.evaluation_batches(batch_size)):
        if index % world != rank:
            continue
        moved = batch.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(moved.model_inputs()).float().cpu()
        for e, ids, values in zip(batch.examples, batch.candidate_ids, logits, strict=True):
            row = {
                "id": e.id, "group_id": e.group_id, "family": e.family, "kind": e.kind,
                "split": e.split, "pair_id": e.pair_id, "variant": e.variant,
                "target": e.target, "candidate_ids": ids,
                "logits": values[:len(ids)].tolist(), "tie_break": "request_order",
                "supervision": e.supervision, "score_target": e.score_target,
            }
            if not all(math.isfinite(v) for v in row["logits"]):
                raise FloatingPointError("Development logits are non-finite.")
            rows.append(row)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [row for part in gathered for row in part]
    model.train(was_training)
    if rank:
        return None
    rows.sort(key=lambda row: row["id"])
    if len(rows) != len(dataset.examples) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Development evaluation omitted or duplicated a question.")
    return {
        "questions": len(rows), "independent_groups": len({r["group_id"] for r in rows}),
        "temperature": 1.0, "calibration_fitted": False, "final_evaluation": False,
        "overall": evaluate_rows([r for r in rows if r["supervision"] == "hard_label"]),
        "supervision": evaluate_supervised(rows),
        "by_family": {
            family: evaluate_supervised([r for r in rows if r["family"] == family])
            for family in sorted({r["family"] for r in rows})
        },
    }


def save_training(model, optimizer, path, step, counters, provenance, device, rank, world,
                  remote):
    rng = {"cpu": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    if rank == 0:
        payload = {
            "format": FORMAT, "config": model.config.to_dict(), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "counters": counters,
            "provenance": provenance, "rank_rng_states": states, "world_size": world,
            "release_gate_passed": False, "temperatures": {},
        }
        record = write_checkpoint(payload, path)
        if remote:
            receipt = remote.publish(path.resolve(strict=True), record)
            atomic_json(path.resolve(strict=True).with_suffix(".json"),
                        {**record, "remote": receipt})
            prune_uploaded_generations(path)
    if world > 1:
        dist.barrier()


def run(args):
    if args.out.exists():
        raise ValueError("Use a fresh output directory; resume into a new directory.")
    started = time.monotonic()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; CPU is only an explicit test mode.")
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This recipe requires BF16-capable CUDA.")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    tokenizer = ScratchTokenizer(args.tokenizer)
    model, parent = load_language_parent(args.initial_mlm, tokenizer, args.minimum_parent_tokens)
    train, dev, data = load_partitions(args.data, tokenizer, model.config)
    model.to(device)
    groups = parameter_groups(
        model, backbone_lr=args.backbone_lr, reader_lr=args.reader_lr,
        freeze_backbone=args.freeze_backbone,
    )
    parameters = [p for group in groups for p in group["params"]]
    optimizer = torch.optim.AdamW(
        groups, betas=(0.9, 0.98), weight_decay=args.weight_decay, fused=device.type == "cuda",
    )
    provenance = {
        "weight_origin": "project_scratch_mlm", "language_parent": parent,
        "tokenizer_sha256": tokenizer.digest,
        "tokenizer_encoding_profile": tokenizer.encoding_profile,
        "data": data, "seed": args.seed, "world_size": world,
        "contexts_per_gpu": args.contexts_per_gpu, "gradient_accumulation": args.accumulation,
        "backbone_lr": args.backbone_lr, "reader_lr": args.reader_lr,
        "freeze_backbone": args.freeze_backbone, "weight_decay": args.weight_decay,
        "schedule_steps": args.schedule_steps, "warmup_steps": args.warmup_steps,
        "objective": "context-weighted categorical CE and observed ordinal-mean squared error",
        "mean_loss_scale": 1.0, "mean_targets_identify_categorical_distribution": False,
        "candidate_policy": "offered candidates; shuffle unordered choices, retain Score order",
        "precision": "bf16_autocast" if device.type == "cuda" else "fp32",
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "source_sha256": {
            name: file_hash(Path(__file__).with_name(name)) for name in
            ("decision_train.py", "model.py", "batching.py", "tokenization.py",
             "schema.py", "checkpoints.py", "metrics.py", "supervision.py", "public_decisions.py")
        },
    }
    step, counters = 0, {
        "tokens": 0, "questions": 0, "context_draws": 0, "optimizer_steps": 0,
        "training_seconds": 0.0,
    }
    resume_rng = None
    if args.resume:
        verify_checkpoint(args.resume, expected_format=FORMAT)
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["format"] != FORMAT or saved["provenance"] != provenance:
            raise ValueError(
                "Resume requires the same parent, data, schedule, code and world size."
            )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, counters = saved["step"], saved["counters"]
        resume_rng = saved["rank_rng_states"][rank]
        del saved
    wrapped = DistributedDataParallel(
        model, device_ids=[local_rank] if device.type == "cuda" else None,
        broadcast_buffers=False,
    ) if world > 1 else model
    torch.manual_seed(args.seed + rank)
    if resume_rng:
        torch.set_rng_state(resume_rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume_rng["cuda"], device)
    if rank == 0:
        args.out.mkdir(parents=True)
        atomic_json(args.out / "run.json", {
            "provenance": provenance,
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_trainable": sum(p.numel() for p in parameters),
            "maximum_stage_steps": args.max_steps, "maximum_stage_seconds": args.max_seconds,
            "quality_gate_passed": False,
        })
    if world > 1:
        dist.barrier()
    remote = S3CheckpointStore(args.checkpoint_s3, region=args.s3_region) if (
        rank == 0 and args.checkpoint_s3
    ) else None
    stopping = {"requested": False, "reason": None}

    def stop(signum, _):
        stopping.update(requested=True, reason=signal.Signals(signum).name)

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}

    def emit(row):
        if rank == 0:
            text = json.dumps(row, allow_nan=False)
            with (args.out / "metrics.jsonl").open("a") as stream:
                stream.write(text + "\n")
            print(text, flush=True)

    def evaluate():
        metrics = development(model, dev, device, rank, world, args.eval_batch_size)
        emit({"event": "development", "step": step, "metrics": metrics})

    last_saved = -1
    try:
        evaluate()
        model.train()
        while step < min(args.max_steps, args.schedule_steps):
            if args.stop_file and args.stop_file.exists():
                stopping.update(requested=True, reason="stop_file")
            if args.max_seconds - (time.monotonic() - started) < args.checkpoint_reserve_seconds:
                stopping.update(requested=True, reason="deadline_reserve")
            stop_flag = torch.tensor(int(stopping["requested"]), device=device)
            if world > 1:
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if stop_flag.item():
                stopping["reason"] = stopping["reason"] or "another_rank_requested_stop"
                break
            update_start = time.monotonic()
            batches = []
            for micro in range(args.accumulation):
                index = (step * args.accumulation + micro) * world + rank
                batch = train.collate(
                    train.training_indices(index, args.contexts_per_gpu, args.seed),
                    shuffle_seed=args.seed + index * 1009,
                )
                batches.append((batch, context_weights(batch, train)))
            weight = reduce_sum(torch.tensor(
                sum(float(w.sum()) for _, w in batches), dtype=torch.float64, device=device,
            ), world)
            scale = min(1.0, (step + 1) / max(1, args.warmup_steps))
            progress = max(0, step - args.warmup_steps) / max(
                1, args.schedule_steps - args.warmup_steps,
            )
            scale *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(9, dtype=torch.float64, device=device)
            for micro, (batch, weights) in enumerate(batches):
                moved, weights = batch.to(device), weights.to(device)
                context = wrapped.no_sync() if world > 1 and micro + 1 < len(batches) else (
                    contextlib.nullcontext()
                )
                with context:
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
                    ):
                        logits = wrapped(moved.model_inputs()).float()
                        losses = supervised_losses(
                            logits, moved.tensors["targets"],
                            moved.tensors["score_targets"].to(device),
                        )
                        weighted_sum = (losses.double() * weights).sum()
                        loss = weighted_sum * world / weight
                    loss.backward()
                totals[0] += weighted_sum.detach()
                totals[1] += weights.sum()
                totals[2] += (
                    logits.argmax(-1).eq(moved.tensors["targets"]).double() * weights
                ).sum()
                totals[3] += len(batch.examples)
                totals[4] += batch.nonpadding_tokens
                hard = moved.tensors["targets"].ne(MEAN_TARGET)
                totals[5] += weights[hard].sum()
                totals[6] += (losses[hard].double() * weights[hard]).sum().detach()
                totals[7] += weights[~hard].sum()
                totals[8] += (losses[~hard].double() * weights[~hard]).sum().detach()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            finite = torch.tensor(int(torch.isfinite(norm)), device=device)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite gradients; preserve the last checkpoint.")
            optimizer.step()
            reduce_sum(totals, world)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - update_start
            objective, contexts, correct, questions, tokens, hard_weight, nll, mean_weight, mse = (
                totals.tolist()
            )
            step += 1
            for key, value in (("tokens", tokens), ("questions", questions),
                               ("context_draws", round(contexts))):
                counters[key] += int(value)
            counters["optimizer_steps"] = step
            counters["training_seconds"] += elapsed
            if step == 1 or step % args.log_every == 0:
                emit({
                    "event": "train", "step": step, **counters,
                    "context_weighted_objective": objective / contexts,
                    "context_weighted_nll": nll / hard_weight if hard_weight else None,
                    "context_weighted_accuracy": correct / hard_weight if hard_weight else None,
                    "context_weighted_ordinal_mean_mse": mse / mean_weight if mean_weight else None,
                    "categorical_context_weight": hard_weight,
                    "ordinal_mean_context_weight": mean_weight,
                    "gradient_norm": float(norm), "update_seconds": elapsed,
                    "nonpadding_input_tokens_per_second": tokens / elapsed,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda" else 0,
                })
            if step % args.eval_every == 0:
                evaluate()
            if step == 1 or step % args.save_every == 0:
                save_training(model, optimizer, args.out / "last.pt", step, counters, provenance,
                              device, rank, world, remote)
                last_saved = step
        if last_saved != step:
            save_training(model, optimizer, args.out / "last.pt", step, counters, provenance,
                          device, rank, world, remote)
        evaluate()
        emit({
            "event": "stage_finished", "step": step, **counters,
            "wall_seconds": time.monotonic() - started,
            "stop_reason": stopping["reason"] or "stage_limit",
            "quality_gate_passed": False, "final_evaluation": False,
        })
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("initial-mlm", "tokenizer", "data", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--minimum-parent-tokens", type=int, default=1_000_000)
    parser.add_argument("--contexts-per-gpu", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--reader-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--schedule-steps", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--checkpoint-reserve-seconds", type=float, default=300)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--checkpoint-s3")
    parser.add_argument("--s3-region")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if (min(args.minimum_parent_tokens, args.contexts_per_gpu, args.accumulation,
            args.schedule_steps, args.max_steps, args.save_every, args.eval_every,
            args.eval_batch_size, args.log_every, args.cpu_threads) < 1
            or args.warmup_steps < 0
            or not 120 < args.max_seconds <= 7 * 3600
            or not 0 <= args.checkpoint_reserve_seconds < args.max_seconds
            or not 0 <= args.weight_decay < 1
            or any(not math.isfinite(x) or x <= 0 for x in (args.backbone_lr, args.reader_lr))):
        parser.error("Use positive counts, finite learning rates and a bounded runtime.")
    run(args)


if __name__ == "__main__":
    main()
