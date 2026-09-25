"""Finite, resumable masked-language pretraining on real mmap language corpora.

Launch with torchrun for DDP. Loss normalization uses the actual masked-token
count across ranks and accumulation microbatches. Validation masks are fixed.
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

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from bobcat.checkpoints import (
    S3CheckpointStore,
    prune_uploaded_generations,
    verify_checkpoint,
    write_checkpoint,
)
from bobcat.corpus import MMapCorpus, atomic_json
from bobcat.model import DecisionModel, ModelConfig
from bobcat.schema import file_hash
from bobcat.tokenization import MASK, ScratchTokenizer


def mask_tokens(ids: torch.Tensor, vocab_size: int, seed: int, probability: float = 0.15):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    eligible = ids.ge(6)
    selected = torch.rand(ids.shape, generator=generator).lt(probability) & eligible
    # Short/remainder documents still contribute a language target.
    for index in torch.where(~selected.any(1))[0].tolist():
        choices = torch.where(eligible[index])[0]
        if not len(choices):
            raise ValueError("Empty language sequence.")
        selected[index, choices[0]] = True
    replacement = torch.rand(ids.shape, generator=generator)
    corrupted = ids.clone()
    corrupted[selected & replacement.lt(0.8)] = MASK
    random_positions = selected & replacement.ge(0.8) & replacement.lt(0.9)
    random_ids = torch.randint(6, vocab_size, ids.shape, generator=generator)
    corrupted[random_positions] = random_ids[random_positions]
    return corrupted, selected, ids[selected]


def trainable_language_parameters(model: DecisionModel) -> list[nn.Parameter]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith(("embedding.", "encoder.", "encoder_norm.", "mlm_transform."))
            or name == "mlm_bias"
        )
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _reduce(value: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


@torch.inference_mode()
def validate(
    model: DecisionModel, corpora: dict, device: torch.device,
    batch_size: int, batches: int, seed: int, rank: int, world: int,
) -> dict:
    was_training = model.training
    model.eval()
    result = {}
    for language, corpus in corpora.items():
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        for batch_index in range(rank, batches, world):
            rng = np.random.default_rng(seed + batch_index)
            original = torch.from_numpy(corpus.sample(rng, batch_size))
            corrupted, selected, labels = mask_tokens(
                original, model.config.vocab_size, seed + 100_000 + batch_index,
            )
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.mlm_logits(corrupted.to(device), selected.to(device)).float()
                labels = labels.to(device)
                loss = F.cross_entropy(logits, labels, reduction="sum")
            totals += torch.stack((loss.double(), labels.numel() * loss.new_ones(()).double(),
                                   logits.argmax(-1).eq(labels).sum().double()))
        _reduce(totals, world)
        loss_sum, count, correct = totals.tolist()
        result[language] = {
            "masked_token_nll": loss_sum / count,
            "masked_token_accuracy": correct / count,
            "masked_tokens": int(count),
            "masked_token_perplexity": math.exp(min(30, loss_sum / count)),
            "perplexity_scope": "masked positions, not autoregressive text perplexity",
        }
    model.train(was_training)
    return result


def checkpoint(
    model, optimizer, path: Path, step: int, counters: dict, provenance: dict,
    device: torch.device, rank: int, world: int, remote: S3CheckpointStore | None = None,
) -> None:
    rng = {"cpu": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    if rank == 0:
        payload = {
            "format": "bobcat-real-mlm-v1", "model": model.state_dict(),
            "config": model.config.to_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "counters": counters, "provenance": provenance,
            "rank_rng_states": states, "world_size": world,
        }
        record = write_checkpoint(payload, path)
        if remote is not None:
            receipt = remote.publish(path.resolve(strict=True), record)
            atomic_json(path.resolve(strict=True).with_suffix(".json"),
                        {**record, "remote": receipt})
            prune_uploaded_generations(path)
            print(json.dumps({"event": "checkpoint_uploaded", "step": step,
                              "sha256": record["sha256"], "bytes": record["bytes"]}), flush=True)
    if world > 1:
        dist.barrier()


def run(args) -> None:
    started = time.monotonic()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; use --device cpu only for explicit tests."
        )
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This training recipe requires BF16-capable CUDA.")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    tokenizer = ScratchTokenizer(args.tokenizer)
    config = ModelConfig(**json.loads(args.config.read_text()))
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Configured and trained vocabularies differ.")
    train_data = {lang: MMapCorpus(args.corpus, lang, "train", tokenizer.digest)
                  for lang in ("ko", "en")}
    validation_data = {lang: MMapCorpus(args.corpus, lang, "validation", tokenizer.digest)
                       for lang in ("ko", "en")}
    if any(data.manifest["sequence_length"] > config.max_context_tokens
           or data.manifest.get("tokenizer_encoding_profile") != tokenizer.encoding_profile
           for data in train_data.values()):
        raise ValueError("Corpus context length or tokenizer encoding profile is incompatible.")
    source_hashes = {name: file_hash(Path(__file__).with_name(name)) for name in (
        "pretrain.py", "model.py", "corpus.py", "tokenization.py", "checkpoints.py",
    )}
    provenance = {
        "initialization": "random", "tokenizer_sha256": tokenizer.digest,
        "tokenizer_encoding_profile": tokenizer.encoding_profile,
        "corpus_manifests": {lang: file_hash(c.manifest_path) for lang, c in train_data.items()},
        "config_sha256": file_hash(args.config), "seed": args.seed,
        "korean_sequence_fraction_target": 0.6,
        "schedule_total_steps": args.schedule_steps, "warmup_steps": args.warmup_steps,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "batch_per_gpu": args.batch_per_gpu, "gradient_accumulation": args.accumulation,
        "world_size": world,
        "precision": "bf16_autocast" if device.type == "cuda" else "fp32",
        "source_sha256": source_hashes,
        "training_objective": "15% MLM with 80/10/10 corruption",
        "decision_reader": "random and frozen during language pretraining",
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
    }
    # Full checksums once on the master before any paid optimization step.
    if rank == 0:
        for data in train_data.values():
            for info in data.manifest["files"].values():
                if file_hash(args.corpus / info["path"]) != info["sha256"]:
                    raise ValueError("Corpus checksum mismatch.")
    if world > 1:
        dist.barrier()
    model = DecisionModel(config).to(device)
    parameters = trainable_language_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98), eps=1e-8, fused=device.type == "cuda",
    )
    step = 0
    counters = {"tokens": 0, "ko_tokens": 0, "en_tokens": 0, "masked_tokens": 0,
                "optimizer_steps": 0, "training_seconds": 0.0}
    resume_rng = None
    if args.resume:
        verify_checkpoint(args.resume)
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["format"] != "bobcat-real-mlm-v1" or saved["provenance"] != provenance:
            raise ValueError("Resume requires the exact corpus, model, schedule, and world size.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, counters = saved["step"], saved["counters"]
        resume_rng = saved["rank_rng_states"][rank]
        del saved
    elif (args.out / "last.pt").exists():
        raise ValueError("Checkpoint exists; explicitly resume it.")
    wrapped = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda"
                                      else None, broadcast_buffers=False)
               if world > 1 else model)
    torch.manual_seed(args.seed + rank)
    if resume_rng is not None:
        torch.set_rng_state(resume_rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume_rng["cuda"], device)
    args.out.mkdir(parents=True, exist_ok=True)
    remote = (S3CheckpointStore(args.checkpoint_s3, region=args.s3_region)
              if args.checkpoint_s3 and rank == 0 else None)
    stopping = {"requested": False, "reason": None}

    def request_stop(signum, _frame):
        stopping.update(requested=True, reason=signal.Signals(signum).name)

    original_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        original_handlers[signum] = signal.signal(signum, request_stop)
    if rank == 0:
        atomic_json(args.out / "run.json", {
            "provenance": provenance,
            "parameters_total": sum(p.numel() for p in model.parameters()),
            "parameters_language_trainable": sum(p.numel() for p in parameters),
            "max_tokens_this_stage": args.max_tokens, "max_seconds_this_boot": args.max_seconds,
            "source_files": source_hashes,
            "hardware": {
                "world_size": world, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
        })

    def emit(row):
        if rank == 0:
            line = json.dumps(row)
            print(line, flush=True)
            with (args.out / "metrics.jsonl").open("a") as stream:
                stream.write(line + "\n")

    # Validation must not consume training RNG state.
    eval_rng = torch.get_rng_state()
    emit({"event": "validation", "step": step, "tokens": counters["tokens"],
          "metrics": validate(model, validation_data, device, args.batch_per_gpu,
                              args.eval_batches, args.seed + 900_000, rank, world)})
    torch.set_rng_state(eval_rng)
    model.train()
    last_saved = step if args.resume else -1
    while counters["tokens"] < args.max_tokens and step < args.schedule_steps:
        remaining = args.max_seconds - (time.monotonic() - started)
        if args.stop_file and args.stop_file.exists():
            stopping.update(requested=True, reason="stop_file")
        if remaining < args.checkpoint_reserve_seconds:
            stopping.update(requested=True, reason="deadline_reserve")
        stop = torch.tensor(int(stopping["requested"]), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            if stopping["reason"] is None:
                stopping["reason"] = "another_rank_requested_stop"
            break
        update_start = time.monotonic()
        microbatches = []
        count = 0
        for micro in range(args.accumulation):
            index = (step * args.accumulation + micro) * world + rank
            language = "ko" if index % 5 < 3 else "en"
            seed = args.seed + index * 1009
            original = torch.from_numpy(train_data[language].sample(
                np.random.default_rng(seed), args.batch_per_gpu,
            ))
            ids, selected, labels = mask_tokens(original, config.vocab_size, seed + 17)
            microbatches.append((language, original, ids, selected, labels))
            count += labels.numel()
        total_count = _reduce(torch.tensor(count, device=device, dtype=torch.float64), world)
        warmup = min(1.0, (step + 1) / max(1, args.warmup_steps))
        progress = max(0, step - args.warmup_steps) / max(
            1, args.schedule_steps - args.warmup_steps,
        )
        lr = args.lr * warmup * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, progress))))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(6, device=device, dtype=torch.float64)
        for micro, (language, original, ids, selected, labels) in enumerate(microbatches):
            context = (wrapped.no_sync() if world > 1 and micro + 1 < args.accumulation
                       else contextlib.nullcontext())
            with context:
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
                ):
                    logits = wrapped({"mlm_ids": ids.to(device),
                                      "mlm_positions": selected.to(device)}).float()
                    targets = labels.to(device)
                    loss_sum = F.cross_entropy(logits, targets, reduction="sum")
                    loss = loss_sum * world / total_count
                loss.backward()
            tokens = int(original.ne(0).sum())
            totals[0] += loss_sum.detach().double()
            totals[1] += labels.numel()
            totals[2] += logits.detach().argmax(-1).eq(targets).sum()
            totals[3] += tokens
            totals[4 if language == "ko" else 5] += tokens
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        finite = torch.tensor(int(torch.isfinite(norm)), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(
                "Non-finite gradient; refusing to overwrite the last checkpoint."
            )
        optimizer.step()
        _reduce(totals, world)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - update_start
        loss_sum, masked, correct, tokens, ko, en = totals.tolist()
        step += 1
        for name, value in [("tokens", tokens), ("ko_tokens", ko), ("en_tokens", en),
                            ("masked_tokens", masked)]:
            counters[name] += int(value)
        counters["optimizer_steps"] = step
        counters["training_seconds"] += elapsed
        if step % args.log_every == 0 or step == 1:
            emit({
                "event": "train", "step": step, **counters,
                "masked_token_nll": loss_sum / masked,
                "masked_token_accuracy": correct / masked, "learning_rate": lr,
                "gradient_norm": float(norm), "update_seconds": elapsed,
                "nonpadding_tokens_per_second": tokens / elapsed,
                "actual_korean_token_fraction": counters["ko_tokens"] / counters["tokens"],
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda" else 0,
            })
        if step % args.eval_every == 0:
            emit({"event": "validation", "step": step, "tokens": counters["tokens"],
                  "metrics": validate(model, validation_data, device, args.batch_per_gpu,
                                      args.eval_batches, args.seed + 900_000, rank, world)})
        if step % args.save_every == 0 or step == 1:
            checkpoint(model, optimizer, args.out / "last.pt", step, counters,
                       provenance, device, rank, world, remote)
            last_saved = step
    if last_saved != step:
        checkpoint(model, optimizer, args.out / "last.pt", step, counters,
                   provenance, device, rank, world, remote)
    emit({"event": "validation", "step": step, "tokens": counters["tokens"],
          "metrics": validate(model, validation_data, device, args.batch_per_gpu,
                              args.eval_batches, args.seed + 900_000, rank, world)})
    emit({"event": "stage_finished", "step": step, **counters,
          "wall_seconds": time.monotonic() - started, "quality_gate_passed": False,
          "stop_reason": stopping["reason"] or "stage_limit"})
    for signum, handler in original_handlers.items():
        signal.signal(signum, handler)
    if world > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "tokenizer", "corpus", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--checkpoint-s3")
    parser.add_argument("--s3-region")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--checkpoint-reserve-seconds", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--max-seconds", type=float, default=18_000)
    parser.add_argument("--schedule-steps", type=int, default=20_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-per-gpu", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_per_gpu, args.accumulation, args.schedule_steps,
           args.eval_batches, args.eval_every, args.save_every, args.log_every) < 1:
        parser.error("Counts must be positive.")
    if not 120 < args.max_seconds <= 7 * 3600:
        parser.error("Run between 120 seconds and 7 hours, leaving the boot auto-stop margin.")
    if not 0 <= args.checkpoint_reserve_seconds < args.max_seconds:
        parser.error("Checkpoint reserve must fit within the run deadline.")
    if args.max_tokens < 1 or args.warmup_steps < 0 or args.cpu_threads < 1:
        parser.error("Token/thread counts must be positive and warmup nonnegative.")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.weight_decay < 1:
        parser.error("Use a positive finite learning rate and weight decay in [0,1).")
    run(args)


if __name__ == "__main__":
    main()
