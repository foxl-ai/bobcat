"""Bobcat Flash student training: gold SFT + distillation on Bobcat 1 probabilities (2026-09-26).

Inputs are rows compiled by `flash_data compile` (or `student_readout compile --piecewise` for
evaluation splits): token IDs, the offered identifier IDs, and supervision. The student is read
exactly like the served model: the LM head at the last prompt position, restricted to the
offered identifiers (Gemma's final logit soft-cap included). Nothing is generated.

Loss per question (mean over the questions of a global step):
  kd    KL(p_teacher || q_student) over the offered candidates, p_teacher = softmax(z_T / T_T)
        with T_T = Bobcat 1's calibration temperature (1.1489) unless --teacher-temperature;
  gold  cross-entropy for hard labels; for Score means the normalised squared error of the
        expected level (the Bobcat 1 recipe).
  total = kd_weight * kd (when a teacher row exists) + gold_weight * gold (when gold exists).
Rows are right-padded into token-budget micro-batches of similar length (causal attention and
causal linear attention never look right, so padding cannot change the last real position).
Full fine-tuning keeps FP32 master weights with BF16 autocast; --lora-rank trains a LoRA
adapter on a frozen BF16 base instead. Runs under torchrun; every rank reads its own shard of
rows and gradients are averaged across ranks by hand after each accumulation window.
--save-every writes out/checkpoint/{model or adapter, state.json} (weights, step, data
cursor; the optimizer restarts on resume, recorded in train-summary.json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np


def load_rows(path: Path, rank: int, world: int, seed: int, limit: int | None = None):
    """This rank's share of rows (by a stable shuffle), with token IDs as int32 arrays."""
    rows = []
    with path.open() as stream:
        for index, line in enumerate(stream):
            if limit and index >= limit:
                break
            if (index * 2654435761 + seed) % 4294967296 % world != rank:
                continue
            record = json.loads(line)
            record["input_ids"] = np.asarray(record["input_ids"], dtype=np.int32)
            rows.append(record)
    return rows


def load_teacher(paths: list[Path], wanted: set[str]) -> dict[str, np.ndarray]:
    teacher = {}
    for path in paths:
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["id"] in wanted:
                    teacher[record["id"]] = np.asarray(record["logits"], dtype=np.float32)
    return teacher


def micro_batches(rows: list[dict], budget: int, max_rows: int, seed: int, epoch: int):
    """Shuffle, then sort within windows of 2,048 rows by length, and cut batches whose padded
    size (rows x longest) stays within `budget` tokens."""
    order = list(range(len(rows)))
    random.Random(seed * 1000 + epoch).shuffle(order)
    batches = []
    for start in range(0, len(order), 2048):
        window = sorted(order[start:start + 2048], key=lambda i: len(rows[i]["input_ids"]))
        current, longest = [], 0
        for i in window:
            length = len(rows[i]["input_ids"])
            if current and (max(longest, length) * (len(current) + 1) > budget
                            or len(current) >= max_rows):
                batches.append(current)
                current, longest = [], 0
            current.append(i)
            longest = max(longest, length)
        if current:
            batches.append(current)
    random.Random(seed * 7 + epoch).shuffle(batches)
    return batches


def row_loss(torch, logits, row, teacher_logits, args):
    """Loss terms for one question; `logits` is the 1-D candidate score vector (float32)."""
    log_q = torch.log_softmax(logits, dim=-1)
    terms, count = logits.new_zeros(()), 0
    if (teacher_logits is not None and args.kd_weight > 0
            and len(teacher_logits) == logits.shape[0]):
        t = torch.as_tensor(teacher_logits, device=logits.device, dtype=torch.float32)
        log_p = torch.log_softmax(t / args.teacher_temperature, dim=-1)
        kd = (log_p.exp() * (log_p - log_q)).sum()
        terms = terms + args.kd_weight * kd
        count += 1
    if args.gold_weight > 0:
        if row["supervision"] == "score_mean" and row.get("score_target") is not None:
            levels = torch.arange(logits.shape[0], dtype=torch.float32, device=logits.device)
            expected = (log_q.exp() * levels).sum()
            gold = ((expected - row["score_target"]) / max(logits.shape[0] - 1, 1)) ** 2
            terms = terms + args.gold_weight * gold
            count += 1
        elif row["supervision"] == "hard_label" and row.get("target") is not None:
            terms = terms + args.gold_weight * (-log_q[row["target"]])
            count += 1
    return terms, count


class Student:
    def __init__(self, model_dir: Path, *, device, torch, lora_rank: int, init: Path | None,
                 gradient_checkpointing: bool = True, train: bool = True,
                 freeze: str | None = None):
        from transformers import AutoConfig, AutoModelForCausalLM

        self.torch, self.device = torch, device
        self.lora = lora_rank > 0
        # Full fine-tunes continue from a saved model folder; LoRA always loads the pinned base
        # and then the saved adapter.
        source = model_dir if (self.lora or init is None) else init
        dtype = torch.bfloat16 if (self.lora or not train) else torch.float32
        options = dict(dtype=dtype, device_map={"": device})
        try:
            model = AutoModelForCausalLM.from_pretrained(source, **options)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(source, **options)
        model.config.use_cache = False
        config = AutoConfig.from_pretrained(model_dir)
        text = getattr(config, "text_config", None) or config
        self.softcap = getattr(text, "final_logit_softcapping", None)
        if self.lora:
            from peft import LoraConfig, PeftModel, get_peft_model

            for parameter in model.parameters():
                parameter.requires_grad_(False)
            if init is not None and (init / "adapter_config.json").exists():
                model = PeftModel.from_pretrained(model, init, is_trainable=train)
            else:
                model = get_peft_model(model, LoraConfig(
                    r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0,
                    target_modules="all-linear",
                    exclude_modules=r".*(visual|vision|merger|lm_head|mtp|audio).*"))
        elif train:
            import re

            pattern = re.compile(freeze) if freeze else None
            for name, parameter in model.named_parameters():
                frozen = any(k in name for k in ("visual", "vision", "audio", "mtp")) or bool(
                    pattern and pattern.search(name))
                parameter.requires_grad_(not frozen)
        if train and gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = model
        base = model.get_base_model() if self.lora else model
        self.backbone = base.get_decoder() if hasattr(base, "get_decoder") else base.model
        self.lm_weight = base.get_output_embeddings().weight

    def forward(self, batch, need_mass: bool = False):
        """Candidate logits (float32) for each row of a right-padded micro-batch."""
        torch = self.torch
        lengths = [len(r["input_ids"]) for r in batch]
        longest = max(lengths)
        ids = np.zeros((len(batch), longest), dtype=np.int64)
        for i, r in enumerate(batch):
            ids[i, :lengths[i]] = r["input_ids"]
        ids = torch.from_numpy(ids).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(batch), device=self.device),
                      torch.tensor(lengths, device=self.device) - 1]
        out = []
        for i, r in enumerate(batch):
            options = torch.as_tensor(r["option_ids"], device=self.device)
            z = self.lm_weight[options].float() @ last[i].float()
            if self.softcap:
                z = torch.tanh(z / self.softcap) * self.softcap
            out.append(z)
        masses = []
        if need_mass:
            full = (last.to(self.lm_weight.dtype) @ self.lm_weight.T).float()
            if self.softcap:
                full = torch.tanh(full / self.softcap) * self.softcap
            full = torch.log_softmax(full, dim=-1)
            for i, r in enumerate(batch):
                options = torch.as_tensor(r["option_ids"], device=self.device)
                masses.append(float(torch.logsumexp(full[i, options], dim=0)))
        return out, masses


def readout_module(student, torch):
    """An nn.Module around the student so DDP sees every forward (gradient sync hooks)."""

    class Readout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = student.model

        def forward(self, batch):
            logits, _ = student.forward(batch)
            return logits

    return Readout()


def setup(torch):
    import torch.distributed as dist

    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    return dist, world, rank, torch.device("cuda", local)


def all_min(torch, dist, world, value: int, device) -> int:
    if world == 1:
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def bf16_config(path: Path) -> None:
    """The weights are stored in BF16; say so in every (nested) config, or a loader that
    reads a sub-config's dtype (vLLM for Gemma 4's text/vision/audio towers) mixes types."""
    def fix(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("dtype", "torch_dtype") and value in ("float32", "float16"):
                    node[key] = "bfloat16"
                else:
                    fix(value)
        return node

    config = fix(json.loads(path.read_text()))
    config["torch_dtype"] = config["dtype"] = "bfloat16"
    path.write_text(json.dumps(config, indent=2) + "\n")


def save_model(student, folder: Path, base_dir: Path) -> None:
    import shutil

    folder.mkdir(parents=True, exist_ok=True)
    if student.lora:
        student.model.save_pretrained(folder / "adapter")
        return
    model = student.model
    state = {k: v.to(student.torch.bfloat16) if v.is_floating_point() else v
             for k, v in model.state_dict().items()}
    model.save_pretrained(folder, state_dict=state, safe_serialization=True,
                          max_shard_size="5GB")
    bf16_config(folder / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "special_tokens_map.json", "generation_config.json", "vocab.json",
                 "merges.txt", "preprocessor_config.json", "processor_config.json",
                 "video_preprocessor_config.json"):
        if (base_dir / name).exists():
            shutil.copy2(base_dir / name, folder / name)


def train(args) -> None:
    import torch

    dist, world, rank, device = setup(torch)
    torch.manual_seed(args.seed + rank)
    student = Student(args.model_dir, device=device, torch=torch, lora_rank=args.lora_rank,
                      init=args.init, train=not args.eval_only,
                      gradient_checkpointing=not args.no_checkpointing, freeze=args.freeze)
    if not args.eval_only:
        fit(args, student, dist, world, rank, device, torch)
    if world > 1:
        dist.barrier()
    if args.eval:
        evaluate(args, student, dist, world, rank, torch)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def fit(args, student, dist, world, rank, device, torch) -> None:
    rows = load_rows(args.train, rank, world, args.seed, args.limit)
    teacher = load_teacher(args.teacher, {r["id"] for r in rows}) if args.teacher else {}
    params = [p for p in student.model.parameters() if p.requires_grad]
    # Gradients are all-reduced by hand after each accumulation window (DDP's reducer
    # asserted under non-reentrant gradient checkpointing with this readout).
    model = readout_module(student, torch)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay, fused=True)
    epochs = max(1, math.ceil(args.epochs))
    plans = [micro_batches(rows, args.tokens_per_batch, args.max_rows_per_batch, args.seed, e)
             for e in range(epochs)]
    per_epoch = all_min(torch, dist, world, min(len(p) for p in plans), device)
    total_micro = int(per_epoch * args.epochs)
    steps = total_micro // args.accumulation
    warmup = max(1, round(steps * args.warmup))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(args.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * (step - warmup)
                                                          / max(1, steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    first = 0
    if args.resume_state:
        state = json.loads(args.resume_state.read_text())
        if state["steps"] != steps:
            raise ValueError("The resume state belongs to a different schedule.")
        first = state["step"] + 1
        for _ in range(first):
            scheduler.step()
    log = (args.out / "train-log.jsonl").open("a") if rank == 0 else None
    if log and first:
        log.write(json.dumps({"resumed_at_step": first, "optimizer": "re-initialised"}) + "\n")
    stats = {"rows": len(rows), "teacher_rows": len(teacher), "micro_batches_per_epoch": per_epoch,
             "steps": steps}
    if log:
        log.write(json.dumps({"rank0": stats, "world": world}) + "\n")
        log.flush()
    started, seen_tokens, seen_rows = time.time(), 0, 0
    model.train()
    step = first - 1
    for step in range(first, steps):
        tick = time.time()
        loss_sum, term_count, question_count = 0.0, 0, 0
        for micro in range(args.accumulation):
            index = step * args.accumulation + micro
            epoch, position = divmod(index, per_epoch)
            batch = [rows[i] for i in plans[epoch % len(plans)][position]]
            with _null():
                logits = model(batch)
                # Keeps the graph connected when no row of the batch has a loss term.
                total = 0.0 * sum(z.sum() for z in logits)
                for z, row in zip(logits, batch, strict=True):
                    term, count = row_loss(torch, z, row, teacher.get(row["id"]), args)
                    total = total + term
                    term_count += count
                # Mean over this rank's questions of the step; ranks are averaged below.
                (total / (len(batch) * args.accumulation)).backward()
            loss_sum += float(total.detach())
            question_count += len(batch)
            seen_tokens += sum(len(r["input_ids"]) for r in batch)
        if world > 1:
            works = []
            for parameter in params:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                works.append(dist.all_reduce(parameter.grad, async_op=True))
            for work in works:
                work.wait()
            for parameter in params:
                parameter.grad.div_(world)
        norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        seen_rows += question_count
        if rank == 0 and args.save_every and (step + 1) % args.save_every == 0 and step + 1 < steps:
            checkpoint(student, args, step, steps)
        if log and (step % args.log_every == 0 or step == steps - 1):
            log.write(json.dumps({
                "step": step, "steps": steps, "loss": loss_sum / max(question_count, 1),
                "grad_norm": norm, "lr": scheduler.get_last_lr()[0],
                "seconds": time.time() - tick, "elapsed": time.time() - started,
                "rank0_tokens": seen_tokens, "rank0_questions": seen_rows,
                "rank0_tokens_per_second": seen_tokens / max(time.time() - started, 1e-6),
                "max_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            }) + "\n")
            log.flush()
        stop = torch.tensor(float(bool(args.max_seconds)
                                  and time.time() - started > args.max_seconds), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            if log:
                log.write(json.dumps({"stopped_at_step": step, "reason": "max_seconds"}) + "\n")
            break
    if rank == 0:
        save_model(student, args.out / "model", args.model_dir)
        (args.out / "train-summary.json").write_text(json.dumps({
            "steps_planned": steps, "steps_run": step + 1, "resumed_at_step": first or None,
            "world_size": world, "accumulation": args.accumulation,
            "tokens_per_batch": args.tokens_per_batch, "lr": args.lr, "epochs": args.epochs,
            "kd_weight": args.kd_weight, "gold_weight": args.gold_weight,
            "teacher_temperature": args.teacher_temperature, "lora_rank": args.lora_rank,
            "full_finetune": not student.lora, "seed": args.seed,
            "trainable_parameters": sum(p.numel() for p in params),
            "seconds": time.time() - started, "rank0_tokens": seen_tokens,
            "rank0_questions": seen_rows, "rank0_rows": len(rows),
            "rank0_teacher_rows": len(teacher),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        }, indent=2) + "\n")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def checkpoint(student, args, step: int, steps: int) -> None:
    import shutil

    folder = args.out / "checkpoint"
    partial = folder.with_name("checkpoint.partial")
    if partial.exists():
        shutil.rmtree(partial)
    save_model(student, partial, args.model_dir)
    (partial / "state.json").write_text(json.dumps({"step": step, "steps": steps}) + "\n")
    if folder.exists():
        shutil.rmtree(folder)
    partial.rename(folder)


def evaluate(args, student, dist, world, rank, torch) -> None:
    from bobcat.schema import json_hash

    student.model.eval()
    for path in args.eval:
        rows = []
        with path.open() as stream:
            for index, line in enumerate(stream):
                if index % world == rank:
                    record = json.loads(line)
                    record["input_ids"] = np.asarray(record["input_ids"], dtype=np.int32)
                    rows.append(record)
        mine = []
        batches = micro_batches(rows, args.eval_tokens_per_batch, 64, 0, 0)
        with torch.inference_mode():
            for batch_index in batches:
                batch = [rows[i] for i in batch_index]
                tick = time.perf_counter()
                logits, masses = student.forward(batch, need_mass=True)
                seconds = (time.perf_counter() - tick) / len(batch)
                for row, z, mass in zip(batch, logits, masses, strict=True):
                    mine.append({"id": row["id"], "input_sha256": row.get("input_sha256")
                                 or json_hash(row["input_ids"].tolist()),
                                 "option_ids": row["option_ids"], "logits": z.tolist(),
                                 "candidate_log_mass": mass, "seconds": seconds,
                                 "input_tokens": len(row["input_ids"])})
        gathered = [None] * world
        if world > 1:
            dist.all_gather_object(gathered, mine)
        else:
            gathered = [mine]
        if rank == 0:
            name = path.name.replace(".compiled.jsonl", "").replace(".jsonl", "")
            with (args.out / f"logits-{name}.jsonl").open("w") as stream:
                for part in gathered:
                    for record in part:
                        stream.write(json.dumps(record) + "\n")
    if rank == 0:
        (args.out / "runtime.json").write_text(json.dumps({
            "loader": f"bobcat.flash_train ({'lora' if student.lora else 'full'})",
            "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                            "visible_gpus": world, "dtype": "bfloat16 autocast"},
            "seconds": {"load": 0.0, "evaluate": 0.0},
        }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="pinned base download")
    parser.add_argument("--init", type=Path, help="start from a saved model / adapter folder")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--teacher", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", default=[])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0, help="0 = full fine-tuning")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--max-rows-per-batch", type=int, default=64)
    parser.add_argument("--eval-tokens-per-batch", type=int, default=32768)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--gold-weight", type=float, default=0.5)
    parser.add_argument("--teacher-temperature", type=float, default=1.1488982760285609)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume-state", type=Path, help="checkpoint/state.json (with --init)")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--limit", type=int, help="first N training rows (smoke tests)")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--freeze", help="full fine-tuning: regex of parameter names kept frozen")
    args = parser.parse_args()
    if args.eval_only == bool(args.train):
        parser.error("Give --train, or --eval-only without it.")
    if args.resume_state and not args.init:
        parser.error("--resume-state needs --init pointing at the checkpoint folder.")
    args.out.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
