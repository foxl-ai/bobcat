"""LoRA training of a pretrained student on typed decisions (PLAN step 3).

Inputs are rows compiled by `student_data compile` or `student_readout compile --piecewise`:
token IDs, the offered identifier IDs, each candidate's last-token position and supervision.
Nothing is generated; the model is read at fixed positions.

Readouts
  vocab    the original LM head at the last position, restricted to the offered identifiers
           (the zero-shot readout).
  pointer  vocab logits plus a residual candidate pointer: a bilinear score between the last
           hidden state and each candidate's last hidden state. The pointer starts at zero,
           so step 0 equals the vocab readout; it can correct identifier and list-position
           effects without depending on how identifiers tokenize.
Objectives (Choice/Noul hard labels; Score means always use the expected-level term)
  sft      categorical cross-entropy;
  brier    half multiclass Brier score;
  rl       on-policy REINFORCE with r = c - stop_grad(p(a)), 4 IID samples per question and a
           leave-one-out baseline. Its expected gradient equals half-Brier's; the arm tests
           the sampled estimator, not new information.
Runs under torchrun. Every rank holds the frozen model plus LoRA; trainable gradients are
all-reduced by hand after each accumulation window. Evaluation writes runtime logits that
`student_readout score --piecewise` turns into the standard summary.
  --eval-only   no update: evaluate the base model (zero LoRA) or a saved adapter;
  --head-only   freeze the backbone and any adapter and train only the pointer head, with the
                backbone run without gradients (tests the pointer without LoRA competing).
  --keep-order  read training rows in file order (a builder arranged them, e.g. so long rows
                share a step across ranks); the default shuffles with --seed.
  --save-every  rank 0 writes out/checkpoint/{adapter,state.pt} every N steps; `--init
                out/checkpoint/adapter --resume out/checkpoint/state.pt` continues at the next
                step with the same row cursor, optimizer moments and learning-rate schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def decision_loss(torch, logits, row: dict, objective: str, generator=None):
    """Loss for one question. `logits` is the 1-D candidate score vector (float32)."""
    count = logits.shape[0]
    log_p = torch.log_softmax(logits, dim=-1)
    p = log_p.exp()
    if row["supervision"] == "score_mean":
        levels = torch.arange(count, dtype=logits.dtype, device=logits.device)
        expected = (p * levels).sum()
        return ((expected - row["score_target"]) / max(count - 1, 1)) ** 2
    target = row["target"]
    if objective == "sft":
        return -log_p[target]
    if objective == "brier":
        onehot = torch.zeros_like(p)
        onehot[target] = 1.0
        return 0.5 * ((p - onehot) ** 2).sum()
    if objective == "rl":
        actions = torch.multinomial(p.detach(), 4, replacement=True, generator=generator)
        reward = (actions == target).to(p.dtype) - p.detach()[actions]
        baseline = (reward.sum() - reward) / 3.0
        return -((reward - baseline) * log_p[actions]).mean()
    raise ValueError(f"Unknown objective {objective}")


class PointerHead:
    """Residual bilinear pointer over candidate end states; zero-initialised output."""

    @staticmethod
    def build(torch, hidden: int, width: int = 256, scale: float = 0.0):
        nn = torch.nn

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(hidden, width, bias=False)
                self.key = nn.Linear(hidden, width, bias=False)
                self.scale = nn.Parameter(torch.tensor(float(scale)))
                nn.init.normal_(self.query.weight, std=hidden ** -0.5)
                nn.init.normal_(self.key.weight, std=hidden ** -0.5)

            def forward(self, last, candidates):
                q = self.query(last.float())
                k = self.key(candidates.float())
                return self.scale * (k @ q) / math.sqrt(q.shape[-1])

        return Head()


class Student:
    def __init__(self, model_dir: Path, *, lora_rank: int, readout: str, adapter: Path | None,
                 device, torch, head_only: bool = False, frozen: bool = False,
                 pointer_scale: float = 0.0):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM

        self.torch, self.readout, self.device = torch, readout, device
        options = dict(dtype=torch.bfloat16, device_map={"": device})
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, **options)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_dir, **options)
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter / "lora",
                                              is_trainable=not (head_only or frozen))
        else:
            config = LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0,
                target_modules="all-linear",
                exclude_modules=r".*(visual|vision|merger|lm_head|mtp).*",
            )
            model = get_peft_model(model, config)
        self.no_backbone_grad = head_only or frozen
        if self.no_backbone_grad:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = model
        base = model.get_base_model()
        self.backbone = base.get_decoder() if hasattr(base, "get_decoder") else base.model
        self.lm_weight = base.get_output_embeddings().weight
        self.head = None
        if readout == "pointer":
            self.head = PointerHead.build(torch, self.lm_weight.shape[1],
                                          scale=pointer_scale).to(device)
            if adapter is not None and (adapter / "pointer.pt").exists():
                self.head.load_state_dict(torch.load(adapter / "pointer.pt", map_location=device))

    def trainable(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.head is not None:
            params += list(self.head.parameters())
        return params

    def scores(self, row: dict, *, need_mass: bool = False):
        torch = self.torch
        ids = torch.tensor([row["input_ids"]], device=self.device)
        if self.no_backbone_grad:
            with torch.no_grad():
                hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state[0]
        else:
            hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state[0]
        last = hidden[-1]
        options = torch.tensor(row["option_ids"], device=self.device)
        logits = (self.lm_weight[options].float() @ last.float())
        if self.head is not None:
            ends = torch.tensor(row["candidate_ends"], device=self.device)
            logits = logits + self.head(last, hidden[ends])
        mass = None
        if need_mass:
            full = torch.log_softmax(self.lm_weight.float() @ last.float(), dim=-1)
            mass = float(torch.logsumexp(full[options], dim=0))
        return logits, mass

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(folder / "lora")
        if self.head is not None:
            self.torch.save(self.head.state_dict(), folder / "pointer.pt")


def setup(torch):
    import torch.distributed as dist

    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    return dist, world, rank, torch.device("cuda", local)


def train(args) -> None:
    import torch

    dist, world, rank, device = setup(torch)
    torch.manual_seed(args.seed + rank)
    student = Student(args.model_dir, lora_rank=args.lora_rank, readout=args.readout,
                      adapter=args.init, device=device, torch=torch, head_only=args.head_only,
                      frozen=args.eval_only, pointer_scale=args.pointer_scale_init)
    if not args.eval_only:
        fit(args, student, dist, world, rank, device, torch)
    if world > 1:
        dist.barrier()
    if args.eval:
        evaluate(args, student, dist, world, rank, torch)
    if world > 1:
        dist.destroy_process_group()


def fit(args, student, dist, world, rank, device, torch) -> None:
    params = student.trainable()
    head_ids = {id(p) for p in (student.head.parameters() if student.head else [])}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in params if id(p) not in head_ids], "lr": args.lr},
        {"params": [p for p in params if id(p) in head_ids], "lr": args.head_lr},
    ], weight_decay=0.0)
    rows = load_rows(args.train)
    if not args.keep_order:
        random.Random(args.seed).shuffle(rows)
    per_step = args.accumulation * world
    steps = args.steps or math.ceil(len(rows) * args.epochs / per_step)
    warmup = max(1, round(steps * 0.03))

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    generator = torch.Generator(device=device).manual_seed(args.seed * 7919 + rank)
    log = (args.out / "train-log.jsonl").open("a") if rank == 0 else None
    started, tokens = time.time(), 0
    names = [[name, list(p.shape)] for name, p in trainable_names(student)]
    first = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state["names"] != names or state["steps"] != steps:
            raise ValueError("The resume state belongs to a different adapter or schedule.")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first = state["step"] + 1
        if log:
            log.write(json.dumps({"resumed_at_step": first}) + "\n")
    cursor = rank + first * per_step
    step = first - 1
    student.model.train()
    for step in range(first, steps):
        tick = time.time()
        losses = []
        for _ in range(args.accumulation):
            row = rows[cursor % len(rows)]
            cursor += world
            logits, _ = student.scores(row)
            loss = decision_loss(torch, logits, row, args.objective, generator)
            (loss / args.accumulation).backward()
            losses.append(float(loss.detach()))
            tokens += len(row["input_ids"])
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if world > 1:
                dist.all_reduce(p.grad)
                p.grad /= world
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0 and args.save_every and (step + 1) % args.save_every == 0 \
                and step + 1 < steps:
            save_checkpoint(torch, student, optimizer, scheduler, args.out / "checkpoint",
                            step=step, steps=steps, names=names)
        if log and (step % args.log_every == 0 or step == steps - 1):
            log.write(json.dumps({
                "step": step, "steps": steps, "loss": sum(losses) / len(losses),
                "grad_norm": norm, "lr": scheduler.get_last_lr()[0],
                "seconds": time.time() - tick, "elapsed": time.time() - started,
                "rank0_tokens": tokens,
                "max_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            }) + "\n")
            log.flush()
        # All ranks must stop at the same step, or the next all-reduce deadlocks.
        stop = torch.tensor(float(bool(args.max_seconds)
                                  and time.time() - started > args.max_seconds), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            if rank == 0:
                log.write(json.dumps({"stopped_at_step": step, "reason": "max_seconds"}) + "\n")
            break
    if rank == 0:
        student.save(args.out / "adapter")
        (args.out / "train-summary.json").write_text(json.dumps({
            "objective": args.objective, "readout": args.readout, "steps_planned": steps,
            "steps_run": step + 1, "examples_seen": (step + 1) * per_step,
            "world_size": world, "accumulation": args.accumulation, "lr": args.lr,
            "head_lr": args.head_lr, "lora_rank": args.lora_rank, "seed": args.seed,
            "init": str(args.init) if args.init else None, "head_only": args.head_only,
            "resumed_at_step": first or None, "keep_order": args.keep_order,
            "pointer_scale_init": args.pointer_scale_init,
            "pointer_scale_final": (float(student.head.scale) if student.head else None),
            "trainable_parameters": sum(p.numel() for p in params),
            "seconds": time.time() - started, "rank0_tokens": tokens,
        }, indent=2) + "\n")


def trainable_names(student):
    named = [(n, p) for n, p in student.model.named_parameters() if p.requires_grad]
    if student.head is not None:
        named += [(f"head.{n}", p) for n, p in student.head.named_parameters()]
    return named


def save_checkpoint(torch, student, optimizer, scheduler, folder: Path, *, step: int,
                    steps: int, names: list) -> None:
    """Adapter plus optimizer/schedule state, replaced atomically (a reader never sees a
    half-written checkpoint)."""
    partial = folder.with_name(folder.name + ".partial")
    if partial.exists():
        import shutil

        shutil.rmtree(partial)
    student.save(partial / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "steps": steps, "names": names}, partial / "state.pt")
    (partial / "step.json").write_text(json.dumps({"step": step, "steps": steps}) + "\n")
    if folder.exists():
        import shutil

        shutil.rmtree(folder)
    partial.rename(folder)


def evaluate(args, student, dist, world, rank, torch) -> None:
    student.model.eval()
    for path in args.eval:
        rows = load_rows(path)
        mine = []
        with torch.inference_mode():
            for index in range(rank, len(rows), world):
                row = rows[index]
                tick = time.perf_counter()
                logits, mass = student.scores(row, need_mass=True)
                mine.append({"id": row["id"], "input_sha256": row["input_sha256"],
                             "option_ids": row["option_ids"], "logits": logits.tolist(),
                             "candidate_log_mass": mass,
                             "seconds": time.perf_counter() - tick,
                             "input_tokens": len(row["input_ids"])})
        gathered = [None] * world
        if world > 1:
            dist.all_gather_object(gathered, mine)
        else:
            gathered = [mine]
        if rank == 0:
            name = path.stem.replace(".compiled", "")
            with (args.out / f"logits-{name}.jsonl").open("w") as stream:
                for part in gathered:
                    for record in part:
                        stream.write(json.dumps(record) + "\n")
    if rank == 0:
        (args.out / "runtime.json").write_text(json.dumps({
            "loader": f"bobcat.student_train ({args.readout}, {args.objective})",
            "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                            "visible_gpus": world, "dtype": "bfloat16 + LoRA"},
            "seconds": {"load": 0.0, "evaluate": 0.0},
        }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", default=[])
    parser.add_argument("--readout", choices=("vocab", "pointer"), default="vocab")
    parser.add_argument("--objective", choices=("sft", "brier", "rl"), default="sft")
    parser.add_argument("--init", type=Path, help="continue from a saved adapter folder")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--pointer-scale-init", type=float, default=0.0)
    parser.add_argument("--keep-order", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="state.pt written by --save-every")
    args = parser.parse_args()
    if args.resume and not args.init:
        parser.error("--resume needs --init pointing at the checkpoint's adapter folder.")
    if args.eval_only == bool(args.train):
        parser.error("Give --train, or --eval-only without it.")
    if args.head_only and args.readout != "pointer":
        parser.error("--head-only trains the pointer head; use --readout pointer.")
    args.out.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
