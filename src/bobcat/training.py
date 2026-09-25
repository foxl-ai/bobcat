from __future__ import annotations

import contextlib
import json
import math
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bobcat.batching import EncodedDataset
from bobcat.metrics import evaluate_rows
from bobcat.model import DecisionModel, ModelConfig
from bobcat.paired import counterfactual_loss
from bobcat.schema import file_hash, read_examples
from bobcat.tokenization import MASK, SPECIAL, ScratchTokenizer


@dataclass
class TrainConfig:
    steps: int = 600
    groups_per_batch: int = 8
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    warmup_steps: int = 30
    seed: int = 17
    device: str = "auto"
    precision: str = "fp32"
    checkpoint_every: int = 200
    log_every: int = 25
    max_train_seconds: float | None = None
    objective: str = "decision"
    group_strategy: str = "context"
    counterfactual_weight: float = 0.0


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        # The Mac stays usable; opt in to MPS explicitly.
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16" or device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("Use fp32, or bf16 on a supporting CUDA GPU.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: DecisionModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_config: TrainConfig,
    provenance: dict,
    counters: dict,
) -> None:
    payload = {
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "train_config": asdict(train_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "provenance": provenance,
        "counters": counters,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path: Path, device: torch.device) -> tuple[DecisionModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["provenance"]["weight_origin"] != "random_initialization":
        raise ValueError("Only project-trained random-origin checkpoints are accepted.")
    model = DecisionModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device), checkpoint


@torch.inference_mode()
def predict(
    model: DecisionModel,
    dataset: EncodedDataset,
    device: torch.device,
    precision: str = "fp32",
    batch_size: int = 48,
) -> list[dict]:
    model.eval()
    rows = []
    for batch in dataset.evaluation_batches(batch_size):
        batch = batch.to(device)
        with autocast_context(device, precision):
            logits = model(batch.model_inputs()).float().cpu()
        for example, ids, values in zip(batch.examples, batch.candidate_ids, logits, strict=True):
            rows.append(
                {
                    "id": example.id,
                    "group_id": example.group_id,
                    "family": example.family,
                    "kind": example.kind,
                    "split": example.split,
                    "pair_id": example.pair_id,
                    "variant": example.variant,
                    "target": example.target,
                    "candidate_ids": ids,
                    "logits": values[: len(ids)].tolist(),
                }
            )
    return rows


def mask_tokens(ids: torch.Tensor, vocab_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    eligible = ids.ge(len(SPECIAL))
    selected = (torch.rand(ids.shape, generator=generator) < 0.15) & eligible
    if not selected.any():
        coordinates = eligible.nonzero()
        if not len(coordinates):
            raise ValueError("MLM batch has no ordinary tokens.")
        selected[tuple(coordinates[0])] = True
    corrupted = ids.clone()
    method = torch.rand(ids.shape, generator=generator)
    corrupted[selected & (method < 0.8)] = MASK
    random_ids = torch.randint(len(SPECIAL), vocab_size, ids.shape, generator=generator)
    replace = selected & (method >= 0.8) & (method < 0.9)
    corrupted[replace] = random_ids[replace]
    return corrupted, selected


def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_path: Path,
    tokenizer_path: Path,
    output: Path,
    *,
    dev_path: Path | None = None,
    resume: Path | None = None,
    init_from: Path | None = None,
    stop_after_steps: int | None = None,
) -> dict:
    if train_config.steps < 1 or train_config.objective not in {"decision", "mlm"}:
        raise ValueError("Invalid step count or training objective.")
    if (
        train_config.counterfactual_weight < 0
        or not math.isfinite(train_config.counterfactual_weight)
        or (train_config.counterfactual_weight > 0 and train_config.objective != "decision")
    ):
        raise ValueError(
            "Counterfactual supervision needs a nonnegative weight and decision training."
        )
    if train_config.max_train_seconds is not None and train_config.max_train_seconds <= 0:
        raise ValueError("Training time budget must be positive.")
    if (output / "last.pt").exists() and resume is None:
        raise ValueError("Run already exists; use --resume or choose a new output path.")
    if resume is not None and init_from is not None:
        raise ValueError("Resume and initialization are mutually exclusive.")
    output.mkdir(parents=True, exist_ok=True)
    device = pick_device(train_config.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_everything(train_config.seed)
    tokenizer = ScratchTokenizer(tokenizer_path)
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary does not match the scratch tokenizer.")
    examples = read_examples(train_path)
    if any(e.supervision != "hard_label" for e in examples):
        raise ValueError("Observed ordinal means require bobcat.decision_train.")
    if any(example.split != "train" for example in examples):
        raise ValueError("Training accepts only train records.")
    dataset = EncodedDataset(examples, tokenizer, model_config)
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    provenance = {
        "weight_origin": "random_initialization",
        "tokenizer_sha256": tokenizer.digest,
        "training_data_sha256": file_hash(train_path),
        "torch_version": str(torch.__version__),
        "initialization_checkpoint_sha256": (
            saved["provenance"]["initialization_checkpoint_sha256"]
            if saved is not None
            else file_hash(init_from) if init_from else None
        ),
    }
    if dev_path is not None:
        dev_examples = read_examples(dev_path)
        if any(e.supervision != "hard_label" for e in dev_examples):
            raise ValueError("Observed ordinal means require bobcat.public_eval.")
        if any(not e.split.startswith("dev_") for e in dev_examples):
            raise ValueError("Training-time evaluation may only use development partitions.")
        if {e.group_id for e in examples} & {e.group_id for e in dev_examples}:
            raise ValueError("Train/development world leakage.")
        dev = EncodedDataset(dev_examples, tokenizer, model_config)
        provenance["development_data_sha256"] = file_hash(dev_path)
    else:
        dev = None
    if init_from:
        model, initial = load_model(init_from, device)
        if model.config.to_dict() != model_config.to_dict():
            raise ValueError("MLM initialization must have the same architecture.")
        if initial["provenance"]["tokenizer_sha256"] != tokenizer.digest:
            raise ValueError("Initialization used a different tokenizer.")
    else:
        model = DecisionModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    start_step = 0
    counters = {
        "optimizer_seconds": 0.0,
        "loop_seconds": 0.0,
        "input_tokens": 0,
        "padded_tokens": 0,
        "examples_seen": 0,
        "contexts_encoded": 0,
        "counterfactual_pairs_seen": 0,
    }
    if saved is not None:
        expected = saved["provenance"]
        if expected != provenance or saved["model_config"] != model_config.to_dict():
            raise ValueError("Resume provenance/configuration mismatch.")
        if saved["train_config"] != asdict(train_config):
            raise ValueError("Exact resume requires the same training configuration.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_step, counters = saved["step"], saved["counters"]
    save_json(
        output / "config.json",
        {
            "model": model_config.to_dict(),
            "training": asdict(train_config),
            "provenance": provenance,
        },
    )
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous_handlers = {}
    for sig in [signal.SIGTERM, signal.SIGINT]:
        previous_handlers[sig] = signal.signal(sig, request_stop)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_loop_seconds = counters["loop_seconds"]
    started = time.perf_counter()
    last_loss = None
    step = start_step
    try:
        model.train()
        while step < train_config.steps:
            if stopped or (stop_after_steps is not None and step >= stop_after_steps):
                break
            if (
                train_config.max_train_seconds is not None
                and counters["loop_seconds"] >= train_config.max_train_seconds
            ):
                break
            indices = dataset.training_indices(
                step, train_config.groups_per_batch, train_config.seed, train_config.group_strategy
            )
            batch = dataset.collate(indices, shuffle_seed=train_config.seed + step)
            if train_config.objective == "mlm":
                clean = batch.tensors["context_ids"]
                corrupted, selected = mask_tokens(
                    clean, tokenizer.vocab_size, train_config.seed + step * 101
                )
                clean, corrupted, selected = (
                    clean.to(device),
                    corrupted.to(device),
                    selected.to(device),
                )
            else:
                batch = batch.to(device)
            warmup = min(1.0, (step + 1) / max(1, train_config.warmup_steps))
            if train_config.max_train_seconds is None:
                progress = max(0, step - train_config.warmup_steps) / max(
                    1, train_config.steps - train_config.warmup_steps
                )
                decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                # A time-budget comparison cannot assume the number of completed steps.
                decay = 1.0
            for group in optimizer.param_groups:
                group["lr"] = train_config.learning_rate * warmup * decay
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            compute_started = time.perf_counter()
            with autocast_context(device, train_config.precision):
                if train_config.objective == "mlm":
                    logits = model.mlm_logits(corrupted, selected)
                    loss = nn.functional.cross_entropy(logits.float(), clean[selected])
                else:
                    logits = model(batch.model_inputs())
                    loss = nn.functional.cross_entropy(logits.float(), batch.tensors["targets"])
                    if train_config.counterfactual_weight > 0:
                        contrast, count = counterfactual_loss(logits.float(), batch)
                        loss = loss + train_config.counterfactual_weight * contrast
                        counters["counterfactual_pairs_seen"] += count
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            counters["optimizer_seconds"] += time.perf_counter() - compute_started
            step += 1
            counters["input_tokens"] += (
                int(clean.ne(0).sum())
                if train_config.objective == "mlm"
                else batch.nonpadding_tokens
            )
            counters["padded_tokens"] += (
                clean.numel() if train_config.objective == "mlm" else batch.padded_tokens
            )
            counters["examples_seen"] += len(indices) if train_config.objective == "decision" else 0
            counters["contexts_encoded"] += batch.unique_contexts
            counters["loop_seconds"] = initial_loop_seconds + time.perf_counter() - started
            last_loss = float(loss.detach())
            if step % train_config.log_every == 0 or step == 1:
                event = {
                    "step": step,
                    "loss": last_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    **counters,
                }
                with (output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
                print(json.dumps(event), flush=True)
            if step % train_config.checkpoint_every == 0:
                save_checkpoint(
                    output / "last.pt", model, optimizer, step, train_config, provenance, counters
                )
        save_checkpoint(
            output / "last.pt", model, optimizer, step, train_config, provenance, counters
        )
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    report = {
        "completed_steps": step,
        "last_training_loss": last_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "precision": train_config.precision,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        **counters,
        "processed_tokens_per_optimizer_second": (
            counters["input_tokens"] / max(counters["optimizer_seconds"], 1e-9)
        ),
        "provenance": provenance,
        "comparison_note": (
            "Equal steps do not imply equal compute; inspect device time and tokens."
        ),
    }
    if dev is not None and train_config.objective == "decision":
        rows = predict(model, dev, device, train_config.precision)
        save_json(output / "development-predictions.json", {"rows": rows})
        report["development"] = evaluate_rows(rows)
    save_json(output / "result.json", report)
    return report
