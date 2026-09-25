"""Fit a small residual head on verified, frozen GLM features.

This CPU-sized study updates no GLM backbone weights. It uses context-weighted
supervision from train only, reports internal development separately, and keeps
model/optimizer/cursor state so a finite run can resume without starting over.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from bobcat.checkpoints import S3CheckpointStore, verify_checkpoint, write_checkpoint
from bobcat.corpus import atomic_json
from bobcat.glm_feature_data import MIXED_PLAN_SCHEMA, RUN_SCHEMA, validate_plan
from bobcat.glm_features import FEATURE_PROFILE, ResidualDecisionHead
from bobcat.metrics import evaluate_rows
from bobcat.schema import file_hash
from bobcat.supervision import MEAN_TARGET, evaluate_supervised, target_tensors
from bobcat.supervision import losses as supervised_losses

FORMAT = "bobcat-glm-residual-v1"


@dataclass
class Features:
    hidden: torch.Tensor
    base_logits: torch.Tensor
    candidate_keep: torch.Tensor
    targets: torch.Tensor
    score_targets: torch.Tensor
    weights: torch.Tensor
    rows: list[dict]
    groups: dict[str, list[list[int]]]
    provenance: dict


def load_features(plan: dict, root: Path) -> Features:
    validate_plan(plan)
    manifest = json.loads((root / "run.json").read_text())
    provenance = manifest.get("scorer_provenance", {})
    if (manifest.get("schema") != RUN_SCHEMA or manifest.get("status") != "completed"
            or manifest.get("plan_sha256") != plan["content_sha256"]
            or manifest.get("completed_groups") != plan["group_count"]
            or manifest.get("completed_questions") != plan["question_count"]
            or manifest.get("calibration_and_public_validation_included") is not False
            or provenance.get("feature_profile") != FEATURE_PROFILE
            or provenance.get("feature_identity_checked_on_every_request") is not True
            or provenance.get("base_weights_updated") is not False
            or not provenance.get("option_head_sha256")):
        raise ValueError("Use complete, identity-verified GLM features from the frozen plan.")
    expected = [row for group in plan["groups"] for row in group["rows"]]
    hidden, bases, masks, targets, means, weights, rows = [], [], [], [], [], [], []
    filenames = set()
    width = max(len(row["candidate_ids"]) for row in expected)
    for record in manifest["files"]:
        name = record["path"]
        path = root / name
        if (Path(name).name != name or name in filenames or path.is_symlink()
                or path.stat().st_size != record["bytes"]
                or file_hash(path) != record["sha256"]):
            raise ValueError("Feature shard checksum or path is invalid.")
        filenames.add(name)
        shard = torch.load(path, map_location="cpu", weights_only=True)
        count = record["questions"]
        batch_rows = expected[len(rows):len(rows) + count]
        if (shard.get("schema") != "bobcat-glm-frozen-feature-shard-v1"
                or shard.get("plan_sha256") != plan["content_sha256"]
                or shard["rows"] != batch_rows or len(batch_rows) != count
                or count < 1):
            raise ValueError("Feature row order or source identity changed.")
        h, base, keep = shard["hidden"], shard["base_logits"], shard["candidate_keep"]
        target, weight = shard["targets"], shard["context_weights"]
        expected_target, expected_mean = target_tensors(batch_rows)
        mean = shard.get("score_targets")
        if mean is None:
            if plan["schema"] == MIXED_PLAN_SCHEMA or expected_target.eq(MEAN_TARGET).any():
                raise ValueError("Mixed-supervision feature shards must preserve ordinal means.")
            mean = torch.zeros(count, dtype=torch.float64)
        counts = torch.tensor([len(r["candidate_ids"]) for r in batch_rows])
        if (h.shape != (count, 4096) or h.dtype != torch.float32
                or not torch.isfinite(h).all()
                or base.ndim != 2 or base.shape[0] != count or base.shape[1] > width
                or base.dtype != torch.float32 or keep.shape != base.shape
                or keep.dtype != torch.bool
                or not torch.equal(keep, torch.arange(base.shape[1])[None] < counts[:, None])
                or not torch.isfinite(base[keep]).all() or not torch.isneginf(base[~keep]).all()
                or target.dtype != torch.long or not torch.equal(target, expected_target)
                or mean.dtype != torch.float64 or not torch.equal(mean, expected_mean)
                or weight.dtype != torch.float64 or not torch.equal(weight, torch.tensor([
                    r["context_weight"] for r in batch_rows
                ], dtype=torch.float64))):
            raise ValueError("Feature tensors, labels or candidate masks are misaligned.")
        hidden.append(h)
        bases.append(F.pad(base, (0, width - base.shape[1]), value=float("-inf")))
        masks.append(F.pad(keep, (0, width - keep.shape[1]), value=False))
        targets.append(target)
        means.append(mean)
        weights.append(weight)
        rows.extend(batch_rows)
    if rows != expected or not filenames:
        raise ValueError("The feature run omitted planned examples.")
    groups = defaultdict(list)
    offset = 0
    for group in plan["groups"]:
        size = len(group["rows"])
        groups[group["split"]].append(list(range(offset, offset + size)))
        offset += size
    if not groups["train"] or not groups["dev_train"]:
        raise ValueError("Both training and internal development are required.")
    return Features(
        torch.cat(hidden), torch.cat(bases), torch.cat(masks),
        torch.cat(targets), torch.cat(means), torch.cat(weights), rows, dict(groups),
        {
            "plan_sha256": plan["content_sha256"],
            "feature_manifest_sha256": file_hash(root / "run.json"),
            "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "scorer": provenance,
        },
    )


@torch.inference_mode()
def development(data: Features, model: ResidualDecisionHead | None = None) -> dict:
    indices = [i for group in data.groups["dev_train"] for i in group]
    rows = []
    for start in range(0, len(indices), 256):
        batch = indices[start:start + 256]
        values = (data.base_logits[batch] if model is None else model(
            data.hidden[batch], data.base_logits[batch], data.candidate_keep[batch],
        ))
        for i, logits in zip(batch, values, strict=True):
            row = data.rows[i]
            rows.append({**row, "logits": logits[:len(row["candidate_ids"])].tolist()})
    hard_rows = [r for r in rows if r.get("supervision", "hard_label") == "hard_label"]
    by_family = {
        family: evaluate_supervised([r for r in rows if r["family"] == family])
        for family in sorted({r["family"] for r in rows})
    }
    correctness = defaultdict(list)
    for row in hard_rows:
        prediction = row["candidate_ids"][max(
            range(len(row["logits"])), key=row["logits"].__getitem__,
        )]
        correctness[row["group_id"]].append(float(prediction == row["target"]))
    return {
        "scope": "Internal task development; not calibration, final or generalization proof.",
        "temperature": 1.0, "calibration_fitted": False,
        "context_weighted_accuracy": sum(
            sum(values) / len(values) for values in correctness.values()
        ) / len(correctness) if correctness else None,
        "independent_contexts": len({r["group_id"] for r in rows}),
        "hard_label_contexts": len(correctness), "questions": len(rows),
        "overall": evaluate_rows(hard_rows), "supervision": evaluate_supervised(rows),
        "by_family": by_family, "rows": rows,
    }


def train(plan: dict, features: Path, out: Path, *, epochs: int = 3, rank: int = 32,
          learning_rate: float = 0.0005, batch_contexts: int = 64, seed: int = 20260922,
          max_seconds: float = 900, save_every: int = 32, resume: Path | None = None,
          stop_after_steps: int | None = None, remote=None) -> dict:
    if (out.exists() or not 1 <= epochs <= 20 or not 1 <= rank <= 128
            or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.1
            or not 1 <= batch_contexts <= 512 or not 0 < max_seconds <= 3600
            or not 1 <= save_every <= 256
            or (stop_after_steps is not None and stop_after_steps < 1)):
        raise ValueError("Use a new output directory and a finite head-training recipe.")
    started = time.monotonic()
    data = load_features(plan, features)
    torch.manual_seed(seed)
    model = ResidualDecisionHead(rank=rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    config = {"hidden_size": 4096, "max_choices": 255, "rank": rank}
    recipe = {
        "epochs": epochs, "learning_rate": learning_rate, "batch_contexts": batch_contexts,
        "seed": seed, "weight_decay": 0.01, "gradient_clip": 1.0,
        "loss": ("context_weighted_cross_entropy_and_observed_mean_mse"
                 if data.targets.eq(MEAN_TARGET).any() else "context_weighted_cross_entropy"),
        "mean_loss_scale": 1.0, "ordinal_mean_units": "original_level_index",
        "mean_supervision_identifies_distribution": False,
        "device": "cpu", "dtype": "float32",
    }
    provenance = {
        **data.provenance, "trainer_sha256": file_hash(Path(__file__)),
        "head_source_sha256": file_hash(Path(__file__).with_name("glm_features.py")),
        "supervision_source_sha256": file_hash(Path(__file__).with_name("supervision.py")),
        "head_config": config, "recipe": recipe,
        "initialization": "base logits plus zero-initialized nonlinear residual",
        "base_weights_updated": False, "calibration_fitted": False,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    step, epoch, cursor = 0, 0, 0
    counters = {"contexts": 0, "questions": 0}
    if resume:
        verify_checkpoint(resume, expected_format=FORMAT)
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved["format"] != FORMAT or saved["provenance"] != provenance:
            raise ValueError("Resume requires identical data, source and optimizer recipe.")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        step, epoch, cursor = saved["step"], saved["epoch"], saved["cursor"]
        counters = saved["counters"]
        if not 0 <= epoch <= epochs or not 0 <= cursor < len(data.groups["train"]):
            raise ValueError("Invalid saved training cursor.")
    out.mkdir(parents=True)
    manifest = {
        "schema": "bobcat-glm-head-training-run-v1", "status": "training",
        "provenance": provenance, "max_seconds": max_seconds,
        "resume_sha256": file_hash(resume) if resume else None,
        "backbone_training_performed": False, "release_gate_passed": False,
    }
    atomic_json(out / "run.json", manifest)
    atomic_json(out / "base-development.json", development(data))

    def save():
        payload = {
            "format": FORMAT, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "counters": dict(counters),
            "epoch": epoch, "cursor": cursor, "rng": torch.get_rng_state(),
            "provenance": provenance,
        }
        path = out / "last.pt"
        record = write_checkpoint(payload, path)
        if remote:
            record["remote"] = remote.publish(path, record)
            atomic_json(path.resolve().with_suffix(".json"), record)
        manifest["last_checkpoint"] = record
        manifest.update(step=step, epoch=epoch, cursor=cursor, counters=dict(counters))
        atomic_json(out / "run.json", manifest)

    try:
        with (out / "training.jsonl").open("x") as log:
            while epoch < epochs:
                if time.monotonic() - started >= max_seconds:
                    manifest["status"] = "deadline"
                    break
                if stop_after_steps is not None and step >= stop_after_steps:
                    manifest["status"] = "step_limit"
                    break
                order = list(range(len(data.groups["train"])))
                random.Random(seed + epoch).shuffle(order)
                batch_groups = order[cursor:cursor + batch_contexts]
                indices = [i for group in batch_groups for i in data.groups["train"][group]]
                optimizer.zero_grad(set_to_none=True)
                # Targets/metadata never enter the residual head.
                logits = model(
                    data.hidden[indices], data.base_logits[indices], data.candidate_keep[indices],
                )
                losses = supervised_losses(
                    logits, data.targets[indices], data.score_targets[indices],
                )
                loss = (losses.double() * data.weights[indices]).sum() / len(batch_groups)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite GLM residual training loss.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite GLM residual gradient.")
                optimizer.step()
                step += 1
                counters["contexts"] += len(batch_groups)
                counters["questions"] += len(indices)
                cursor += len(batch_groups)
                if cursor == len(order):
                    epoch, cursor = epoch + 1, 0
                log.write(json.dumps({
                    "step": step, "completed_epochs": epoch, "next_context": cursor,
                    "loss": float(loss.detach()), "gradient_norm": float(norm),
                    "seconds": time.monotonic() - started,
                }, allow_nan=False) + "\n")
                log.flush()
                if step % save_every == 0:
                    save()
            else:
                manifest["status"] = "completed"
        save()
        atomic_json(out / "head-development.json", development(data, model))
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["head_optimizer_steps"] = step
        atomic_json(out / "run.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--remote")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--batch-contexts", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = train(
        json.loads(args.plan.read_text()), args.features, args.out, resume=args.resume,
        epochs=args.epochs, rank=args.rank, learning_rate=args.learning_rate,
        batch_contexts=args.batch_contexts, max_seconds=args.max_seconds,
        remote=S3CheckpointStore(args.remote) if args.remote else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
