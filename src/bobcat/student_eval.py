"""Measure a verified research checkpoint on unchanged public and architecture suites."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from bobcat.architecture_probes import run_suite
from bobcat.checkpoints import verify_checkpoint
from bobcat.corpus import atomic_json
from bobcat.public_eval import run as public_run
from bobcat.schema import file_hash
from bobcat.serve import StudentScorer


def run(checkpoint: Path, tokenizer: Path, public_suite: Path, out: Path, *,
        device: str = "cuda", architecture_suite: Path | None = None,
        max_seconds: float = 1800, external_suite: Path | None = None) -> dict:
    if out.exists() or not 0 < max_seconds <= 3600:
        raise ValueError("Use a new evaluation directory with at most one hour.")
    started = time.monotonic()
    target = checkpoint.resolve(strict=True)
    identity = verify_checkpoint(target)
    if identity["format"] not in {"bobcat-real-mlm-v1", "bobcat-real-decisions-v1"}:
        raise ValueError("Evaluate an actual Bobcat language or decision checkpoint.")
    if device not in {"cuda", "cpu"}:
        raise ValueError("Use CUDA, or explicit CPU software-test mode.")
    if device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("The GPU evaluation requires BF16-capable CUDA.")
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    out.mkdir(parents=True)
    record = {
        "schema": "bobcat-student-development-v1", "status": "loading",
        "started_at": datetime.now(UTC).isoformat(), "max_seconds": max_seconds,
        "checkpoint_sha256": identity["sha256"], "checkpoint_format": identity["format"],
        "checkpoint_step": identity["step"], "checkpoint_counters": identity["counters"],
        "tokenizer_sha256": file_hash(tokenizer), "evaluator_sha256": file_hash(Path(__file__)),
        "device": device, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "gpu_count_used": 1 if device == "cuda" else 0,
        "visible_gpu_count": torch.cuda.device_count() if device == "cuda" else 0,
        "final_evaluation": False, "release_gate_passed": False,
        "calibration_refitted": False, "evaluations": {},
    }
    atomic_json(out / "evaluation.json", record)
    try:
        scorer = StudentScorer(target, tokenizer, device=device, allow_unvalidated=True)
        record["scorer_provenance"] = scorer.provenance
        suites = [("public", public_suite, public_run)]
        if external_suite:
            suites.append(("korean_external", external_suite, public_run))
        if architecture_suite:
            suites.append(("architecture", architecture_suite, run_suite))
        for index, (name, path, evaluator) in enumerate(suites):
            remaining = max_seconds - (time.monotonic() - started)
            if remaining < 1:
                record["evaluations"][name] = {"status": "not_run_deadline"}
                continue
            record["status"] = "evaluating_" + name
            atomic_json(out / "evaluation.json", record)
            suite = json.loads(path.read_text())
            result = evaluator(
                suite, scorer, out / name,
                max_seconds=min(1800, remaining / (len(suites) - index)),
            )
            record["evaluations"][name] = result
        record["status"] = (
            "completed" if all(
                item["status"] == "completed"
                and not item.get("failed_questions") and not item.get("failed_cases")
                for item in record["evaluations"].values()
            ) else "completed_with_failures_or_deadline"
        )
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        record.update(
            finished_at=datetime.now(UTC).isoformat(),
            wall_seconds=time.monotonic() - started,
            peak_gpu_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        )
        atomic_json(out / "evaluation.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "tokenizer", "public-suite", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--architecture-suite", type=Path)
    parser.add_argument("--external-suite", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    result = run(
        args.checkpoint, args.tokenizer, args.public_suite, args.out, device=args.device,
        architecture_suite=args.architecture_suite, max_seconds=args.max_seconds,
        external_suite=args.external_suite,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "checkpoint_sha256", "wall_seconds", "release_gate_passed",
    )}))


if __name__ == "__main__":
    main()
