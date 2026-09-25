"""Mine train-only errors with serial repeats; never relabel data with model outputs."""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np

from bobcat.checkpoint_metrics import enrich
from bobcat.corpus import atomic_json
from bobcat.glm_serving_checkpoint import CompiledReadout, distribution_comparison
from bobcat.glm_training_targets import verify_training_partition
from bobcat.schema import file_hash


def error_kind(row, prediction):
    if (row["split"] != "train" or prediction["id"] != row["id"]
            or prediction["input_sha256"] != row["input_sha256"]
            or len(prediction["logits"]) != len(row["candidate_ids"])
            or prediction["native_completion_tokens"] != 0):
        raise ValueError("Mine intact training inputs without substituting supervision.")
    scored = enrich({**row, **prediction})
    if row["supervision"] == "hard_label":
        return "categorical_error" if not scored["correct"] else None
    if row["supervision"] == "score_mean":
        logits = np.asarray(prediction["logits"], dtype=np.float64)
        probs = np.exp(logits - np.logaddexp.reduce(logits))
        mean = float(probs @ np.arange(len(probs)))
        normalized = abs(mean - row["score_mean"]) / (len(probs) - 1)
        return "ordinal_mean_error_over_0.2" if normalized > .2 else None
    raise ValueError("Unsupported observed supervision.")


def remaining_rows(rows, start_index):
    if type(start_index) is not int or not 0 <= start_index <= len(rows):
        raise ValueError("Continuation must identify an exact position in the frozen curriculum.")
    return rows[start_index:]


def run(curriculum, development_suite, out, *, client, model_path, seconds, provenance,
        start_index=0):
    """Read all batch predictions, recheck up to four observed errors per batch twice."""
    if out.exists() or not 180 <= seconds <= 1800:
        raise ValueError("Use a fresh, bounded mining directory.")
    manifest, rows, proof = verify_training_partition(curriculum, development_suite)
    total_questions = len(rows)
    rows = remaining_rows(rows, start_index)
    out.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + seconds
    record = {
        "schema": "bobcat-glm-hard-case-mining-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "partition_proof": proof,
        "provenance": provenance, "planned_questions": len(rows),
        "planned_prompt_tokens": sum(row["input_tokens"] for row in rows),
        "full_curriculum_prompt_tokens": manifest["statistics"]["train"]["prompt_tokens"],
        "full_curriculum_questions": total_questions, "start_index": start_index,
        "completed_questions": 0, "native_requests": 0, "processed_prompt_tokens": 0,
        "unique_prompt_tokens": 0, "rechecked_cases": 0, "repeat_confirmed_errors": 0,
        "batch_size": 8, "max_rechecks_per_batch": 4, "serial_repeats_per_selected_error": 2,
        "score_error_threshold": .2, "observed_labels_replaced": False,
        "teacher_targets_for_training_published": False, "calibration_fitted": False,
        "weights_updated": False, "release_gate_passed": False,
        "runtime_stability_is_not_assumed": True,
        "selection": "frozen train curriculum from start_index; first four errors in each batch",
        "limits": "Repetition confirms this runtime's error, not an expert-vetted dataset error.",
        "counts": {}, "by_language": {}, "by_task": {},
    }
    atomic_json(out / "run.json", record)
    counts, languages, tasks = Counter(), Counter(), Counter()
    try:
        scorer = CompiledReadout(client, model_path=model_path, run_id="hard-case-mining")
        with (out / "predictions.jsonl").open("x") as stream, (
            out / "rechecks.jsonl").open("x") as checked:
            for offset in range(0, len(rows), 8):
                if deadline - time.monotonic() < 65:
                    record["status"] = "partial_time_budget"
                    break
                batch = rows[offset:offset + 8]
                predictions, measured = scorer.read(batch, label=f"batch-{offset}")
                record["native_requests"] += 1
                record["processed_prompt_tokens"] += sum(r["input_tokens"] for r in batch)
                errors = []
                for row, prediction in zip(batch, predictions, strict=True):
                    kind = error_kind(row, prediction)
                    item = {
                        **prediction, "split": "train", "candidate_ids": row["candidate_ids"],
                        "source_request_sha256": row["source_request_sha256"],
                        "batch_offset": start_index + offset, "error_kind": kind,
                        "observed_labels_replaced": False,
                    }
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[kind or "within_declared_error_threshold"] += 1
                    languages[row["language"]] += 1
                    tasks[row["task"]] += 1
                    if kind:
                        errors.append((row, prediction, kind))
                stream.flush()
                record["completed_questions"] += len(batch)
                record["unique_prompt_tokens"] += sum(r["input_tokens"] for r in batch)
                for index, (row, original, kind) in enumerate(errors[:4]):
                    if deadline - time.monotonic() < 125:
                        break
                    repeated, timings = [], []
                    for repeat in range(2):
                        values, timing = scorer.read(
                            [row], label=f"recheck-{offset}-{index}-{repeat}")
                        repeated.append(values[0])
                        timings.append(timing)
                        record["native_requests"] += 1
                        record["processed_prompt_tokens"] += row["input_tokens"]
                    repeated_kinds = [error_kind(row, value) for value in repeated]
                    confirmed = all(value == kind for value in repeated_kinds)
                    result = {
                        "id": row["id"], "group_id": row["group_id"],
                        "input_sha256": row["input_sha256"], "language": row["language"],
                        "language_origin": row["language_origin"], "task": row["task"],
                        "kind": row["kind"], "candidate_ids": row["candidate_ids"],
                        "observed_supervision": {
                            "type": row["supervision"], "target_index": row["target_index"],
                            "score_mean": row["score_mean"],
                        },
                        "batch_prediction": original, "serial_predictions": repeated,
                        "batch_measurement": measured, "serial_measurements": timings,
                        "serial_repeat_comparison": distribution_comparison(
                            repeated[:1], repeated[1:]),
                        "batch_vs_serial": distribution_comparison([original], repeated[:1]),
                        "error_kind": kind, "serial_error_kinds": repeated_kinds,
                        "repeat_confirmed_error": confirmed,
                        "automatic_relabel_or_distillation_eligibility": False,
                    }
                    checked.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    checked.flush()
                    record["rechecked_cases"] += 1
                    record["repeat_confirmed_errors"] += int(confirmed)
                record.update(
                    updated_at=datetime.now(UTC).isoformat(), counts=dict(counts),
                    by_language=dict(languages), by_task=dict(tasks))
                atomic_json(out / "run.json", record)
            else:
                record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - started,
                      next_index=start_index + record["completed_questions"],
                      counts=dict(counts), by_language=dict(languages), by_task=dict(tasks))
        record["files"] = {
            path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in out.glob("*.jsonl")}
        atomic_json(out / "run.json", record)
    return record
