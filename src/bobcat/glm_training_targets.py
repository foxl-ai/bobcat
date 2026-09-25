"""Collect GLM decision distributions on frozen training inputs, never on final data.

These are teacher predictions, not ground truth or a calibrated distribution.
Existing labels remain separate. Collection does not update any model weights.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bobcat.corpus import atomic_json
from bobcat.glm_native_evaluate import read_suite
from bobcat.glm_native_train import read_curriculum
from bobcat.glm_serving_checkpoint import CompiledReadout, distribution_comparison
from bobcat.schema import file_hash


def verify_training_partition(curriculum, development_suite):
    manifest, data = read_curriculum(curriculum)
    suite, development = read_suite(development_suite)
    groups = {row["group_id"] for row in development}
    overlaps = groups & {row["group_id"] for row in data["train"]}
    if overlaps:
        raise ValueError("Teacher collection cannot include development components.")
    return manifest, data["train"], {
        "curriculum_sha256": file_hash(curriculum / "manifest.json"),
        "train_file_sha256": file_hash(curriculum / "train.jsonl"),
        "development_suite_sha256": file_hash(development_suite / "manifest.json"),
        "development_components_checked": len(groups),
        "training_components": len(data["train"]),
        "development_overlap_components": 0,
        "producer_separate_train_dev_check": True,
        "pretrained_historical_exposure_excluded": False,
        "source_revision": suite["source_revision"],
    }


def target_record(row, prediction, *, teacher):
    if (row["split"] != "train" or row["id"] != prediction["id"]
            or row["input_sha256"] != prediction["input_sha256"]
            or len(prediction["logits"]) != len(row["candidate_ids"])
            or prediction["native_completion_tokens"] != 0):
        raise ValueError("Teacher logits must align with the complete training question.")
    scores = np.asarray(prediction["logits"], dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite teacher output.")
    # Preserve the raw candidate log probabilities as well as stable normalization.
    log_probabilities = scores - np.logaddexp.reduce(scores)
    predicted = int(scores.argmax())
    return {
        "id": row["id"], "group_id": row["group_id"], "split": "train",
        "source_request_sha256": row["source_request_sha256"],
        "input_sha256": row["input_sha256"], "input_tokens": row["input_tokens"],
        "language": row["language"], "language_origin": row["language_origin"],
        "task": row["task"], "kind": row["kind"], "candidate_ids": row["candidate_ids"],
        "option_token_ids": row["option_token_ids"],
        "observed_supervision": {
            "type": row["supervision"], "target_index": row["target_index"],
            "score_mean": row["score_mean"],
        },
        "teacher": teacher,
        "teacher_raw_candidate_logprobs": scores.tolist(),
        "teacher_conditional_logprobs": log_probabilities.tolist(),
        "teacher_conditional_probabilities": np.exp(log_probabilities).tolist(),
        "teacher_argmax_index": predicted,
        "teacher_matches_observed_label": (
            predicted == row["target_index"] if row["supervision"] == "hard_label" else None
        ),
        "teacher_distribution_is_ground_truth": False,
        "teacher_calibration_fitted": False,
        "native_completion_tokens": prediction["native_completion_tokens"],
        "training_eligibility": "requires_teacher_quality_review_and_explicit_loss_mixture",
    }


def collect(curriculum: Path, development_suite: Path, out: Path, *, client,
            model_path: str, teacher: dict, seconds: float, batch_size: int,
            on_shard=None, verify_every_batches: int = 0):
    if (out.exists() or not 65 < seconds <= 7200 or batch_size not in (1, 8)
            or verify_every_batches not in (0, 128)):
        raise ValueError("Use a new, bounded target collection with explicit batch policy.")
    if (teacher.get("source_repo") != "zai-org/GLM-5.3-Flash"
            or teacher.get("weights_manifest_sha256") is None
            or teacher.get("jev_outputs_used") is not False):
        raise ValueError("Identify the actual independently developed GLM teacher.")
    started, batches = time.monotonic(), []
    manifest, rows, partition = verify_training_partition(curriculum, development_suite)
    if teacher["source_revision"] != partition["source_revision"]:
        raise ValueError("Teacher and compiled input tokenizers have different revisions.")
    deadline = started + seconds
    out.mkdir(parents=True)
    record = {
        "schema": "bobcat-glm-training-targets-v1", "status": "collecting",
        "started_at": datetime.now(UTC).isoformat(), "teacher": teacher,
        "partition_proof": partition, "planned_questions": len(rows),
        "planned_prompt_tokens": manifest["statistics"]["train"]["prompt_tokens"],
        "completed_questions": 0, "processed_prompt_tokens": 0,
        "batch_size": batch_size, "weight_updates": 0, "training_performed": False,
        "evaluation_or_release_claim": False, "calibration_fitted": False,
        "shards": [], "counts_by_language": {}, "prompt_tokens_by_language": {},
        "stability_monitor_every_batches": verify_every_batches, "stability_checks": [],
    }
    atomic_json(out / "collection.json", record)
    pending, counts, tokens = [], Counter(), Counter()

    def commit():
        if not pending:
            return
        name = f"targets-{len(record['shards']):05d}.jsonl"
        path = out / name
        with path.open("x") as stream:
            for value in pending:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        shard = {"path": name, "rows": len(pending), "sha256": file_hash(path),
                 "bytes": path.stat().st_size}
        if on_shard is not None:
            try:
                shard["versioned_export"] = on_shard(path)
            except Exception as error:
                # Retain the completed local file and its checksum for recovery.
                # Never rewrite its immutable S3 key after an ambiguous upload.
                shard["export_error"] = f"{type(error).__name__}: {error}"[:1500]
        record["shards"].append(shard)
        pending.clear()
        record["counts_by_language"] = dict(counts)
        record["prompt_tokens_by_language"] = dict(tokens)
        atomic_json(out / "collection.json", record)

    try:
        scorer = CompiledReadout(client, model_path=model_path, run_id="training-targets")
        record["server_model_info"] = scorer.model_info
        for offset in range(0, len(rows), batch_size):
            # One ordinary native request can use up to the separate 60s HTTP limit.
            monitor = bool(
                verify_every_batches and (offset // batch_size) % verify_every_batches == 0)
            if deadline - time.monotonic() < (125 if monitor else 65):
                record["status"] = "partial_time_budget"
                break
            batch = rows[offset:offset + batch_size]
            predictions, measured = scorer.read(batch, label=f"train-{offset}")
            if monitor:
                # Recheck an actual training question in isolation. For batch8,
                # this also tests the current batch/serial boundary. A failure
                # does not become a silently discarded low-confidence example.
                repeated, echo_measurement = scorer.read([batch[0]], label=f"echo-{offset}")
                check = distribution_comparison(predictions[:1], repeated)
                record["stability_checks"].append({
                    "offset": offset, "comparison": check,
                    "first_prediction": predictions[0], "repeated_prediction": repeated[0],
                    "measurement": echo_measurement,
                })
                if not check["passed"]:
                    record["runtime_stability_violation"] = True
                    raise ValueError("Actual training inputs failed the runtime stability monitor.")
            values = [target_record(row, prediction, teacher=teacher)
                      for row, prediction in zip(batch, predictions, strict=True)]
            pending.extend(values)
            batches.append(measured)
            record["completed_questions"] += len(values)
            record["processed_prompt_tokens"] += sum(row["input_tokens"] for row in batch)
            for row in batch:
                counts[row["language"]] += 1
                tokens[row["language"]] += row["input_tokens"]
            if len(pending) >= 256:
                commit()
        else:
            record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        commit()
        atomic_json(out / "measurements.json", batches)
        record.update(
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            counts_by_language=dict(counts), prompt_tokens_by_language=dict(tokens),
            completed_shard_rows=sum(row["rows"] for row in record["shards"]),
            all_shards_exported=bool(record["shards"]) and all(
                row.get("versioned_export", {}).get("fresh_get_verified") is True
                for row in record["shards"]
            ),
        )
        atomic_json(out / "collection.json", record)
    return record
