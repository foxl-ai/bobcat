"""Audit collected teacher predictions against preserved training supervision.

This checks data integrity and describes teacher errors. It neither authorizes
distillation nor turns training predictions into a held-out quality evaluation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from bobcat.checkpoint_metrics import aggregate, enrich
from bobcat.glm_training_targets import verify_training_partition
from bobcat.schema import file_hash

IDENTITY_FIELDS = (
    "id", "group_id", "split", "source_request_sha256", "input_sha256", "input_tokens",
    "language", "language_origin", "task", "kind", "candidate_ids", "option_token_ids",
)


def validate_target(target, original, teacher):
    """Reject altered labels, candidate order, normalized values, or teacher identity."""
    if (original["split"] != "train"
            or any(target.get(key) != original[key] for key in IDENTITY_FIELDS)
            or target.get("teacher") != teacher
            or target.get("observed_supervision") != {
                "type": original["supervision"], "target_index": original["target_index"],
                "score_mean": original["score_mean"],
            }
            or target.get("teacher_distribution_is_ground_truth") is not False
            or target.get("teacher_calibration_fitted") is not False
            or target.get("native_completion_tokens") != 0
            or target.get("training_eligibility")
            != "requires_teacher_quality_review_and_explicit_loss_mixture"):
        raise ValueError("Teacher identity, input, or observed supervision changed.")
    raw = np.asarray(target["teacher_raw_candidate_logprobs"], dtype=np.float64)
    normalized = np.asarray(target["teacher_conditional_logprobs"], dtype=np.float64)
    probs = np.asarray(target["teacher_conditional_probabilities"], dtype=np.float64)
    count = len(original["candidate_ids"])
    if (raw.shape != (count,) or normalized.shape != raw.shape or probs.shape != raw.shape
            or count < 2 or not np.isfinite(raw).all()
            or not np.isfinite(normalized).all() or not np.isfinite(probs).all()):
        raise ValueError("Retain every finite teacher candidate value.")
    expected = raw - np.logaddexp.reduce(raw)
    if (not np.allclose(normalized, expected, rtol=0, atol=1e-10)
            or not np.allclose(probs, np.exp(expected), rtol=0, atol=1e-10)
            or abs(probs.sum() - 1) > 1e-10
            or target["teacher_argmax_index"] != int(raw.argmax())):
        raise ValueError("Teacher probabilities do not match the recorded raw readout.")
    if original["supervision"] == "hard_label":
        if not 0 <= original["target_index"] < count:
            raise ValueError("Observed categorical label is outside the offered candidates.")
        matches = int(raw.argmax()) == original["target_index"]
    elif original["supervision"] == "score_mean":
        if (original["kind"] != "ordinal"
                or not np.isfinite(original["score_mean"])
                or not 0 <= original["score_mean"] <= count - 1):
            raise ValueError("Preserve an ordinal mean without inventing category gold.")
        matches = None
    else:
        raise ValueError("The target audit does not recognize this supervision.")
    if target["teacher_matches_observed_label"] is not matches:
        raise ValueError("Teacher disagreement was changed or hidden.")
    return enrich({
        **{key: original[key] for key in IDENTITY_FIELDS},
        "supervision": original["supervision"], "target_index": original["target_index"],
        "score_mean": original["score_mean"], "logits": raw.tolist(),
    })


def describe(rows):
    if not rows or len({row["group_id"] for row in rows}) != len(rows):
        raise ValueError("Describe nonempty, distinct training components.")
    groups = {"all": rows}
    for field in ("language", "language_origin", "task", "kind"):
        for value in sorted({row[field] for row in rows}):
            groups[f"{field}/{value}"] = [row for row in rows if row[field] == value]
    for language, origin in sorted({(row["language"], row["language_origin"]) for row in rows}):
        groups[f"language_origin/{language}/{origin}"] = [
            row for row in rows
            if row["language"] == language and row["language_origin"] == origin
        ]
    for count in sorted({len(row["logits"]) for row in rows}):
        groups[f"candidate_count/{count}"] = [
            row for row in rows if len(row["logits"]) == count
        ]
    incorrect = [
        {"id": row["id"], "group_id": row["group_id"], "task": row["task"],
         "language": row["language"], "language_origin": row["language_origin"],
         "candidate_count": len(row["logits"]), "teacher_index": row["prediction"],
         "observed_target_index": row["target_index"],
         "teacher_top_probability": row["top_probability"]}
        for row in rows if row["supervision"] == "hard_label" and not row["correct"]
    ]
    return {
        "schema": "bobcat-glm-training-target-quality-audit-v1",
        "purpose": "training-data review; not held-out teacher generalization",
        "questions": len(rows), "unique_training_components": len(rows),
        "prompt_tokens": sum(row["input_tokens"] for row in rows),
        "candidate_count_counts": dict(sorted(Counter(len(row["logits"]) for row in rows).items())),
        "subsets": {key: aggregate(values) for key, values in groups.items()},
        "disagreements_with_observed_labels": incorrect,
        "teacher_disagreements_removed": False,
        "observed_labels_replaced": False,
        "training_predictions_may_include_teacher_training_examples": True,
        "student_training_authorized_by_this_audit": False,
        "calibration_fitted": False, "weights_updated": False,
        "held_out_generalization_claim": False, "release_gate_passed": False,
    }


def audit(folder: Path, curriculum: Path, development_suite: Path):
    _, originals, partition = verify_training_partition(curriculum, development_suite)
    path = folder / "collection.json"
    collection = json.loads(path.read_text())
    teacher = collection["teacher"]
    if (collection.get("schema") != "bobcat-glm-training-targets-v1"
            or collection.get("status") not in ("completed", "partial_time_budget", "failed")
            or not collection.get("finished_at")
            or collection.get("partition_proof") != partition
            or collection.get("weight_updates") != 0
            or teacher.get("source_repo") != "zai-org/GLM-5.3-Flash"
            or teacher.get("source_revision") != partition["source_revision"]
            or teacher.get("jev_outputs_used") is not False
            or not teacher.get("weights_manifest_sha256")):
        raise ValueError("Require a terminal target collection bound to its training partition.")
    rows, names, evidence = [], set(), {}
    for shard in collection["shards"]:
        name = shard["path"]
        source = folder / name
        if (Path(name).name != name or name in names or source.is_symlink()
                or source.stat().st_size != shard["bytes"]
                or file_hash(source) != shard["sha256"]):
            raise ValueError("A target shard changed or was referenced more than once.")
        names.add(name)
        targets = [json.loads(line) for line in source.read_text().splitlines()]
        if len(targets) != shard["rows"] or len(rows) + len(targets) > len(originals):
            raise ValueError("Target shard count differs from the frozen collection.")
        for target in targets:
            rows.append(validate_target(target, originals[len(rows)], teacher))
        evidence[name] = shard["sha256"]
    if (len(rows) != collection["completed_questions"]
            or len(rows) != collection["completed_shard_rows"]
            or sum(row["input_tokens"] for row in rows) != collection["processed_prompt_tokens"]
            or collection["planned_questions"] != len(originals)
            or collection["status"] == "completed" and len(rows) != len(originals)):
        raise ValueError("Partial or missing teacher outputs must stay in the denominator.")
    result = describe(rows)
    result.update(
        teacher=teacher, collection_status=collection["status"],
        planned_questions=len(originals), collected_fraction=len(rows) / len(originals),
        selection="fixed curriculum prefix; partial collection may be unrepresentative",
        partition_proof=partition, collection_sha256=file_hash(path),
        target_shards_sha256=evidence,
    )
    return result
