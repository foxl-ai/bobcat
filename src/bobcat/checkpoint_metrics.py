"""Paired development monitoring; never fits calibration or passes release gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bobcat.corpus import atomic_json
from bobcat.glm_native_evaluate import read_suite
from bobcat.metrics import basic_metrics, probabilities, scored_row
from bobcat.schema import file_hash


def read_training_completed(folder: Path, label: str, curriculum_folder: Path):
    """Read a trainer-published dev snapshot without accessing live optimizer state.

    The immutable curriculum supplies input hashes and language provenance that
    older rank logs omit. IDs and all logged supervision must match it exactly.
    """
    from bobcat.glm_native_train import read_curriculum

    _, data = read_curriculum(curriculum_folder)
    expected = data["dev_train"]
    return _read_training_snapshot(folder, label, expected, curriculum_folder)


def read_training_suite_completed(
    folder: Path, label: str, curriculum_folder: Path, suite_folder: Path,
):
    """Read a trainer's completed expanded evaluation, with independent split checks."""
    from bobcat.glm_native_train import read_curriculum

    _, data = read_curriculum(curriculum_folder)
    suite, expected = read_suite(suite_folder)
    if {row["group_id"] for row in expected} & {row["group_id"] for row in data["train"]}:
        raise ValueError("A training component appears in the expanded monitoring suite.")
    expected_monitor = {
        "manifest_sha256": file_hash(suite_folder / "manifest.json"),
        "records_sha256": suite["records_sha256"],
        "components": len(expected), "training_overlap": False, "final_test": False,
    }
    return _read_training_snapshot(
        folder, label, expected, curriculum_folder, monitoring_suite=expected_monitor,
    )


def _read_training_snapshot(
    folder, label, expected, curriculum_folder, *, monitoring_suite=None,
):
    marker = json.loads((folder / f"evaluation-{label}-complete.json").read_text())
    names = {f"{label}-rank-{rank}.json" for rank in range(8)}
    if (marker.get("label") != label
            or marker.get("curriculum_sha256") != file_hash(curriculum_folder / "manifest.json")
            or marker.get("questions") != len(expected)
            or marker.get("prompt_tokens") != sum(row["input_tokens"] for row in expected)
            or marker.get("optimizer_updated") is not False
            or marker.get("rng_restored") is not True
            or marker.get("final_test") is not False
            or marker.get("monitoring_suite") != monitoring_suite
            or set(marker.get("files", {})) != names):
        raise ValueError("Use a completed dev snapshot bound to its immutable curriculum.")
    observed = {}
    for name, digest in marker["files"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("A completed training evaluation shard changed.")
        for row in json.loads((folder / name).read_text()):
            if row["id"] in observed:
                raise ValueError("Duplicate development prediction.")
            observed[row["id"]] = row
    if set(observed) != {row["id"] for row in expected}:
        raise ValueError("Every planned development question must remain in the denominator.")
    fields = ("id", "group_id", "task", "language", "kind",
              "target_index", "score_mean", "supervision")
    results = []
    for gold in expected:
        row = observed[gold["id"]]
        if (any(row[field] != gold[field] for field in fields)
                or ("input_sha256" in row and row["input_sha256"] != gold["input_sha256"])
                or (monitoring_suite is not None and "input_sha256" not in row)
                or len(row["logits"]) != len(gold["option_token_ids"])
                or not np.isfinite(np.asarray(row["logits"], dtype=np.float64)).all()):
            raise ValueError("Prediction and frozen development supervision differ.")
        results.append({
            **{field: gold[field] for field in (*fields, "input_sha256", "language_origin")},
            "logits": row["logits"], "loss": row["loss"],
            "input_tokens": gold["input_tokens"], "checkpoint": label,
            "input_identity_source": (
                "immutable_monitoring_suite" if monitoring_suite is not None
                else "immutable_producer_curriculum"
            ),
        })
    return results, marker


def read_completed(folder: Path, label: str, suite_folder: Path):
    manifest, expected = read_suite(suite_folder)
    marker = json.loads((folder / f"{label}-complete.json").read_text())
    if (marker.get("checkpoint") != label
            or marker.get("suite_sha256") != file_hash(suite_folder / "manifest.json")
            or marker.get("source_revision") != manifest["source_revision"]
            or marker.get("evaluation_role") != "development_monitoring"
            or marker.get("training_modified") is not False
            or marker.get("rows") != len(expected)
            or set(marker.get("files", {})) != {
                f"{label}-rank-{rank}.json" for rank in range(8)
            }):
        raise ValueError("Evaluation is incomplete or belongs to another frozen suite.")
    rows = {}
    for name, digest in marker["files"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("A completed evaluation rank file changed.")
        for row in json.loads((folder / name).read_text()):
            if row["id"] in rows:
                raise ValueError("Duplicate monitoring result.")
            rows[row["id"]] = row
    if set(rows) != {row["id"] for row in expected}:
        raise ValueError("Missing/extra monitoring predictions must not leave the denominator.")
    ordered = []
    fields = ("id", "group_id", "input_sha256", "kind", "task", "language",
              "language_origin", "supervision", "target_index", "score_mean")
    for gold in expected:
        row = rows[gold["id"]]
        if (any(row[field] != gold[field] for field in fields)
                or len(row["logits"]) != len(gold["option_token_ids"])
                or row["checkpoint"] != label
                or row["suite_sha256"] != marker["suite_sha256"]
                or not np.isfinite(np.asarray(row["logits"], dtype=np.float64)).all()):
            raise ValueError("Monitoring predictions and frozen supervision are misaligned.")
        ordered.append(row)
    return ordered, marker


def enrich(row):
    probs = probabilities(row["logits"])
    if row["supervision"] == "hard_label":
        return scored_row({
            **row, "candidate_ids": list(range(len(probs))), "target": row["target_index"],
            "tie_break": "request_order",
        })
    if row["supervision"] != "score_mean":
        raise ValueError("Unknown monitoring supervision.")
    estimate = float(np.dot(probs, np.arange(len(probs))))
    return {**row, "estimate": estimate, "absolute_error": abs(estimate - row["score_mean"]),
            "normalized_absolute_error": abs(estimate - row["score_mean"]) / (len(probs) - 1)}


def aggregate(rows):
    categorical = [row for row in rows if row["supervision"] == "hard_label"]
    ordinal = [row for row in rows if row["supervision"] == "score_mean"]
    result = {"rows": len(rows), "categorical": basic_metrics(categorical),
              "ordinal_mean": {"count": len(ordinal)}}
    if ordinal:
        result["ordinal_mean"].update(
            mae=float(np.mean([row["absolute_error"] for row in ordinal])),
            normalized_mae=float(np.mean([row["normalized_absolute_error"] for row in ordinal])),
        )
    if categorical:
        tasks = sorted({row["task"] for row in categorical})
        result["macro_task_accuracy"] = float(np.mean([
            np.mean([row["correct"] for row in categorical if row["task"] == task])
            for task in tasks
        ]))
        accepted = [row for row in categorical if row["top_probability"] >= .9]
        result["raw_pmax_0_9_diagnostic"] = {
            "accepted": len(accepted), "coverage": len(accepted) / len(categorical),
            "observed_error": float(np.mean([not row["correct"] for row in accepted]))
            if accepted else None,
            "deployment_policy": False, "calibration_fitted": False,
        }
    return result


def paired_change(before, after, *, metric, seed=20260923, samples=2000):
    if not before:
        return {"count": 0}
    differences = np.asarray([b[metric] - a[metric]
                              for a, b in zip(before, after, strict=True)], dtype=np.float64)
    # The suite has exactly one judgment per distinct source component.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(samples, len(differences)))
    interval = np.quantile(differences[draws].mean(1), [.025, .975]).tolist()
    return {"count": len(differences), "delta": float(differences.mean()),
            "paired_component_bootstrap_95pct": interval, "samples": samples,
            "seed": seed, "repeated_monitoring_adjusted": False}


def compare(before_rows, after_rows):
    if ([r["id"] for r in before_rows] != [r["id"] for r in after_rows]
            or len({r["group_id"] for r in before_rows}) != len(before_rows)):
        raise ValueError("Compare the same ordered, unique monitoring components.")
    before, after = list(map(enrich, before_rows)), list(map(enrich, after_rows))
    for a, b in zip(before, after, strict=True):
        if any(a[k] != b[k] for k in (
            "group_id", "input_sha256", "supervision", "target_index", "score_mean",
            "task", "kind", "language", "language_origin",
        )) or len(a["logits"]) != len(b["logits"]):
            raise ValueError("The paired judgments use different inputs or supervision.")
    subsets = {"all": list(range(len(before)))}
    for field in ("language", "language_origin", "task", "kind"):
        for value in sorted({r[field] for r in before}):
            subsets[f"{field}/{value}"] = [i for i, r in enumerate(before) if r[field] == value]
    for count in sorted({len(r["logits"]) for r in before}):
        subsets[f"candidate_count/{count}"] = [
            i for i, row in enumerate(before) if len(row["logits"]) == count
        ]
    result = {}
    for name, indices in subsets.items():
        a, b = [before[i] for i in indices], [after[i] for i in indices]
        ac, bc = ([r for r in rows if r["supervision"] == "hard_label"] for rows in (a, b))
        ao, bo = ([r for r in rows if r["supervision"] == "score_mean"] for rows in (a, b))
        result[name] = {
            "baseline": aggregate(a), "checkpoint": aggregate(b),
            "accuracy_change": paired_change(ac, bc, metric="correct"),
            "ordinal_normalized_mae_change": paired_change(
                ao, bo, metric="normalized_absolute_error",
            ),
        }
        if ac:
            result[name]["answer_flip_fraction"] = float(np.mean([
                x["prediction"] != y["prediction"] for x, y in zip(ac, bc, strict=True)
            ]))
    warnings = []
    for task in sorted({r["task"] for r in after}):
        rows = [r for r in after if r["task"] == task and r["supervision"] == "hard_label"]
        if rows:
            counts = Counter(r["prediction"] for r in rows)
            if (len(rows) >= 20 and max(counts.values()) / len(rows) >= .95
                    and len({r["target_index"] for r in rows}) > 1):
                warnings.append({"task": task, "reason": "near_constant_option_position",
                                 "counts": dict(counts), "review_required": True})
    return {
        "schema": "bobcat-paired-checkpoint-monitoring-v1",
        "evaluation_role": "development_monitoring", "final_evaluation": False,
        "release_gate_passed": False, "calibration_fitted": False,
        "training_modified": False, "subsets": result, "diagnostic_warnings": warnings,
        "scope": "Repeated development comparisons guide research; confidence intervals are "
                 "descriptive and do not authorize a fresh-test or Jev-equivalence claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument(
        "--input-format", choices=("evaluator", "trainer-expanded"), default="evaluator",
    )
    parser.add_argument("--curriculum", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve existing monitoring reports.")
    if args.input_format == "trainer-expanded":
        if args.curriculum is None:
            parser.error("Expanded trainer monitoring requires the immutable curriculum.")
        def read(label):
            return read_training_suite_completed(
                args.evaluations, label, args.curriculum, args.suite,
            )
    else:
        def read(label):
            return read_completed(args.evaluations, label, args.suite)
    baseline, base_marker = read(args.baseline)
    current, marker = read(args.checkpoint)
    result = compare(baseline, current)
    result.update(checkpoint=args.checkpoint, baseline_marker=base_marker, marker=marker)
    atomic_json(args.out, result)


if __name__ == "__main__":
    main()
