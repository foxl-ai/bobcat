"""Recompute crossed identifier effects from frozen native prediction evidence.

These are descriptive development diagnostics. Repeats, candidate-count variants
and language variants are not independent new benchmark items.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from bobcat.corpus import atomic_json
from bobcat.glm_identifier_controls import ARMS, validate
from bobcat.protocol import probabilities
from bobcat.schema import file_hash

THRESHOLDS = (0.8, 0.9, 0.95)
CONTRASTS = {
    "repeat_noise": ("base", "base_repeat"),
    "order_at_base_binding": ("base", "order_only"),
    "binding_at_base_order": ("base", "identifier_only"),
    "order_at_changed_binding": ("identifier_only", "both"),
    "binding_at_changed_order": ("order_only", "both"),
    "both_changed": ("base", "both"),
}


def _prediction(case, row):
    for field in (
        "id", "semantic_group", "block", "arm", "language", "gold",
        "labels_in_order", "identifier_slots",
    ):
        if row.get(field) != case[field]:
            raise ValueError(f"Prediction metadata does not match the frozen case: {field}")
    if row["status"] == "failed":
        return None
    labels, scores = case["labels_in_order"], row.get("raw_scores", [])
    if (row["status"] != "scored" or len(scores) != len(labels)
            or any(type(x) not in (int, float) for x in scores)):
        raise ValueError("A scored response needs every finite candidate logit.")
    values = probabilities(scores)
    distribution = dict(zip(labels, values, strict=True))
    selected = max(range(len(values)), key=values.__getitem__)
    choice = labels[selected]
    recorded = row.get("probabilities", {})
    if (set(recorded) != set(labels) or any(
            not math.isclose(recorded[key], value, rel_tol=1e-10, abs_tol=1e-12)
            for key, value in distribution.items())
            or row.get("prediction") != choice or row.get("correct") != (choice == case["gold"])):
        raise ValueError("Recorded probabilities or decisions disagree with the raw logits.")
    return {
        "probabilities": distribution, "prediction": choice, "correct": choice == case["gold"],
        "gold_probability": distribution[case["gold"]], "top_probability": values[selected],
        "selected_list_index": selected,
        "selected_identifier_slot": case["identifier_slots"][selected],
        "gold_list_index": labels.index(case["gold"]),
        "gold_identifier_slot": case["identifier_slots"][labels.index(case["gold"])],
    }


def _contrast(before, after):
    if before is None or after is None:
        return {"status": "missing_or_failed_arm"}
    p, q = before["probabilities"], after["probabilities"]
    if p.keys() != q.keys():
        raise ValueError("Contrast requires the same candidate meanings.")
    return {
        "status": "compared",
        "tv": sum(abs(p[k] - q[k]) for k in p) / 2,
        "argmax_changed": before["prediction"] != after["prediction"],
        "correct_delta": int(after["correct"]) - int(before["correct"]),
        "gold_probability_delta": after["gold_probability"] - before["gold_probability"],
        "thresholds": {
            str(t): {
                "crossing": (before["top_probability"] >= t) != (after["top_probability"] >= t),
                "action_changed": (
                    before["prediction"] if before["top_probability"] >= t else None
                ) != (after["prediction"] if after["top_probability"] >= t else None),
                "wrong_execution_delta": int(after["top_probability"] >= t and not after["correct"])
                - int(before["top_probability"] >= t and not before["correct"]),
            } for t in THRESHOLDS
        },
    }


def analyze_identifiers(suite, observations):
    validate(suite)
    cases = {case["id"]: case for case in suite["cases"]}
    seen, predictions, grouped = set(), {}, defaultdict(dict)
    for row in observations:
        if row.get("id") not in cases or row["id"] in seen:
            raise ValueError("Unknown or duplicate prediction; do not pool reruns.")
        seen.add(row["id"])
        predictions[row["id"]] = _prediction(cases[row["id"]], row)
    for case in cases.values():
        grouped[(case["semantic_group"], case["block"])][case["arm"]] = case

    arm_counts = {}
    for arm in ARMS:
        planned = [c for c in cases.values() if c["arm"] == arm]
        scored = [predictions[c["id"]] for c in planned if predictions.get(c["id"]) is not None]
        attempted = sum(c["id"] in seen for c in planned)
        correct = sum(r["correct"] for r in scored)
        arm_counts[arm] = {
            "planned": len(planned), "attempted": attempted, "scored": len(scored),
            "failed": attempted - len(scored), "unattempted": len(planned) - attempted,
            "correct": correct,
            # Failed attempts stay in the denominator; unattempted items are unknown.
            "accuracy_on_attempts": correct / attempted if attempted else None,
            "accuracy_complete_plan": correct / len(planned) if attempted == len(planned) else None,
            "selected_identifier_slots": dict(sorted(Counter(
                row["selected_identifier_slot"] for row in scored
            ).items())),
            "selected_list_indices": dict(sorted(Counter(
                row["selected_list_index"] for row in scored
            ).items())),
        }

    blocks = []
    for (group, block), arms in sorted(grouped.items()):
        anchor = arms["base"]
        values = {arm: predictions.get(case["id"]) for arm, case in arms.items()}
        contrasts = {
            name: _contrast(values[first], values[second])
            for name, (first, second) in CONTRASTS.items()
        }
        row = {
            "semantic_group": group, "block": block, "language": anchor["language"],
            "candidate_count": len(anchor["labels_in_order"]), "contrasts": contrasts,
            "arms": {
                arm: {k: v for k, v in value.items() if k != "probabilities"}
                if value else None for arm, value in values.items()
            },
        }
        if all(values[arm] is not None for arm in
               ("base", "order_only", "identifier_only", "both")):
            row["factorial"] = {}
            for field in ("gold_probability", "correct"):
                b, o, i, both = (float(values[arm][field]) for arm in (
                    "base", "order_only", "identifier_only", "both",
                ))
                row["factorial"][field] = {
                    "mean_order_effect": ((o - b) + (both - i)) / 2,
                    "mean_identifier_effect": ((i - b) + (both - o)) / 2,
                    "interaction": both - i - o + b,
                }
        blocks.append(row)

    # Keep each language/count slice distinct; repeated blocks remain explicitly counted.
    slices = defaultdict(list)
    for row in blocks:
        slices[(row["language"], row["candidate_count"])].append(row)
    summaries = []
    for (language, count), rows in sorted(slices.items()):
        summary = {
            "language": language, "candidate_count": count, "blocks": len(rows),
            "semantic_groups": len({row["semantic_group"] for row in rows}), "contrasts": {},
        }
        for name in CONTRASTS:
            complete = [row["contrasts"][name] for row in rows
                        if row["contrasts"][name]["status"] == "compared"]
            summary["contrasts"][name] = {
                "compared_blocks": len(complete),
                "missing_or_failed_blocks": len(rows) - len(complete),
                "mean_tv": mean(x["tv"] for x in complete) if complete else None,
                "max_tv": max((x["tv"] for x in complete), default=None),
                "argmax_flips": sum(x["argmax_changed"] for x in complete),
                "correct_delta_sum": sum(x["correct_delta"] for x in complete),
                "mean_gold_probability_delta": (
                    mean(x["gold_probability_delta"] for x in complete) if complete else None
                ),
                "thresholds": {
                    str(t): {
                        "crossings": sum(x["thresholds"][str(t)]["crossing"] for x in complete),
                        "action_changes": sum(
                            x["thresholds"][str(t)]["action_changed"] for x in complete
                        ),
                        "wrong_execution_delta_sum": sum(
                            x["thresholds"][str(t)]["wrong_execution_delta"] for x in complete
                        ),
                    } for t in THRESHOLDS
                },
            }
        summaries.append(summary)
    return {
        "schema": "bobcat-glm-identifier-analysis-v1",
        "suite_content_sha256": suite["content_sha256"],
        "planned_calls": len(cases), "attempted_calls": len(seen),
        "scored_calls": sum(x is not None for x in predictions.values()),
        "semantic_groups": len({key[0] for key in grouped}), "paired_blocks": len(grouped),
        "arm_counts": arm_counts, "by_language_and_count": summaries, "blocks": blocks,
        "temperature": 1.0, "calibration_fitted": False, "release_gate_passed": False,
        "confidence_intervals": None,
        "scope": "Descriptive crossed development probes; no independent-item or population "
        "claim. Repeats, count variants and bilingual worlds share source constructions.",
        "policy_scope": "Fixed diagnostic thresholds, not a calibrated production policy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve the previous analysis; use a new output.")
    suite_path = args.run_dir / "suite.json"
    rows_path = args.run_dir / "predictions.jsonl"
    run_path = args.run_dir / "run.json"
    manifest = json.loads(run_path.read_text())
    suite = json.loads(suite_path.read_text())
    if (manifest["schema"] != "bobcat-glm-identifier-run-v1"
            or manifest["suite_sha256"] != suite["content_sha256"]
            or manifest["predictions_sha256"] != file_hash(rows_path)
            or manifest["temperature"] != 1.0):
        raise ValueError("The run's predictions, suite or calibration changed.")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    result = analyze_identifiers(suite, rows)
    if (manifest["planned_calls"] != result["planned_calls"]
            or manifest["attempted_calls"] != result["attempted_calls"]
            or manifest["failed_calls"] != result["attempted_calls"] - result["scored_calls"]):
        raise ValueError("The run counts disagree with the recovered raw predictions.")
    result.update(
        analyzed_at=datetime.now(UTC).isoformat(), run_status=manifest["status"],
        analysis_source_sha256=file_hash(Path(__file__)),
        inputs_sha256={name: file_hash(args.run_dir / name)
                       for name in ("suite.json", "predictions.jsonl", "run.json")},
    )
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in (
        "run_status", "planned_calls", "attempted_calls", "scored_calls",
    )}))


if __name__ == "__main__":
    main()
