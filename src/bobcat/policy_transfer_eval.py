"""Matched model-vs-decomposition evaluation for the synthetic policy diagnostic.

The strong control predicts a source category once and pushes that distribution
through the known policy interpreter. It does not ask the model to generate a
policy parser. Its applicability is limited to this construction-known policy
suite; it is not a solution to arbitrary natural-language business rules.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bobcat.corpus import atomic_json
from bobcat.policy_transfer import FAMILIES, change_program, execute, make_program
from bobcat.schema import file_hash, json_hash


def reconstruct_program(row: dict, *, seed: int) -> dict:
    instructions = row["request"]["questions"]["decision"]["instructions"]
    categories = list(instructions["category_definitions"])
    design_seed = json_hash([seed, row["source"]["original_input_sha256"], "policy-design"])
    family = random.Random(design_seed).choice(FAMILIES[row["split"]])
    program, metadata = make_program(categories, design_seed, family)
    program = change_program(program, row["policy_view"])
    if (
        family != row["policy_family"]
        or metadata != row["request"]["state"]["workflow_metadata"]
        or json_hash(program) != row["source"]["policy_program_sha256"]
        or json_hash(row["request"]) != row["input_sha256"]
    ):
        raise ValueError("The original policy construction, seed or input binding changed.")
    return program


def probabilities(prediction: dict, candidate_ids: list[str], request_sha: str) -> np.ndarray:
    if prediction.get("source_request_sha256") != request_sha:
        raise ValueError("Predictions must bind the exact original request.")
    labels = prediction.get("candidate_ids")
    if (
        not isinstance(labels, list)
        or len(labels) != len(set(labels))
        or set(labels) != set(candidate_ids)
        or ("probabilities" in prediction) == ("logits" in prediction)
    ):
        raise ValueError("Need every original candidate and either logits or probabilities.")
    values = np.asarray(prediction.get("logits", prediction.get("probabilities")), dtype=np.float64)
    if values.shape != (len(labels),) or not np.isfinite(values).all():
        raise ValueError("All candidate values must be present and finite.")
    if "logits" in prediction:
        values = np.exp(values - values.max())
        values /= values.sum()
    elif np.any(values < 0) or np.any(values > 1) or abs(float(values.sum()) - 1) > 1e-6:
        raise ValueError("A categorical probability distribution must sum to one.")
    return values[[labels.index(label) for label in candidate_ids]]


def decomposition_prediction(row: dict, source_prediction: dict, *, seed: int) -> dict:
    program = reconstruct_program(row, seed=seed)
    instructions = row["request"]["questions"]["decision"]["instructions"]
    categories = list(instructions["category_definitions"])
    source_p = probabilities(source_prediction, categories, row["source"]["original_input_sha256"])
    metadata = row["request"]["state"]["workflow_metadata"]
    route_p = dict.fromkeys(program["routes"], 0.0)
    for category, probability in zip(categories, source_p, strict=True):
        route_p[execute(program, category, metadata)] += float(probability)
    if row["kind"] == "choice":
        result = [route_p[label] for label in row["candidate_ids"]]
    elif row["kind"] == "ordinal":
        result = [route_p[route] for route in program["routes"]]
    else:
        # Recover the independently chosen Noul probe; never use row["target"].
        design_seed = json_hash([seed, row["source"]["original_input_sha256"], "policy-design"])
        rng = random.Random(design_seed)
        rng.choice(FAMILIES[row["split"]])
        probe = rng.choice(program["routes"])
        result = [1 - route_p[probe], route_p[probe]]
    return {
        "id": row["id"],
        "source_request_sha256": row["input_sha256"],
        "candidate_ids": row["candidate_ids"],
        "probabilities": result,
        "method": "source_category_distribution_then_known_policy",
    }


def evaluate(rows: list[dict], predictions: list[dict]) -> dict:
    if not rows or any(row["split"] != "dev_train" for row in rows):
        raise ValueError("Evaluate the declared development rule holdout; never call it final.")
    reference = {row["id"]: row for row in rows}
    if len(reference) != len(rows):
        raise ValueError("Duplicate reference observations.")
    predicted = {row["id"]: row for row in predictions}
    if len(predicted) != len(predictions) or not set(predicted) <= set(reference):
        raise ValueError("Duplicate predictions or predictions from another evaluation cohort.")
    good, errors, groups = {}, Counter(), defaultdict(list)
    components = defaultdict(list)
    nll, brier, ordinal_errors, chosen_p, correctness = [], [], [], [], []
    for row in rows:
        uid = row["id"]
        try:
            p = probabilities(predicted[uid], row["candidate_ids"], row["input_sha256"])
        except (KeyError, ValueError, TypeError) as error:
            errors["missing" if uid not in predicted else type(error).__name__] += 1
            correct = False
        else:
            target = row["candidate_ids"].index(row["target"])
            chosen = int(p.argmax())
            correct = chosen == target
            good[uid] = {
                "probabilities": p,
                "choice": row["candidate_ids"][chosen],
                "correct": correct,
            }
            # Count failures in accuracy; distribution metrics report their
            # valid-response denominator and the explicit floor used for log(0).
            nll.append(-math.log(max(float(p[target]), 1e-12)))
            brier.append(float(np.square(p - np.eye(len(p))[target]).sum()))
            chosen_p.append(float(p[chosen]))
            correctness.append(correct)
            if row["kind"] == "ordinal":
                ordinal_errors.append(abs(float(p @ np.arange(len(p))) - target))
        components[row["group_id"]].append(correct)
        for axis in ("language", "kind", "task", "policy_family", "policy_view"):
            groups[f"{axis}:{row[axis]}"].append(correct)
        if row["source_text_required"]:
            groups["source_text_required"].append(correct)
        if row["language"] == "ko" and row["language_origin"] == "native":
            groups["native_korean_source_text"].append(correct)

    paired = defaultdict(dict)
    for row in rows:
        paired[row["policy_pair_id"]][row["policy_view"]] = row
    interventions = {}
    for view in ("renamed_routes", "changed_clause", "permuted_options"):
        expected_changes, both_correct, changed_correctly, stable_correctly = 0, 0, 0, 0
        pairs, usable, tv_values, threshold_crossings = 0, 0, [], 0
        for variants in paired.values():
            if "original" not in variants or view not in variants:
                continue
            a, b = variants["original"], variants[view]
            pairs += 1
            gold_change = a["target"] != b["target"]
            expected_changes += gold_change
            if a["id"] not in good or b["id"] not in good:
                continue
            usable += 1
            left, right = good[a["id"]], good[b["id"]]
            both = left["correct"] and right["correct"]
            both_correct += both
            changed_correctly += gold_change and both
            stable_correctly += not gold_change and both
            if view == "permuted_options":
                indices = [b["candidate_ids"].index(label) for label in a["candidate_ids"]]
                aligned = right["probabilities"][indices]
                tv_values.append(float(np.abs(left["probabilities"] - aligned).sum() / 2))
                threshold_crossings += (float(left["probabilities"].max()) >= 0.9) != (
                    float(aligned.max()) >= 0.9
                )
        interventions[view] = {
            "pairs": pairs,
            "valid_both": usable,
            "expected_label_changes": expected_changes,
            "both_correct": both_correct,
            "both_correct_rate_with_failures": both_correct / pairs if pairs else None,
            "changed_and_both_correct": changed_correctly,
            "unchanged_and_both_correct": stable_correctly,
            "permutation_max_tv": max(tv_values, default=None),
            "permutation_mean_tv": float(np.mean(tv_values)) if tv_values else None,
            "permutation_pmax_0_9_threshold_crossings": threshold_crossings,
        }
    confidence = np.asarray(chosen_p)
    truth = np.asarray(correctness, dtype=np.float64)
    ece = 0.0
    # Assign each probability to exactly one bin, including floating-point
    # values exactly on a boundary. Repeated lo + .1 comparisons can overlap.
    bins = np.clip(np.searchsorted(np.linspace(0, 1, 11), confidence, side="right") - 1, 0, 9)
    for bin_id in range(10):
        selected = bins == bin_id
        if selected.any():
            ece += float(
                selected.sum()
                / len(confidence)
                * abs(confidence[selected].mean() - truth[selected].mean())
            )
    correct_count = sum(row["correct"] for row in good.values())
    return {
        "schema": "bobcat-policy-transfer-evaluation-v1",
        "fresh_final_evaluation": False,
        "questions": len(rows),
        "components": len(components),
        "source_observations": len({r["source"]["original_id"] for r in rows}),
        "valid_responses": len(good),
        "invalid_or_missing": dict(errors),
        "correct": correct_count,
        "accuracy_with_failures": correct_count / len(rows),
        "component_macro_accuracy": float(np.mean([np.mean(v) for v in components.values()])),
        "distribution_metric_denominator": len(good),
        "nll_probability_floor": 1e-12,
        "nll": float(np.mean(nll)) if nll else None,
        "brier": float(np.mean(brier)) if brier else None,
        "ece_10_equal_width": ece if len(good) else None,
        "ordinal_expected_level_mae": float(np.mean(ordinal_errors)) if ordinal_errors else None,
        "groups": {
            key: {
                "questions": len(values),
                "correct": int(sum(values)),
                "accuracy_with_failures": float(np.mean(values)),
            }
            for key, values in sorted(groups.items())
        },
        "interventions": interventions,
        "scope": (
            "Generated policies and metadata on real source text; "
            "not native workflow ground truth."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-category-control", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new immutable evaluation report path.")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    rows_path = args.dataset / "dev_train.jsonl"
    if file_hash(rows_path) != manifest["files"][rows_path.name]["sha256"]:
        raise ValueError("The frozen rule evaluation data changed.")
    rows = [json.loads(line) for line in rows_path.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    if args.source_category_control:
        source_predictions = {row["source_request_sha256"]: row for row in predictions}
        if len(source_predictions) != len(predictions):
            raise ValueError("A source category may only be predicted once in this control.")
        predictions = [
            decomposition_prediction(
                row,
                source_predictions[row["source"]["original_input_sha256"]],
                seed=manifest["seed"],
            )
            for row in rows
            if row["source"]["original_input_sha256"] in source_predictions
        ]
    result = evaluate(rows, predictions)
    result.update(
        dataset_manifest_sha256=file_hash(args.dataset / "manifest.json"),
        predictions_file_sha256=file_hash(args.predictions),
        method="source_category_then_known_policy"
        if args.source_category_control
        else "joint_model",
    )
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "questions",
                    "components",
                    "valid_responses",
                    "correct",
                    "accuracy_with_failures",
                    "component_macro_accuracy",
                    "nll",
                    "brier",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
