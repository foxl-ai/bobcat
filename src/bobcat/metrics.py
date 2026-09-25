from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import beta

from bobcat.schema import SENTINELS


def probabilities(logits: list[float], temperature: float = 1.0) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    array = np.asarray(logits, dtype=np.float64) / temperature
    if not np.isfinite(array).all():
        raise ValueError("Valid candidate logits must be finite.")
    return np.exp(array - logsumexp(array))


def scored_row(row: dict, temperature: float = 1.0) -> dict:
    if row.get("supervision", "hard_label") != "hard_label" or row.get("target") is None:
        raise ValueError("Categorical accuracy/NLL require a category, not an ordinal mean.")
    probs = probabilities(row["logits"], temperature)
    ids = row["candidate_ids"]
    # Legacy synthetic evaluations use an ID tie-break. Public typed responses
    # use request order; keep their evaluator consistent with the returned answer.
    tie_break = row.get("tie_break", "candidate_id")
    if tie_break == "request_order":
        index = max(range(len(ids)), key=lambda i: probs[i])
    elif tie_break == "candidate_id":
        index = min(range(len(ids)), key=lambda i: (-probs[i], ids[i]))
    else:
        raise ValueError("Unknown evaluation tie-break rule.")
    gold = ids.index(row["target"])
    scaled = np.asarray(row["logits"], dtype=np.float64) / temperature
    # Work in log space: exp(log_p) can underflow to zero even when log_p is
    # finite. Clipping that zero would hide the worst confidently wrong answers.
    nll = float(logsumexp(scaled - scaled[gold]))
    top_probability = float(probs[index])
    return {
        **row,
        "probabilities": probs.tolist(),
        "prediction": ids[index],
        "top_probability": top_probability,
        # Preserve the original synthetic-report key for existing consumers.
        # This alias is NOT the System One adapter's concentration statistic.
        "confidence": top_probability,
        "confidence_semantics": "legacy_alias_of_top_probability",
        "correct": ids[index] == row["target"],
        "nll": nll,
        "gold_probability_underflowed": bool(probs[gold] == 0),
        "brier": float(np.square(probs - np.eye(len(ids))[gold]).sum()),
    }


def basic_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    accuracy = float(np.mean([r["correct"] for r in rows]))
    ece = 0.0
    bins = []
    for bin_index in range(10):
        lower = bin_index / 10
        selected = [r for r in rows if min(9, int(r["top_probability"] * 10)) == bin_index]
        if selected:
            conf = float(np.mean([r["top_probability"] for r in selected]))
            acc = float(np.mean([r["correct"] for r in selected]))
            ece += len(selected) / len(rows) * abs(conf - acc)
            bins.append(
                {
                    "lower": float(lower),
                    "count": len(selected),
                    "accuracy": acc,
                    "mean_top_probability": conf,
                    # Legacy report alias, not public adapter confidence.
                    "mean_confidence": conf,
                }
            )
    return {
        "count": len(rows),
        "accuracy": accuracy,
        "nll": float(np.mean([r["nll"] for r in rows])),
        "nll_method": "log_softmax_from_raw_logits",
        "gold_probability_underflow_count": sum(r["gold_probability_underflowed"] for r in rows),
        "brier_sum": float(np.mean([r["brier"] for r in rows])),
        "ece_10_equal_width_bins": float(ece),
        "calibration_signal": "top_probability",
        "legacy_confidence_semantics": "alias_of_top_probability_not_adapter_confidence",
        "calibration_bins": bins,
    }


def grouped_interval(rows: list[dict], seed: int = 73, samples: int = 1000) -> list[float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(float(row["correct"]))
    arrays = [np.array(values) for values in groups.values()]
    if not arrays:
        return [0.0, 0.0]
    sums = np.array([values.sum() for values in arrays])
    counts = np.array([len(values) for values in arrays])
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(arrays), size=(samples, len(arrays)))
    estimates = sums[selections].sum(1) / counts[selections].sum(1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def evaluate_rows(
    raw_rows: list[dict], temperature: float = 1.0, policy: dict | None = None
) -> dict:
    rows = [scored_row(row, temperature) for row in raw_rows]
    report = basic_metrics(rows)
    report["temperature"] = temperature
    report["accuracy_95pct_world_bootstrap_interval"] = grouped_interval(rows)
    for field in ["family", "kind"]:
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        report[f"by_{field}"] = {name: basic_metrics(group) for name, group in groups.items()}
    report["macro_family_accuracy"] = (
        float(np.mean([value["accuracy"] for value in report["by_family"].values()]))
        if rows
        else 0.0
    )
    categories = defaultdict(list)
    for row in rows:
        category = row["target"] if row["target"] in SENTINELS else "candidate"
        categories[category].append(row)
    report["by_target_type"] = {name: basic_metrics(group) for name, group in categories.items()}
    pairs = defaultdict(dict)
    for row in rows:
        if row.get("pair_id"):
            pairs[row["pair_id"]][row["variant"]] = row
    changed_pairs = [
        (pair["base"], pair["counterfactual"])
        for pair in pairs.values()
        if "base" in pair
        and "counterfactual" in pair
        and pair["base"]["target"] != pair["counterfactual"]["target"]
    ]
    report["counterfactuals"] = {
        "gold_changed_pairs": len(changed_pairs),
        "both_correct": float(
            np.mean([first["correct"] and second["correct"] for first, second in changed_pairs])
        )
        if changed_pairs
        else None,
        "prediction_changed": float(
            np.mean(
                [first["prediction"] != second["prediction"] for first, second in changed_pairs]
            )
        )
        if changed_pairs
        else None,
    }
    # This curve is descriptive only. It never chooses a deployment threshold.
    eligible = sorted(
        [row for row in rows if row["prediction"] not in SENTINELS],
        key=lambda row: -row["top_probability"],
    )
    curve = []
    for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]:
        subset = eligible[: max(1, math.ceil(len(eligible) * fraction))]
        curve.append(
            {
                "coverage": len(subset) / len(rows) if rows else 0,
                "risk": float(np.mean([not r["correct"] for r in subset])) if subset else None,
            }
        )
    report["diagnostic_risk_coverage"] = curve
    if policy is not None:
        accepted = [
            row
            for row in rows
            if policy["threshold"] is not None
            and row["prediction"] not in SENTINELS
            and row["top_probability"] >= policy["threshold"]
        ]
        report["frozen_referral_policy"] = {
            "threshold": policy["threshold"],
            "accepted": len(accepted),
            "coverage": len(accepted) / len(rows) if rows else 0,
            "observed_risk": (
                float(np.mean([not row["correct"] for row in accepted])) if accepted else None
            ),
            "note": "Calibration guarantee assumes the same distribution; not an OOD guarantee.",
        }
    return report


def fit_temperature(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("Temperature fitting requires an independent calibration partition.")
    if any(row["split"] != "cal_temperature" for row in rows):
        raise ValueError("Only cal_temperature may fit temperature.")

    def objective(log_temperature: float) -> float:
        return float(
            np.mean([scored_row(row, float(np.exp(log_temperature)))["nll"] for row in rows])
        )

    result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed.")
    return float(np.exp(result.x))


def binomial_upper(errors: int, count: int, delta: float) -> float:
    if count == 0 or errors == count:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, count - errors))


def fit_referral_policy(
    rows: list[dict], temperature: float, target_risk: float = 0.05, delta: float = 0.05
) -> dict:
    if not rows or any(row["split"] != "cal_policy" for row in rows):
        raise ValueError("Only the separate cal_policy partition may select referral thresholds.")
    if not 0 < target_risk < 1 or not 0 < delta < 1:
        raise ValueError("Risk and delta must be in (0, 1).")
    # Questions/counterfactuals in a world are correlated. Select one observation per
    # world by a label-independent hash before applying a binomial confidence bound.
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    independent = [
        min(group, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        for group in groups.values()
    ]
    scored = [scored_row(row, temperature) for row in independent]
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
    candidates = []
    for threshold in thresholds:
        accepted = [
            row
            for row in scored
            if row["prediction"] not in SENTINELS and row["top_probability"] >= threshold
        ]
        errors = sum(not row["correct"] for row in accepted)
        upper = binomial_upper(errors, len(accepted), delta / len(thresholds))
        candidates.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "errors": errors,
                "risk_upper_bound": upper,
            }
        )
    valid = [row for row in candidates if row["risk_upper_bound"] <= target_risk]
    best = max(valid, key=lambda row: row["accepted"]) if valid else None
    return {
        "threshold": best["threshold"] if best else None,
        "target_risk": target_risk,
        "delta": delta,
        "independent_world_samples": len(scored),
        "method": "one-per-world Clopper-Pearson upper bound with fixed-grid Bonferroni correction",
        "status": "supported_on_calibration_distribution" if best else "refer_all",
        "assumptions": (
            "Nominal bound for i.i.d. correctly labelled worlds. Stratified synthetic samples "
            "and distribution shift require separate validation; this is not a safety certificate."
        ),
        "candidates": candidates,
    }
