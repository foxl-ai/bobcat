"""Cross-fitted development diagnostic, never a deployable calibration artifact.

This asks whether an expensive update improves probabilities beyond a scalar
temperature. The development labels remain development labels. It deliberately
does not call or relax the production calibration partition checks.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from bobcat.checkpoint_metrics import aggregate, enrich


def crossfit_categorical(rows, *, folds=5, seed=2026092301):
    if (type(folds) is not int or folds < 2 or len(rows) < 2 * folds
            or len({row["group_id"] for row in rows}) != len(rows)
            or len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("Use distinct development components and nonempty fitting folds.")
    for row in rows:
        values = np.asarray(row["logits"], dtype=np.float64)
        if (row["supervision"] != "hard_label" or values.ndim != 1
                or len(values) < 2 or not np.isfinite(values).all()
                or type(row["target_index"]) is not int
                or not 0 <= row["target_index"] < len(values)):
            raise ValueError("Only finite categorical logits with native labels are supported.")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['group_id']}".encode()).hexdigest(),
    )
    assignment = {row["group_id"]: index % folds for index, row in enumerate(ordered)}
    fitting, temperatures = [], {}
    for fold in range(folds):
        train = [row for row in rows if assignment[row["group_id"]] != fold]
        held = [row for row in rows if assignment[row["group_id"]] == fold]
        train_groups = {row["group_id"] for row in train}
        held_groups = {row["group_id"] for row in held}
        if train_groups & held_groups:
            raise ValueError("A component crossed an optimization/held-out boundary.")

        def objective(log_temperature, fitting_rows=tuple(train)):
            temperature = np.exp(log_temperature)
            return float(np.mean([
                logsumexp(np.asarray(row["logits"], dtype=np.float64) / temperature)
                - row["logits"][row["target_index"]] / temperature
                for row in fitting_rows
            ]))

        result = minimize_scalar(objective, bounds=(-3., 3.), method="bounded")
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Scalar temperature optimization failed.")
        temperature = float(np.exp(result.x))
        fitting.append({
            "fold": fold, "temperature": temperature,
            "training_components": sorted(train_groups),
            "held_out_components": sorted(held_groups),
            "training_nll": float(result.fun), "log_temperature_bounds": [-3., 3.],
        })
        temperatures[fold] = temperature
    scored = []
    for row in rows:
        fold = assignment[row["group_id"]]
        scored.append({
            **row, "logits": (
                np.asarray(row["logits"], dtype=np.float64) / temperatures[fold]
            ).tolist(),
            "development_crossfit_fold": fold,
            "development_crossfit_temperature": temperatures[fold],
        })
    metrics = aggregate(list(map(enrich, scored)))
    metrics["raw_pmax_0_9_diagnostic"]["calibration_fitted"] = True
    return {
        "schema": "bobcat-crossfit-development-temperature-control-v1",
        "role": "exploratory_repeated_development_only",
        "folds": fitting, "seed": seed, "questions": len(rows),
        "metrics": metrics, "predictions": scored,
        "backbone_parameter_updates": 0, "scalar_temperatures_fitted": folds,
        "native_deployment_calibration": False, "release_gate_passed": False,
        "selection_bias_removed": False, "pretrained_contamination_removed": False,
        "ordinal_score_support": False,
        "caveat": "Cross-fitting holds out optimization components, but the suite was "
                  "already used for development and checkpoint selection. No final claim.",
    }
