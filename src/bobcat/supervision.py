"""Keep observed categorical labels separate from observed ordinal means."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from bobcat.metrics import evaluate_rows

MEAN_TARGET = -100


def target_values(row: dict) -> tuple[int, float]:
    labels = row["candidate_ids"]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Supervision requires distinct offered candidates.")
    mode = row.get("supervision", "hard_label")
    if mode == "score_mean":
        value = row.get("score_target")
        if (row.get("kind") != "ordinal" or row.get("target") is not None
                or labels != [str(i) for i in range(len(labels))]
                or type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= len(labels) - 1):
            raise ValueError("An ordinal mean needs ordered levels and its original scalar target.")
        return MEAN_TARGET, float(value)
    if (mode != "hard_label" or row.get("target") not in labels
            or row.get("score_target") is not None):
        raise ValueError("Use an observed categorical label or an explicit ordinal mean.")
    return labels.index(row["target"]), 0.0


def target_tensors(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    values = [target_values(row) for row in rows]
    return (
        torch.tensor([index for index, _ in values], dtype=torch.long),
        torch.tensor([mean for _, mean in values], dtype=torch.float64),
    )


def losses(logits: torch.Tensor, targets: torch.Tensor,
           mean_values: torch.Tensor) -> torch.Tensor:
    """CE for categorical outcomes; squared error of expectation for ordinal means.

    Mean-only supervision does not identify the six-way label distribution.
    It must not be used to claim categorical calibration or invent vote counts.
    """
    if (logits.ndim != 2 or targets.shape != logits.shape[:1]
            or mean_values.shape != targets.shape or targets.dtype != torch.long
            or not torch.isfinite(mean_values).all()):
        raise ValueError("Misaligned supervision tensors.")
    mean_mask = targets.eq(MEAN_TARGET)
    result = F.cross_entropy(logits, targets, reduction="none", ignore_index=MEAN_TARGET)
    if not mean_mask.any():
        return result
    selected = logits[mean_mask]
    probabilities = selected.softmax(-1)
    levels = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
    prediction = (probabilities * levels).sum(-1)
    result = result.double()
    result[mean_mask] = (prediction.double() - mean_values[mean_mask]).square()
    return result


def evaluate_supervised(rows: list[dict]) -> dict:
    hard, means = [], []
    for row in rows:
        index, target = target_values(row)
        logits = np.asarray(row["logits"], dtype=np.float64)
        if logits.shape != (len(row["candidate_ids"]),) or not np.isfinite(logits).all():
            raise ValueError("Supervised evaluation needs every finite candidate score.")
        if index != MEAN_TARGET:
            hard.append(row)
            continue
        mass = np.exp(logits - logits.max())
        p = mass / mass.sum()
        means.append((float(p @ np.arange(len(p))), target))
    mean_metrics = {
        "questions": len(means), "mae": None, "mse": None, "pearson": None,
        "categorical_distribution_observed": False,
        "categorical_nll": None, "categorical_ece": None,
        "probability_calibration_verified": False,
    }
    if means:
        prediction, target = np.asarray(means).T
        mean_metrics.update(
            mae=float(np.abs(prediction - target).mean()),
            mse=float(np.square(prediction - target).mean()),
        )
        if len(means) > 1 and np.std(prediction) > 0 and np.std(target) > 0:
            mean_metrics["pearson"] = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "questions": len(rows), "hard_label_questions": len(hard),
        "hard_label_metrics": evaluate_rows(hard), "ordinal_mean_metrics": mean_metrics,
        "mean_scores_are_not_categorical_labels": True,
    }
