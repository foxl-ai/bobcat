"""Preserve gradient checks and FP64 reduction order with fewer device synchronizations."""

from __future__ import annotations

import math
import statistics
import time


def gradient_norm_squared(gradients, *, batched):
    """Return the same left-to-right Python FP64 sum as the original trainer.

    Only finite flags and one scalar per gradient are copied to the host.
    Gradients, their elementwise reductions, clipping and optimizer are unchanged.
    """
    import torch

    gradients = list(gradients)
    if not gradients or any(gradient is None for gradient in gradients):
        raise ValueError("Require every intended trainable gradient.")
    local = [gradient.to_local() if hasattr(gradient, "to_local") else gradient
             for gradient in gradients]
    if any(gradient.device != local[0].device for gradient in local):
        raise ValueError("All local gradient statistics must belong to one device.")
    if not batched:
        total = 0.
        for gradient in local:
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError("Non-finite gradient.")
            total += float(gradient.double().square().sum())
        return total
    checks = torch.stack([torch.isfinite(gradient).all() for gradient in local])
    partials = torch.stack([gradient.double().square().sum() for gradient in local])
    if not bool(checks.all()):
        raise ValueError("Non-finite gradient.")
    values = partials.cpu().tolist()
    # Python 3.12+ sum(float_values) uses a different summation algorithm.
    # Keep exactly the original explicit scalar-add order.
    total = 0.
    for value in values:
        total += value
    return total


def compare_gradient_statistics(gradients, *, device):
    """Same live gradients, balanced timings, no forward/backward or parameter update."""
    import torch

    gradients = list(gradients)
    samples = []
    reference = None
    exact = True
    for index, batched in enumerate((False, True, False, True, True, False)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begin = time.perf_counter()
        value = gradient_norm_squared(gradients, batched=batched)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - begin
        if reference is None:
            reference = value
        else:
            exact &= value == reference
        samples.append({
            "batched": batched, "warmup": index < 2, "seconds": elapsed,
            "local_gradient_norm_squared": value,
        })
    return {
        "schema": "bobcat-gradient-statistics-admission-v1",
        "finite": math.isfinite(reference), "local_value_exact": exact,
        "gradient_tensors": len(gradients), "samples": samples,
        "gradient_modified": False, "optimizer_updated": False,
        "additional_model_forward_backward": False,
    }


def admit_gradient_statistics(reports, *, minimum_speedup=1.05):
    if (len(reports) != 8 or {row.get("rank") for row in reports} != set(range(8))
            or not math.isfinite(minimum_speedup) or minimum_speedup < 1):
        raise ValueError("Require eight unique ranks and an explicit performance threshold.")
    samples = []
    for index, batched in enumerate((False, True, False, True, True, False)):
        seconds = []
        for report in reports:
            if len(report.get("samples", [])) != 6:
                raise ValueError("The warmup and balanced gradient-statistics samples "
                                 "are incomplete.")
            row = report["samples"][index]
            if (row["batched"] is not batched or row["warmup"] is not (index < 2)
                    or not math.isfinite(row["seconds"]) or row["seconds"] <= 0):
                raise ValueError("The gradient-statistics timing protocol changed.")
            seconds.append(row["seconds"])
        samples.append({"batched": batched, "warmup": index < 2,
                        "slowest_rank_seconds": max(seconds)})
    reference = statistics.median(row["slowest_rank_seconds"] for row in samples
                                  if not row["warmup"] and not row["batched"])
    batched = statistics.median(row["slowest_rank_seconds"] for row in samples
                                if not row["warmup"] and row["batched"])
    exact = all(row.get("local_value_exact") is True and row.get("finite") is True
                and row.get("gradient_modified") is False
                and row.get("optimizer_updated") is False for row in reports)
    speedup = reference / batched
    return {
        "activated": exact and speedup >= minimum_speedup,
        "all_rank_values_exact": exact, "reference_median_seconds": reference,
        "batched_median_seconds": batched, "measured_speedup": speedup,
        "minimum_speedup": minimum_speedup, "samples": samples,
        "timing_scope": "gradient statistics only, slowest rank; warmups excluded",
        "whole_training_speedup_measured": False,
        "fallback": None if exact and speedup >= minimum_speedup else "original_scalar_checks",
    }
