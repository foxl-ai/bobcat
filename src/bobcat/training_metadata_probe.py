"""Compare an input-metadata optimization without taking an optimizer step.

The callback performs the same forward/loss with caching disabled or enabled.
Both modes warm up before the balanced ABBA timing sequence. Exact loss,
readout and adapter-gradient equality is required; a local pass is not a
distributed admission or a service-latency claim.
"""

from __future__ import annotations

import statistics
import time


def distributed_metadata_admission(reports, *, minimum_speedup=1.05):
    """Admit only exact all-rank results with faster synchronous step timings."""
    import math

    if (len(reports) != 8 or {row.get("rank") for row in reports} != set(range(8))
            or not math.isfinite(minimum_speedup) or minimum_speedup < 1):
        raise ValueError("Require the complete eight-rank metadata comparison.")
    samples = []
    for index, enabled in enumerate((False, True, False, True, True, False)):
        timings = []
        for report in reports:
            if len(report.get("samples", [])) != 6:
                raise ValueError("Every rank must complete both warmups and the ABBA comparison.")
            sample = report["samples"][index]
            seconds = sample["forward_backward_seconds"]
            if (sample["cache_enabled"] is not enabled
                    or sample["warmup"] is not (index < 2)
                    or sample.get("optimizer_updated") is not False
                    or not math.isfinite(seconds) or seconds <= 0):
                raise ValueError("The distributed metadata timing sequence changed.")
            timings.append(seconds)
        samples.append({
            "cache_enabled": enabled, "warmup": index < 2,
            "slowest_rank_seconds": max(timings),
        })
    reference = statistics.median(
        row["slowest_rank_seconds"] for row in samples
        if not row["warmup"] and not row["cache_enabled"]
    )
    cached = statistics.median(
        row["slowest_rank_seconds"] for row in samples
        if not row["warmup"] and row["cache_enabled"]
    )
    exact = all(
        row.get("local_numerical_gate_passed") is True
        and row.get("finite") is True and row.get("readout_exact") is True
        and row.get("loss_exact") is True and not row.get("different_gradient_tensors")
        and row.get("adapter_weights_unchanged") is True and row.get("rng_restored") is True
        and row.get("optimizer_updates") == 0 for row in reports
    )
    speedup = reference / cached
    return {
        "all_ranks_numerical_gate_passed": exact,
        "activated": exact and speedup >= minimum_speedup,
        "minimum_speedup": minimum_speedup, "measured_speedup": speedup,
        "reference_median_seconds": reference, "cached_median_seconds": cached,
        "samples": samples, "performance_samples_per_mode": 2,
        "timing_scope": "slowest rank per same-input forward/backward, two warmups excluded",
        "fallback": None if exact and speedup >= minimum_speedup else "native_uncached_metadata",
        "service_latency_measured": False, "quality_gate_passed": False,
    }


def compare_metadata_backward(model, parameters, compute, *, device):
    import torch

    if not parameters or any(parameter.grad is not None for parameter in parameters.values()):
        raise ValueError("Run metadata admission with a nonempty adapter and no pending gradient.")

    def local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    def copies(values):
        return {name: local(value).detach().cpu().clone() for name, value in values.items()}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    before = copies(parameters)
    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    reference = None
    samples = []
    differences = set()
    finite = True
    readout_exact = loss_exact = True
    try:
        model.train()
        # Two warmups followed by ABBA. Compilation/warmup is not a speedup.
        for index, enabled in enumerate((False, True, False, True, True, False)):
            for parameter in parameters.values():
                parameter.grad = None
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            sync()
            begin = time.perf_counter()
            output = compute(enabled)
            loss, logits = output["loss"], output["logits"]
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("Metadata probe requires a finite scalar loss.")
            loss.backward()
            sync()
            elapsed = time.perf_counter() - begin
            if any(parameter.grad is None for parameter in parameters.values()):
                raise ValueError("The metadata probe missed an intended adapter gradient.")
            gradients = copies({name: parameter.grad for name, parameter in parameters.items()})
            current = {
                "gradients": gradients, "loss": loss.detach().cpu().clone(),
                "logits": logits.detach().cpu().clone(),
            }
            finite &= bool(torch.isfinite(current["logits"]).all()) and all(
                bool(torch.isfinite(value).all()) for value in gradients.values()
            )
            if reference is None:
                reference = current
            else:
                differences.update(
                    name for name, value in gradients.items()
                    if not torch.equal(reference["gradients"][name], value)
                )
                readout_exact &= torch.equal(reference["logits"], current["logits"])
                loss_exact &= torch.equal(reference["loss"], current["loss"])
            samples.append({
                "cache_enabled": enabled, "warmup": index < 2,
                "forward_backward_seconds": elapsed, "optimizer_updated": False,
            })
        unchanged = all(
            torch.equal(before[name], value) for name, value in copies(parameters).items()
        )
        if not unchanged:
            raise ValueError("Metadata admission unexpectedly changed an adapter weight.")
        measured = {
            enabled: [row["forward_backward_seconds"] for row in samples
                      if not row["warmup"] and row["cache_enabled"] == enabled]
            for enabled in (False, True)
        }
        return {
            "schema": "bobcat-training-metadata-admission-v1",
            "local_numerical_gate_passed": (
                finite and loss_exact and readout_exact and not differences
            ),
            "finite": finite, "readout_exact": readout_exact, "loss_exact": loss_exact,
            "different_gradient_tensors": sorted(differences),
            "adapter_weights_unchanged": unchanged, "optimizer_updates": 0,
            "rng_restored": True, "samples": samples,
            "reference_median_seconds": statistics.median(measured[False]),
            "cached_median_seconds": statistics.median(measured[True]),
            "timing_scope": "same-input forward/backward; excludes CPU evidence copies",
            "performance_samples_per_mode": 2,
            "distributed_gate_passed": False, "service_latency_measured": False,
        }
    finally:
        for parameter in parameters.values():
            parameter.grad = None
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
