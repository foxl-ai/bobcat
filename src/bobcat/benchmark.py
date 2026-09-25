from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from bobcat.batching import EncodedDataset
from bobcat.model import DecisionModel
from bobcat.training import autocast_context, synchronize


@torch.inference_mode()
def benchmark_model(
    model: DecisionModel,
    data: EncodedDataset,
    device: torch.device,
    precision: str = "fp32",
    repetitions: int = 50,
) -> dict:
    if repetitions < 5:
        raise ValueError("Use at least five measured repetitions.")
    model.eval()
    indices = data.groups[data.group_keys[0]]
    shared = data.collate(indices).to(device)
    separate = data.collate(indices, reuse_context=False).to(device)
    variants = {
        "batched_full_forward": lambda: model(shared.model_inputs()),
    }
    if model.config.architecture == "shared":
        with autocast_context(device, precision):
            memory = model.encode_state(shared.tensors["context_ids"])
        variants["batched_without_context_reuse"] = lambda: model(separate.model_inputs())
        variants["resident_state_queries_only"] = lambda: model.decide(
            memory,
            shared.tensors["context_index"],
            shared.tensors["schema_ids"],
            shared.tensors["candidate_keep"],
            shared.tensors["schema_candidate_mask"],
        )
    measurements = {}
    for name, call in variants.items():
        for _ in range(5):
            with autocast_context(device, precision):
                call()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repetitions):
            synchronize(device)
            start = time.perf_counter()
            with autocast_context(device, precision):
                call()
            synchronize(device)
            durations.append((time.perf_counter() - start) * 1000)
        measurements[name] = {
            "p50_ms": statistics.median(durations),
            "p95_ms": float(np.quantile(durations, 0.95)),
            "mean_ms": statistics.mean(durations),
            "repetitions": repetitions,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
    return {
        "architecture": model.config.architecture,
        "reader_passes": model.config.reader_passes,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "precision": precision,
        "questions": len(indices),
        "shared_contexts": shared.unique_contexts,
        "context_shape": list(shared.tensors["context_ids"].shape),
        "schema_shape": list(shared.tensors["schema_ids"].shape),
        "joint_shape": list(shared.tensors["joint_ids"].shape),
        "measurements": measurements,
        "measurement_scope": (
            "Eager model calls on resident tensors; excludes tokenization, transfer, network, "
            "queueing and service overhead. Cached query timing excludes state encoding."
        ),
    }
