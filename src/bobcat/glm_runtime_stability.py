"""Matched repeat/interleave controls for a retained native GLM serving process."""

from __future__ import annotations

import time
from pathlib import Path

from bobcat.checkpoint_metrics import aggregate, enrich
from bobcat.corpus import atomic_json
from bobcat.glm_native_evaluate import read_suite
from bobcat.glm_serving_checkpoint import (
    CompiledReadout,
    distribution_comparison,
    latency_summary,
)
from bobcat.schema import file_hash

PAD_ALLOCATION = "buffer_init = torch.zeros if deterministic else torch.empty"


def zero_padding_patch(source: str):
    """Use the vendor's existing initialized allocation for scatter-padding slots."""
    if (source.count(PAD_ALLOCATION) != 1
            or source.count("input_tensor = buffer_init(") != 1
            or source.count("input_tensor_scale = buffer_init(") != 1
            or source.count("m_indices = buffer_init(") != 1):
        raise ValueError("The pinned DeepGEMM padding allocation changed.")
    return source.replace(PAD_ALLOCATION, "buffer_init = torch.zeros", 1)


def matched_controls(suite: Path, out: Path, *, client, model_path, run_id, seconds=240):
    if out.exists() or not 90 <= seconds <= 420:
        raise ValueError("Use a new, bounded runtime control directory.")
    _, available = read_suite(suite)
    rows = available[:64]
    if len(rows) != 64:
        raise ValueError("The fixed runtime study requires 64 development components.")
    out.mkdir(parents=True)
    start = time.monotonic()
    deadline = start + seconds
    record = {
        "schema": "bobcat-glm-runtime-stability-v1", "run_id": run_id,
        "status": "running", "planned_questions": 64,
        "suite_sha256": file_hash(suite / "manifest.json"),
        "inputs": "first 64 frozen development components; not selected by observed errors",
        "warmup": "one full pass of the same 64 prompts before measurement",
        "readout_mode": "prefill_only", "model_weights_modified": False,
        "release_gate_passed": False, "calibration_fitted": False,
    }
    atomic_json(out / "controls.json", record)
    scorer = CompiledReadout(client, model_path=model_path, run_id=run_id)
    measurements, predictions = [], {}

    def request(batch, label):
        if deadline - time.monotonic() < 65:
            raise TimeoutError("Runtime control deadline leaves no ordinary HTTP allowance.")
        values, measured = scorer.read(batch, label=label)
        measurements.append({**measured, "phase": label.split("-")[0]})
        return values

    try:
        # No labels are sent. Warming every control shape is recorded, not hidden.
        for index, row in enumerate(rows):
            request([row], f"warm-{index}")
        for label, ordered, batch_size in (
            ("serial", rows, 1), ("reverse", list(reversed(rows)), 1), ("batch", rows, 8),
        ):
            values = []
            for index in range(0, len(ordered), batch_size):
                values.extend(request(ordered[index:index + batch_size], f"{label}-{index}"))
            by_id = {row["id"]: row for row in values}
            predictions[label] = [by_id[row["id"]] for row in rows]
            atomic_json(out / f"{label}.json", predictions[label])
        record.update(
            status="completed",
            serial_repeat=distribution_comparison(predictions["serial"], predictions["reverse"]),
            serial_vs_batch=distribution_comparison(predictions["serial"], predictions["batch"]),
            metrics=aggregate(list(map(enrich, predictions["serial"]))),
            latency=latency_summary([m for m in measurements if m["phase"] == "serial"]),
        )
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        atomic_json(out / "measurements.json", measurements)
        atomic_json(out / "controls.json", record)
    return record
