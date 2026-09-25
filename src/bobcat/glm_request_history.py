"""Separate immediate repeat variation from variation after an unrelated request."""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from bobcat.corpus import atomic_json
from bobcat.glm_native_evaluate import read_suite
from bobcat.glm_serving_checkpoint import CompiledReadout, distribution_comparison
from bobcat.schema import file_hash


def history_schedule(rows):
    """Use all first 64 frozen components, without selecting known failures."""
    if (len(rows) != 64 or len({r["group_id"] for r in rows}) != 64
            or any(r["split"] != "dev_train" for r in rows)):
        raise ValueError("Require 64 distinct frozen development components.")
    return [
        (row, rows[(index + 31) % 64])
        for index, row in enumerate(rows)
    ]


def summarize(completed):
    pairs = {
        "immediate_repeat": ("a0", "a1"),
        "after_unrelated_request": ("a1", "a2"),
        "post_intervention_repeat": ("a2", "a3"),
    }
    return {
        name: distribution_comparison(
            [case[left] for case in completed], [case[right] for case in completed])
        for name, (left, right) in pairs.items()
    }


def run(suite, out, *, client, model_path, seconds=360):
    if out.exists() or not 120 <= seconds <= 420:
        raise ValueError("Use a fresh directory and a bounded final-window study.")
    _, available = read_suite(suite)
    schedule = history_schedule(available[:64])
    out.mkdir(parents=True)
    start = time.monotonic()
    deadline = start + seconds
    record = {
        "schema": "bobcat-glm-request-history-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "suite_sha256": file_hash(suite / "manifest.json"),
        "selection": "first 64 frozen development components, no error-based selection",
        "sequence": ["a0", "a1", "unrelated_b", "a2", "a3"],
        "intervention": "component (index + 31) modulo 64",
        "planned_components": 64, "completed_components": 0,
        "planned_native_requests": 320, "attempted_native_requests": 0,
        "native_requests": 0, "processed_prompt_tokens": 0,
        "weights_updated": False, "calibration_fitted": False,
        "training_or_release": False, "release_gate_passed": False,
        "causal_kernel_identification_claimed": False,
        "scope": "serial zero-generation HTTP; cache off; fixed existing server",
    }
    atomic_json(out / "run.json", record)
    complete = []
    try:
        scorer = CompiledReadout(client, model_path=model_path, run_id="request-history")
        with (out / "observations.jsonl").open("x") as stream:
            for index, (original, intervening) in enumerate(schedule):
                case = {}
                for phase in record["sequence"]:
                    if deadline - time.monotonic() < 65:
                        record["status"] = "partial_time_budget"
                        break
                    row = intervening if phase == "unrelated_b" else original
                    record["attempted_native_requests"] += 1
                    values, measured = scorer.read([row], label=f"{index}-{phase}")
                    record["native_requests"] += 1
                    record["processed_prompt_tokens"] += row["input_tokens"]
                    case[phase] = values[0]
                    stream.write(json.dumps({
                        "case_index": index, "phase": phase,
                        "anchor_id": original["id"], "anchor_group_id": original["group_id"],
                        "prediction": values[0], "measurement": measured,
                    }, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                if record["status"] == "partial_time_budget":
                    break
                complete.append(case)
                record["completed_components"] = len(complete)
                record["updated_at"] = datetime.now(UTC).isoformat()
                atomic_json(out / "run.json", record)
            else:
                record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1200])
        raise
    finally:
        record["comparisons_completed_subset"] = summarize(complete)
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - start)
        atomic_json(out / "completed-cases.json", complete)
        atomic_json(out / "run.json", record)
    return record
