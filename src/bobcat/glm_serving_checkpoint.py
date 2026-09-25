"""Measure frozen compiled judgments through the native zero-generation API.

The suite is development monitoring, not a fresh release test. Tokenization is
excluded from these HTTP timings; a separately recorded typed smoke request
exercises compilation and the public response adapter.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bobcat.checkpoint_metrics import aggregate, enrich
from bobcat.corpus import atomic_json
from bobcat.glm_native_evaluate import read_suite
from bobcat.glm_readout import CompiledRequest, extract_scores, native_payload
from bobcat.metrics import probabilities
from bobcat.schema import file_hash

RESULT_FIELDS = (
    "id", "group_id", "input_sha256", "kind", "task", "language",
    "language_origin", "supervision", "target_index", "score_mean", "input_tokens",
)


class CompiledReadout:
    def __init__(self, client, *, model_path, run_id):
        self.client, self.run_id = client, run_id
        response = client.get("/get_model_info")
        response.raise_for_status()
        self.model_info = response.json()
        if self.model_info.get("model_path") != model_path:
            raise ValueError("Native endpoint loaded a different model mount.")

    def read(self, rows, *, label):
        if not 1 <= len(rows) <= 8:
            raise ValueError("Use between one and eight explicitly independent questions.")
        compiled = CompiledRequest(
            input_ids=[row["input_ids"] for row in rows],
            option_token_ids=[row["option_token_ids"] for row in rows],
            shared_prefix_tokens=0,
            logical_input_tokens=sum(row["input_tokens"] for row in rows),
        )
        payload = native_payload(compiled, readout_mode="prefill_only")
        # Defense in depth for this uncached experiment, not a shared-state claim.
        # IDs and gold never become model input, and this salt has no model tokens.
        payload["cache_salt"] = [f"{self.run_id}:{label}:{i}" for i in range(len(rows))]
        started = time.perf_counter()
        response = self.client.post("/generate", json=payload)
        response.raise_for_status()
        values = response.json()
        elapsed = time.perf_counter() - started
        if not isinstance(values, list) or len(values) != len(rows):
            raise ValueError("Native response changed the number of planned questions.")
        predictions = []
        for gold, value in zip(rows, values, strict=True):
            if value.get("text") not in ("", None) or value.get("output_ids") not in ([], None):
                raise ValueError("Zero-generation readout returned generated output.")
            logits = extract_scores(
                value["meta_info"], gold["option_token_ids"], gold["input_tokens"],
                readout_mode="prefill_only",
            )
            cached = value["meta_info"].get("cached_tokens")
            if cached not in (None, 0):
                raise ValueError("The explicitly uncached comparison reused prompt tokens.")
            predictions.append({
                **{name: gold[name] for name in RESULT_FIELDS}, "logits": logits,
                "native_completion_tokens": value["meta_info"]["completion_tokens"],
                "native_scored_positions": 1, "cached_tokens_reported": cached,
                "checkpoint": self.run_id,
            })
        return predictions, {
            "native_http_seconds": elapsed, "questions": len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "maximum_input_tokens": max(row["input_tokens"] for row in rows),
            "candidate_counts": [len(row["option_token_ids"]) for row in rows],
            "scope": "native HTTP incl serialization, queue and model; excludes tokenization",
            "native_completion_tokens": 0, "decoded_token_loop_verified_by_trace": False,
            "sequential_latency_per_question": elapsed if len(rows) == 1 else None,
        }


def distribution_comparison(before, after):
    if ([row["id"] for row in before] != [row["id"] for row in after]
            or any(a["input_sha256"] != b["input_sha256"]
                   for a, b in zip(before, after, strict=True))):
        raise ValueError("Compare identical ordered questions.")
    tvs = [
        float(np.abs(probabilities(a["logits"]) - probabilities(b["logits"])).sum() / 2)
        for a, b in zip(before, after, strict=True)
    ]
    flips = sum(np.argmax(a["logits"]) != np.argmax(b["logits"])
                for a, b in zip(before, after, strict=True))
    return {
        "questions": len(tvs), "maximum_probability_tv": max(tvs, default=0.),
        "mean_probability_tv": float(np.mean(tvs)) if tvs else 0.,
        "argmax_changes": int(flips), "maximum_tv_limit": .001,
        "allowed_argmax_changes": 0,
        "passed": bool(tvs and max(tvs) <= .001 and flips == 0),
        "quality_or_release_gate": False,
    }


def latency_summary(measurements):
    serial = [row for row in measurements if row["questions"] == 1]
    output = {}
    for label, low, high in (
        ("all", 0, 32768), ("under_512", 0, 512), ("512_to_1023", 512, 1024),
        ("1024_to_2047", 1024, 2048), ("2048_to_32767", 2048, 32768),
    ):
        times = [row["native_http_seconds"] for row in serial
                 if low <= row["input_tokens"] < high]
        if times:
            output[label] = {
                "requests": len(times), "p50_seconds": float(np.quantile(times, .5)),
                "p95_seconds": float(np.quantile(times, .95)),
                "p99_seconds": float(np.quantile(times, .99)),
            }
    return {"serial": output, "warmup_included": False,
            "scope": "precompiled native HTTP; full public API latency not measured"}


def run(suite_folder, out, *, client, model_path, run_id, seconds):
    if seconds <= 0 or seconds > 1200 or out.exists():
        raise ValueError("Use a new directory and at most 20 minutes of measured inference.")
    suite, rows = read_suite(suite_folder)
    out.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + seconds
    record = {
        "schema": "bobcat-glm-compiled-serving-evaluation-v1",
        "started_at": datetime.now(UTC).isoformat(), "status": "starting",
        "run_id": run_id, "model_path": model_path,
        "suite_sha256": file_hash(suite_folder / "manifest.json"),
        "suite_content_sha256": suite["content_sha256"],
        "planned_questions": len(rows), "completed_questions": 0,
        "evaluation_role": "development_monitoring",
        "training_or_calibration": False, "release_gate_passed": False,
        "maximum_seconds": seconds, "readout_mode": "prefill_only",
        "shared_state_reuse_claimed": False,
    }
    predictions, timings = [], []
    atomic_json(out / "evaluation.json", record)

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("The frozen serving-evaluation allowance expired.")

    try:
        scorer = CompiledReadout(client, model_path=model_path, run_id=run_id)
        record["model_info"] = scorer.model_info
        # Distinct lengths and candidate widths compile before measured requests.
        warm = sorted({0, len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4})
        warmups = []
        for index in warm:
            check_time()
            _, measured = scorer.read([rows[index]], label=f"warm-{index}")
            warmups.append(measured)
        atomic_json(out / "warmup.json", warmups)
        record["status"] = "evaluating_serial"
        with (out / "predictions.jsonl").open("x") as pred_stream, (
            out / "measurements.jsonl"
        ).open("x") as time_stream:
            for index, row in enumerate(rows):
                check_time()
                result, measured = scorer.read([row], label=f"eval-{index}")
                predictions.extend(result)
                timings.append(measured)
                pred_stream.write(json.dumps(result[0], ensure_ascii=False, allow_nan=False) + "\n")
                time_stream.write(json.dumps(measured, allow_nan=False) + "\n")
                record["completed_questions"] = len(predictions)
                if (index + 1) % 32 == 0:
                    pred_stream.flush()
                    time_stream.flush()
                    record["updated_at"] = datetime.now(UTC).isoformat()
                    atomic_json(out / "evaluation.json", record)
        count = min(32, len(rows))
        repeated, batches = [], []
        for index in range(count):
            check_time()
            values, _ = scorer.read([rows[index]], label=f"repeat-{index}")
            repeated.extend(values)
        for offset in range(0, count, 8):
            check_time()
            values, measured = scorer.read(rows[offset:offset + 8], label=f"batch-{offset}")
            batches.extend(values)
            atomic_json(out / f"batch-{offset:03d}.json", {
                "predictions": values, "measurement": measured,
                "seconds_per_question_is_batch_amortization_not_request_latency": True,
            })
        atomic_json(out / "repeated.json", repeated)
        record["serial_repeat"] = distribution_comparison(predictions[:count], repeated)
        record["serial_vs_batch"] = distribution_comparison(predictions[:count], batches)
        scored = list(map(enrich, predictions))
        record["metrics"] = aggregate(scored)
        record["by_language"] = {
            language: aggregate([row for row in scored if row["language"] == language])
            for language in sorted({row["language"] for row in scored})
        }
        record["latency"] = latency_summary(timings)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(out / "evaluation.json", record)
    return record


def main():
    import httpx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    with httpx.Client(base_url=args.upstream, timeout=60) as client:
        record = run(args.suite, args.out, client=client, model_path=args.model_path,
                     run_id=args.run_id, seconds=args.seconds)
    print(json.dumps({key: record[key] for key in (
        "status", "completed_questions", "metrics", "latency", "serial_repeat", "serial_vs_batch",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
