"""Collect bounded SGLang traces on an exclusively owned localhost server.

The API was checked against the source recovered from the pinned Bobcat image.
Trace collection is evidence for a later analysis, not proof of prefix reuse.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash, json_hash

IMAGE = "lmsysorg/sglang@sha256:d246b29aa6543ab28859439d4143bbf553b44cc74b29a98f4ee8768b401de252"
PROFILE_SOURCE_SHA256 = "8e4a29923065c2c674151e94d1d4355ee295f1fca0104d36d5de70e5f7e22fe2"
MAX_TRACE_BYTES = 512 * 1024**2


def _union_duration(intervals):
    """Union on one device's trace clock; never sum this across GPUs as latency."""
    total, left, right = 0.0, None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0.0)


def summarize_trace(path: Path, *, max_bytes=MAX_TRACE_BYTES):
    """Read one bounded Chrome trace, preserving overlap and missing-GPU evidence."""
    if not 1 <= max_bytes <= MAX_TRACE_BYTES:
        raise ValueError("A positive bounded trace allowance is required.")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("Use a regular, bounded trace file.")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("Decompressed trace exceeds its byte allowance.")
    parsed = json.loads(payload)
    events = parsed.get("traceEvents") if isinstance(parsed, dict) else parsed
    if not isinstance(events, list):
        raise ValueError("A Chrome trace must contain traceEvents.")
    names, categories, kernels, shapes = Counter(), Counter(), [], Counter()
    durations, intervals = {}, {}
    annotations = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Malformed trace event.")
        if event.get("ph") != "X":
            continue
        name, category = event.get("name", ""), event.get("cat", "")
        if not isinstance(name, str) or not isinstance(category, str):
            raise ValueError("Malformed trace event name or category.")
        start, duration = event.get("ts"), event.get("dur")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) for x in (start, duration)) or duration < 0:
            raise ValueError("A complete event needs finite timestamps and nonnegative duration.")
        if not math.isfinite(start + duration):
            raise ValueError("A complete event needs a finite end timestamp.")
        categories[category] += 1
        args = event.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Malformed trace event arguments.")
        if category == "kernel":
            # Each device is analyzed separately even if a trace contains several.
            device = str(args.get("device", event.get("pid", "unknown")))
            kernels.append(event)
            names[name] += 1
            durations[device] = durations.get(device, 0.0) + duration
            intervals.setdefault(device, []).append((start, start + duration))
        if category == "cpu_op" and "Input Dims" in args:
            shapes[(name, json.dumps(args["Input Dims"], sort_keys=True))] += 1
        if category == "user_annotation" and len(annotations) < 128:
            annotations.append({"name": name, "ts": start, "dur": duration, "args": args})
    return {
        "file": path.name, "sha256": file_hash(path),
        "compressed_bytes": path.stat().st_size, "decoded_bytes": len(payload),
        "events": len(events), "complete_event_categories": dict(categories),
        "gpu_kernel_events": len(kernels),
        "per_device": {
            device: {
                "sum_kernel_duration_us": durations[device],
                "union_kernel_active_us": _union_duration(spans),
                "first_to_last_kernel_us": max(b for _, b in spans) - min(a for a, _ in spans),
            } for device, spans in intervals.items()
        },
        "most_frequent_kernel_names": [
            {"name": name, "count": count} for name, count in names.most_common(100)
        ],
        "operator_shape_counts": [
            {"name": name, "input_dims": json.loads(shape), "count": count}
            for (name, shape), count in shapes.most_common(128)
        ],
        "annotations_first_128": annotations,
        "timing_scope": "Per-device trace clock; kernel sums include overlap and are not latency.",
        "physical_prefix_reuse_verified": False,
    }


def collect_rank_traces(directory: Path, profile_id: str, expected_ranks: int):
    if not re.fullmatch(r"[a-f0-9]{32}", profile_id) or not 1 <= expected_ranks <= 64:
        raise ValueError("Use an opaque profile ID and a bounded expected TP rank count.")
    summaries, seen = [], set()
    for path in sorted(directory.glob(f"{profile_id}-TP-*.trace.json.gz")):
        match = re.fullmatch(
            re.escape(profile_id) + r"-TP-(\d+)(?:-(?:DP|PP|EP)-\d+)*\.trace\.json\.gz",
            path.name,
        )
        if match is None:
            raise ValueError("Unexpected profile rank filename.")
        rank = int(match[1])
        if rank in seen or rank >= expected_ranks:
            raise ValueError("Duplicate or unexpected TP rank; do not combine different replicas.")
        seen.add(rank)
        summaries.append({"tp_rank": rank, **summarize_trace(path)})
    missing = sorted(set(range(expected_ranks)) - seen)
    return {
        "traces": summaries, "missing_tp_ranks": missing,
        "gpu_kernel_profile_collected": (
            not missing and all(row["gpu_kernel_events"] > 0 for row in summaries)
        ),
        "physical_prefix_reuse_verified": False,
    }


def profile_generation(
    client, payload: dict, host_trace_root: Path, server_trace_root: str, *,
    expected_ranks: int, deadline: float, dedicated_server=False,
):
    """Profile exactly one native generation request, without automatic HTTP retries.

    Caller owns the server exclusively and supplies an existing writable mount
    visible under both trace roots. A failed/uncertain stop requires ending that
    dedicated server before another experiment; it must not be silently reused.
    """
    server_root = PurePosixPath(server_trace_root)
    remaining = deadline - time.monotonic()
    if (not dedicated_server or client.base_url.host not in ("127.0.0.1", "::1", "localhost")
            or not host_trace_root.is_dir() or not server_root.is_absolute()
            or ".." in server_root.parts or not 1 <= expected_ranks <= 64
            or not 10 <= remaining <= 600):
        raise ValueError("Use an exclusive localhost server, mapped roots and a finite deadline.")
    completion_limit = payload.get("sampling_params", {}).get("max_new_tokens")
    if (type(completion_limit) is not int or completion_limit not in (0, 1)
            or not isinstance(payload.get("input_ids"), list) or not payload["input_ids"]):
        raise ValueError("Profile a frozen, nonempty one-position decision batch.")
    if completion_limit == 0 and (
        payload.get("return_logprob") is not True
        or payload.get("logprob_start_len") != -1
        or not isinstance(payload.get("token_ids_logprob"), list)
        or len(payload["token_ids_logprob"]) != len(payload["input_ids"])
        or any(not ids for ids in payload["token_ids_logprob"])
    ):
        raise ValueError("A zero-completion profile must request the first-position scores.")
    profile_id = uuid.uuid4().hex
    directory = host_trace_root / profile_id
    directory.mkdir()
    record = {
        "schema": "bobcat-glm-profile-v1", "profile_id": profile_id, "status": "starting",
        "started_at": datetime.now(UTC).isoformat(), "expected_tp_ranks": expected_ranks,
        "adapter_sha256": file_hash(Path(__file__)),
        "reviewed_image": IMAGE, "reviewed_profiler_source_sha256": PROFILE_SOURCE_SHA256,
        "runtime_image_verified_by_this_helper": False,
        "request_payload_sha256": json_hash(payload),
        "compiled_input_sha256": json_hash(payload["input_ids"]),
        "native_sequences": len(payload["input_ids"]),
        "requested_new_tokens_per_sequence": completion_limit,
        "zero_decode_steps_verified": False,
        "gpu_kernel_profile_collected": False, "physical_prefix_reuse_verified": False,
        "start_attempted": False, "stop_attempted": False,
        "profiler_stop_confirmed": False, "server_quiescence_verified": False,
        "timing_scope": "Instrumentation run; not optimized latency.",
    }
    output = directory / "profile.json"
    atomic_json(output, record)

    def post(endpoint, **kwargs):
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Finite profiling deadline reached.")
        response = client.post(endpoint, timeout=min(60, left), **kwargs)
        response.raise_for_status()
        return response

    started, start_uncertain, failure, response_body = False, False, None, None
    try:
        record["start_attempted"] = True
        atomic_json(output, record)
        try:
            response = post("/start_profile", json={
                "output_dir": str(server_root / profile_id), "profile_id": profile_id,
                "activities": ["CPU", "GPU"], "record_shapes": True, "with_stack": False,
                "merge_profiles": False, "detailed_annotations": True,
                # Manual bounded stop avoids the num_steps auto-stop race.
            })
        except Exception as error:
            # An explicit HTTP rejection does not grant ownership of a profiler.
            start_uncertain = getattr(error, "response", None) is None
            raise
        started = True
        record.update(status="recording", start_response=response.text[:500])
        atomic_json(output, record)
        began = time.monotonic()
        response_body = post("/generate", json=payload).json()
        record["instrumented_generate_wall_seconds"] = time.monotonic() - began
    except BaseException as error:
        failure = error
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
    finally:
        if started or start_uncertain:
            record["stop_attempted"] = True
            atomic_json(output, record)
            try:
                stopped = post("/stop_profile")
                record.update(stop_response=stopped.text[:500], profiler_stop_confirmed=True)
            except Exception as error:
                record.update(stop_error_type=type(error).__name__, stop_error=str(error)[:1000])
                failure = failure or error
        if failure is None:
            try:
                record.update(collect_rank_traces(directory, profile_id, expected_ranks))
                record["status"] = (
                    "collected" if record["gpu_kernel_profile_collected"]
                    else "missing_gpu_evidence"
                )
            except Exception as error:
                failure = error
                record.update(error_type=type(error).__name__, error=str(error)[:1000])
        if failure is not None:
            record["status"] = "failed"
        record["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(output, record)
    if failure is not None:
        raise failure
    return response_body, record
