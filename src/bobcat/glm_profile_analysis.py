"""Verify recovered rank traces and compare actual ModelRunner step annotations.

The annotation grammar comes from the source recovered from the pinned image,
not from a newer serving release. Token-shape aggregates are not FLOP counts.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_profile import MAX_TRACE_BYTES, collect_rank_traces
from bobcat.schema import file_hash

CONDITIONS = ("isolated_batch", "shared_batch", "warm_shared_batch")
STEP_FIELDS = {
    "bs", "toks", "c", "g",
    "c_sq", "c_sk", "c_sqsq", "c_sqsk", "g_sq", "g_sk", "g_sqsq", "g_sqsk",
}


def extract_model_steps(path: Path, expected_sha256: str):
    """Read all step spans, including later chunks beyond the preview's first 128 annotations."""
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_TRACE_BYTES
            or file_hash(path) != expected_sha256):
        raise ValueError("Recovered trace checksum or file boundary does not match.")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        payload = stream.read(MAX_TRACE_BYTES + 1)
    if len(payload) > MAX_TRACE_BYTES:
        raise ValueError("Decompressed trace exceeds its bounded allowance.")
    document = json.loads(payload)
    events = document["traceEvents"] if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ValueError("Trace events must be a list.")
    steps, operator_shapes = [], Counter()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid trace event.")
        if event.get("ph") != "X":
            continue
        if event.get("cat") == "cpu_op" and "Input Dims" in event.get("args", {}):
            operator_shapes[(event["name"], json.dumps(
                event["args"]["Input Dims"], sort_keys=True, separators=(",", ":"),
            ))] += 1
        if event.get("cat") != "user_annotation" or not event.get("name", "").startswith("step["):
            continue
        match = re.fullmatch(r"step\[(\w+)((?: \w+=\d+)*)\]", event["name"])
        if match is None:
            raise ValueError("Unrecognized step span; review the actual installed source.")
        fields = {}
        for term in match[2].strip().split():
            key, value = term.split("=")
            if key not in STEP_FIELDS or key in fields:
                raise ValueError("Unknown or duplicate step-span field.")
            fields[key] = int(value)
        if "bs" not in fields:
            raise ValueError("A step annotation must identify its batch size.")
        if match[1] == "EXTEND" and "c_sq" in fields and fields.get("toks") != fields["c_sq"]:
            raise ValueError("EXTEND token count disagrees with its detailed query aggregate.")
        steps.append({
            "mode": match[1], "fields": fields, "annotation": event["name"],
            "cpu_span_start_us": event.get("ts"), "cpu_span_duration_us": event.get("dur"),
        })
    detailed = bool(steps) and all(
        "c_sq" in row["fields"] or "g_sq" in row["fields"] for row in steps
    )
    return {
        "trace_sha256": expected_sha256, "steps": steps, "model_forward_spans": len(steps),
        "mode_counts": dict(Counter(row["mode"] for row in steps)),
        "all_spans_have_detailed_token_shapes": detailed,
        "token_shape_sums": {
            key: sum(row["fields"].get(key, 0) for row in steps)
            for key in sorted(STEP_FIELDS - {"bs", "toks", "c", "g"})
        } if detailed else None,
        "operator_shape_counts": [
            {"name": name, "input_dims": json.loads(shape), "count": count}
            for (name, shape), count in sorted(operator_shapes.items())
        ],
        "scope": "ModelRunner input shapes plus CPU operator shapes; not measured FLOPs. "
        "CPU span durations are not GPU latency.",
    }


def analyze_profiles(observations, trace_root, *, expected_ranks=8, expected_blocks=2):
    if not 1 <= expected_ranks <= 64 or not 1 <= expected_blocks <= 4:
        raise ValueError("Use bounded, explicitly expected rank and repeat counts.")
    seen, runs, hashes = set(), {}, set()
    for row in observations:
        key = (row["block"], row["condition"])
        if (key in seen or key[0] not in range(expected_blocks) or key[1] not in CONDITIONS):
            raise ValueError("Unknown or duplicate profile arm; preserve separate reruns.")
        seen.add(key)
        profile = row["profile"]
        if (profile["compiled_input_sha256"] != row["compiled_input_sha256"]
                or not profile["profiler_stop_confirmed"]):
            raise ValueError("Profile inputs or profiler ownership/stop evidence do not match.")
        hashes.add(row["compiled_input_sha256"])
        identifier = profile["profile_id"]
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Use the exact opaque profile directory.")
        directory = trace_root / identifier
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Recovered profile directory is missing or indirect.")
        recovered = collect_rank_traces(directory, identifier, expected_ranks)
        expected = {(t["tp_rank"], t["file"]): t["sha256"] for t in profile["traces"]}
        actual = {(t["tp_rank"], t["file"]): t["sha256"] for t in recovered["traces"]}
        if (actual != expected or len(expected) != expected_ranks
                or not recovered["gpu_kernel_profile_collected"]):
            raise ValueError("Expected GPU rank trace evidence is missing or changed.")
        ranks = []
        for trace in recovered["traces"]:
            ranks.append({
                "tp_rank": trace["tp_rank"], "gpu_kernel_events": trace["gpu_kernel_events"],
                "per_device": trace["per_device"],
                **extract_model_steps(directory / trace["file"], trace["sha256"]),
            })
        runs[key] = {
            "block": key[0], "condition": key[1], "case_id": row["case_id"],
            "compiled_input_sha256": row["compiled_input_sha256"],
            "logical_input_tokens": row["logical_input_tokens"],
            "native_prompt_tokens": row["native_prompt_tokens"],
            "cached_tokens_reported": row["cached_tokens_reported"],
            "instrumented_generate_wall_seconds": profile["instrumented_generate_wall_seconds"],
            "ranks": ranks,
        }
    if len(hashes) > 1:
        raise ValueError("The compared profile arms must have identical compiled input tokens.")
    expected_arms = {(b, c) for b in range(expected_blocks) for c in CONDITIONS}
    pairs = []
    for block in range(expected_blocks):
        baseline = runs.get((block, "isolated_batch"))
        for condition in CONDITIONS[1:]:
            other = runs.get((block, condition))
            if baseline is None or other is None:
                pairs.append({"block": block, "condition": condition, "status": "missing_arm"})
                continue
            rank_pairs = []
            for first, second in zip(baseline["ranks"], other["ranks"], strict=True):
                if first["tp_rank"] != second["tp_rank"]:
                    raise ValueError("Never align different tensor-parallel ranks.")
                a, b = first["token_shape_sums"], second["token_shape_sums"]
                rank_pairs.append({
                    "tp_rank": first["tp_rank"],
                    "reference_forward_spans": first["model_forward_spans"],
                    "condition_forward_spans": second["model_forward_spans"],
                    "reference_context_query_tokens": a["c_sq"] if a else None,
                    "condition_context_query_tokens": b["c_sq"] if b else None,
                    "context_query_token_ratio": b["c_sq"] / a["c_sq"]
                    if a and b and a["c_sq"] else None,
                    # Device clocks can overlap; retain devices, never sum ranks as latency.
                    "reference_per_device": first["per_device"],
                    "condition_per_device": second["per_device"],
                })
            pairs.append({
                "block": block, "condition": condition, "status": "compared", "ranks": rank_pairs,
                "reference_instrumented_wall_seconds": baseline[
                    "instrumented_generate_wall_seconds"
                ],
                "condition_instrumented_wall_seconds": other[
                    "instrumented_generate_wall_seconds"
                ],
            })
    return {
        "schema": "bobcat-glm-profile-analysis-v1",
        "status": "completed" if seen == expected_arms else "partial",
        "profiles_recovered": len(runs), "expected_profiles": len(expected_arms),
        "rank_traces_sha256_verified": len(runs) * expected_ranks,
        "expected_rank_traces": len(expected_arms) * expected_ranks,
        "missing_arms": [{"block": b, "condition": c} for b, c in sorted(expected_arms - seen)],
        "runs": [runs[key] for key in sorted(runs)], "comparisons": pairs,
        "physical_prefix_reuse_verified": False, "release_gate_passed": False,
        "scope": "Matched, instrumented development calls. Trace and shape evidence requires "
        "interpretation alongside numerical consistency and remaining kernel/communication work.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve prior analyses; use a new path.")
    records = json.loads(args.profiles.read_text())
    result = analyze_profiles(records["observations"], args.trace_root)
    result.update(
        analyzed_at=datetime.now(UTC).isoformat(),
        profiles_file_sha256=file_hash(args.profiles), source_sha256=file_hash(Path(__file__)),
    )
    atomic_json(args.out, result)
    print(json.dumps({k: result[k] for k in (
        "status", "profiles_recovered", "rank_traces_sha256_verified",
    )}))


if __name__ == "__main__":
    main()
