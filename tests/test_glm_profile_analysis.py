import gzip
import json

import pytest

from bobcat.glm_profile import collect_rank_traces
from bobcat.glm_profile_analysis import analyze_profiles, extract_model_steps
from bobcat.schema import file_hash


def make_trace(path, tokens, *, detailed=True):
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "layer", "ts": n, "dur": 1}
        for n in range(140)
    ]
    name = f"step[EXTEND bs=2 toks={tokens}"
    if detailed:
        name += f" c_sq={tokens} c_sk=1000 c_sqsq={tokens**2} c_sqsk={tokens * 1000}"
    events.extend([
        {"ph": "X", "cat": "user_annotation", "name": name + "]", "ts": 150, "dur": 200},
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 160, "dur": 100,
         "args": {"device": 0, "stream": 1}},
        {"ph": "X", "cat": "kernel", "name": "nccl", "ts": 170, "dur": 30,
         "args": {"device": 0, "stream": 2}},
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)


def test_step_spans_are_not_lost_after_first_128_layer_annotations(tmp_path):
    path = tmp_path / "trace.json.gz"
    make_trace(path, 400)
    result = extract_model_steps(path, file_hash(path))
    assert result["model_forward_spans"] == 1
    assert result["token_shape_sums"]["c_sq"] == 400
    make_trace(path, 400, detailed=False)
    result = extract_model_steps(path, file_hash(path))
    assert result["all_spans_have_detailed_token_shapes"] is False
    assert result["token_shape_sums"] is None


def profile_rows(root):
    rows = []
    for index, (condition, tokens) in enumerate((
        ("isolated_batch", 400), ("shared_batch", 300), ("warm_shared_batch", 100),
    )):
        identifier = f"{index:032x}"
        directory = root / identifier
        for rank in range(2):
            make_trace(directory / f"{identifier}-TP-{rank}.trace.json.gz", tokens)
        rows.append({
            "block": 0, "condition": condition, "case_id": "one-case",
            "compiled_input_sha256": "abc", "logical_input_tokens": 250,
            "native_prompt_tokens": 400, "cached_tokens_reported": [0, 0],
            "profile": {
                **collect_rank_traces(directory, identifier, 2),
                "profile_id": identifier, "compiled_input_sha256": "abc",
                "profiler_stop_confirmed": True, "instrumented_generate_wall_seconds": 1.0,
            },
        })
    return rows


def test_rank_shapes_compared_without_summing_devices_as_latency(tmp_path):
    rows = profile_rows(tmp_path)
    result = analyze_profiles(rows, tmp_path, expected_ranks=2, expected_blocks=1)
    assert result["rank_traces_sha256_verified"] == 6
    shared, warm = result["comparisons"]
    assert shared["ranks"][0]["context_query_token_ratio"] == 0.75
    assert warm["ranks"][0]["context_query_token_ratio"] == 0.25
    assert warm["ranks"][0]["reference_per_device"]["0"]["union_kernel_active_us"] == 100
    assert warm["ranks"][0]["reference_per_device"]["0"]["sum_kernel_duration_us"] == 130
    assert result["physical_prefix_reuse_verified"] is False
    partial = analyze_profiles(rows[:2], tmp_path, expected_ranks=2, expected_blocks=1)
    assert partial["status"] == "partial" and partial["expected_rank_traces"] == 6


def test_input_mismatch_and_changed_rank_trace_fail_validation(tmp_path):
    rows = profile_rows(tmp_path)
    rows[1]["compiled_input_sha256"] = "changed"
    with pytest.raises(ValueError, match="inputs"):
        analyze_profiles(rows, tmp_path, expected_ranks=2, expected_blocks=1)
    rows[1]["compiled_input_sha256"] = "abc"
    path = tmp_path / rows[1]["profile"]["profile_id"] / rows[1]["profile"]["traces"][0]["file"]
    make_trace(path, 999)
    with pytest.raises(ValueError, match="changed"):
        analyze_profiles(rows, tmp_path, expected_ranks=2, expected_blocks=1)
