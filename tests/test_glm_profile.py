import gzip
import json
import time

import httpx
import pytest

from bobcat.glm_profile import collect_rank_traces, profile_generation, summarize_trace


def write_trace(path, *, gpu=True):
    events = [
        {"ph": "X", "cat": "cpu_op", "name": "aten::linear", "ts": 1, "dur": 40,
         "args": {"Input Dims": [[8, 16], [16, 16]]}},
    ]
    if gpu:
        # Two streams overlap: summing 20 + 20 is not 30us of active device time.
        events.extend([
            {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 10, "dur": 20,
             "args": {"device": 0, "stream": 1}},
            {"ph": "X", "cat": "kernel", "name": "nccl", "ts": 20, "dur": 20,
             "args": {"device": 0, "stream": 2}},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy", "ts": 5, "dur": 2},
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": events}, stream)


def test_overlap_and_cpu_events_are_not_reported_as_gpu_latency(tmp_path):
    path = tmp_path / "trace.json.gz"
    write_trace(path)
    summary = summarize_trace(path)
    assert summary["gpu_kernel_events"] == 2
    assert summary["per_device"]["0"] == {
        "sum_kernel_duration_us": 40.0, "union_kernel_active_us": 30.0,
        "first_to_last_kernel_us": 30,
    }
    assert summary["operator_shape_counts"][0]["input_dims"] == [[8, 16], [16, 16]]
    assert summary["physical_prefix_reuse_verified"] is False


def test_missing_rank_or_cpu_only_trace_cannot_pass_collection(tmp_path):
    identifier = "a" * 32
    write_trace(tmp_path / f"{identifier}-TP-0-EP-0.trace.json.gz")
    record = collect_rank_traces(tmp_path, identifier, 2)
    assert record["missing_tp_ranks"] == [1]
    assert record["gpu_kernel_profile_collected"] is False
    write_trace(tmp_path / f"{identifier}-TP-1-EP-1.trace.json.gz", gpu=False)
    record = collect_rank_traces(tmp_path, identifier, 2)
    assert not record["missing_tp_ranks"] and not record["gpu_kernel_profile_collected"]
    write_trace(tmp_path / f"{identifier}-TP-0-DP-1-EP-0.trace.json.gz")
    with pytest.raises(ValueError, match="Duplicate"):
        collect_rank_traces(tmp_path, identifier, 2)


def test_decompression_limit_and_invalid_duration_fail(tmp_path):
    path = tmp_path / "oversized.json.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(b" " * 10000)
    with pytest.raises(ValueError, match="Decompressed"):
        summarize_trace(path, max_bytes=512)
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps([
        {"ph": "X", "cat": "kernel", "name": "bad", "ts": 0, "dur": float("nan")},
    ]))
    with pytest.raises(ValueError, match="finite"):
        summarize_trace(path)


def exercise(tmp_path, *, failure=None, dedicated=True, hostname="127.0.0.1"):
    calls, state = [], {}

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/start_profile":
            state["start"] = json.loads(request.content)
            if failure == "rejected":
                return httpx.Response(409, text="Another profile is active.")
            if failure == "start_timeout":
                raise httpx.ReadTimeout("Response lost", request=request)
            return httpx.Response(200, text="Start profiling.\n")
        if request.url.path == "/generate":
            if failure == "generation":
                return httpx.Response(500, text="Generation failed.")
            completion_limit = json.loads(request.content)["sampling_params"]["max_new_tokens"]
            return httpx.Response(
                200, json=[{"meta_info": {"completion_tokens": completion_limit}}],
            )
        assert request.url.path == "/stop_profile"
        if failure == "stop":
            raise httpx.ReadTimeout("Export timed out", request=request)
        if failure != "missing":
            identifier = state["start"]["profile_id"]
            for rank in range(2):
                write_trace(
                    tmp_path / identifier / f"{identifier}-TP-{rank}-EP-{rank}.trace.json.gz",
                )
        return httpx.Response(200, text="Stop profiling. This will take some time.\n")

    client = httpx.Client(
        base_url=f"http://{hostname}", transport=httpx.MockTransport(respond),
    )
    args = (
        client, {"input_ids": [[1, 2], [1, 3]],
                 "sampling_params": {"max_new_tokens": 1}},
        tmp_path, "/root/.cache/bobcat-profiles",
    )
    kwargs = {"expected_ranks": 2, "deadline": time.monotonic() + 120,
              "dedicated_server": dedicated}
    return calls, state, args, kwargs


def test_one_position_profile_preserves_evidence_without_claiming_reuse(tmp_path):
    calls, state, args, kwargs = exercise(tmp_path)
    response, record = profile_generation(*args, **kwargs)
    assert calls == ["/start_profile", "/generate", "/stop_profile"]
    assert response[0]["meta_info"]["completion_tokens"] == 1
    assert state["start"]["activities"] == ["CPU", "GPU"]
    assert "num_steps" not in state["start"]
    assert record["status"] == "collected"
    assert record["gpu_kernel_profile_collected"] is True
    assert record["physical_prefix_reuse_verified"] is False
    assert record["profiler_stop_confirmed"] is True
    assert record["server_quiescence_verified"] is False
    saved = json.loads((tmp_path / record["profile_id"] / "profile.json").read_text())
    assert saved == record


def test_prefill_profile_preserves_zero_completion_without_claiming_zero_decode(tmp_path):
    calls, _, args, kwargs = exercise(tmp_path)
    args[1].update(
        sampling_params={"max_new_tokens": 0}, return_logprob=True,
        logprob_start_len=-1, token_ids_logprob=[[3, 4], [3, 4]],
    )
    response, record = profile_generation(*args, **kwargs)
    assert calls == ["/start_profile", "/generate", "/stop_profile"]
    assert response[0]["meta_info"]["completion_tokens"] == 0
    assert record["requested_new_tokens_per_sequence"] == 0
    assert record["zero_decode_steps_verified"] is False


def test_zero_completion_without_readout_never_starts_profiler(tmp_path):
    calls, _, args, kwargs = exercise(tmp_path)
    args[1]["sampling_params"]["max_new_tokens"] = 0
    with pytest.raises(ValueError, match="first-position"):
        profile_generation(*args, **kwargs)
    assert not calls


@pytest.mark.parametrize("failure", ["generation", "stop", "rejected", "start_timeout"])
def test_errors_preserve_ownership_and_cleanup_state(tmp_path, failure):
    calls, _, args, kwargs = exercise(tmp_path, failure=failure)
    with pytest.raises(httpx.HTTPError):
        profile_generation(*args, **kwargs)
    expected_stop = failure != "rejected"
    assert ("/stop_profile" in calls) == expected_stop
    record = json.loads(next(tmp_path.glob("*/profile.json")).read_text())
    assert record["status"] == "failed"
    assert record["gpu_kernel_profile_collected"] is False
    assert record["profiler_stop_confirmed"] == (failure in ("generation", "start_timeout"))
    if failure in ("rejected", "start_timeout"):
        assert "/generate" not in calls


def test_200_responses_without_rank_files_do_not_prove_collection(tmp_path):
    _, _, args, kwargs = exercise(tmp_path, failure="missing")
    _, record = profile_generation(*args, **kwargs)
    assert record["status"] == "missing_gpu_evidence"
    assert record["gpu_kernel_profile_collected"] is False


@pytest.mark.parametrize("settings", [{"dedicated": False}, {"hostname": "remote.example"}])
def test_shared_or_remote_servers_are_not_modified(tmp_path, settings):
    calls, _, args, kwargs = exercise(tmp_path, **settings)
    with pytest.raises(ValueError, match="exclusive localhost"):
        profile_generation(*args, **kwargs)
    assert not calls and not list(tmp_path.iterdir())
