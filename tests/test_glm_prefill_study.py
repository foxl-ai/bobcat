import copy
import json
import time

import httpx
import pytest
from test_glm_cache_eval import small_suite
from test_glm_readout import compiler as compiler

from bobcat import glm_prefill_study as study
from bobcat.protocol import parse_request
from bobcat.schema import json_hash


def sample_plan():
    case = small_suite()["cases"][0]
    plan = {
        "schema": study.SCHEMA,
        "cases": [{
            "id": case["id"], "language": case["language"],
            "request": case["request"], "source": "unit-fixture",
        }],
        "repeats": 2, "profile_case_id": case["id"], "profile_blocks": 1,
        "expected_tp_ranks": 8, "max_study_seconds": 300,
    }
    plan["content_sha256"] = json_hash(plan)
    return plan


def transport(*, fail_readout=False, stale_once=False):
    requests, events = [], []

    def respond(request):
        events.append(request.url.path)
        if request.url.path == "/v1/loads":
            if stale_once and events.count("/v1/loads") == 1:
                stamp, running = 0, 0
            else:
                stamp, running = time.time(), 0
            return httpx.Response(200, json={
                "num_accelerators": 8, "loads": [{
                    "dp_rank": 0, "timestamp": stamp,
                    "num_running_reqs": running, "num_waiting_reqs": 0,
                }],
            })
        if request.url.path == "/flush_cache":
            return httpx.Response(200)
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        assert request.url.path == "/generate"
        body = json.loads(request.content)
        requests.append(body)
        if fail_readout:
            raise httpx.ReadTimeout("Result unknown", request=request)
        return httpx.Response(200, json=[
            {"text": "", "output_ids": [], "meta_info": {
                "prompt_tokens": len(ids),
                "completion_tokens": body["sampling_params"]["max_new_tokens"],
                "output_token_ids_logprobs": [
                    [[-0.7 - i, token, None] for i, token in enumerate(options)]
                ],
                "cached_tokens": 0,
            }} for ids, options in zip(
                body["input_ids"], body["token_ids_logprob"], strict=True,
            )
        ])

    client = httpx.Client(
        base_url="http://127.0.0.1", transport=httpx.MockTransport(respond),
    )
    return client, requests, events


def test_checksum_and_full_input_preflight_precede_server_work(compiler, tmp_path):
    plan = sample_plan()
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    checked = study.load_plan(path)
    compiled, rows = study.preflight(checked, compiler[0])
    state, qs = parse_request(plan["cases"][0]["request"])
    assert compiled[rows[0]["case_id"]].input_ids == compiler[0].compile(state, qs).input_ids
    bad = copy.deepcopy(plan)
    bad["repeats"] = 100
    bad["content_sha256"] = json_hash({k: v for k, v in bad.items() if k != "content_sha256"})
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="finite"):
        study.load_plan(path)
    plan["cases"][0]["request"]["state"]["changed"] = True
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="checksummed"):
        study.load_plan(path)


def test_preparation_is_only_exact_state_prefix_and_waits_before_branch(compiler):
    plan = sample_plan()
    compiled, rows = study.preflight(plan, compiler[0])
    value = compiled[rows[0]["case_id"]]
    client, requests, events = transport()
    row = study.run_arm(
        client, value, "prefill_only", "state_prepared", "fixture", time.monotonic() + 120,
    )
    assert len(requests) == 2
    primer, decision = requests
    assert primer["input_ids"] == [value.input_ids[0][:value.shared_prefix_tokens]]
    assert primer["sampling_params"]["max_new_tokens"] == 0
    assert decision["input_ids"] == value.input_ids
    assert decision["token_ids_logprob"] == value.option_token_ids
    assert set(decision["cache_salt"]) == set(primer["cache_salt"]) == {"fixture"}
    first, second = [i for i, endpoint in enumerate(events) if endpoint == "/generate"]
    assert "/v1/loads" in events[first + 1:second]
    assert row["decision"]["native_completion_tokens"] == 0
    assert row["decision"]["native_scored_positions"] == len(value.input_ids)
    assert row["decision"]["zero_decode_steps_verified"] is False


def test_stale_idle_snapshot_does_not_release_the_next_branch():
    client, _, events = transport(stale_once=True)
    result = study.wait_scheduler_idle(
        client, time.monotonic() + 60, not_before=time.time() - 1,
    )
    assert events == ["/v1/loads", "/v1/loads"]
    assert len(result["snapshots"]) == 2
    assert result["scheduler_reported_idle"] and not result["gpu_streams_synchronized"]


def test_failed_primer_prevents_suffix_and_entire_followup_study(compiler, tmp_path):
    plan = sample_plan()
    values, rows = study.preflight(plan, compiler[0])
    client, requests, _ = transport(fail_readout=True)
    with pytest.raises(httpx.ReadTimeout):
        study.run_arm(
            client, values[rows[0]["case_id"]], "prefill_only", "state_prepared",
            "failure", time.monotonic() + 120,
        )
    assert len(requests) == 1  # No suffix was submitted after an uncertain primer.
    requests.clear()
    out = tmp_path / "run"
    with pytest.raises(httpx.ReadTimeout):
        study.run(
            plan, compiler[0], client, "/models/glm", out, tmp_path,
            max_seconds=300, dedicated_server=True,
        )
    record = json.loads((out / "study.json").read_text())
    assert record["status"] == "failed"
    assert record["observations_completed"] == record["profiles_completed"] == 0
    assert len(requests) == 1


def test_all_arms_keep_same_inputs_and_profile_parent_cost(compiler, tmp_path, monkeypatch):
    plan = sample_plan()
    client, _, _ = transport()
    profile_calls = []

    def profile(native, payload, host_root, server_root, **kwargs):
        profile_calls.append(copy.deepcopy(payload))
        body = native.post("/generate", json=payload).json()
        return body, {
            "gpu_kernel_profile_collected": True, "profiler_stop_confirmed": True,
            "instrumented_generate_wall_seconds": 1.0,
            "traces": [{"tp_rank": r} for r in range(8)],
        }

    monkeypatch.setattr(study, "profile_generation", profile)
    out = tmp_path / "study"
    result = study.run(
        plan, compiler[0], client, "/models/glm", out, tmp_path,
        max_seconds=300, dedicated_server=True,
    )
    assert result["status"] == "completed"
    assert result["observations_completed"] == 12 and result["warmup_completed"] == 2
    assert result["profiles_completed"] == 6 and result["profile_trace_count"] == 64
    assert len(profile_calls) == 8  # Six decisions plus two explicit parent profiles.
    rows = [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()]
    assert len({row["decision"]["compiled_input_sha256"] for row in rows}) == 1
    assert all(row["including_state_preparation_seconds"] >=
               row["decision"]["native_http_seconds"] for row in rows)
    pairs = json.loads((out / "comparisons.json").read_text())["comparisons"]
    assert {pair["kind"] for pair in pairs} == {
        "readout_mode", "cache_condition", "exact_input_repeat",
    }
    assert all(change["tv"] == 0 for pair in pairs for change in pair["question_changes"])
    assert not result["release_gate_passed"] and not result["physical_prefix_reuse_verified"]
    broken = copy.deepcopy(rows)
    broken[0]["decision"]["compiled_input_sha256"] = "changed"
    with pytest.raises(ValueError, match="inputs"):
        study.compare_observations(broken)
