import copy
import json
import time

import httpx
import pytest
from test_glm_cache_eval import small_suite
from test_glm_readout import compiler as compiler

from bobcat import glm_diagnostics as diagnostics
from bobcat.schema import file_hash, json_hash
from bobcat.workflow_probes import build_suite


def plan_fixture(root):
    suites = {}
    for name in ("identifier", "workflow", "cache"):
        p = root / f"{name}.json"
        p.write_text(json.dumps({"fixture": name}))
        suites[name] = {"path": p.name, "sha256": file_hash(p)}
    plan = {
        "schema": diagnostics.SCHEMA, "suites": suites,
        "allowances_seconds": {"identifier": 600, "workflow": 300, "cache": 1500, "profile": 900},
        "cache_repeats": 2, "profile_case_id": "fixture", "profile_blocks": 2,
        "expected_tp_ranks": 8,
    }
    plan["content_sha256"] = json_hash(plan)
    p = root / "plan.json"
    p.write_text(json.dumps(plan))
    return p, plan


def test_plan_rejects_modified_suite_and_oversized_work_before_any_server(tmp_path):
    path, plan = plan_fixture(tmp_path)
    loaded, suites = diagnostics.load_plan(path, tmp_path)
    assert loaded == plan and set(suites) == {"identifier", "workflow", "cache"}
    (tmp_path / "cache.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="suite changed"):
        diagnostics.load_plan(path, tmp_path)
    bad = copy.deepcopy(plan)
    bad["allowances_seconds"]["cache"] = 99999
    bad["content_sha256"] = json_hash({k: v for k, v in bad.items() if k != "content_sha256"})
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="finite"):
        diagnostics.load_plan(path, tmp_path)


def test_workflow_preflight_covers_wrong_upstream_answers_without_replacing_gold():
    suite = build_suite()
    payloads = diagnostics.workflow_inputs(suite)
    assert len(payloads) == 112  # 7 inputs/world, including every possible route.
    for case in suite["cases"]:
        assignments = {
            p["state"]["recorded_assignment"]["route"]
            for p in payloads
            if p["state"]["facts"] == case["state"]["facts"]
            and p["state"]["destinations"] == case["state"]["destinations"]
            and "recorded_assignment" in p["state"]
        }
        assert assignments == set(case["state"]["destinations"])
        assert len(assignments - {case["gold"]["route"]}) == 2
    assert all("gold" not in p and "id" not in p for p in payloads)


def test_incomplete_study_never_starts_next_gpu_experiment(tmp_path, monkeypatch):
    _, plan = plan_fixture(tmp_path)
    monkeypatch.setattr(diagnostics, "preflight", lambda *a: {})
    monkeypatch.setattr(diagnostics, "run_identifiers", lambda *a, **kw: {"status": "failed"})

    def forbidden(*a, **kw):
        pytest.fail("No study may follow a transport/deadline failure on this server.")

    for name in ("run_workflows", "run_cache", "run_profiles"):
        monkeypatch.setattr(diagnostics, name, forbidden)
    client = httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(forbidden))
    result = diagnostics.run(
        plan, {"identifier": {}}, None, client, "/models/glm",
        tmp_path / "run", tmp_path, max_seconds=300, dedicated_server=True,
    )
    assert result["status"] == "incomplete"
    assert set(result["studies"]) == {"identifier"}
    assert result["training_performed"] is result["release_gate_passed"] is False


@pytest.mark.parametrize("missing_rank", [False, True])
def test_profiles_preserve_inputs_change_only_cache_and_require_every_rank(
    compiler, tmp_path, monkeypatch, missing_rank,
):
    case_suite = small_suite()
    plan = {"profile_case_id": case_suite["cases"][0]["id"],
            "profile_blocks": 2, "expected_tp_ranks": 8}
    profiled, native = [], []

    def result(body):
        return [
            {"meta_info": {
                "prompt_tokens": len(ids), "completion_tokens": 1, "cached_tokens": 0,
                "output_token_ids_logprobs": [
                    [[-0.7 - i, token, None] for i, token in enumerate(options)]
                ],
            }} for ids, options in zip(body["input_ids"], body["token_ids_logprob"], strict=True)
        ]

    def respond(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        if request.url.path == "/flush_cache":
            return httpx.Response(200)
        body = json.loads(request.content)
        native.append(body)
        return httpx.Response(200, json=result(body))

    def profile(client, payload, host_root, server_root, **kw):
        assert kw["dedicated_server"] and kw["expected_ranks"] == 8
        assert host_root == tmp_path and server_root == "/bobcat-profiles"
        profiled.append(copy.deepcopy(payload))
        return result(payload), {
            "gpu_kernel_profile_collected": not missing_rank,
            "profiler_stop_confirmed": True,
            "traces": [{"tp_rank": r} for r in range(7 if missing_rank else 8)],
        }

    monkeypatch.setattr(diagnostics, "profile_generation", profile)
    client = httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(respond))
    out = tmp_path / "profiles-summary"
    out.mkdir()
    args = (plan, {"cache": case_suite}, compiler[0], client, "/models/glm",
            out, tmp_path, time.monotonic() + 300)
    if missing_rank:
        with pytest.raises(RuntimeError, match="Incomplete GPU evidence"):
            diagnostics.run_profiles(*args)
        assert len(profiled) == 1
        return
    record = diagnostics.run_profiles(*args)
    assert record["profile_trace_count"] == 48 and record["instrumented_calls"] == 6
    assert record["physical_prefix_reuse_verified"] is False
    for payload in profiled:
        assert payload["input_ids"] == profiled[0]["input_ids"]
        assert payload["token_ids_logprob"] == profiled[0]["token_ids_logprob"]
        assert payload["sampling_params"]["max_new_tokens"] == 1
    assert sum(len(set(p["cache_salt"])) == 8 for p in profiled) == 2
    assert len(native) == 2  # Only the two warm arms have a separate priming question.
    assert all(p["input_ids"][0] not in profiled[0]["input_ids"] for p in native)
