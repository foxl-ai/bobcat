import copy
import json
import math
import time

import httpx
import pytest
from test_glm_readout import compiler as compiler

from bobcat.glm_cache_eval import (
    SaltedTransport,
    build_cases,
    distribution_changes,
    run,
    validate_cases,
)
from bobcat.schema import json_hash


def small_suite():
    suite = build_cases()
    suite["cases"] = [suite["cases"][2]]  # Korean, eight distinct suffixes.
    suite["content_sha256"] = json_hash({
        k: v for k, v in suite.items() if k != "content_sha256"
    })
    return suite


def test_salt_changes_only_transport_and_isolates_sequential_requests():
    seen = []
    client = httpx.Client(
        base_url="http://fixture",
        transport=httpx.MockTransport(
            lambda r: seen.append(json.loads(r.content)) or httpx.Response(200, json={}),
        ),
    )
    wrapper = SaltedTransport(client, time.monotonic() + 60, "case")
    payload = {"input_ids": [[1, 2], [1, 3]], "token_ids_logprob": [[10], [10]]}
    expected = copy.deepcopy(payload)
    wrapper.post("/generate", json=payload)
    wrapper.post("/generate", json=payload)
    assert payload == expected
    assert set(seen[0]["cache_salt"]).isdisjoint(seen[1]["cache_salt"])
    wrapper.shared = True
    wrapper.post("/generate", json=payload)
    assert seen[-1]["cache_salt"] == ["case", "case"]
    assert all({k: v for k, v in row.items() if k != "cache_salt"} == expected for row in seen)


def test_same_argmax_can_cross_an_execution_threshold():
    result = distribution_changes(
        [[math.log(0.89), math.log(0.11)]],
        [[math.log(0.91), math.log(0.09)]],
    )[0]
    assert result["argmax_changed"] is False
    assert result["tv"] == pytest.approx(0.02)
    assert result["top_probability_threshold_crossings"]["0.9"] is True
    assert distribution_changes([[1000, 0]], [[1003, 3]])[0]["tv"] == 0


def test_distinct_suffixes_and_exact_standalone_inputs(compiler):
    suite = small_suite()
    records = validate_cases(suite, compiler[0])
    assert records[0]["questions"] == len(set(records[0]["prompt_sha256"])) == 8
    broken = copy.deepcopy(suite)
    q = broken["cases"][0]["request"]["questions"]
    q["q1"] = copy.deepcopy(q["q0"])
    broken["content_sha256"] = json_hash({
        k: v for k, v in broken.items() if k != "content_sha256"
    })
    with pytest.raises(ValueError, match="distinct"):
        validate_cases(broken, compiler[0])


def test_execution_records_all_arms_without_claiming_gpu_reuse(compiler, tmp_path):
    seen, resets = [], []

    def respond(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        if request.url.path == "/flush_cache":
            resets.append(request.url.path)
            return httpx.Response(200, text="Cache flushed.")
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=[
            {"meta_info": {
                "prompt_tokens": len(ids), "completion_tokens": 1, "cached_tokens": 0,
                "output_token_ids_logprobs": [
                    [[-0.7 - i, token, None] for i, token in enumerate(options)]
                ],
            }} for ids, options in zip(body["input_ids"], body["token_ids_logprob"], strict=True)
        ])

    client = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(respond))
    out = tmp_path / "result"
    result = run(small_suite(), compiler[0], client, "/models/glm", out,
                 repeats=1, max_seconds=60, dedicated_server=True)
    assert result["status"] == "completed" and result["attempted_arms"] == 6
    assert len(resets) == 6 and len(seen) == 29
    assert result["physical_prefix_reuse_verified"] is False
    observations = [
        json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()
    ]
    warmed = [row for row in observations if row["condition"].startswith("warm")]
    assert len(warmed) == 2 and all("priming_native_call" in row for row in warmed)
    for primer in (body for body in seen if len(body["input_ids"]) == 1):
        following = [body for body in seen if body is not primer
                     and body["cache_salt"][0] == primer["cache_salt"][0]]
        if following:
            assert all(primer["input_ids"][0] != ids for body in following
                       for ids in body["input_ids"])
    comparisons = json.loads((out / "comparisons.json").read_text())
    assert all(change["tv"] == 0 for row in comparisons for change in row["probability_changes"])


def test_shared_server_or_invalid_suite_does_not_flush(compiler, tmp_path):
    client = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(
        lambda request: pytest.fail("Preflight failure must not touch the server."),
    ))
    with pytest.raises(ValueError, match="dedicated"):
        run(small_suite(), compiler[0], client, "/models/glm", tmp_path / "shared")
    broken = small_suite()
    broken["cases"][0]["request"]["state"] = "Changed after freezing."
    with pytest.raises(ValueError, match="checksummed"):
        run(broken, compiler[0], client, "/models/glm", tmp_path / "bad",
            dedicated_server=True)


def test_failed_cache_reset_does_not_run_or_claim_a_completed_arm(compiler, tmp_path):
    def respond(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        assert request.url.path == "/flush_cache"
        return httpx.Response(400, text="Running requests prevent cache reset.")
    client = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(respond))
    out = tmp_path / "busy"
    with pytest.raises(httpx.HTTPStatusError):
        run(small_suite(), compiler[0], client, "/models/glm", out,
            dedicated_server=True, repeats=1)
    result = json.loads((out / "run.json").read_text())
    assert result["status"] == "failed" and result["attempted_arms"] == 0


def test_native_timeout_stops_before_another_cache_condition(compiler, tmp_path):
    calls = []

    def respond(request):
        calls.append(request.url.path)
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        if request.url.path == "/flush_cache":
            return httpx.Response(200)
        raise httpx.ReadTimeout("server may still be running", request=request)

    client = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(respond))
    out = tmp_path / "timeout"
    with pytest.raises(RuntimeError, match="Remote quiescence is unverified"):
        run(small_suite(), compiler[0], client, "/models/glm", out,
            dedicated_server=True, repeats=1)
    assert calls.count("/generate") == calls.count("/flush_cache") == 1
    result = json.loads((out / "run.json").read_text())
    assert result["status"] == "failed" and result["failed_arms"] == 1
    assert result["attempted_arms"] == 1
