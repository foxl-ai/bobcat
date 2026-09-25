import copy
import json

import httpx
import pytest
from test_glm_readout import compiler as compiler

from bobcat.glm_identifier_controls import (
    build,
    compile_case,
    ordered_hash,
    preflight,
    run,
    validate,
)
from bobcat.schema import json_hash


def small_source():
    request = {
        "model": "bobcat-latest", "state": {"max_fee": 3},
        "questions": {"decision": {
            "type": "choice", "instructions": "수수료가 가장 작은 후보를 고르세요.",
            "criteria": {"안": {"fee": 1}, "밖": {"fee": 3}, "옆": {"fee": 2}},
        }},
    }
    cases = []
    for index, order in enumerate((["안", "밖", "옆"], ["옆", "안", "밖"])):
        changed = copy.deepcopy(request)
        changed["questions"]["decision"]["criteria"] = {
            key: request["questions"]["decision"]["criteria"][key] for key in order
        }
        cases.append({
            "id": f"not_for_the_model_{index}", "group": "many-ko-fixture",
            "family": "many_choices", "language": "ko", "gold": "안", "request": changed,
        })
    source = {"schema": "bobcat-architecture-probes-v2", "cases": cases}
    source["content_sha256"] = json_hash(source)
    return source


def test_crossed_controls_keep_meanings_and_bind_identifiers_to_labels(compiler):
    model = compiler[0]
    suite = build(small_source(), blocks=1)
    cases = {case["arm"]: case for case in suite["cases"]}
    original_ids, original_labels = model.identifier_ids[:], model.identifiers[:]
    compiled = {arm: compile_case(model, case) for arm, case in cases.items()}
    bindings = {
        arm: dict(zip(case["labels_in_order"], compiled[arm].option_token_ids[0], strict=True))
        for arm, case in cases.items()
    }
    assert bindings["base"] == bindings["order_only"] == bindings["base_repeat"]
    assert bindings["identifier_only"] == bindings["both"] != bindings["base"]
    assert compiled["base"] == compiled["base_repeat"]
    assert compiled["order_only"].input_ids != compiled["base"].input_ids
    assert model.identifier_ids == original_ids and model.identifiers == original_labels
    assert len(preflight(suite, model)) == 5


def test_order_preserving_checksum_catches_object_reordering():
    suite = build(small_source(), blocks=1)
    broken = copy.deepcopy(suite)
    q = broken["cases"][0]["request"]["questions"]["decision"]
    q["criteria"] = dict(reversed(list(q["criteria"].items())))
    assert json_hash(broken) == json_hash(suite)  # Canonical JSON alone misses this change.
    with pytest.raises(ValueError, match="order-preserving"):
        validate(broken)


def test_gold_and_evaluator_metadata_never_enter_compiled_input(compiler):
    case = build(small_source(), blocks=1)["cases"][0]
    expected = compile_case(compiler[0], case)
    changed = copy.deepcopy(case)
    changed.update(gold="metadata_secret", id="private_case_id", semantic_group="private_group")
    assert compile_case(compiler[0], changed) == expected


def test_invalid_bijection_and_confounded_arm_are_rejected(compiler):
    suite = build(small_source(), blocks=1)
    broken = copy.deepcopy(suite)
    case = next(row for row in broken["cases"] if row["arm"] == "order_only")
    case["identifier_slots"] = [0, 0, 1]
    with pytest.raises(ValueError, match="bijective"):
        compile_case(compiler[0], case)
    broken["content_sha256"] = ordered_hash({
        key: value for key, value in broken.items() if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="binding"):
        validate(broken)


def native_response(model, seen, fail_at=None):
    def respond(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == fail_at:
            raise httpx.ReadTimeout("owned fixture timeout", request=request)
        return httpx.Response(200, json=[
            {"meta_info": {
                "prompt_tokens": len(ids), "completion_tokens": 1,
                "output_token_ids_logprobs": [[[
                    0.0 if token == model.identifier_ids[0] else -9.0, token, None,
                ] for token in options]],
            }}
            for ids, options in zip(body["input_ids"], body["token_ids_logprob"], strict=True)
        ])
    return respond


def test_native_results_map_back_to_meanings_not_list_positions(compiler, tmp_path):
    model, seen = compiler[0], []
    suite = build(small_source(), blocks=1)
    client = httpx.Client(
        base_url="http://fixture", transport=httpx.MockTransport(native_response(model, seen)),
    )
    out = tmp_path / "result"
    manifest = run(suite, model, client, "/models/glm", out, max_seconds=30)
    rows = {r["arm"]: r for r in (
        json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()
    )}
    assert manifest["status"] == "completed" and manifest["attempted_calls"] == 5
    assert rows["base"]["prediction"] == rows["order_only"]["prediction"]
    assert rows["identifier_only"]["prediction"] == rows["both"]["prediction"]
    assert rows["base"]["prediction"] != rows["identifier_only"]["prediction"]
    assert manifest["release_gate_passed"] is False
    assert manifest["remote_quiescence_proven"] is False
    assert len({row["cache_salt"][0] for row in seen}) == 5


def test_native_timeout_stops_without_retry_or_more_requests(compiler, tmp_path):
    model, seen = compiler[0], []
    client = httpx.Client(
        base_url="http://fixture",
        transport=httpx.MockTransport(native_response(model, seen, fail_at=2)),
    )
    manifest = run(
        build(small_source(), blocks=1), model, client, "/models/glm",
        tmp_path / "failure", max_seconds=30,
    )
    assert len(seen) == manifest["attempted_calls"] == 2
    assert manifest["status"] == "failed" and manifest["failed_calls"] == 1
    assert manifest["remote_quiescence_proven"] is False
