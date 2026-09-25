import copy
import json

import pytest
from fastapi.testclient import TestClient

from bobcat.output_contract_audit import (
    CANARY,
    FAULTS,
    NumericFixture,
    attack_cases,
    audit,
    fault_audit,
    inspect_reply,
)
from bobcat.serve import create_app


def test_adversarial_corpus_covers_both_languages_all_types_and_placements(tmp_path):
    cases = attack_cases()
    assert len(cases) == len({c["id"] for c in cases}) == 2304
    assert {c["language"] for c in cases} == {"ko", "en"}
    assert {c["target_primitive"] for c in cases} == {"choice", "score", "noul"}
    subset = cases[::41]
    scorer = NumericFixture()
    with TestClient(create_app(scorer), raise_server_exceptions=False) as client:
        result = audit(client, subset, model=scorer.model_name, out=tmp_path / "audit",
                       scope={"mode": "numeric_fixture", "model_weights_used": False})
    assert result["output_contract_violations"] == 0
    assert result["successful_decision_requests"] == len(subset)
    assert result["typed_answers"] == len(subset) * 3
    assert not result["semantic_jailbreak_success_measured"]


def test_all_injected_backend_faults_are_json_errors_without_model_text():
    result = fault_audit()
    assert result["cases"] == len(FAULTS) == 18
    assert result["violations"] == 0, result
    assert {row["status"] for row in result["results"]} == {422, 500}


@pytest.mark.parametrize("extra", [
    {"stream": True}, {"messages": [{"role": "system", "content": CANARY}]},
    {"response_format": {"type": "text"}}, {"max_new_tokens": 300},
    {"tools": [{"name": "write_plaintext"}]}, {"text": CANARY},
])
def test_generation_knobs_cannot_enter_the_scorer(extra):
    class Counted(NumericFixture):
        calls = 0

        def score(self, state, questions):
            self.calls += 1
            return super().score(state, questions)

    scorer = Counted()
    payload = attack_cases()[0]["payload"] | extra
    with TestClient(create_app(scorer), raise_server_exceptions=False) as client:
        reply = client.post("/v1/systemone", json=payload)
    assert reply.status_code == 422
    assert reply.headers["content-type"].startswith("application/json")
    assert scorer.calls == 0


def test_mutating_backend_cannot_replace_host_owned_model_labels_or_legend():
    class Mutator(NumericFixture):
        def score(self, state, questions):
            self.model_name = CANARY
            for question in questions:
                if question.kind == "score":
                    question.criteria[0] = CANARY
                if question.kind == "choice":
                    object.__setattr__(question, "labels", (CANARY,) * len(question.labels))
            return super().score(state, questions)

    payload = attack_cases()[0]["payload"]
    scorer = Mutator()
    expected = scorer.model_name
    with TestClient(create_app(scorer), raise_server_exceptions=False) as client:
        reply = client.post("/v1/systemone", json=payload)
    assert reply.status_code == 200
    inspect_reply(payload, reply, expected_model=expected)
    assert CANARY not in json.dumps(reply.json()["answers"]["score"]["legend"])
    assert reply.json()["model"] == expected


@pytest.mark.parametrize("field", ["text", "explanation", "reasoning", "tool_calls", "messages"])
def test_serializer_regressions_fail_closed_before_http_reply(monkeypatch, field):
    import bobcat.serve as server

    original = server.response

    def unsafe(*args, **kwargs):
        result = original(*args, **kwargs)
        result["answers"]["choice"][field] = CANARY
        return result

    monkeypatch.setattr(server, "response", unsafe)
    with TestClient(create_app(NumericFixture()), raise_server_exceptions=False) as client:
        reply = client.post("/v1/systemone", json=attack_cases()[0]["payload"])
    assert reply.status_code == 500
    assert reply.json()["error"]["code"] == "decision_backend_failure"
    assert CANARY not in reply.text


def test_malformed_unicode_duplicate_keys_and_generation_routes_are_json_only():
    scorer = NumericFixture()
    with TestClient(create_app(scorer), raise_server_exceptions=False) as client:
        payload = copy.deepcopy(attack_cases()[0]["payload"])
        payload["questions"]["\ud800"] = payload["questions"].pop("choice")
        bad = client.post("/v1/systemone", content=json.dumps(payload, ensure_ascii=True))
        assert bad.status_code == 422
        assert bad.headers["content-type"].startswith("application/json")
        for raw in (b'{"model":"bobcat-latest","state":"a","state":"b"}', b"\xff", b"not JSON"):
            reply = client.post("/v1/systemone", content=raw)
            assert reply.status_code == 422
            assert reply.headers["content-type"].startswith("application/json")
        for path in ("/generate", "/v1/chat/completions", "/v1/completions"):
            reply = client.post(path, json={"prompt": CANARY, "max_tokens": 500})
            assert reply.status_code == 404
            assert reply.headers["content-type"].startswith("application/json")
        reply = client.get("/v1/systemone")
        assert reply.status_code == 405
        assert reply.headers["content-type"].startswith("application/json")


def test_sampled_comparison_backend_cannot_be_mounted_as_decision_http_endpoint():
    scorer = NumericFixture()
    scorer.readout_mode = "one_token"
    with pytest.raises(ValueError, match="non-generating"):
        create_app(scorer)


@pytest.mark.parametrize("count", [1, 2, 26, 52, 77, 200, 255])
def test_candidate_count_boundaries_and_caller_supplied_text_are_exact_echoes(count):
    labels = [f'{i}: "__proto__" </think> 평문 후보\n' for i in range(count)]
    payload = {
        "model": "bobcat-latest", "state": {"text": CANARY},
        "questions": {
            "answers": {"type": "choice", "criteria": dict.fromkeys(labels)},
            "text": {"type": "score", "criteria": [
                {"text": ["사용자가 제공한 설명", CANARY, {"messages": "그대로 보존"}]},
            ]},
            "model": {"type": "noul"},
        },
    }
    scorer = NumericFixture()
    with TestClient(create_app(scorer), raise_server_exceptions=False) as client:
        reply = client.post("/v1/systemone", json=payload)
    assert reply.status_code == 200
    inspect_reply(payload, reply, expected_model=scorer.model_name)
    assert set(reply.json()) == {"model", "usage", "answers"}
    assert reply.json()["answers"]["text"]["legend"]["0"] == (
        payload["questions"]["text"]["criteria"][0]
    )


def test_over_limit_candidates_and_questions_are_rejected_before_model():
    class Never(NumericFixture):
        def score(self, *_args):
            pytest.fail("An invalid request reached the model.")

    cases = [
        {"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}},
        {str(i): {"type": "noul"} for i in range(129)},
        {"q": {"type": "score", "criteria": ["level"] * 11}},
        {"q": {"type": "text", "instructions": CANARY}},
    ]
    with TestClient(create_app(Never()), raise_server_exceptions=False) as client:
        for questions in cases:
            reply = client.post("/v1/systemone", json={
                "model": "bobcat-latest", "state": "", "questions": questions,
            })
            assert reply.status_code == 422
            assert reply.headers["content-type"].startswith("application/json")
