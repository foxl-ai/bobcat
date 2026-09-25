import copy

import pytest
import torch

from bobcat.model import DecisionModel
from bobcat.protocol import CONFIDENCE_PROFILE, parse_request

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from bobcat.serve import StudentScorer, create_app  # noqa: E402


class FixtureScorer:
    model_name = "bobcat-contract-fixture"
    release_gate_passed = False
    temperatures = {}

    def score(self, state, questions):
        return [[0.0] * len(q.labels) for q in questions], 42


def test_http_contract_with_real_model_identity_and_public_confidence_profile():
    client = TestClient(create_app(FixtureScorer()))
    result = client.post("/v1/systemone", json={
        "model": "jev-latest", "state": {},
        "questions": {"일치": {"type": "noul", "criteria": {"true": "일치하는 경우"}}},
    })
    assert result.status_code == 200
    assert result.headers["X-Bobcat-Confidence-Method"] == CONFIDENCE_PROFILE
    assert result.json() == {
        "model": "bobcat-contract-fixture",
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "answers": {"일치": {"type": "noul", "noul": 0.5}},
    }
    assert client.get("/health").json()["release_gate_passed"] is False


def test_http_validation_and_backend_failure_have_different_statuses():
    client = TestClient(create_app(FixtureScorer()))
    invalid = client.post("/v1/systemone", content=b'{"state":"a","state":"b"}')
    assert invalid.status_code == 422
    assert "Duplicate" in invalid.json()["detail"]
    assert client.post("/v1/systemone", json={
        "state": "", "questions": {"q": {"type": "noul"}},
    }).status_code == 422
    assert client.post("/v1/systemone", content=b" " * (2 * 1024 * 1024 + 1)).status_code == 413

    class BrokenScorer(FixtureScorer):
        def score(self, state, questions):
            return [[float("nan"), 0.0]], 42

    broken = TestClient(create_app(BrokenScorer()), raise_server_exceptions=False)
    result = broken.post("/v1/systemone", json={
        "model": "bobcat-latest", "state": "", "questions": {"q": {"type": "noul"}},
    })
    assert result.status_code == 500


def test_student_question_isolation_id_independence_singletons_and_length_errors(
    corpus, small_config, tmp_path,
):
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "config": small_config.to_dict(), "model": DecisionModel(small_config).state_dict(),
        "provenance": {"tokenizer_sha256": corpus[2].digest},
    }, checkpoint)
    with pytest.raises(ValueError, match="release gate"):
        StudentScorer(checkpoint, corpus[2].path, device="cpu")
    scorer = StudentScorer(
        checkpoint, corpus[2].path, device="cpu", allow_unvalidated=True,
    )
    payload = {
        "model": "bobcat-latest", "state": "The policy permits a refund.",
        "questions": {
            "first": {"type": "choice", "instructions": "Select a team.",
                      "criteria": {"billing": None, "shipping": None}},
            "second": {"type": "noul", "instructions": "Does the state include a secret code?"},
        },
    }
    state, original = parse_request(payload)
    before, _ = scorer.score(state, original)
    changed = copy.deepcopy(payload)
    changed["questions"]["first"]["instructions"] = "The secret is ZEBRA-7741."
    _, modified = parse_request(changed)
    after, _ = scorer.score(state, modified)
    torch.testing.assert_close(
        torch.tensor(before[1]), torch.tensor(after[1]), atol=1e-6, rtol=1e-5,
    )
    renamed = copy.deepcopy(payload)
    renamed["questions"] = dict(zip(["unrelated1", "unrelated2"],
                                   renamed["questions"].values(), strict=True))
    _, renamed_questions = parse_request(renamed)
    renamed_scores, _ = scorer.score(state, renamed_questions)
    assert renamed_scores == before
    alone, _ = scorer.score(state, original[1:])
    torch.testing.assert_close(
        torch.tensor(before[1]), torch.tensor(alone[0]), atol=1e-6, rtol=1e-5,
    )
    client = TestClient(create_app(scorer))
    single = client.post("/v1/systemone", json={
        "model": "bobcat-latest", "state": "",
        "questions": {"q": {"type": "choice", "criteria": {"only": None}}},
    })
    assert single.status_code == 200
    assert single.json()["answers"]["q"]["probabilities"] == {"only": 1.0}
    assert single.json()["model"].startswith("bobcat-scratch-research-")
    too_long = client.post("/v1/systemone", json={
        **payload, "state": "not a short policy " * 1000,
    })
    assert too_long.status_code == 422
    assert "truncate" in too_long.json()["detail"]
