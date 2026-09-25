import copy
import json

import httpx
import pytest
import torch
from test_glm_readout import compiler as compiler
from test_glm_readout import questions

from bobcat.glm_features import (
    FrozenOptionHead,
    ResidualDecisionHead,
    ResidualFeatureScorer,
    VerifiedFeatureScorer,
)
from bobcat.glm_readout import SGLangScorer
from bobcat.protocol import RequestLimitError, parse_request, response
from bobcat.schema import file_hash


@pytest.fixture
def basis(compiler, tmp_path):
    compiler = compiler[0]
    compiler.source["files"].append({"path": "weights.safetensors", "sha256": "c" * 64})
    torch.manual_seed(911)
    payload = {
        "format": "bobcat-glm-option-head-v1", "base_repo": compiler.source["repo"],
        "base_revision": compiler.source["revision"], "head_tensor": "lm_head.weight",
        "head_shape": [154880, 4096], "verified_shard": "weights.safetensors",
        "verified_shard_sha256": "c" * 64, "token_ids": compiler.identifier_ids,
        "weights": torch.randn(len(compiler.identifier_ids), 4096).bfloat16() / 64,
    }
    path = tmp_path / "option-head.pt"
    torch.save(payload, path)
    return FrozenOptionHead(path, file_hash(path), compiler), path, compiler


def test_projection_uses_relative_logits_and_rejects_wrong_hidden_position(basis):
    basis = basis[0]
    torch.manual_seed(12)
    hidden = torch.randn(4096).bfloat16().double()
    scores = basis.weights[:7] @ hidden
    native = (scores - scores.logsumexp(0) - 2).tolist()
    rows = [{"meta_info": {"hidden_states": hidden.tolist()}}]
    features, checks = basis.verify([native], rows)
    torch.testing.assert_close(features[0], hidden.float())
    assert checks[0]["max_candidate_relative_logit_error"] < 1e-12
    wrong = copy.deepcopy(rows)
    wrong[0]["meta_info"]["hidden_states"][0] += 5.0
    with pytest.raises(ValueError, match="mismatch"):
        basis.verify([native], wrong)
    with pytest.raises(ValueError, match="strict"):
        basis.verify([native], rows, max_absolute_error=1.0)


def test_corrupt_or_wrong_tokenizer_basis_is_rejected(basis):
    _, path, compiler = basis
    with pytest.raises(ValueError, match="checksum"):
        FrozenOptionHead(path, "0" * 64, compiler)
    compiler.identifier_ids = compiler.identifier_ids[::-1]
    with pytest.raises(ValueError, match="provenance"):
        FrozenOptionHead(path, file_hash(path), compiler)


def test_native_feature_capture_keeps_question_rows_and_scores_aligned(basis):
    basis, _, compiler = basis
    seen = []
    hidden = [torch.arange(4096).double() / 4096, torch.arange(4096).double() / -8192]

    def respond(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        body = json.loads(request.content)
        seen.append(body)
        rows = []
        for vector, prompt, ids in zip(
            hidden, body["input_ids"], body["token_ids_logprob"], strict=True,
        ):
            logits = basis.weights[:len(ids)] @ vector
            values = (logits - logits.logsumexp(0) - 3).tolist()
            rows.append({"text": "discard this sampled output", "meta_info": {
                "prompt_tokens": len(prompt), "completion_tokens": 1,
                "hidden_states": vector.tolist(),
                "output_token_ids_logprobs": [
                    [[value, token_id, None]
                     for value, token_id in zip(values, ids, strict=True)]
                ],
            }})
        return httpx.Response(200, json=rows)

    client = httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(respond))
    native = SGLangScorer(compiler, "http://fixture", "/models/glm", client=client)
    scorer = VerifiedFeatureScorer(native, basis)
    state, query = questions()
    logits, features, tokens = scorer.extract(state, query)
    assert seen[0]["return_hidden_states"] == "last"
    assert seen[0]["sampling_params"]["max_new_tokens"] == 1
    torch.testing.assert_close(features, torch.stack(hidden).float())
    assert len(logits) == 2 and tokens == compiler.compile(state, query).logical_input_tokens
    assert all(c["max_candidate_relative_logit_error"] < 1e-12
               for c in scorer.last_measurement["feature_identity"])
    assert scorer.provenance["base_weights_updated"] is False


def test_residual_head_starts_at_base_then_learns_with_frozen_features():
    torch.manual_seed(39)
    model = ResidualDecisionHead(hidden_size=12, max_choices=7, rank=4)
    features = torch.randn(6, 12)
    base = torch.zeros(6, 7)
    keep = torch.tensor([[True, True, True, False, False, False, False]] * 3
                        + [[True] * 7] * 3)
    logits = model(features, base, keep)
    assert torch.equal(logits[keep], base[keep])
    assert torch.isneginf(logits[~keep]).all()
    original = features.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    target = torch.tensor([0, 1, 2, 3, 4, 5])
    initial = torch.nn.functional.cross_entropy(logits, target)
    for _ in range(40):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(features, base, keep), target)
        loss.backward()
        optimizer.step()
    assert loss < initial / 3
    assert torch.equal(features, original)
    assert features.grad is None
    assert model(features, base, keep).argmax(-1).tolist() == target.tolist()


@pytest.mark.parametrize("adapted", [False, True])
def test_singletons_keep_limits_and_do_not_disable_other_questions_head(basis, adapted):
    basis, _, compiler = basis
    calls = []
    hidden = torch.arange(4096).double() / 4096

    def respond(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/models/glm"})
        body = json.loads(request.content)
        calls.append(body)
        rows = []
        for prompt, ids in zip(body["input_ids"], body["token_ids_logprob"], strict=True):
            logits = basis.weights[:len(ids)] @ hidden
            values = (logits - logits.logsumexp(0) - 3).tolist()
            rows.append({"text": "", "meta_info": {
                "prompt_tokens": len(prompt), "completion_tokens": 1,
                "hidden_states": hidden.tolist(),
                "output_token_ids_logprobs": [
                    [[value, token_id, None]
                     for value, token_id in zip(values, ids, strict=True)]
                ],
            }})
        return httpx.Response(200, json=rows)

    native = SGLangScorer(
        compiler, "http://fixture", "/models/glm",
        client=httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(respond)),
    )
    verified = VerifiedFeatureScorer(native, basis)
    scorer = verified
    if adapted:
        # Exercise the inference path with a nonzero residual, without claiming
        # this test-created head is a trained/releasable checkpoint.
        scorer = ResidualFeatureScorer.__new__(ResidualFeatureScorer)
        scorer.features = verified
        scorer.model = ResidualDecisionHead().eval()
        with torch.no_grad():
            scorer.model.down.weight.fill_(0.0001)
            scorer.model.up.weight[0].fill_(0.1)
        scorer.provenance = {"head_checkpoint_sha256": "test-only"}
    state, ordinary = questions()
    expected, expected_tokens = scorer.score(state, ordinary)
    if adapted:
        unmodified, _ = verified.score(state, ordinary)
        assert expected[0][0] != unmodified[0][0]
    _, singletons = parse_request({
        "model": "bobcat-latest", "state": state, "questions": {
            "only_choice": {"type": "choice", "criteria": {"유일": None}},
            "only_score": {"type": "score", "criteria": [{"설명": ["단일 단계"]}]},
        },
    })
    mixed = [singletons[0], ordinary[0], singletons[1], ordinary[1]]
    scores, tokens = scorer.score(state, mixed)
    assert scores == [[0.0], expected[0], [0.0], expected[1]]
    assert tokens == expected_tokens
    assert len(calls[-1]["input_ids"]) == len(ordinary)
    assert scorer.last_measurement["host_resolved_singleton_indices"] == [0, 2]
    assert scorer.last_measurement["native_question_indices"] == [1, 3]
    assert scorer.last_measurement["model_scored_questions"] == 2
    assert scorer.last_measurement["model_scored_logical_input_tokens"] == expected_tokens
    assert scorer.last_measurement["validated_logical_request_tokens"] > expected_tokens
    before_calls = len(calls)
    scores, tokens = scorer.score(state, singletons)
    assert scores == [[0.0], [0.0]] and tokens == 0 and len(calls) == before_calls
    assert scorer.last_measurement["native_question_indices"] == []
    assert scorer.last_measurement["native_prompt_tokens"] == 0
    assert scorer.last_measurement["native_completion_tokens"] == 0
    answers = response("fixture", singletons, scores, {}, tokens)["answers"]
    assert answers["only_choice"]["probabilities"] == {"유일": 1.0}
    assert answers["only_score"] == {
        "type": "score", "score": 0.0, "legend": {"0": {"설명": ["단일 단계"]}},
        "probabilities": {"0": 1.0}, "confidence": 1.0,
    }
    with pytest.raises(ValueError, match="at least two"):
        verified.extract(state, singletons)
    with pytest.raises(ValueError, match="at least two"):
        verified.extract_many([(state, ordinary), (state, singletons)])
    assert len(calls) == before_calls
    # A singleton cannot hide an otherwise excessive total request.
    compiler.max_request_tokens = compiler.compile(state, mixed).logical_input_tokens - 1
    with pytest.raises(RequestLimitError, match="request limit"):
        scorer.score(state, mixed)
    assert len(calls) == before_calls
