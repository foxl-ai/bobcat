import pytest
import torch

from bobcat.decision_projection import (
    DecisionProjection,
    RetainedVocabularyDecisionProjection,
    install_materialized_decision_projection,
)


def test_selected_vocabulary_rows_preserve_logits_and_backbone_gradient():
    torch.manual_seed(14)
    original = torch.nn.Linear(16, 300, dtype=torch.float64)
    original.requires_grad_(False)
    projection = DecisionProjection.from_linear(original, [199, 8, 27, 4])
    x = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
    requested = [27, 199, 4]
    expected = original(x)[:, requested]
    actual = projection(x, requested)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    expected.square().sum().backward()
    grad = x.grad.clone()
    x.grad = None
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, grad, atol=1e-12, rtol=1e-12)
    assert not any(p.requires_grad for p in projection.parameters())
    assert projection.weight.numel() == 64


def test_unseen_option_is_an_error_not_a_fake_logit():
    head = torch.nn.Linear(4, 300)
    projection = DecisionProjection.from_linear(head, [1, 7])
    with pytest.raises(ValueError, match="no original"):
        projection(torch.ones(1, 4), [1, 99])
    with pytest.raises(ValueError, match="distinct"):
        projection(torch.ones(1, 4), [1, 1])


def test_native_bank_handles_ragged_candidates_and_preserves_mixed_loss_gradient():
    torch.manual_seed(2314)
    original = torch.nn.Linear(16, 400, dtype=torch.float64).requires_grad_(False)
    identifiers = [301, 5, 42, 0, 199]
    projection = DecisionProjection.from_linear(original, identifiers)
    choices = [[199, 42, 301], [0, 5], [301, 0, 5, 42, 199], [42]]
    hidden = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
    bank = projection(hidden)
    actual = projection.select_batch(bank, choices)
    expected = [original(hidden)[index, ids].float() for index, ids in enumerate(choices)]
    assert bank.shape == (4, 5)  # No 400-wide vocabulary result exists on the new path.
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def loss(values):
        return (
            -values[0].log_softmax(-1)[1]
            -values[1].log_softmax(-1)[0]
            + ((values[2].softmax(-1) * torch.arange(5)).sum() - 2.7).square()
        ) / 3

    loss(expected).backward()
    expected_gradient = hidden.grad.clone()
    hidden.grad = None
    loss(actual).backward()
    torch.testing.assert_close(hidden.grad, expected_gradient, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="one decision"):
        projection.select_batch(bank[:-1], choices)


def test_projection_checkpoint_retains_identifier_meanings():
    original = torch.nn.Linear(4, 300).requires_grad_(False)
    projection = DecisionProjection.from_linear(original, [99, 7, 8])
    restored = DecisionProjection.from_linear(original, [99, 7, 8])
    restored.load_state_dict(projection.state_dict())
    x = torch.randn(2, 4)
    torch.testing.assert_close(restored(x), projection(x), rtol=0, atol=0)
    wrong = DecisionProjection.from_linear(original, [7, 99, 8])
    with pytest.raises(RuntimeError, match="identifier order"):
        wrong.load_state_dict(projection.state_dict())


def test_native_install_keeps_embeddings_and_rejects_tied_heads():
    class Model(torch.nn.Module):
        def __init__(self, tied=False):
            super().__init__()
            self.embedding = torch.nn.Embedding(300, 4)
            self.lm_head = torch.nn.Linear(4, 300, bias=False)
            if tied:
                self.lm_head.weight = self.embedding.weight
            self.requires_grad_(False)

        def get_input_embeddings(self):
            return self.embedding

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, head):
            self.lm_head = head

    model = Model()
    embedding = model.embedding
    head = install_materialized_decision_projection(model, [11, 42])
    assert model.lm_head is head and model.embedding is embedding
    assert head.weight.shape == (2, 4)
    with pytest.raises(ValueError, match="tied"):
        install_materialized_decision_projection(Model(tied=True), [11, 42])


def test_retained_native_head_changes_only_gemm_not_checkpoint_ownership():
    torch.manual_seed(143)
    original = torch.nn.Linear(16, 400, dtype=torch.float64).requires_grad_(False)
    projection = RetainedVocabularyDecisionProjection(original, [301, 5, 42, 0, 199])
    assert projection.weight is original.weight and projection.bias is original.bias
    assert projection.state_dict().keys() == original.state_dict().keys()
    choices = [[199, 42, 301], [0, 5], [301, 0, 5, 42, 199]]
    hidden = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    expected = projection.select_batch(projection(hidden), choices)
    sum(v.square().sum() for v in expected).backward()
    gradient = hidden.grad.clone()
    hidden.grad = None
    projection.decision_only = True
    logits = projection(hidden)
    assert logits.shape == (3, 5)
    actual = projection.select_batch(logits, choices)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    sum(v.square().sum() for v in actual).backward()
    torch.testing.assert_close(hidden.grad, gradient, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="no original"):
        projection.select_batch(logits, [[7], choices[1], choices[2]])


def test_retained_native_head_supports_meta_construction_and_original_state_load():
    original = torch.nn.Linear(16, 400, bias=False, device="meta").requires_grad_(False)
    projection = RetainedVocabularyDecisionProjection(original, [3, 11])
    assert projection.weight is original.weight and projection.weight.is_meta
    materialized = torch.nn.Linear(16, 400, bias=False).requires_grad_(False)
    projection.load_state_dict(materialized.state_dict(), assign=True)
    projection.decision_only = True
    x = torch.randn(2, 16)
    torch.testing.assert_close(projection(x), materialized(x)[:, [3, 11]])


def test_retained_head_audits_the_same_hidden_without_changing_reference_output():
    original = torch.nn.Linear(16, 400).requires_grad_(False)
    projection = RetainedVocabularyDecisionProjection(original, [301, 5, 42, 0, 199])
    x = torch.randn(1, 1, 16)
    projection.audit_option_ids = [199, 42, 301]
    output = projection(x)
    torch.testing.assert_close(output, original(x), atol=0, rtol=0)
    assert projection.last_audit["same_hidden"]
    assert projection.last_audit["finite"]
    assert projection.last_audit["argmax_equal"]
    assert projection.last_audit["maximum_probability_tv"] < 1e-6
    assert not projection.last_audit["production_latency_measured"]
    projection.audit_option_ids = None
    projection.decision_only = True
    assert projection(x).shape == (1, 1, 5)
