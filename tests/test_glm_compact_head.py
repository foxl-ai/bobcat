import pytest
import torch
from torch import nn

from bobcat.decision_projection import RetainedVocabularyDecisionProjection
from bobcat.glm_compact_head import (
    NativeDecisionProjection,
    compare_frozen_body_hashes,
    install_uninitialized_native_decision_projection,
)


class NativeShell(nn.Module):
    def __init__(self, *, device="cpu", tied=False):
        super().__init__()
        self.embedding = nn.Embedding(64, 16, device=device)
        self.lm_head = nn.Linear(16, 64, bias=False, device=device)
        if tied:
            self.lm_head.weight = self.embedding.weight
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head


def test_output_storage_is_reduced_without_changing_language_embedding():
    model = NativeShell()
    original = model.lm_head.weight.detach().clone()
    embedding = model.embedding
    values = embedding.weight.detach().clone()
    selected = [41, 2, 17, 0, 63]
    head = install_uninitialized_native_decision_projection(model, selected)
    assert head.weight.shape == (5, 16)
    assert model.embedding is embedding and torch.equal(embedding.weight, values)
    # Initialization is an explicit separate operation; construction is not a restore.
    head.weight.copy_(original[selected])
    hidden = torch.randn(2, 16, requires_grad=True)
    actual = head.select_batch(head(hidden), [[2, 41], [63, 17, 0]])
    full = nn.functional.linear(hidden, original)
    expected = [full[0, [2, 41]], full[1, [63, 17, 0]]]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), hidden,
                                      retain_graph=True)[0]
    expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), hidden)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert model.lm_head.weight.numel() == 5 * 16
    assert model.embedding.weight.numel() == 64 * 16
    assert not model.lm_head.weight.requires_grad


def test_meta_construction_precedes_any_distributed_output_allocation():
    model = NativeShell(device="meta")
    head = install_uninitialized_native_decision_projection(model, [1, 4, 7])
    assert head.weight.is_meta and head.weight.shape == (3, 16)
    assert set(model.state_dict()) == {"embedding.weight", "lm_head.weight"}
    assert model.embedding.weight.shape == (64, 16)


def test_tied_input_embeddings_cannot_be_repurposed_as_compact_output():
    with pytest.raises(ValueError, match="Tied embeddings"):
        install_uninitialized_native_decision_projection(NativeShell(tied=True), [1, 4])


@pytest.mark.parametrize("requested", [[[4, 4]], [[999]], [[]]])
def test_unknown_or_duplicate_candidates_are_not_filled_or_silently_removed(requested):
    head = NativeDecisionProjection(16, [1, 4, 7], original_vocabulary=64,
                                    dtype=torch.float32, device="cpu")
    with pytest.raises(ValueError):
        head.select_batch(torch.zeros(1, 3), requested)


def test_original_head_can_audit_exported_rows_without_replacing_its_parameters():
    original = nn.Linear(16, 64, bias=False).requires_grad_(False)
    identifiers = [9, 2, 41]
    head = RetainedVocabularyDecisionProjection(original, identifiers)
    before = set(head.state_dict())
    head.external_audit_bank = original.weight[identifiers].detach().clone()
    head.audit_option_ids = [41, 9]
    hidden = torch.randn(2, 16)
    torch.testing.assert_close(head(hidden), original(hidden))
    assert head.last_audit["external_original_rows_equal"]
    assert head.last_audit["external_maximum_probability_tv"] < 1e-6
    assert head.last_audit["external_argmax_equal"]
    assert set(head.state_dict()) == before and head.weight is original.weight
    head.external_audit_bank[0, 0] += 1
    head(hidden)
    assert not head.last_audit["external_original_rows_equal"]


def test_compact_audit_detects_loaded_weight_corruption_without_repairing_it():
    head = NativeDecisionProjection(16, [9, 2, 41], original_vocabulary=64,
                                    dtype=torch.bfloat16, device="cpu")
    before_keys = set(head.state_dict())
    pointer = head.weight.data_ptr()
    head.external_audit_bank = head.weight.detach().clone()
    head.audit_option_ids = [41, 9]
    hidden = torch.ones(2, 16, dtype=torch.bfloat16)
    head(hidden)
    assert head.last_audit["loaded_rows_bitwise_equal"]
    assert head.last_audit["maximum_probability_tv"] == 0
    with torch.no_grad():
        head.weight[0] += 1
    actual = head(hidden)
    assert not head.last_audit["loaded_rows_bitwise_equal"]
    assert head.last_audit["maximum_logit_difference"] > 1
    assert torch.equal(actual, nn.functional.linear(hidden, head.weight))
    assert head.weight.data_ptr() == pointer and set(head.state_dict()) == before_keys


def test_compact_body_comparison_excludes_only_the_intended_head():
    original = {"lm_head.weight": "1" * 64, "layer.weight": "2" * 64, "router": "3" * 64}
    compact = {**original, "lm_head.weight": "4" * 64}
    assert compare_frozen_body_hashes(compact, original)["body_values_exact"]
    compact["router"] = "5" * 64
    comparison = compare_frozen_body_hashes(compact, original)
    assert not comparison["body_values_exact"]
    assert comparison["different_body_tensors"] == ["router"]
    with pytest.raises(ValueError):
        compare_frozen_body_hashes({"lm_head.weight": "4" * 64}, original)
