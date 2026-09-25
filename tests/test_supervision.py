import copy

import pytest
import torch

from bobcat.supervision import evaluate_supervised, losses, target_tensors, target_values


def mean_row(value=2.5):
    return {
        "id": "fixture", "group_id": "fixture", "kind": "ordinal",
        "target": None, "score_target": value, "supervision": "score_mean",
        "candidate_ids": [str(i) for i in range(6)],
    }


def test_ordinal_mean_preserves_fraction_and_does_not_invent_a_distribution():
    row = mean_row(26 / 7)
    target, mean = target_values(row)
    assert target == -100 and mean == 26 / 7
    wrong = copy.deepcopy(row)
    wrong["target"] = "4"
    with pytest.raises(ValueError, match="scalar"):
        target_values(wrong)
    wrong = copy.deepcopy(row)
    wrong["candidate_ids"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        target_values(wrong)


def test_equal_means_have_equal_loss_despite_different_uncertainty():
    rows = [mean_row(), mean_row()]
    target, mean = target_tensors(rows)
    # Uniform and almost-bimodal predictions both have mean 2.5.
    logits = torch.tensor([[0.] * 6, [5., -5., -5., -5., -5., 5.]], requires_grad=True)
    actual = losses(logits, target, mean)
    torch.testing.assert_close(actual, torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0)
    report = evaluate_supervised([{**r, "logits": v} for r, v in zip(
        rows, logits.detach().tolist(), strict=True,
    )])
    assert report["hard_label_questions"] == 0
    assert report["ordinal_mean_metrics"]["mae"] == pytest.approx(0)
    assert report["ordinal_mean_metrics"]["categorical_ece"] is None


def test_mixed_ce_and_mean_loss_use_only_their_observed_target_type():
    rows = [
        {"candidate_ids": ["no", "yes"], "kind": "boolean", "target": "yes"},
        mean_row(4.2),
    ]
    target, mean = target_tensors(rows)
    logits = torch.tensor([[0., 1., -torch.inf, -torch.inf, -torch.inf, -torch.inf],
                           [0., 0., 0., 0., 0., 0.]], requires_grad=True)
    values = losses(logits, target, mean)
    assert float(values[0].detach()) == pytest.approx(float(torch.nn.functional.cross_entropy(
        logits[:1, :2], torch.tensor([1]),
    ).detach()))
    assert float(values[1].detach()) == pytest.approx((2.5 - 4.2) ** 2)
    values.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1, 5] < 0 and logits.grad[1, 0] > 0
    assert torch.equal(logits.grad[0, 2:], torch.zeros(4))
