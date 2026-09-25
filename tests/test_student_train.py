import itertools

import pytest

torch = pytest.importorskip("torch")

from bobcat.student_train import PointerHead, decision_loss  # noqa: E402


def test_sft_brier_and_score_losses():
    logits = torch.tensor([2.0, 0.0, -1.0])
    p = torch.softmax(logits, 0)
    hard = {"supervision": "hard_label", "target": 1}
    assert torch.isclose(decision_loss(torch, logits, hard, "sft"), -torch.log(p[1]))
    onehot = torch.tensor([0.0, 1.0, 0.0])
    brier = decision_loss(torch, logits, hard, "brier")
    assert torch.isclose(brier, 0.5 * ((p - onehot) ** 2).sum())
    mean = {"supervision": "score_mean", "score_target": 1.5}
    expected = (p * torch.arange(3.0)).sum()
    assert torch.isclose(decision_loss(torch, logits, mean, "rl"), ((expected - 1.5) / 2) ** 2)


def test_rl_expected_gradient_matches_half_brier():
    logits = torch.tensor([0.3, -0.2, 0.9], dtype=torch.float64, requires_grad=True)
    row = {"supervision": "hard_label", "target": 0}
    p = torch.softmax(logits, 0).detach()
    exact = torch.zeros(3, dtype=torch.float64)
    # Enumerate every ordered group of 4 IID actions with its probability.
    for actions in itertools.product(range(3), repeat=4):
        weight = torch.prod(p[list(actions)])
        log_p = torch.log_softmax(logits, 0)
        reward = (torch.tensor(actions) == 0).double() - p[list(actions)]
        baseline = (reward.sum() - reward) / 3
        loss = -((reward - baseline) * log_p[list(actions)]).mean()
        exact += weight * torch.autograd.grad(loss, logits)[0]
    brier = decision_loss(torch, logits, row, "brier")
    target = torch.autograd.grad(brier, logits)[0]
    assert torch.allclose(exact, target, atol=1e-12)


def test_pointer_head_starts_as_a_zero_residual():
    head = PointerHead.build(torch, hidden=8, width=4)
    out = head(torch.randn(8), torch.randn(3, 8))
    assert out.shape == (3,) and torch.all(out == 0)
