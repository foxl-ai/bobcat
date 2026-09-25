import pytest
import torch

from bobcat.rl_calibration import (
    categorical_brier,
    marginal_brier_feedback,
    outcome_policy_loss,
)


def test_full_action_expectation_matches_direct_brier_gradient():
    logits = torch.tensor([.2, -.4, 1.1], dtype=torch.float64, requires_grad=True)
    logs = logits.log_softmax(-1)
    actions = torch.arange(3)
    feedback = marginal_brier_feedback(logs.detach(), 1, actions)
    estimator = -(logs.detach().exp() * feedback["advantages"] * logs).sum()
    sampled_gradient = torch.autograd.grad(estimator, logits, retain_graph=True)[0]
    direct_gradient = torch.autograd.grad(categorical_brier(logits, 1), logits)[0]
    assert torch.allclose(sampled_gradient, direct_gradient, atol=1e-12, rtol=1e-12)
    assert feedback["is_typesafe_disclosed_rlcd_recipe"] is False


def test_calibrated_population_is_stationary_while_correctness_reward_is_not():
    # Population P(y=true)=.7. The proper score's expected gradient vanishes
    # at .7; maximizing sampled correctness still pushes toward 100% true.
    logits = torch.tensor([.3, .7], dtype=torch.float64).log().requires_grad_()
    proper = .3 * categorical_brier(logits, 0) + .7 * categorical_brier(logits, 1)
    correct = -(.3 * logits.softmax(-1)[0] + .7 * logits.softmax(-1)[1])
    assert torch.allclose(torch.autograd.grad(proper, logits, retain_graph=True)[0],
                          torch.zeros_like(logits), atol=1e-12)
    assert torch.autograd.grad(correct, logits)[0][1] < 0


def test_feedback_is_frozen_and_rejects_invalid_or_mean_target():
    old = torch.tensor([.4, .6]).log().requires_grad_()
    feedback = marginal_brier_feedback(old, 1, torch.tensor([0, 1, 1]))
    assert not feedback["advantages"].requires_grad
    with pytest.raises(ValueError):
        marginal_brier_feedback(old, .5, torch.tensor([0]))
    with pytest.raises(ValueError):
        categorical_brier(old, .5)
    with pytest.raises(ValueError):
        marginal_brier_feedback(torch.tensor([0., 0.]), 1, torch.tensor([0]))


def test_loo_group_expectation_equals_half_brier_gradient():
    # Enumerate every possible IID group, including duplicates, instead of a
    # stochastic tolerance check. All probabilities of sampling are detached.
    import itertools

    logits = torch.tensor([.2, -.4, 1.1], dtype=torch.float64, requires_grad=True)
    probabilities = logits.detach().softmax(-1)
    expectation = logits.sum() * 0
    for group in itertools.product(range(3), repeat=3):
        actions = torch.tensor(group)
        outcomes = (actions == 1).double()
        result = outcome_policy_loss(logits, actions, outcomes)
        expectation = expectation + probabilities[actions].prod() * result["loss"]
        assert not result["advantages"].requires_grad
    actual = torch.autograd.grad(expectation, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(.5 * categorical_brier(logits, 1), logits)[0]
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_correctness_control_has_different_expected_objective():
    import itertools

    logits = torch.tensor([.3, .7], dtype=torch.float64).log().requires_grad_()
    probabilities = logits.detach().softmax(-1)
    expected_loss = logits.sum() * 0
    for truth, prevalence in enumerate((.3, .7)):
        for group in itertools.product(range(2), repeat=2):
            actions = torch.tensor(group)
            result = outcome_policy_loss(logits, actions, (actions == truth).double(),
                                         calibrated=False)
            expected_loss = expected_loss + prevalence * probabilities[actions].prod() \
                * result["loss"]
    # Accuracy-only RL pushes the already calibrated .7 distribution toward 1.
    assert torch.autograd.grad(expected_loss, logits)[0][1] < 0


def test_duplicate_samples_and_invalid_outcomes():
    logits = torch.tensor([-.1, .8], requires_grad=True)
    result = outcome_policy_loss(logits, torch.tensor([1, 1, 1, 1]), torch.ones(4))
    assert result["zero_group_advantage"]
    assert result["full_label_observed_by_learner"] is False
    with pytest.raises(ValueError):
        outcome_policy_loss(logits, torch.tensor([0, 1]), torch.tensor([.3, 1.]))
    with pytest.raises(ValueError):
        outcome_policy_loss(logits, torch.tensor([1]), torch.tensor([1.]))
