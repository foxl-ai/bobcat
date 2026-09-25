"""Proper-score policy gradients, not TypeSafe's undisclosed training recipe.

The outcome-minus-probability / leave-one-out experiment follows Anthony Maio's
independent eve-rlcd implementation (MIT), pinned at
57a179b7b1bedc80f65bf42ccda129dd1888272f. It requires only sampled correctness
feedback. Its expected on-policy loss gradient equals half the Brier gradient.
No clipping, group standard-deviation normalization or reward differentiation
is part of that identity. Bobcat's retention loss is a separate regularizer.

For S(p,y) = -sum_i (p_i-y_i)^2, the action score 2*(y_a-p_a)
has E_a[score_a * grad log p_a] = grad S at the behavior policy.
The probability in the score is detached behavior data. Subtracting its exact
action-independent expectation is a variance-reduction baseline.

This is a sampled estimator of an available supervised Brier gradient. PPO
clipping, a finite batch, KL regularization and off-policy reuse can change that
gradient; no calibration guarantee is claimed. A direct Brier control is
mandatory. This formula has not been disclosed as Jev's RLCD algorithm.
"""

from __future__ import annotations

import torch


def outcome_policy_loss(logits, actions, outcomes, *, calibrated=True):
    """REINFORCE using only the observed outcomes of sampled actions.

    The caller samples IID with replacement from these exact logits, then asks
    an environment for correctness. No full label or unchosen reward is an
    argument to this learner. Leave-one-out excludes a sample's own reward,
    unlike a group mean which shrinks the estimator by (G-1)/G.
    """
    if (logits.ndim != 1 or not bool(torch.isfinite(logits).all())
            or actions.ndim != 1 or actions.dtype != torch.long
            or len(actions) < 2 or actions.shape != outcomes.shape
            or bool(((actions < 0) | (actions >= len(logits))).any())
            or not bool(((outcomes == 0) | (outcomes == 1)).all())
            or outcomes.requires_grad):
        raise ValueError("Require finite logits and at least two legal sampled binary outcomes.")
    scores = logits if logits.dtype == torch.float64 else logits.float()
    logs = scores.log_softmax(-1)
    taken = logs.detach().exp()[actions]
    rewards = outcomes.to(logs).detach() - taken if calibrated else outcomes.to(logs).detach()
    baseline = (rewards.sum() - rewards) / (len(rewards) - 1)
    advantages = (rewards - baseline).detach()
    return {
        "loss": -(advantages * logs[actions]).mean(),
        "rewards": rewards.detach(), "advantages": advantages,
        "sampled_probability": taken, "sampled_correctness": outcomes.detach(),
        "zero_group_advantage": bool((advantages.abs() < 1e-12).all()),
        "full_label_observed_by_learner": False,
        "reward_probability_differentiated": False,
        "method": "outcome_minus_probability_reinforce_loo" if calibrated
        else "correctness_only_reinforce_loo",
        "is_typesafe_disclosed_rlcd_recipe": False,
    }


def categorical_brier(logits, target_index):
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Require finite scores for the complete candidate set.")
    if type(target_index) is not int or not 0 <= target_index < logits.numel():
        raise ValueError("Brier scoring needs an observed category, not an ordinal mean.")
    probabilities = logits.float().softmax(-1) if logits.dtype != torch.float64 \
        else logits.softmax(-1)
    gold = torch.zeros_like(probabilities)
    gold[target_index] = 1.
    return (probabilities - gold).square().sum()


def marginal_brier_feedback(old_log_probs, target_index, actions):
    """Return frozen behavior-policy scores and the exact per-question baseline."""
    if (old_log_probs.ndim != 1 or not bool(torch.isfinite(old_log_probs).all())
            or not torch.allclose(old_log_probs.logsumexp(-1),
                                  old_log_probs.new_zeros(()), atol=1e-5, rtol=0)
            or type(target_index) is not int
            or not 0 <= target_index < old_log_probs.numel()
            or actions.ndim != 1 or actions.dtype != torch.long or not actions.numel()
            or bool(((actions < 0) | (actions >= old_log_probs.numel())).any())):
        raise ValueError("Feedback needs normalized old probabilities and sampled legal actions.")
    probabilities = old_log_probs.detach().exp()
    gold = torch.zeros_like(probabilities)
    gold[target_index] = 1.
    marginal = 2 * (gold - probabilities)
    baseline = (probabilities * marginal).sum()
    rewards = marginal[actions].detach()
    advantages = rewards - baseline.detach()
    return {
        "rewards": rewards, "advantages": advantages,
        "exact_action_independent_baseline": baseline.detach(),
        "negative_brier_score": -(probabilities - gold).square().sum(),
        "unbiased_brier_gradient_at_behavior_policy": True,
        "reward_is_policy_dependent_marginal_score": True,
        "is_typesafe_disclosed_rlcd_recipe": False,
    }
