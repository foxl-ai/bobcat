import math

import pytest
import torch

from bobcat.rl_objective import (
    clipped_actor_critic_loss,
    generalized_advantages,
    masked_log_probabilities,
)


def test_illegal_high_logit_receives_neither_probability_nor_gradient():
    logits = torch.tensor([[0.0, 1_000_000.0, 0.0]], requires_grad=True)
    legal = torch.tensor([[True, False, True]])
    logs = masked_log_probabilities(logits, legal)
    assert torch.equal(logs.exp(), torch.tensor([[.5, 0., .5]]))
    (-logs[0, 0]).backward()
    assert torch.equal(logits.grad, torch.tensor([[-.5, 0., .5]]))
    with pytest.raises(ValueError, match="legal action"):
        masked_log_probabilities(logits, torch.zeros_like(legal))


def test_advantages_separate_true_end_rollout_cutoff_and_finite_horizon():
    advantages, returns = generalized_advantages(
        torch.tensor([0., 1., 2., 3.], dtype=torch.float64),
        torch.tensor([.1, .2, .3, .4], dtype=torch.float64),
        torch.tensor([.2, 99., 7., 8.], dtype=torch.float64),
        terminated=torch.tensor([False, True, False, False]),
        truncated=torch.tensor([False, False, True, True]),
        bootstrap_allowed=torch.tensor([True, False, True, False]),
        discount=1., trace_decay=1.,
    )
    # Episode one returns 1, regardless of the next episode's critic value 99.
    # A collector cutoff retains value 7; an exhausted environment budget does not.
    assert torch.allclose(returns, torch.tensor([1., 1., 9., 3.], dtype=torch.float64))
    assert torch.allclose(advantages, returns - torch.tensor([.1, .2, .3, .4]))


def loss_inputs(probabilities):
    logits = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()
    count = len(probabilities)
    return dict(
        logits=logits, legal=torch.ones_like(logits, dtype=torch.bool),
        actions=torch.zeros(count, dtype=torch.long),
        old_action_log_probs=torch.full((count,), math.log(.5), dtype=torch.float64),
        advantages=torch.ones(count, dtype=torch.float64),
        values=torch.zeros(count, dtype=torch.float64, requires_grad=True),
        old_values=torch.zeros(count, dtype=torch.float64),
        returns=torch.zeros(count, dtype=torch.float64),
        reference_log_probs=torch.full_like(logits, math.log(.5)),
    )


def test_clipping_stops_both_excessive_positive_and_negative_policy_updates():
    inputs = loss_inputs([[.9, .1], [.1, .9]])
    inputs["advantages"] = torch.tensor([1., -1.], dtype=torch.float64)
    result = clipped_actor_critic_loss(
        **inputs, value_coefficient=0., reference_kl_coefficient=0.,
    )
    # The hand-computed clipped surrogates are +1.2 and -0.8.
    assert result["actor_loss"].item() == pytest.approx(-.2)
    result["loss"].backward()
    assert torch.equal(inputs["logits"].grad, torch.zeros_like(inputs["logits"]))
    assert result["clip_fraction"].item() == 1.
    assert result["calibrated_event_probability"] is False


def test_valid_on_policy_update_has_correct_direction_without_target_gradients():
    inputs = loss_inputs([[.5, .5]])
    for name in ("old_action_log_probs", "advantages", "old_values", "returns",
                 "reference_log_probs"):
        inputs[name].requires_grad_()
    result = clipped_actor_critic_loss(**inputs)
    result["loss"].backward()
    assert inputs["logits"].grad[0, 0] < 0
    assert inputs["logits"].grad[0, 1] > 0
    assert result["approximate_old_policy_kl"].item() == 0.
    for name in ("old_action_log_probs", "advantages", "old_values", "returns",
                 "reference_log_probs"):
        assert inputs[name].grad is None


def test_reference_on_different_candidate_set_or_scale_is_rejected():
    inputs = loss_inputs([[.5, .5]])
    inputs["reference_log_probs"] = torch.tensor([[0., 0.]], dtype=torch.float64)
    with pytest.raises(ValueError, match="normalized"):
        clipped_actor_critic_loss(**inputs)
    inputs = loss_inputs([[.5, .5]])
    inputs["legal"][0, 0] = False
    with pytest.raises(ValueError, match="not legal"):
        clipped_actor_critic_loss(**inputs)


def test_terminal_state_cannot_accidentally_bootstrap_into_next_episode():
    with pytest.raises(ValueError, match="cannot truncate or bootstrap"):
        generalized_advantages(
            torch.tensor([1.]), torch.tensor([.1]), torch.tensor([900.]),
            terminated=torch.tensor([True]), truncated=torch.tensor([False]),
            bootstrap_allowed=torch.tensor([True]),
        )


def test_padded_candidate_does_not_make_anchored_loss_nan():
    inputs = loss_inputs([[.5, .5]])
    inputs["logits"] = torch.tensor([[0., 99., 0.]], requires_grad=True)
    inputs["legal"] = torch.tensor([[True, False, True]])
    inputs["values"] = torch.zeros(1, dtype=torch.bfloat16, requires_grad=True)
    # Neither an invalid entry nor the full-vocabulary distribution defines
    # the legal policy. A BF16 critic must still accumulate its loss in FP32.
    inputs["reference_log_probs"] = masked_log_probabilities(
        torch.zeros_like(inputs["logits"]), inputs["legal"],
    ).detach()
    result = clipped_actor_critic_loss(**inputs)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(inputs["logits"].grad).all()
    assert inputs["logits"].grad[0, 1] == 0
