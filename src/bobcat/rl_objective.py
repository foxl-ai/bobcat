"""PPO arithmetic for the planned finite decision-policy experiment.

This module does not load GLM, collect rollouts or train a model. Policy action
probabilities are not calibrated event probabilities. A native actor, critic,
on-policy collector, supervised control and independent evaluation are required.
"""

from __future__ import annotations

import math

import torch


def _vector(name, value, count):
    if value.shape != (count,) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point vector of length {count}.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite value.")


def masked_log_probabilities(logits, legal):
    """Normalize over the complete legal set; never silently shortlist actions."""
    if logits.ndim != 2 or not logits.is_floating_point() or not logits.shape[0]:
        raise ValueError("Require a nonempty [decisions, candidates] logit matrix.")
    if legal.shape != logits.shape or legal.dtype != torch.bool:
        raise ValueError("The legal-action mask must match the logits.")
    if not bool(legal.any(dim=-1).all()) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Every decision needs a legal action and finite logits.")
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    return logits.masked_fill(~legal, -torch.inf).log_softmax(dim=-1)


def generalized_advantages(
    rewards, values, next_values, *, terminated, truncated, bootstrap_allowed,
    discount=0.99, trace_decay=0.95,
):
    """GAE for one chronological lane, which may contain several episodes.

    An administrative rollout cutoff bootstraps from the next-state critic.
    A true terminal state, or this environment's exhausted finite decision
    budget, does not. The caller must make that distinction explicitly.
    A truncation always stops the advantage trace from entering the next episode.
    """
    count = rewards.numel()
    if not count:
        raise ValueError("An empty trajectory cannot define advantages.")
    for name, value in (
        ("rewards", rewards), ("values", values), ("next_values", next_values),
    ):
        _vector(name, value, count)
    for flag in (terminated, truncated, bootstrap_allowed):
        if flag.shape != (count,) or flag.dtype != torch.bool:
            raise ValueError("Trajectory boundary flags must be boolean vectors.")
    if bool((terminated & truncated).any()) or bool((terminated & bootstrap_allowed).any()):
        raise ValueError("A true terminal state cannot truncate or bootstrap.")
    if bool((~(terminated | truncated) & ~bootstrap_allowed).any()):
        raise ValueError("An ongoing transition must retain its next-state value.")
    if not 0 <= discount <= 1 or not 0 <= trace_decay <= 1:
        raise ValueError("Discount and trace decay must be in [0, 1].")
    if not bool(terminated[-1] | truncated[-1]):
        raise ValueError("Declare the final rollout boundary explicitly.")
    # Old critic values and rewards are data, not differentiable actor targets.
    rewards, values, next_values = (
        value.detach() for value in (rewards, values, next_values)
    )
    delta = rewards + discount * torch.where(
        bootstrap_allowed, next_values, torch.zeros_like(next_values)
    ) - values
    advantages = torch.empty_like(delta)
    tail = torch.zeros((), dtype=delta.dtype, device=delta.device)
    for index in range(count - 1, -1, -1):
        continues = ~(terminated[index] | truncated[index])
        tail = delta[index] + discount * trace_decay * continues * tail
        advantages[index] = tail
    return advantages, advantages + values


def clipped_actor_critic_loss(
    logits, legal, actions, old_action_log_probs, advantages, values, old_values,
    returns, reference_log_probs, *, policy_clip=0.2, value_clip=0.2,
    value_coefficient=0.5, reference_kl_coefficient=0.01, entropy_coefficient=0.0,
):
    """PPO clipped surrogate plus critic loss and an explicit frozen-policy anchor.

    Inputs must come from actual on-policy transitions, not relabeled gold rows.
    A separate supervised replay loss is required by the Bobcat experiment plan.
    The returned mean is local: distributed training must weight valid sample
    counts correctly, rather than averaging unequal per-rank means.
    """
    log_probs = masked_log_probabilities(logits, legal)
    count = logits.shape[0]
    if actions.shape != (count,) or actions.dtype != torch.long:
        raise ValueError("Actions must be one integer candidate index per decision.")
    if bool(((actions < 0) | (actions >= logits.shape[1])).any()):
        raise ValueError("A sampled action is outside the offered candidate set.")
    if not bool(legal.gather(1, actions[:, None]).all()):
        raise ValueError("A sampled action was not legal at rollout time.")
    for name, value in (
        ("old_action_log_probs", old_action_log_probs), ("advantages", advantages),
        ("values", values), ("old_values", old_values), ("returns", returns),
    ):
        _vector(name, value, count)
    settings = (
        policy_clip, value_clip, value_coefficient,
        reference_kl_coefficient, entropy_coefficient,
    )
    if not all(math.isfinite(value) and value >= 0 for value in settings):
        raise ValueError("PPO coefficients must be finite and nonnegative.")
    if not 0 < policy_clip < 1:
        raise ValueError("Policy clipping must be strictly between 0 and 1.")
    if bool((old_action_log_probs > 1e-6).any()):
        raise ValueError("Old log probabilities must not be positive.")
    if reference_log_probs.shape != logits.shape:
        raise ValueError("The frozen reference must score exactly the same legal actions.")
    reference = reference_log_probs.detach().to(log_probs).masked_fill(~legal, -torch.inf)
    if not bool(torch.isfinite(reference[legal]).all()) or not torch.allclose(
        reference.logsumexp(dim=-1), torch.zeros_like(log_probs[:, 0]), atol=1e-5, rtol=0,
    ):
        raise ValueError("The reference must be a finite, normalized legal distribution.")
    selected = log_probs.gather(1, actions[:, None]).squeeze(1)
    log_ratio = selected - old_action_log_probs.detach().to(log_probs)
    if bool((log_ratio.abs() > 60).any()):
        raise ValueError("Policy drift is too large for a numerically valid PPO update.")
    ratio = log_ratio.exp()
    detached_advantages = advantages.detach().to(log_probs)
    surrogate = torch.minimum(
        ratio * detached_advantages,
        ratio.clamp(1 - policy_clip, 1 + policy_clip) * detached_advantages,
    )
    actor_loss = -surrogate.mean()
    values = values.to(log_probs)
    old_values = old_values.detach().to(log_probs)
    returns = returns.detach().to(log_probs)
    clipped_values = old_values + (
        values - old_values
    ).clamp(-value_clip, value_clip)
    critic_loss = 0.5 * torch.maximum(
        (values - returns).square(),
        (clipped_values - returns).square(),
    ).mean()
    # Replace invalid log entries before multiplying by zero probability:
    # 0 * (-inf) would otherwise contaminate both values and gradients.
    safe_logs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    safe_reference = torch.where(legal, reference, torch.zeros_like(reference))
    probabilities = log_probs.exp()
    entropy = -(probabilities * safe_logs).sum(dim=-1).mean()
    reference_kl = (probabilities * (safe_logs - safe_reference)).sum(dim=-1).mean()
    total = (
        actor_loss + value_coefficient * critic_loss
        + reference_kl_coefficient * reference_kl - entropy_coefficient * entropy
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("A non-finite PPO loss must not reach the optimizer.")
    return {
        "loss": total, "actor_loss": actor_loss, "critic_loss": critic_loss,
        "reference_kl": reference_kl, "entropy": entropy,
        "approximate_old_policy_kl": (ratio - 1 - log_ratio).mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > policy_clip).float().mean().detach(),
        "sample_count": count,
        "calibrated_event_probability": False,
    }
