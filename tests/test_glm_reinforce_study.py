from types import SimpleNamespace

import torch

from bobcat.glm_reinforce_study import update
from bobcat.rl_bandit import DecisionBandit


def test_native_update_boundary_uses_sampled_feedback_without_oracle(monkeypatch):
    row = {
        "supervision": "hard_label", "target_index": 1,
        "input_ids": [2, 3], "option_token_ids": [5, 6],
        "candidate_ids": ["approve", "review"], "group_id": "example",
    }
    environment = DecisionBandit([row])
    parameter = torch.nn.Parameter(torch.tensor([.2, -.1]))
    optimizer = torch.optim.AdamW([parameter], lr=.01, foreach=False)
    experiment = SimpleNamespace(
        bandit=environment, target_cursor=0, rank=0, world=1, device="cpu", torch=torch,
        optimizer=optimizer, critic_optimizer=optimizer, updates=0, transitions=0,
        args=SimpleNamespace(samples_per_question=4, replay_coefficient=0),
        sampling=torch.Generator().manual_seed(14), loop={"cursor": 0},
    )
    observed = []

    def forward(payload):
        observed.append(payload)
        assert "target_index" not in payload
        return parameter * 1., None

    def step():
        norm = float(parameter.grad.norm())
        optimizer.step()
        experiment.updates += 1
        return norm

    def forbidden(_):
        raise AssertionError("The RL learner requested an unchosen label.")

    experiment.forward, experiment.clip_and_step = forward, step
    monkeypatch.setattr(environment, "reveal_for_supervised_control", forbidden)
    result = update(experiment, "proper_score_reinforce")
    assert len(observed) == 1
    assert experiment.transitions == 4 and experiment.updates == 1
    assert experiment.target_cursor == 1 and experiment.loop["cursor"] == 1
    assert result["full_label_observed_by_learner"] is False
    assert len(result["sampled_rewards"]) == 4


def test_supervised_control_is_exact_half_brier_without_rl_samples():
    from bobcat.rl_calibration import categorical_brier

    row = {"supervision": "hard_label", "target_index": 1, "input_ids": [1],
           "option_token_ids": [5, 6], "candidate_ids": ["a", "b"]}
    parameter = torch.nn.Parameter(torch.tensor([.2, -.1]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    expected = torch.autograd.grad(.5 * categorical_brier(parameter, 1), parameter)[0]
    experiment = SimpleNamespace(
        bandit=DecisionBandit([row]), target_cursor=0, rank=0, world=1,
        optimizer=optimizer, critic_optimizer=optimizer, updates=0, transitions=0,
        args=SimpleNamespace(replay_coefficient=0), loop={"cursor": 0},
        forward=lambda _: (parameter * 1., None),
        clip_and_step=lambda: float(parameter.grad.norm()),
    )
    result = update(experiment, "direct_brier")
    assert torch.equal(parameter.grad, expected)
    assert experiment.transitions == 0
    assert result["full_label_observed_by_learner"] is True
