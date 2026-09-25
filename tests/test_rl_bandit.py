import pytest
import torch

from bobcat.rl_bandit import DecisionBandit


def test_gold_is_not_in_observation_and_only_sampled_outcomes_return():
    row = {
        "supervision": "hard_label", "target_index": 2,
        "input_ids": [1, 9], "option_token_ids": [7, 8, 10],
        "candidate_ids": ["billing", "shipping", "technical"],
        "id": "train-1", "group_id": "component-1", "language": "ko",
    }
    env = DecisionBandit([row])
    observation = env.observation(0)
    assert set(observation) == {"input_ids", "option_token_ids", "candidate_ids"}
    assert env.step(0, torch.tensor([1, 2, 1, 2])).tolist() == [0, 1, 0, 1]
    assert env.reveal_for_supervised_control(0) == 2
    observation["input_ids"].clear()
    assert env.observation(0)["input_ids"] == [1, 9]
    with pytest.raises(ValueError):
        env.step(0, torch.tensor([3]))
    with pytest.raises(ValueError):
        DecisionBandit([row | {"supervision": "score_mean"}])
