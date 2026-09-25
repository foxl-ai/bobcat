import copy

import pytest

from bobcat.rl_collection import WorkflowLane, validate_world_splits, visible_teacher_actions
from bobcat.rl_workflow import EvidenceWorkflow, World, make_world
from bobcat.schema import json_hash


def test_lane_resume_reproduces_visible_trajectory_and_next_world():
    lane = WorkflowLane(first_seed=1000, world_count=256, language="ko", lane=2)
    snapshot = lane.state_dict()
    first = []
    for _ in range(15):
        payload = lane.observation()
        action = visible_teacher_actions(payload)[0]
        first.append(lane.advance(action))
    after = lane.state_dict()
    lane.load_state_dict(snapshot)
    second = []
    for _ in range(15):
        action = visible_teacher_actions(lane.observation())[0]
        second.append(lane.advance(action))
    assert first == second and after == lane.state_dict()


def test_language_pairs_use_same_component_without_crossing_split():
    ko = WorkflowLane(first_seed=100, world_count=16, language="ko", lane=1)
    en = WorkflowLane(first_seed=100, world_count=16, language="en", lane=1)
    assert ko.environment.world.component_id == en.environment.world.component_id
    assert ko.seed == en.seed == 101
    assert validate_world_splits(100, 16, 1000, 16)["component_overlap"] == 0
    with pytest.raises(ValueError, match="cross"):
        validate_world_splits(100, 16, 110, 16)


def test_unread_hidden_records_cannot_change_visible_teacher_targets():
    world = make_world(10001, "en")
    changed = World(world.component_id, world.language, world.rules, world.initial_facts,
                    {name: {"secret_outcome": "arbitrary"} for name in world.documents},
                    world.order_seed, world.maximum_steps)
    first, second = EvidenceWorkflow(world).observation(), EvidenceWorkflow(changed).observation()
    assert first == second
    assert visible_teacher_actions(first) == visible_teacher_actions(second)


def test_saved_lane_rejects_forged_state_or_schedule():
    lane = WorkflowLane(first_seed=1000, world_count=256, language="en", lane=0)
    state = lane.state_dict()
    for key, value in (("seed", -1), ("observation_sha256", "wrong"),
                       ("episode", -1), ("schema", "untrusted")):
        invalid = {**state, key: value}
        with pytest.raises(ValueError):
            lane.load_state_dict(invalid)
    forged = copy.deepcopy(state)
    forged["settings"]["language"] = "ko"
    with pytest.raises(ValueError):
        lane.load_state_dict(forged)


def test_visible_teacher_earns_supported_reward_in_both_languages():
    success = 0
    for language in ("ko", "en"):
        for seed in range(128):
            env = EvidenceWorkflow(make_world(seed, language))
            while not (env.terminated or env.truncated):
                payload = env.observation()
                before = json_hash(payload)
                options = visible_teacher_actions(payload)
                assert json_hash(payload) == before
                result = env.step(options[-1])
            success += result["info"]["success"]
    assert success == 256
