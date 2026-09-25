import copy
from dataclasses import replace

import pytest

from bobcat.protocol import parse_request
from bobcat.rl_workflow import EvidenceWorkflow, Rule, World, make_world, visible_rule_policy


def fixture(*, evidence=True, maximum_steps=4):
    return World(
        component_id="held-out-fixture", language="ko",
        rules=(Rule("경과일", "at_most", 7), Rule("확인", "equals", True)),
        initial_facts={"경과일": 2}, documents={"확인서": evidence},
        order_seed=17, maximum_steps=maximum_steps,
    )


def action(env, operation):
    return next(name for name, value in env.legal_actions().items()
                if value["operation"] == operation)


def test_hidden_evidence_does_not_enter_initial_observation_or_candidate_order():
    yes = EvidenceWorkflow(fixture(evidence={"확인": True}))
    no = EvidenceWorkflow(fixture(evidence={"확인": False}))
    absent = EvidenceWorkflow(fixture(evidence=None))
    assert yes.observation() == no.observation() == absent.observation()
    state, questions = parse_request(yes.observation())
    assert "component_id" not in state and "outcome" not in state
    assert questions[0].kind == "choice"
    yes.step(action(yes, "read"))
    no.step(action(no, "read"))
    assert yes.observation() != no.observation()


def test_guessing_correct_hidden_answer_without_reading_is_not_rewarded():
    env = EvidenceWorkflow(fixture(evidence={"확인": True}))
    result = env.step(action(env, "approve"))
    assert result["reward"] == -1 and result["terminated"]
    assert result["info"]["failure"] == "unsupported_commitment"
    with pytest.raises(ValueError):
        env.observation()


@pytest.mark.parametrize("evidence,decision", [
    ({"확인": True}, "approve"), ({"확인": False}, "reject"), (None, "refer"),
])
def test_action_reveals_evidence_and_changes_correct_terminal_decision(evidence, decision):
    env = EvidenceWorkflow(fixture(evidence=evidence))
    first = env.step(action(env, "read"))
    assert first["reward"] == -.03 and not first["terminated"]
    assert not any(value["operation"] == "read" for value in env.legal_actions().values())
    last = env.step(action(env, decision))
    assert last["reward"] == 1 and last["info"]["success"]
    assert last["terminated"] and not last["truncated"]


def test_invalid_action_does_not_advance_time_or_disclose_evidence():
    env = EvidenceWorkflow(fixture(evidence={"확인": True}))
    before = copy.deepcopy(env.__dict__)
    with pytest.raises(ValueError):
        env.step("not offered")
    assert env.__dict__ == before


def test_step_limit_is_truncation_and_not_a_successful_terminal_disposition():
    world = fixture(evidence={"확인": True}, maximum_steps=2)
    world = replace(world, documents={"확인서": {"확인": True}, "추가": {}})
    env = EvidenceWorkflow(world)
    env.step(action(env, "read"))
    result = env.step(action(env, "read"))
    assert result["truncated"] and not result["terminated"]
    assert result["reward"] == -1.03 and not result["info"]["success"]


def test_visible_policy_handles_every_generated_bilingual_world_without_private_state():
    for seed in range(128):
        for language in ("ko", "en"):
            env = EvidenceWorkflow(make_world(seed, language))
            while not env.terminated and not env.truncated:
                result = env.step(visible_rule_policy(copy.deepcopy(env.observation())))
            assert result["info"]["success"]
            assert env.steps <= 3
        assert make_world(seed, "ko").component_id == make_world(seed, "en").component_id
