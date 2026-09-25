"""Visible-input compilation and replayable lanes for the native RL pilot.

The environment's hidden records are never returned as model features. A lane
checkpoint contains its seed and observed action trace; restoration replays the
same host environment rather than copying unverified outcome fields.
"""

from __future__ import annotations

import copy

from bobcat.protocol import parse_request
from bobcat.rl_workflow import EvidenceWorkflow, Rule, make_world
from bobcat.schema import json_hash


class WorkflowLane:
    def __init__(self, *, first_seed, world_count, language, lane, lane_count=4):
        if (type(first_seed) is not int or first_seed < 0 or type(world_count) is not int
                or world_count < lane_count or world_count % lane_count
                or type(lane) is not int or not 0 <= lane < lane_count
                or language not in ("ko", "en")):
            raise ValueError("Use a finite, evenly partitioned bilingual world schedule.")
        self.settings = dict(first_seed=first_seed, world_count=world_count,
                             language=language, lane=lane, lane_count=lane_count)
        self.episode = 0
        self.trace = []
        self._reset()

    def _reset(self):
        p = self.settings
        offset = p["lane"] + self.episode * p["lane_count"]
        if offset >= p["world_count"]:
            raise ValueError("The frozen training world schedule is exhausted.")
        self.seed = p["first_seed"] + offset
        self.environment = EvidenceWorkflow(make_world(self.seed, p["language"]))
        self.trace = []

    def observation(self):
        return self.environment.observation()

    def advance(self, action):
        result = self.environment.step(action)
        self.trace.append(action)
        terminal_trace = list(self.trace)
        result["episode_trace"] = terminal_trace
        result["world_seed"] = self.seed
        if result["terminated"] or result["truncated"]:
            self.episode += 1
            self._reset()
        return result

    def state_dict(self):
        return {
            "schema": "bobcat-visible-workflow-lane-v1", "settings": dict(self.settings),
            "episode": self.episode, "seed": self.seed, "trace": list(self.trace),
            "observation_sha256": json_hash(self.observation()),
        }

    def load_state_dict(self, state):
        if (state.get("schema") != "bobcat-visible-workflow-lane-v1"
                or state.get("settings") != self.settings
                or type(state.get("episode")) is not int or state["episode"] < 0):
            raise ValueError("The lane belongs to a different frozen schedule.")
        self.episode = state["episode"]
        self._reset()
        if self.seed != state["seed"]:
            raise ValueError("The lane seed does not match its episode cursor.")
        for action in state["trace"]:
            result = self.environment.step(action)
            if result["terminated"] or result["truncated"]:
                raise ValueError("A saved active lane cannot contain a completed episode.")
            self.trace.append(action)
        if json_hash(self.observation()) != state["observation_sha256"]:
            raise ValueError("The restored visible environment changed.")


def compile_observation(payload, compiler):
    """Pass only the same public request used by a typed inference client."""
    state, questions = parse_request(copy.deepcopy(payload))
    if len(questions) != 1 or questions[0].kind != "choice":
        raise ValueError("This finite policy environment emits exactly one Choice.")
    compiled = compiler.compile(state, questions)
    inputs = {"input_ids": compiled.input_ids[0],
              "option_token_ids": compiled.option_token_ids[0]}
    if len(questions[0].labels) != len(inputs["option_token_ids"]):
        raise ValueError("Action labels and original vocabulary positions disagree.")
    return {
        **inputs, "input_sha256": json_hash(inputs),
        "observation_sha256": json_hash(payload),
        "candidate_ids": list(questions[0].labels),
        "input_tokens": len(inputs["input_ids"]),
    }


def visible_teacher_actions(payload):
    """All equally valid next actions, derived only from observed facts and rules."""
    state = payload["state"]
    rules = state.get("policy", state.get("정책"))
    facts = state.get("observed_facts", state.get("확인한 사실"))
    choices = payload["questions"]["next_action"]["criteria"]
    values = [Rule(**rule).matches(facts) for rule in rules]
    operation = "reject" if False in values else (
        "approve" if all(value is True for value in values) else "read"
    )
    selected = [name for name, value in choices.items() if value["operation"] == operation]
    if not selected and operation == "read":
        selected = [name for name, value in choices.items() if value["operation"] == "refer"]
    if not selected:
        raise ValueError("No supported visible teacher action exists.")
    return selected


def validate_world_splits(train_start, train_count, evaluation_start, evaluation_count):
    fields = (train_start, train_count, evaluation_start, evaluation_count)
    if any(type(value) is not int or value < 0 for value in fields):
        raise ValueError("Use nonnegative integer world ranges.")
    if not train_count or not evaluation_count:
        raise ValueError("Both training and evaluation need whole world components.")
    train = set(range(train_start, train_start + train_count))
    evaluation = set(range(evaluation_start, evaluation_start + evaluation_count))
    if train & evaluation:
        raise ValueError("Bilingual workflow components cross the evaluation boundary.")
    return {"training_worlds": train_count, "evaluation_worlds": evaluation_count,
            "component_overlap": 0, "same_generator_holdout": True,
            "unseen_task_generalization_claimed": False}
