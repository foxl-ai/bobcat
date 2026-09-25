"""Local evidence-gathering environment for a later on-policy decision experiment.

Actions reveal records or commit a disposition. Hidden evidence and the
evaluator's outcome never enter observations. This module neither invokes a
model nor trains one, and has no external service or payment side effects.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from bobcat.schema import json_hash


@dataclass(frozen=True)
class Rule:
    field: str
    operation: str
    value: int | bool

    def matches(self, facts):
        if self.field not in facts or facts[self.field] is None:
            return None
        actual = facts[self.field]
        if self.operation == "equals":
            return actual == self.value
        if self.operation == "at_most":
            return actual <= self.value
        raise ValueError("Unknown host policy operator.")


@dataclass(frozen=True)
class World:
    component_id: str
    language: str
    rules: tuple[Rule, ...]
    initial_facts: dict
    documents: dict
    order_seed: int
    maximum_steps: int = 4


def make_world(seed: int, language: str) -> World:
    if type(seed) is not int or language not in ("ko", "en"):
        raise ValueError("Use an explicit world seed and Korean or English.")
    rng = random.Random(seed)
    days = "접수 후 경과일" if language == "ko" else "days_since_request"
    payment = "결제 확인" if language == "ko" else "payment_confirmed"
    identity = "신원 확인" if language == "ko" else "identity_verified"
    receipt = "결제 기록" if language == "ko" else "payment_record"
    verification = "신원 기록" if language == "ko" else "identity_record"
    threshold = rng.choice((3, 7, 14, 30))
    elapsed = rng.randint(0, threshold + 7)
    paid, identified = rng.random() < .8, rng.random() < .8
    unavailable = rng.choice((None, None, None, receipt, verification))
    # Use one component for Korean/English views of the same underlying world.
    return World(
        component_id=f"evidence-workflow-{seed}",
        language=language, rules=(Rule(days, "at_most", threshold),
                                 Rule(payment, "equals", True),
                                 Rule(identity, "equals", True)),
        initial_facts={days: elapsed},
        documents={
            receipt: None if unavailable == receipt else {payment: paid},
            verification: None if unavailable == verification else {identity: identified},
        },
        order_seed=seed ^ 0xB0BCA7,
    )


class EvidenceWorkflow:
    """Only the visible trace determines whether a commitment has sufficient evidence."""

    def __init__(self, world: World):
        if (world.language not in ("ko", "en") or not 2 <= world.maximum_steps <= 16
                or not world.rules or not world.documents):
            raise ValueError("Require a bounded, nonempty local workflow.")
        self.world = copy.deepcopy(world)
        self.facts = copy.deepcopy(world.initial_facts)
        self.read_documents = set()
        self.unavailable_documents = set()
        self.steps = 0
        self.terminated = self.truncated = False
        self.committed = None

    def legal_actions(self):
        if self.terminated or self.truncated:
            return {}
        ko = self.world.language == "ko"
        choices = {
            "승인" if ko else "approve": {"operation": "approve"},
            "거절" if ko else "reject": {"operation": "reject"},
            "검토 요청" if ko else "refer": {"operation": "refer"},
        }
        for document in self.world.documents:
            if document not in self.read_documents:
                name = f"{document} 조회" if ko else f"read {document}"
                choices[name] = {"operation": "read", "document": document}
        items = list(choices.items())
        random.Random(self.world.order_seed + self.steps).shuffle(items)
        return dict(items)

    def observation(self):
        if self.terminated or self.truncated:
            raise ValueError("A completed workflow has no next decision.")
        ko = self.world.language == "ko"
        state = {
            "정책" if ko else "policy": [
                {"field": rule.field, "operation": rule.operation, "value": rule.value}
                for rule in self.world.rules
            ],
            "확인한 사실" if ko else "observed_facts": copy.deepcopy(self.facts),
            "조회한 기록" if ko else "read_records": sorted(self.read_documents),
            "열 수 없는 기록" if ko else "unavailable_records": sorted(self.unavailable_documents),
            "남은 행동 횟수" if ko else "remaining_steps": self.world.maximum_steps - self.steps,
        }
        instructions = (
            "필요한 기록을 조회하고 정책에 따라 처리하세요. 모든 조건이 확인되면 승인, "
            "확인된 사실이 조건을 위반하면 거절하세요. 모든 기록을 조회했지만 열 수 없는 "
            "기록 때문에 판단할 수 없으면 검토 요청을 선택하세요. 근거 없이 결론을 내리면 "
            "실패입니다. 조회는 0.03의 비용이 들며 정확한 처리는 1, 잘못된 처리는 -1입니다."
            if ko else
            "Read evidence and apply the supplied policy. Approve after all conditions are "
            "verified; reject when observed evidence violates a condition. Refer only after "
            "reading all records if unavailable evidence prevents a decision. Unsupported "
            "commitments fail. Each read costs 0.03; a correct disposition earns 1, a wrong "
            "one -1."
        )
        return {
            "model": "bobcat-rl-research",
            "state": state,
            "questions": {"next_action": {
                "type": "choice", "instructions": instructions,
                "criteria": self.legal_actions(),
            }},
        }

    def supported_disposition(self):
        """Host evaluator over evidence actually observed, never a model input field."""
        known = [rule.matches(self.facts) for rule in self.world.rules]
        if False in known:
            return "reject"
        if all(value is True for value in known):
            return "approve"
        if (len(self.read_documents) == len(self.world.documents)
                and self.unavailable_documents):
            return "refer"
        return None

    def step(self, action):
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError("An invalid action must not change the workflow.")
        before = json_hash(self.observation())
        operation = legal[action]
        self.steps += 1
        success = False
        failure = None
        if operation["operation"] == "read":
            document = operation["document"]
            self.read_documents.add(document)
            evidence = self.world.documents[document]
            if evidence is None:
                self.unavailable_documents.add(document)
            else:
                self.facts.update(copy.deepcopy(evidence))
            reward = -.03
        else:
            supported = self.supported_disposition()
            success = operation["operation"] == supported
            reward = 1. if success else -1.
            failure = None if success else (
                "unsupported_commitment" if supported is None else "wrong_disposition"
            )
            self.committed = operation["operation"]
            self.terminated = True
        if not self.terminated and self.steps >= self.world.maximum_steps:
            self.truncated = True
            reward -= 1.
            failure = "step_limit"
        return {
            "observation": None if self.terminated or self.truncated else self.observation(),
            "reward": reward, "terminated": self.terminated, "truncated": self.truncated,
            "info": {
                "component_id": self.world.component_id, "action": action,
                "before_observation_sha256": before, "success": success,
                "failure": failure, "external_side_effects": False,
            },
        }


def visible_rule_policy(payload):
    """Deterministic control using only the same public observation as the learned actor."""
    state = payload["state"]
    rules = state.get("policy", state.get("정책"))
    facts = state.get("observed_facts", state.get("확인한 사실"))
    available = payload["questions"]["next_action"]["criteria"]
    values = [Rule(**rule).matches(facts) for rule in rules]
    desired = "reject" if False in values else "approve" if all(
        value is True for value in values
    ) else "read"
    choices = [name for name, operation in available.items()
               if operation["operation"] == desired]
    if not choices:
        choices = [name for name, operation in available.items()
                   if operation["operation"] == "refer"]
    if not choices:
        raise ValueError("The visible policy has no legal next action.")
    return sorted(choices)[0]
