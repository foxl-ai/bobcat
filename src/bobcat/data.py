"""Generated policy worlds, a finite-domain oracle, and partition audits.

This is a software/reasoning testbed, not a general-language benchmark.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from bobcat.schema import INSUFFICIENT, NONE, Choice, Example, file_hash, write_examples

TRAIN_FAMILIES = ("lookup", "threshold", "conjunction", "ordinal")
DEV_OOD_FAMILIES = ("priority", "interval")
TEST_OOD_FAMILIES = ("nested", "exception")
GENERATOR_VERSION = "policy-worlds-v1"
ORACLE_VERSION = "candidate-outcome-v2"


def evaluate_condition(condition: dict, facts: dict) -> bool:
    op = condition["op"]
    if op == "and":
        return all(evaluate_condition(c, facts) for c in condition["args"])
    if op == "or":
        return any(evaluate_condition(c, facts) for c in condition["args"])
    if op == "not":
        return not evaluate_condition(condition["arg"], facts)
    left = facts[condition["field"]]
    if left is None:
        raise ValueError("The oracle must enumerate missing facts before evaluation.")
    right = condition["value"]
    if op == "eq":
        return left == right
    if op == "le":
        return left <= right
    if op == "ge":
        return left >= right
    raise ValueError(f"Unknown condition operator: {op}")


def execute_known(program: dict, facts: dict) -> str:
    for rule in program["rules"]:
        if evaluate_condition(rule["when"], facts):
            return rule["action"]
    return program["default"]


def possible_outcomes(program: dict) -> set[str]:
    missing = [name for name, value in program["facts"].items() if value is None]
    domains = [program["domains"][name] for name in missing]
    outcomes = set()
    for values in itertools.product(*domains):
        facts = {**program["facts"], **dict(zip(missing, values, strict=True))}
        outcomes.add(execute_known(program, facts))
    return outcomes


def oracle_target(program: dict, query: dict, choices: list[Choice]) -> str:
    outcomes = possible_outcomes(program)
    if query["kind"] == "boolean":
        outcomes = {"true" if outcome == query["action"] else "false" for outcome in outcomes}
    decisions = {
        next((choice.id for choice in choices if choice.text == outcome), NONE)
        for outcome in outcomes
    }
    if len(decisions) != 1:
        return INSUFFICIENT
    return next(iter(decisions))


def condition_text(condition: dict) -> str:
    op = condition["op"]
    if op in {"and", "or"}:
        return "(" + f" {op} ".join(condition_text(c) for c in condition["args"]) + ")"
    if op == "not":
        return f"not ({condition_text(condition['arg'])})"
    operator = {"eq": "equals", "le": "is at most", "ge": "is at least"}[op]
    return f"{condition['field']} {operator} {str(condition['value']).lower()}"


def render_program(program: dict) -> str:
    lines = ["Apply the first matching rule. A later rule never overrides an earlier rule."]
    for index, rule in enumerate(program["rules"], 1):
        lines.append(f"Rule {index}: if {condition_text(rule['when'])}, select {rule['action']}.")
    lines.append(f"If no rule matches, select {program['default']}.")
    lines.append("Observed facts:")
    for key, value in program["facts"].items():
        shown = "unknown" if value is None else str(value).lower()
        lines.append(f"{key}: {shown}.")
    return "\n".join(lines)


def atom(field: str, value: object, op: str = "eq") -> dict:
    return {"op": op, "field": field, "value": value}


def make_program(family: str, rng: random.Random) -> tuple[dict, list[str]]:
    # Nonce action names are independent of the split, position, and answer.
    names = rng.sample(
        ["amber", "birch", "cedar", "coral", "dawn", "elm", "fern", "jade", "lark", "moss"],
        5,
    )
    suffix = rng.randrange(1000, 9999)
    actions = [f"{name}-{suffix}" for name in names]
    rng.shuffle(actions)
    a, b, c = actions[:3]
    score = rng.randrange(0, 21)
    threshold = rng.randrange(4, 17)
    flags = {name: bool(rng.randrange(2)) for name in ["verified", "sealed", "urgent"]}
    domains = {name: [False, True] for name in flags}
    facts = dict(flags)
    facts["score"] = score
    domains["score"] = list(range(21))
    default = b
    if family == "lookup":
        facts["signal"] = rng.choice(["north", "south", "east"])
        domains["signal"] = ["north", "south", "east"]
        rules = [
            {"when": atom("signal", direction), "action": action}
            for direction, action in zip(rng.sample(domains["signal"], 2), [a, c], strict=True)
        ]
    elif family == "threshold":
        rules = [{"when": atom("score", threshold, rng.choice(["le", "ge"])), "action": a}]
    elif family == "conjunction":
        condition = {
            "op": rng.choice(["and", "or"]),
            "args": [atom("verified", True), atom("sealed", True)],
        }
        rules = [{"when": condition, "action": a}]
    elif family == "ordinal":
        actions = ["low", "medium", "high"]
        default = "high"
        rules = [
            {"when": atom("score", threshold - 2, "le"), "action": "low"},
            {"when": atom("score", threshold + 2, "le"), "action": "medium"},
        ]
        # A harmless case token avoids accidental identical prompts across partitions.
        facts["case"] = f"{names[0]}-{suffix}"
        domains["case"] = [facts["case"]]
    elif family == "priority":
        rules = [
            {"when": atom("urgent", True), "action": c},
            {"when": atom("score", threshold, "le"), "action": a},
        ]
    elif family == "interval":
        condition = {
            "op": "and",
            "args": [
                atom("score", threshold - 3, "ge"),
                atom("score", threshold + 3, "le"),
            ],
        }
        rules = [{"when": condition, "action": a}]
    elif family == "nested":
        condition = {
            "op": "and",
            "args": [
                {"op": "or", "args": [atom("verified", True), atom("urgent", True)]},
                {"op": "not", "arg": atom("sealed", True)},
            ],
        }
        rules = [
            {"when": condition, "action": a},
            {"when": atom("score", threshold, "ge"), "action": c},
        ]
    elif family == "exception":
        condition = {
            "op": "and",
            "args": [
                atom("verified", True),
                {
                    "op": "not",
                    "arg": {
                        "op": "or",
                        "args": [
                            atom("urgent", True),
                            atom("score", threshold, "le"),
                        ],
                    },
                },
            ],
        }
        rules = [{"when": condition, "action": a}]
    else:
        raise ValueError(f"Unknown family: {family}")
    return {"facts": facts, "domains": domains, "rules": rules, "default": default}, actions


def change_one_clause(program: dict, actions: list[str], rng: random.Random) -> dict:
    changed = copy.deepcopy(program)
    representative = {
        key: value if value is not None else program["domains"][key][0]
        for key, value in program["facts"].items()
    }
    current = execute_known(program, representative)
    alternative = rng.choice([action for action in actions if action != current])
    for rule in changed["rules"]:
        if evaluate_condition(rule["when"], representative):
            rule["action"] = alternative
            break
    else:
        changed["default"] = alternative
    return changed


def world_examples(index: int, split: str, family: str, seed: int) -> list[Example]:
    rng = random.Random(seed + index * 104729)
    program, actions = make_program(family, rng)
    mode = index % 8
    # Missing evidence is not automatically "insufficient": enumerate all completions.
    if mode in {0, 1, 2}:
        field = {
            "lookup": "signal",
            "conjunction": "verified",
            "nested": "sealed",
            "exception": "urgent",
        }.get(family, "score")
        program["facts"][field] = None
    variants = [program, change_one_clause(program, actions, rng)]
    kind = "ordinal" if family == "ordinal" else "choice"
    offered = list(actions[: rng.randint(2, len(actions))])
    if mode == 3 and kind != "ordinal":
        outcomes = possible_outcomes(program)
        if len(outcomes) == 1:
            answer = next(iter(outcomes))
            offered = [action for action in actions if action != answer]
    if kind == "ordinal":
        offered = list(actions)
    while len(offered) < 2:
        offered.append(actions[-1])
    group = f"world-{seed}-{index:08d}"
    queries = [
        {"kind": kind},
        {"kind": "boolean", "action": actions[0]},
        {"kind": "boolean", "action": actions[1]},
    ]
    examples = []
    for variant, selected in zip(["base", "counterfactual"], variants, strict=True):
        context = render_program(selected)
        for qindex, query in enumerate(queries):
            if query["kind"] == "boolean":
                texts = ["true", "false"]
                instruction = (
                    f"Is the action selected by the policy {query['action']}? "
                    "Answer true or false using the stated rules and facts."
                )
            else:
                texts = list(offered)
                instruction = (
                    "Choose the level selected by the policy. The ordered levels are low, "
                    "medium, high."
                    if query["kind"] == "ordinal"
                    else "Which action is selected by the stated policy and observed facts?"
                )
            choices = [Choice(f"c{n}", text) for n, text in enumerate(texts)]
            rng.shuffle(choices)
            target = oracle_target(selected, query, choices)
            examples.append(
                Example(
                    id=f"{group}-{variant}-q{qindex}",
                    group_id=group,
                    family=family,
                    split=split,
                    context=context,
                    instruction=instruction,
                    choices=choices,
                    target=target,
                    kind=query["kind"],
                    pair_id=f"{group}-q{qindex}",
                    variant=variant,
                    metadata={
                        "generator": GENERATOR_VERSION,
                        "program": selected,
                        "query": query,
                        "mode": mode,
                        "source": "procedural finite-domain policies",
                    },
                )
            )
    return examples


def audit_partitions(partitions: dict[str, list[Example]]) -> dict:
    groups: dict[str, set[str]] = defaultdict(set)
    prompts: dict[str, set[str]] = defaultdict(set)
    contexts: dict[str, set[str]] = defaultdict(set)
    identities: set[str] = set()
    summaries = {}
    for split, examples in partitions.items():
        targets: Counter = Counter()
        for example in examples:
            example.validate()
            if example.id in identities:
                raise ValueError(f"Duplicate record ID: {example.id}")
            identities.add(example.id)
            if example.split != split:
                raise ValueError("Split field does not match its partition.")
            groups[example.group_id].add(split)
            prompts[example.input_fingerprint()].add(split)
            contexts[example.context_id].add(split)
            targets[example.target if example.target in {NONE, INSUFFICIENT} else "candidate"] += 1
            metadata = example.metadata
            if (
                oracle_target(metadata["program"], metadata["query"], example.choices)
                != example.target
            ):
                raise ValueError(f"Oracle disagreement: {example.id}")
        summaries[split] = {
            "examples": len(examples),
            "worlds": len({e.group_id for e in examples}),
            "contexts": len({e.context_id for e in examples}),
            "families": dict(Counter(e.family for e in examples)),
            "kinds": dict(Counter(e.kind for e in examples)),
            "targets": dict(targets),
        }
    for name, mapping in [("world", groups), ("prompt", prompts), ("context", contexts)]:
        overlaps = [key for key, splits in mapping.items() if len(splits) > 1]
        if overlaps:
            raise ValueError(f"Cross-split {name} leakage: {overlaps[:3]}")
    training = {e.family for e in partitions.get("train", [])}
    dev_ood = {e.family for e in partitions.get("dev_ood", [])}
    test_ood = {e.family for e in partitions.get("test_ood", [])}
    if training & (dev_ood | test_ood) or dev_ood & test_ood:
        raise ValueError("Generator holdouts overlap.")
    return {
        "generator_version": GENERATOR_VERSION,
        "oracle_version": ORACLE_VERSION,
        "cross_split_world_overlap": 0,
        "cross_split_prompt_overlap": 0,
        "cross_split_context_overlap": 0,
        "oracle_checked": len(identities),
        "partitions": summaries,
        "limitation": "Synthetic rule execution only; not evidence of general language competence.",
    }


def generate_dataset(
    output: Path, train_worlds: int = 1200, eval_worlds: int = 160, seed: int = 41
) -> dict:
    if train_worlds < 8 or eval_worlds < 8:
        raise ValueError("Use at least eight worlds per partition.")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Dataset output must be empty; do not overwrite a frozen partition.")
    specs = {
        "train": (train_worlds, TRAIN_FAMILIES),
        "dev_iid": (eval_worlds, TRAIN_FAMILIES),
        "dev_ood": (eval_worlds, DEV_OOD_FAMILIES),
        "cal_temperature": (eval_worlds, TRAIN_FAMILIES + DEV_OOD_FAMILIES),
        "cal_policy": (eval_worlds, TRAIN_FAMILIES + DEV_OOD_FAMILIES),
        "test_iid": (eval_worlds, TRAIN_FAMILIES),
        "test_ood": (eval_worlds, TEST_OOD_FAMILIES),
    }
    partitions = {}
    index = 0
    for split, (count, families) in specs.items():
        examples = []
        for offset in range(count):
            # Decouple family from index % 8 status modes.
            family = families[(offset // 8 + offset % len(families)) % len(families)]
            examples.extend(world_examples(index, split, family, seed))
            index += 1
        partitions[split] = examples
    audit = audit_partitions(partitions)
    for split, examples in partitions.items():
        write_examples(output / f"{split}.jsonl", examples)
    manifest = {
        **audit,
        "seed": seed,
        "source_rights": "Project-authored synthetic templates and finite-domain rule programs.",
        "files": {
            split: {"path": f"{split}.jsonl", "sha256": file_hash(output / f"{split}.jsonl")}
            for split in partitions
        },
        "final_test_policy": (
            "Use dev_* for experiments. Unlock test_* only after a frozen decision."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
