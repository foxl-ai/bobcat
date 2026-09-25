import copy
import json

import pytest

from bobcat.data import atom, audit_partitions, oracle_target, possible_outcomes
from bobcat.schema import INSUFFICIENT, NONE, Choice, read_examples
from bobcat.tokenization import ScratchTokenizer


def test_missing_fact_can_be_irrelevant():
    program = {
        "facts": {"verified": False, "sealed": None},
        "domains": {"verified": [False, True], "sealed": [False, True]},
        "rules": [
            {
                "when": {
                    "op": "and",
                    "args": [
                        atom("verified", True),
                        atom("sealed", True),
                    ],
                },
                "action": "approve",
            }
        ],
        "default": "deny",
    }
    assert possible_outcomes(program) == {"deny"}
    choices = [Choice("a", "approve"), Choice("b", "deny")]
    assert oracle_target(program, {"kind": "choice"}, choices) == "b"
    program["facts"]["verified"] = True
    assert oracle_target(program, {"kind": "choice"}, choices) == INSUFFICIENT
    program["facts"]["sealed"] = False
    assert (
        oracle_target(
            program,
            {"kind": "choice"},
            [
                Choice("a", "approve"),
                Choice("b", "review"),
            ],
        )
        == NONE
    )


def test_missing_number_preserves_correlations():
    # x cannot be both <=2 and >=4. Independent unknown-condition sampling is wrong.
    program = {
        "facts": {"score": None},
        "domains": {"score": list(range(7))},
        "rules": [
            {
                "when": {
                    "op": "and",
                    "args": [
                        atom("score", 2, "le"),
                        atom("score", 4, "ge"),
                    ],
                },
                "action": "impossible",
            }
        ],
        "default": "certain",
    }
    assert possible_outcomes(program) == {"certain"}


def test_none_is_known_when_every_possible_answer_is_outside_the_schema():
    program = {
        "facts": {"signal": None},
        "domains": {"signal": ["north", "south"]},
        "rules": [{"when": atom("signal", "north"), "action": "A"}],
        "default": "B",
    }
    # The precise action is unknown, but C and D are both provably inapplicable.
    assert possible_outcomes(program) == {"A", "B"}
    assert (
        oracle_target(
            program,
            {"kind": "choice"},
            [
                Choice("c", "C"),
                Choice("d", "D"),
            ],
        )
        == NONE
    )


def test_split_manifest_and_oracle(corpus):
    root, _, _ = corpus
    partitions = {path.stem: read_examples(path) for path in (root / "data").glob("*.jsonl")}
    report = audit_partitions(partitions)
    manifest = json.loads((root / "data" / "manifest.json").read_text())
    assert report["oracle_checked"] == manifest["oracle_checked"]
    assert report["cross_split_context_overlap"] == 0
    assert not (
        set(report["partitions"]["train"]["families"])
        & set(report["partitions"]["test_ood"]["families"])
    )


def test_group_and_prompt_leakage_is_rejected(corpus):
    example = copy.deepcopy(corpus[1][0])
    duplicate = copy.deepcopy(example)
    duplicate.id += "-different-id"
    duplicate.split = "dev_iid"
    with pytest.raises(ValueError, match="leakage"):
        audit_partitions({"train": [example], "dev_iid": [duplicate]})


def test_wrong_gold_is_rejected(corpus):
    example = copy.deepcopy(corpus[1][0])
    example.target = NONE if example.target != NONE else INSUFFICIENT
    with pytest.raises(ValueError, match="Oracle disagreement"):
        audit_partitions({"train": [example]})


def test_tokenizer_never_fits_development_data(corpus, tmp_path):
    dev = read_examples(corpus[0] / "data" / "dev_iid.jsonl")
    with pytest.raises(ValueError, match="training partition"):
        ScratchTokenizer.train(dev, tmp_path / "forbidden.json")
    # Byte alphabet must still represent unseen Korean text without unknown tokens.
    assert 1 not in corpus[2].encode("환불 조건이 충족되지 않았습니다.")
