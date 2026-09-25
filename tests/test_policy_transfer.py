import copy
import json

import pytest

from bobcat.policy_transfer import FAMILIES, change_program, execute, expand, make_program
from bobcat.schema import json_hash


def source_row(split="train", target="경제"):
    request = {
        "model": "bobcat-latest",
        "state": {"뉴스 제목": "무역 수지가 개선되었다."},
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "뉴스 제목의 주제를 고르세요.",
                "criteria": {"경제": "경제 기사", "정치": "정치 기사", "사회": "사회 기사"},
            }
        },
    }
    return {
        "id": "private-id-ONLY-METADATA",
        "observation_id": "original-observation",
        "group_id": "shared-original-component",
        "task": "klue_ynat",
        "language": "ko",
        "language_origin": "native",
        "kind": "choice",
        "source_split": "train",
        "split": split,
        "request": request,
        "candidate_ids": ["경제", "정치", "사회"],
        "target": target,
        "score_target": None,
        "supervision": "hard_label",
        "input_sha256": json_hash(request),
        "source": {
            "license": "CC-BY-SA-4.0",
            "source_label": target,
            "annotation_origin": "upstream_human",
            "secret": "GOLD-CANARY-187",
        },
    }


def test_rule_precedence_conjunction_and_nonmatching_exception():
    program = {
        "routes": ["A", "B", "C"],
        "mapping": {"finance": "A", "sports": "B"},
        "exceptions": [
            {"categories": ["finance"], "attributes": {"channel": "web"}, "destination": "B"},
            {
                "categories": ["finance"],
                "attributes": {"channel": "web", "tier": "priority"},
                "destination": "C",
            },
        ],
    }
    assert execute(program, "finance", {"channel": "web", "tier": "priority"}) == "C"
    assert execute(program, "finance", {"channel": "web", "tier": "standard"}) == "B"
    assert execute(program, "finance", {"channel": "app", "tier": "priority"}) == "A"
    reversed_program = change_program(program, "changed_clause")
    assert execute(reversed_program, "finance", {"channel": "web", "tier": "priority"}) == "B"
    assert execute(program, "finance", {"channel": "web", "tier": "priority"}) == "C"


def test_policy_inputs_do_not_depend_on_gold_or_include_metadata():
    first = expand(source_row(target="경제"), seed=9)
    second = expand(source_row(target="정치"), seed=9)
    assert [r["request"] for r in first] == [r["request"] for r in second]
    assert any(a["target"] != b["target"] for a, b in zip(first, second, strict=True))
    for row in first:
        serialized = json.dumps(row["request"], ensure_ascii=False)
        assert "GOLD-CANARY" not in serialized
        assert "private-id-ONLY-METADATA" not in serialized
        assert "source_label" not in serialized
        assert row["request"]["state"]["source_text"] == source_row()["request"]["state"]
        assert row["group_id"] == "shared-original-component"
        assert row["source"]["original_text_unmodified"]
        assert not row["fresh_final_evaluation"]


def test_holdout_generators_are_disjoint_and_all_primitives_have_valid_gold():
    for split in FAMILIES:
        for seed in range(10):
            rows = expand(source_row(split=split), seed=seed)
            assert {r["policy_family"] for r in rows} <= set(FAMILIES[split])
            assert {r["kind"] for r in rows} == {"choice", "boolean", "ordinal"}
            assert all(r["target"] in r["candidate_ids"] for r in rows)
            by_key = {(r["policy_view"], r["kind"]): r for r in rows}
            for view in ("original", "renamed_routes", "changed_clause"):
                choice = by_key[view, "choice"]
                score = by_key[view, "ordinal"]
                instructions = score["request"]["questions"]["decision"]["instructions"]
                assert str(instructions["destination_levels"][choice["target"]]) == score["target"]
            assert (
                by_key["original", "choice"]["target"]
                != by_key["renamed_routes", "choice"]["target"]
            )
            if split == "dev_train":
                original, permuted = (
                    by_key["original", "choice"],
                    by_key["permuted_options", "choice"],
                )
                assert original["target"] == permuted["target"]
                assert original["candidate_ids"] == permuted["candidate_ids"][::-1]
                assert (
                    original["request"]["questions"]["decision"]["instructions"]
                    == (permuted["request"]["questions"]["decision"]["instructions"])
                )
    assert not set(FAMILIES["train"]) & set(FAMILIES["dev_train"])


def test_initial_policy_always_requires_source_semantics():
    for family in {f for families in FAMILIES.values() for f in families}:
        for seed in range(20):
            program, metadata = make_program(["negative", "positive"], str(seed), family)
            assert len({execute(program, c, metadata) for c in program["mapping"]}) == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(source_split="test"),
        lambda r: r.update(split="cal_temperature"),
        lambda r: r.update(candidate_ids=r["candidate_ids"][::-1]),
        lambda r: r["request"].update(state="changed after checksum"),
        lambda r: r.update(target="outside"),
    ],
)
def test_partition_label_and_integrity_errors_are_rejected(mutation):
    row = copy.deepcopy(source_row())
    mutation(row)
    with pytest.raises(ValueError):
        expand(row, seed=1)
