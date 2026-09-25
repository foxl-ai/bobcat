import json
from collections import Counter
from pathlib import Path

from bobcat import stress
from bobcat.protocol import parse_request

ROOT = Path(__file__).resolve().parents[1]


def dev_row(index, kind="choice", labels=("a", "b", "c", "d"), task="product_search"):
    if kind == "noul":
        question = {"type": "noul", "instructions": "맞나?",
                    "criteria": {"true": "맞다", "false": "아니다"}}
        labels = ("no", "yes")
    else:
        question = {"type": "choice", "instructions": "고르라.",
                    "criteria": {label: f"{label} 설명" for label in labels}}
    return {"id": f"row{index}", "task": task, "family": "f", "split": "dev",
            "request": {"model": "bobcat-latest",
                        "state": {"문서": f"본문 {index} " * 5, "질의": "무엇?"},
                        "questions": {"q": question}},
            "candidate_ids": list(labels), "target": labels[-1]}


def answer_for(case, label):
    (qid, spec), = case["payload"]["questions"].items()
    if spec["type"] == "noul":
        return {qid: {"type": "noul", "noul": 0.9 if label == "yes" else 0.1}}
    labels = list(spec["criteria"])
    probs = {x: (0.7 if x == label else 0.3 / (len(labels) - 1)) for x in labels}
    return {qid: {"type": "choice", "choice": label, "probabilities": probs,
                  "confidence": 0.5}}


def test_general_probes_are_valid_bilingual_pairs():
    cases = stress.general_cases(ROOT / "configs/general-probes-v1.json")
    concepts = Counter(c["meta"]["concept"] for c in cases)
    assert set(concepts.values()) == {2}
    for case in cases:
        _, questions = parse_request(case["payload"])
        assert {q.id for q in questions} == set(case["expect"])


def test_tool_ood_catalog_builds_with_the_eval_generator():
    cases = stress.tool_cases(ROOT / "configs/tool-situations-v2-ood.json", 1)
    families = Counter(c["meta"]["family"] for c in cases)
    assert set(families) == {"scope", "irreversible", "plan_mismatch"}
    v1 = {s["id"] for s in json.loads((ROOT / "configs/tool-situations-v1.json").read_text())
          ["situations"]}
    assert not v1 & {c["meta"]["situation"] for c in cases}


def test_variants_keep_candidates_and_targets():
    rows = [dev_row(i) for i in range(4)] + [dev_row(9, kind="noul", task="product_tool_call")]
    perm = stress.permutation_cases(rows, 3, 10)
    assert len(perm) == 16
    for case in perm:
        (spec,) = case["payload"]["questions"].values()
        assert sorted(spec["criteria"]) == ["a", "b", "c", "d"]
    orders = {c["meta"]["variant"]: list(next(iter(c["payload"]["questions"].values()))
                                         ["criteria"]) for c in perm if c["meta"]["base"] == "row0"}
    assert orders["reversed"] == orders["original"][::-1]
    distract = stress.distractor_cases(rows, 10)
    front = next(c for c in distract if c["meta"]["variant"] == "distractors_front")
    assert list(next(iter(front["payload"]["questions"].values()))["criteria"])[:2] == \
        list(stress.DISTRACTORS["described"])
    inject = stress.injection_cases(rows, 3, 10)
    noul = [c for c in inject if c["meta"]["base"] == "row9"]
    assert {c["meta"]["attack_target"] for c in noul} == {"no"}
    note = next(c for c in noul if c["meta"]["variant"] == "state_note")
    assert "'false'" in note["payload"]["state"]["시스템 메모"]


def test_summary_measures_flips_and_attack_success():
    rows = [dev_row(i) for i in range(2)]
    cases = (stress.permutation_cases(rows, 3, 2) + stress.distractor_cases(rows, 2)
             + stress.injection_cases(rows, 3, 2))
    responses = {}
    for case in cases:
        label = "d"
        if case["meta"]["variant"] == "reversed" and case["meta"]["base"] == "row0":
            label = "a"
        if case["suite"] == "injection" and case["meta"]["variant"] != "clean":
            label = case["meta"]["attack_target"]
        if case["meta"]["variant"] == "distractors_end":
            label = "바나나 시세"
        body = {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 0},
                "answers": answer_for(case, label)}
        responses[case["id"]] = {"status": 200, "body": body, "violation": None}
    summary = stress.summarize(cases, responses)
    assert summary["structural"]["violations"] == 0
    assert summary["permutation"]["same_answer_all_orders"] == 0.5
    assert summary["distractor"]["picked_distractor"] == 0.5
    assert summary["injection"]["attack_success_all"] == 1.0
    assert summary["injection"]["accuracy_by_variant"]["clean"] == 1.0
