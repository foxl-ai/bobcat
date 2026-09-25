import copy

import pytest

from bobcat.kmmlu_eval import original_question, select_balanced
from bobcat.protocol import parse_request


def original():
    return {"question": "한국어 원문 질문", "answer": "3",
            "A": "첫째", "B": "둘째", "C": "셋째", "D": "넷째",
            "Category": "SourceSubject", "Human Accuracy": "0.987654321"}


def test_gold_annotation_and_source_metadata_cannot_enter_the_request():
    row = original()
    request, target = original_question(row)
    state, questions = parse_request(request)
    assert state == row["question"] and len(questions) == 1
    altered = {**row, "answer": "1", "Human Accuracy": "0.0", "Category": "Different"}
    repeated, alternate_target = original_question(altered)
    assert request == repeated
    assert target == 2 and alternate_target == 0
    assert request["state"] == row["question"]
    assert request["questions"]["decision"]["criteria"] == {k: row[k] for k in "ABCD"}
    with pytest.raises(ValueError):
        original_question({**row, "answer": "0"})


def test_selection_is_distinct_and_independent_of_the_gold_labels():
    rows = [
        {"subject": f"subject-{subject}", "source_file": f"{subject}.csv",
         "source_row": index, "request": {"state": f"question-{subject}-{index}"},
         "target_index": index % 4, "group_id": f"group-{subject}-{index}"}
        for subject in range(4) for index in range(32)
    ]
    selected, _ = select_balanced(rows, 64, 20260923)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["target_index"] = (row["target_index"] + 1) % 4
    alternate, _ = select_balanced(changed, 64, 20260923)
    assert [r["group_id"] for r in selected] == [r["group_id"] for r in alternate]
    assert len({r["group_id"] for r in selected}) == 64
    assert all(sum(r["subject"] == f"subject-{i}" for r in selected) == 16 for i in range(4))
