import copy
import json

import pytest

from bobcat.public_decisions import checked_file, partition, request_record
from bobcat.schema import file_hash


def sts(index, left, right, *, split="train", value=26 / 7):
    return request_record("klue_sts", split, index, {
        "guid": f"PRIVATE-SOURCE-{index}", "sentence1": left, "sentence2": right,
        "labels": {"real-label": value, "label": round(value, 1), "binary-label": 1},
    }, {"annotation_id": "PRIVATE-ANNOTATOR"}, [])


def prior(index, text, split):
    row = sts(index, text, f"existing counterpart {index}")
    row.update(id=f"prior-{index}", group_id=f"prior-group-{index}", split=split)
    return row


def test_source_mean_ids_and_annotations_do_not_change_the_model_request():
    first = sts(0, "민수는 서울에 갑니다.", "민수가 서울로 이동합니다.")
    second = sts(7, "민수는 서울에 갑니다.", "민수가 서울로 이동합니다.", value=1.0)
    assert first["request"] == second["request"]
    assert first["target"] is None and first["score_target"] == 26 / 7
    assert first["supervision"] == "score_mean"
    assert first["candidate_ids"] == list("012345")
    assert "PRIVATE" not in json.dumps(first["request"])


def test_banking_keeps_77_candidates_and_uses_original_category_annotation():
    categories = [f"intent_{i}" for i in range(77)]
    raw = {"text": "The card charge appears twice.", "category": "intent_76"}
    first = request_record("banking77", "train", 0, raw, {}, categories)
    second = request_record("banking77", "train", 9, {**raw, "category": "intent_0"},
                            {}, categories)
    assert first["request"] == second["request"]
    assert first["candidate_ids"] == categories and first["target"] == "intent_76"
    with pytest.raises(ValueError, match="77"):
        request_record("banking77", "train", 0, raw, {}, categories[:-1])


def test_boolq_and_arc_use_explicit_input_allowlists():
    boolean = {"question": "Is it open?", "passage": "It opened today.", "answer": True}
    first = request_record("boolq", "train", 0, boolean, {}, [])
    other = request_record("boolq", "test", 9, {**boolean, "answer": False}, {}, [])
    assert first["request"] == other["request"]
    assert first["candidate_ids"] == ["no", "yes"] and first["target"] == "yes"
    arc = {"id": "PRIVATE-ANSWER-ID", "question": "Which is hot?",
           "choices": {"label": ["1", "2", "3"], "text": ["Fire", "Ice", "Snow"]},
           "answerKey": "1", "explanation": "GOLD-EXPLANATION"}
    question = request_record("arc_easy", "train", 0, arc, {}, [])
    changed = request_record("arc_easy", "test", 1, {**arc, "answerKey": "2"}, {}, [])
    assert question["request"] == changed["request"]
    assert question["candidate_ids"] == ["1", "2", "3"]
    assert "PRIVATE" not in json.dumps(question["request"])
    assert "GOLD" not in json.dumps(question["request"])


def test_new_data_cannot_bridge_frozen_partitions_or_import_public_labels_into_train():
    old = [prior(0, "existing training input", "train"),
           prior(1, "existing temperature input", "cal_temperature")]
    new = [
        sts(0, "existing training input", "existing temperature input"),
        sts(1, "existing training input", "public sentence", split="validation"),
        sts(2, "independent left", "independent right"),
    ]
    kept, audit = partition(new, old)
    assert audit["removal_counts"] == {"bridges_frozen_partitions": 2}
    assert {r["id"] for r in kept} == {"prior-0", "prior-1", "klue_sts:train:2"}
    assert {r["id"]: r["split"] for r in kept if r["id"].startswith("prior")} == {
        "prior-0": "train", "prior-1": "cal_temperature",
    }
    assert audit["text_and_component_split_overlap"] == 0
    reverse, _ = partition(list(reversed(copy.deepcopy(new))), copy.deepcopy(old))
    assert sorted((r["id"], r["group_id"], r["split"]) for r in kept) == sorted(
        (r["id"], r["group_id"], r["split"]) for r in reverse
    )


def test_transitive_public_overlap_conflicts_and_duplicate_inputs_are_quarantined():
    new = [
        sts(0, "A", "B"), sts(1, "B", "C"), sts(2, "C", "D", split="validation"),
        sts(3, "E", "F", value=1), sts(4, "E", "F", value=4),
        sts(5, "G", "H"), sts(6, "G", "H"),
    ]
    kept, audit = partition(new, [])
    assert audit["removal_counts"] == {
        "shares_public_input_component": 2, "conflicting_annotations": 2, "duplicate_input": 1,
    }
    assert len(kept) == 2
    assert next(r for r in kept if r["source_split"] == "validation")["split"] == "dev_public"
    assert all(r["score_target"] == 26 / 7 for r in kept)


def test_changed_source_file_cannot_pass_recorded_size_and_hash(tmp_path):
    path = tmp_path / "fixture.csv"
    path.write_text("text,category\ncustomer,account\n")
    item = {"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
    assert checked_file(tmp_path, item) == path
    path.write_text("text,category\ncustomer,payment\n")
    with pytest.raises(ValueError, match="identity"):
        checked_file(tmp_path, item)
