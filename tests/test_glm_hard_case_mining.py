import pytest

from bobcat.glm_hard_case_mining import error_kind, remaining_rows


def case(**changes):
    row = {
        "id": "a", "group_id": "a", "split": "train", "input_sha256": "hash",
        "candidate_ids": ["no", "yes"], "supervision": "hard_label", "target_index": 1,
        "score_mean": None, "kind": "choice", "task": "test", "language": "ko",
        "language_origin": "native", "input_tokens": 10, **changes,
    }
    pred = {**row, "logits": [3., -1.], "native_completion_tokens": 0}
    return row, pred


def test_only_preserved_gold_defines_errors():
    row, pred = case()
    assert error_kind(row, pred) == "categorical_error"
    pred["logits"] = [-1., 3.]
    assert error_kind(row, pred) is None


def test_score_mean_uses_expected_value_not_rounded_category():
    row, pred = case(supervision="score_mean", kind="ordinal", target_index=-1, score_mean=.5)
    pred["logits"] = [0., 0.]
    assert error_kind(row, pred) is None
    pred["logits"] = [20., -20.]
    assert error_kind(row, pred) == "ordinal_mean_error_over_0.2"


def test_development_or_misaligned_logits_are_not_mined():
    row, pred = case(split="dev_train")
    with pytest.raises(ValueError, match="training"):
        error_kind(row, pred)


def test_continuation_keeps_only_unprocessed_components():
    rows = [{"id": str(i)} for i in range(20)]
    assert remaining_rows(rows, 8) == rows[8:]
    assert remaining_rows(rows, len(rows)) == []
    for invalid in (-1, 21, True, 8.5):
        with pytest.raises(ValueError):
            remaining_rows(rows, invalid)
    row, pred = case()
    pred["input_sha256"] = "different"
    with pytest.raises(ValueError, match="training"):
        error_kind(row, pred)
