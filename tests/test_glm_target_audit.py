import copy

import pytest

from bobcat.glm_target_audit import describe, validate_target
from bobcat.glm_training_targets import target_record


def fixture(*, ordinal=False):
    source = {
        "id": "component:question", "group_id": "component", "split": "train",
        "source_request_sha256": "source", "input_sha256": "input", "input_tokens": 128,
        "language": "ko", "language_origin": "native", "task": "policy", "kind": "choice",
        "candidate_ids": ["거절", "허용"], "option_token_ids": [32, 33],
        "supervision": "hard_label", "target_index": 1, "score_mean": None,
    }
    if ordinal:
        source.update(kind="ordinal", supervision="score_mean", target_index=-100,
                      score_mean=.4)
    teacher = {"source_repo": "zai-org/GLM-5.3-Flash", "model": "scoped_256"}
    target = target_record(source, {
        "id": source["id"], "input_sha256": source["input_sha256"], "logits": [4., 0.],
        "native_completion_tokens": 0,
    }, teacher=teacher)
    return source, target, teacher


def test_confident_teacher_error_stays_an_error_without_relabeling():
    source, target, teacher = fixture()
    original = copy.deepcopy(target)
    result = describe([validate_target(target, source, teacher)])
    assert result["subsets"]["all"]["categorical"]["accuracy"] == 0
    assert result["subsets"]["all"]["raw_pmax_0_9_diagnostic"]["observed_error"] == 1
    assert len(result["disagreements_with_observed_labels"]) == 1
    assert not result["student_training_authorized_by_this_audit"]
    assert not result["observed_labels_replaced"]
    assert not result["held_out_generalization_claim"]
    assert target == original


@pytest.mark.parametrize("changed", [
    {"candidate_ids": ["허용", "거절"]},
    {"input_sha256": "other"}, {"teacher_distribution_is_ground_truth": True},
    {"teacher_matches_observed_label": True}, {"teacher_argmax_index": 1},
    {"teacher_conditional_probabilities": [.5, .5]},
    {"observed_supervision": {"type": "hard_label", "target_index": 0, "score_mean": None}},
])
def test_changed_candidate_meaning_labels_or_probabilities_cannot_pass(changed):
    source, target, teacher = fixture()
    with pytest.raises(ValueError):
        validate_target({**target, **changed}, source, teacher)


def test_mean_score_never_invents_a_category_or_calibration_label():
    source, target, teacher = fixture(ordinal=True)
    result = describe([validate_target(target, source, teacher)])
    assert result["subsets"]["all"]["ordinal_mean"]["count"] == 1
    assert result["subsets"]["all"]["ordinal_mean"]["normalized_mae"] > .3
    assert result["subsets"]["all"]["categorical"]["count"] == 0
    assert result["disagreements_with_observed_labels"] == []
    assert target["observed_supervision"]["score_mean"] == .4


def test_repeated_component_does_not_inflate_teacher_review():
    source, target, teacher = fixture()
    value = validate_target(target, source, teacher)
    with pytest.raises(ValueError, match="distinct training components"):
        describe([value, value])
