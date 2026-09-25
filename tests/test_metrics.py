import copy
import math

import pytest

from bobcat.metrics import (
    basic_metrics,
    evaluate_rows,
    fit_referral_policy,
    fit_temperature,
    scored_row,
)


def row(index=0, split="cal_policy"):
    return {
        "id": str(index),
        "group_id": f"world-{index}",
        "family": "test",
        "kind": "choice",
        "split": split,
        "target": "a",
        "candidate_ids": ["a", "b", "__none__", "__insufficient__"],
        "logits": [6.0, 0.0, 0.0, 0.0],
        "variant": "base",
        "pair_id": "",
    }


def test_calibration_refuses_evaluation_labels():
    with pytest.raises(ValueError, match="cal_temperature"):
        fit_temperature([row(split="test_ood")])
    with pytest.raises(ValueError, match="cal_policy"):
        fit_referral_policy([row(split="dev_iid")], 1)


def test_calibration_can_reduce_overconfidence():
    rows = [row(i, "cal_temperature") for i in range(100)]
    for item in rows[:30]:
        item["target"] = "b"
    temperature = fit_temperature(rows)
    assert temperature > 1
    assert evaluate_rows(rows, temperature)["nll"] < evaluate_rows(rows)["nll"]


def test_policy_needs_independent_support_not_repeated_questions():
    duplicates = [row(i) for i in range(200)]
    for item in duplicates:
        item["group_id"] = "one-world"
    policy = fit_referral_policy(duplicates, 1)
    assert policy["independent_world_samples"] == 1
    assert policy["threshold"] is None
    independent = fit_referral_policy([row(i) for i in range(200)], 1)
    assert independent["threshold"] is not None


def test_semantic_outcomes_never_become_automatic_actions():
    rows = [row(i) for i in range(200)]
    for item in rows:
        item["logits"] = [0, 0, 6, 0]
        item["target"] = "__none__"
    policy = fit_referral_policy(rows, 1)
    assert policy["threshold"] is None
    assert evaluate_rows(rows, policy=policy)["frozen_referral_policy"]["coverage"] == 0


def test_ece_bins_partition_boundary_values():
    rows = []
    for value in [i / 10 for i in range(11)]:
        scored = scored_row(row())
        scored["top_probability"] = value
        rows.append(scored)
    result = basic_metrics(rows)
    assert [b["count"] for b in result["calibration_bins"]] == [1] * 9 + [2]
    assert result["calibration_signal"] == "top_probability"


def test_ece_uses_prediction_probability_not_adapter_concentration():
    example = row()
    example["candidate_ids"] = ["a", "b", "c"]
    example["logits"] = [math.log(0.8), math.log(0.1), math.log(0.1)]
    scored = scored_row(example)
    # The API's concentration for this distribution is 0.7. It must not become
    # a correctness probability even if a caller supplies it under this name.
    scored["confidence"] = 0.7
    metrics = basic_metrics([scored])
    assert metrics["ece_10_equal_width_bins"] == pytest.approx(0.2)
    assert metrics["calibration_bins"][0]["mean_top_probability"] == pytest.approx(0.8)


def test_nll_preserves_extreme_errors_when_output_probabilities_underflow():
    example = row()
    example["target"] = "b"
    example["logits"] = [1000.0, 0.0, 0.0, 0.0]
    scored = scored_row(example)
    assert scored["probabilities"][1] == 0.0
    assert scored["nll"] == pytest.approx(1000.0)
    metrics = evaluate_rows([example])
    assert metrics["nll_method"] == "log_softmax_from_raw_logits"
    assert metrics["gold_probability_underflow_count"] == 1
    assert scored_row(example, temperature=2.0)["nll"] == pytest.approx(500.0)


def test_exact_ties_are_stable_under_permutation():
    original = row()
    original["logits"] = [0, 0, 0, 0]
    permuted = copy.deepcopy(original)
    permuted["candidate_ids"].reverse()
    assert scored_row(original)["prediction"] == scored_row(permuted)["prediction"]
