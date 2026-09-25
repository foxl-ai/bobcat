import copy

import numpy as np
import pytest

from bobcat.policy_transfer import execute, expand
from bobcat.policy_transfer_eval import (
    decomposition_prediction,
    evaluate,
    probabilities,
    reconstruct_program,
)
from bobcat.schema import json_hash


def source():
    request = {
        "model": "bobcat-latest",
        "state": "대출 금리가 내려갔다.",
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "분야를 분류하라.",
                "criteria": {"경제": "경제", "정치": "정치", "사회": "사회", "세계": "세계"},
            }
        },
    }
    return {
        "id": "text1",
        "observation_id": "text1",
        "group_id": "cluster1",
        "task": "klue_ynat",
        "kind": "choice",
        "language": "ko",
        "language_origin": "native",
        "split": "dev_train",
        "source_split": "train",
        "supervision": "hard_label",
        "candidate_ids": ["경제", "정치", "사회", "세계"],
        "target": "경제",
        "score_target": None,
        "request": request,
        "input_sha256": json_hash(request),
        "source": {"license": "fixture"},
    }


def source_prediction(p=(0.7, 0.1, 0.1, 0.1)):
    row = source()
    return {
        "source_request_sha256": row["input_sha256"],
        "candidate_ids": row["candidate_ids"],
        "probabilities": list(p),
    }


def test_decomposition_pushes_all_class_mass_through_exceptions_and_preserves_types():
    rows = expand(source(), seed=5)
    for row in rows:
        program = reconstruct_program(row, seed=5)
        result = decomposition_prediction(row, source_prediction(), seed=5)
        p = probabilities(result, row["candidate_ids"], row["input_sha256"])
        assert p.sum() == pytest.approx(1)
        expected = dict.fromkeys(program["routes"], 0.0)
        for category, mass in zip(source()["candidate_ids"], (0.7, 0.1, 0.1, 0.1), strict=True):
            expected[execute(program, category, row["request"]["state"]["workflow_metadata"])] += (
                mass
            )
        if row["kind"] == "choice":
            assert p.tolist() == pytest.approx([expected[c] for c in row["candidate_ids"]])
        if row["kind"] == "ordinal":
            assert p.tolist() == pytest.approx([expected[c] for c in program["routes"]])


def test_decomposition_prediction_cannot_consult_the_gold_label():
    row = expand(source(), seed=8)[0]
    altered = copy.deepcopy(row)
    altered["target"] = "DO-NOT-READ-GOLD"
    altered["source"]["original_category"] = "DO-NOT-READ-GOLD"
    assert decomposition_prediction(row, source_prediction(), seed=8) == (
        decomposition_prediction(altered, source_prediction(), seed=8)
    )


def test_known_interpreter_oracle_is_only_a_diagnostic_and_failed_answers_count():
    rows = expand(source(), seed=7)
    oracle = [
        decomposition_prediction(row, source_prediction((1, 0, 0, 0)), seed=7) for row in rows
    ]
    report = evaluate(rows, oracle)
    assert report["accuracy_with_failures"] == 1
    assert report["components"] == 1
    assert not report["fresh_final_evaluation"]
    assert report["interventions"]["permuted_options"]["permutation_max_tv"] == 0
    assert report["interventions"]["renamed_routes"]["both_correct_rate_with_failures"] == 1
    missing = evaluate(rows, oracle[:-1])
    assert missing["correct"] == len(rows) - 1
    assert missing["accuracy_with_failures"] == (len(rows) - 1) / len(rows)
    assert missing["invalid_or_missing"] == {"missing": 1}
    assert missing["distribution_metric_denominator"] == len(rows) - 1


def test_drifted_program_or_prediction_is_rejected():
    row = expand(source(), seed=7)[0]
    with pytest.raises(ValueError, match="construction"):
        reconstruct_program(row, seed=111)
    prediction = source_prediction()
    prediction["source_request_sha256"] = "another-source"
    with pytest.raises(ValueError, match="exact original request"):
        decomposition_prediction(row, prediction, seed=7)


def test_all_candidates_required_and_order_is_restored():
    value = {"source_request_sha256": "r", "candidate_ids": ["b", "a"], "logits": [0.0, 2.0]}
    p = probabilities(value, ["a", "b"], "r")
    assert p[0] > p[1]
    with pytest.raises(ValueError):
        probabilities(value, ["a", "b", "c"], "r")
    value["logits"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        probabilities(value, ["a", "b"], "r")


def test_every_probability_boundary_is_counted_in_exactly_one_ece_bin():
    rows = expand(source(), seed=7)
    predictions = []
    selected_probabilities = []
    for row in rows:
        k = len(row["candidate_ids"])
        target = row["candidate_ids"].index(row["target"])
        confidence = 0.6 if k > 2 else 0.9
        p = [(1 - confidence) / (k - 1)] * k
        p[target] = confidence
        selected_probabilities.append(confidence)
        predictions.append(
            {
                "id": row["id"],
                "source_request_sha256": row["input_sha256"],
                "candidate_ids": row["candidate_ids"],
                "probabilities": p,
            }
        )
    report = evaluate(rows, predictions)
    assert report["ece_10_equal_width"] == pytest.approx(1 - np.mean(selected_probabilities))
