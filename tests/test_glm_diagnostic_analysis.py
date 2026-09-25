import copy
import math

import pytest
from test_glm_identifier_controls import small_source

from bobcat.glm_diagnostic_analysis import analyze_identifiers
from bobcat.glm_identifier_controls import build


def evidence(suite, *, prefer_identifier=False):
    rows = []
    for case in suite["cases"]:
        chosen = (
            case["labels_in_order"][case["identifier_slots"].index(0)]
            if prefer_identifier else case["gold"]
        )
        p = {label: 0.9 if label == chosen else 0.05 for label in case["labels_in_order"]}
        rows.append({
            key: copy.deepcopy(case[key]) for key in (
                "id", "semantic_group", "block", "arm", "language", "gold",
                "labels_in_order", "identifier_slots",
            )
        })
        rows[-1].update(
            status="scored", raw_scores=[math.log(p[k]) for k in case["labels_in_order"]],
            probabilities=p, prediction=chosen, correct=chosen == case["gold"],
        )
    return rows


def test_semantic_alignment_does_not_treat_permuted_columns_as_changed_probabilities():
    suite = build(small_source(), blocks=2)
    result = analyze_identifiers(suite, evidence(suite))
    assert result["planned_calls"] == 10 and result["semantic_groups"] == 1
    assert result["paired_blocks"] == 2
    for row in result["blocks"]:
        assert all(c["tv"] == pytest.approx(0) and not c["argmax_changed"]
                   for c in row["contrasts"].values())
        assert row["factorial"]["correct"]["interaction"] == 0
    assert result["confidence_intervals"] is None
    assert result["release_gate_passed"] is False


def test_slot_bias_is_separated_from_list_order_and_identical_repeat():
    suite = build(small_source(), blocks=1)
    result = analyze_identifiers(suite, evidence(suite, prefer_identifier=True))
    row = result["blocks"][0]["contrasts"]
    assert row["repeat_noise"]["tv"] == 0
    assert row["order_at_base_binding"]["tv"] == 0
    # The first identifier changes its semantic meaning in the crossed manipulation.
    assert row["binding_at_base_order"]["tv"] == pytest.approx(0.85)
    assert row["binding_at_base_order"]["argmax_changed"] is True
    assert row["binding_at_base_order"]["thresholds"]["0.8"]["action_changed"] is True
    assert row["binding_at_base_order"]["thresholds"]["0.8"]["crossing"] is False


def test_failed_attempts_are_not_dropped_and_unattempted_rows_are_not_claimed_scored():
    suite = build(small_source(), blocks=2)
    rows = evidence(suite)
    victim = next(row for row in rows if row["arm"] == "base_repeat")
    victim["status"] = "failed"
    result = analyze_identifiers(suite, rows)
    count = result["arm_counts"]["base_repeat"]
    assert count["failed"] == 1 and count["accuracy_complete_plan"] == 0.5
    partial = analyze_identifiers(suite, [row for row in rows if row is not victim])
    count = partial["arm_counts"]["base_repeat"]
    assert count["unattempted"] == 1 and count["accuracy_complete_plan"] is None
    assert partial["scored_calls"] == 9
    assert partial["by_language_and_count"][0]["contrasts"]["repeat_noise"][
        "missing_or_failed_blocks"
    ] == 1


@pytest.mark.parametrize("corruption", ["metadata", "probabilities", "prediction", "duplicate"])
def test_corrupt_or_pooled_evidence_is_rejected(corruption):
    suite = build(small_source(), blocks=1)
    rows = evidence(suite)
    if corruption == "metadata":
        rows[0]["identifier_slots"] = rows[0]["identifier_slots"][::-1]
    elif corruption == "probabilities":
        rows[0]["probabilities"][rows[0]["gold"]] = 0.8
    elif corruption == "prediction":
        rows[0]["prediction"] = "invented"
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        analyze_identifiers(suite, rows)
