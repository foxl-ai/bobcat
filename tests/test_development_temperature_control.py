import copy

import numpy as np
import pytest

from bobcat.development_temperature_control import crossfit_categorical


def examples():
    return [{
        "id": str(index), "group_id": f"component-{index}", "task": "synthetic",
        "language": "ko", "kind": "choice", "supervision": "hard_label",
        "target_index": int(index % 4 == 0), "logits": [8., 0.],
    } for index in range(40)]


def test_crossfit_held_labels_do_not_choose_their_own_temperature():
    rows = examples()
    first = crossfit_categorical(rows)
    held = set(first["folds"][0]["held_out_components"])
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["group_id"] in held:
            row["target_index"] = 1 - row["target_index"]
    second = crossfit_categorical(changed)
    assert first["folds"][0] == second["folds"][0]
    for fold in first["folds"]:
        assert not set(fold["training_components"]) & set(fold["held_out_components"])
    assert set().union(*(set(fold["held_out_components"]) for fold in first["folds"])) == {
        row["group_id"] for row in rows
    }


def test_scalar_control_preserves_choices_and_is_not_release_calibration():
    rows = examples()
    result = crossfit_categorical(rows)
    assert all(fold["temperature"] > 1 for fold in result["folds"])
    assert all(
        np.argmax(before["logits"]) == np.argmax(after["logits"])
        for before, after in zip(rows, result["predictions"], strict=True)
    )
    assert result["metrics"]["categorical"]["nll"] < 1
    assert not result["native_deployment_calibration"]
    assert not result["release_gate_passed"]
    assert rows == examples()


def test_crossfit_rejects_duplicate_components_and_score_means():
    rows = examples()
    rows[1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="distinct"):
        crossfit_categorical(rows)
    rows = examples()
    rows[0]["supervision"] = "score_mean"
    with pytest.raises(ValueError, match="categorical"):
        crossfit_categorical(rows)
