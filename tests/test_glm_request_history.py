import pytest

from bobcat.glm_request_history import history_schedule, summarize


def test_intervention_covers_each_component_without_self_intervention():
    rows = [{"id": str(i), "group_id": str(i), "split": "dev_train"} for i in range(64)]
    plan = history_schedule(rows)
    assert [a["id"] for a, b in plan] == [r["id"] for r in rows]
    assert sorted(b["id"] for a, b in plan) == sorted(r["id"] for r in rows)
    assert all(a["group_id"] != b["group_id"] for a, b in plan)
    rows[-1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError):
        history_schedule(rows)


def test_history_change_is_separated_from_repeat_change():
    def value(logits):
        return {"id": "a", "input_sha256": "same", "logits": logits}
    result = summarize([{
        "a0": value([3., 0.]), "a1": value([3., 0.]),
        "unrelated_b": {"id": "b", "input_sha256": "different", "logits": [0., 3.]},
        "a2": value([0., 3.]), "a3": value([0., 3.]),
    }])
    assert result["immediate_repeat"]["passed"]
    assert result["post_intervention_repeat"]["passed"]
    assert result["after_unrelated_request"]["argmax_changes"] == 1
    assert result["after_unrelated_request"]["maximum_probability_tv"] > .9


def test_different_prompt_cannot_be_misreported_as_history_variation():
    a = {"id": "a", "input_sha256": "one", "logits": [1., 0.]}
    changed = {**a, "input_sha256": "two"}
    with pytest.raises(ValueError):
        summarize([{"a0": a, "a1": a, "a2": changed, "a3": changed}])
