import pytest

from bobcat.checkpoint_metrics import compare


def row(name, logits, *, mean=None, language="ko"):
    return {
        "id": name, "group_id": name, "input_sha256": name, "task": "sample",
        "language": language, "language_origin": "native", "kind": "choice",
        "supervision": "hard_label" if mean is None else "score_mean",
        "score_mean": 0. if mean is None else mean,
        "target_index": 0 if mean is None else -100, "logits": logits,
    }


def test_ordinal_mean_not_in_categorical_accuracy_or_nll():
    before = [row("a", [3., 0.]), row("b", [0., 0.], mean=1.)]
    after = [row("a", [0., 3.]), row("b", [0., 3.], mean=1.)]
    report = compare(before, after)
    all_rows = report["subsets"]["all"]
    assert all_rows["checkpoint"]["categorical"]["count"] == 1
    assert all_rows["accuracy_change"]["delta"] == -1
    assert all_rows["ordinal_normalized_mae_change"]["delta"] < 0
    assert report["release_gate_passed"] is False
    assert report["training_modified"] is False


def test_changed_gold_or_missing_component_cannot_improve_report():
    before = [row("a", [3., 0.]), row("b", [0., 0.])]
    with pytest.raises(ValueError):
        compare(before, before[:1])
    after = [dict(r) for r in before]
    after[0]["target_index"] = 1
    with pytest.raises(ValueError):
        compare(before, after)


def test_reports_korean_regression_even_when_english_improves():
    before = [row("ko", [3., 0.]), row("en", [0., 3.], language="en")]
    after = [row("ko", [0., 3.]), row("en", [3., 0.], language="en")]
    metrics = compare(before, after)["subsets"]
    assert metrics["all"]["accuracy_change"]["delta"] == 0
    assert metrics["language/ko"]["accuracy_change"]["delta"] == -1
    assert metrics["language/en"]["accuracy_change"]["delta"] == 1
