import copy
import json
from pathlib import Path

import pytest

from bobcat.klue import convert
from bobcat.klue_eval import compare_runs, freeze, run
from bobcat.metrics import scored_row
from bobcat.schema import file_hash, write_examples


def frozen_fixture(tmp_path):
    config = json.loads((
        Path(__file__).parents[1] / "configs/korean-decisions-klue-v1.json"
    ).read_text())
    rows = []
    for i in range(4):
        rows.extend([
            {"task": "nli", "upstream_split": "validation", "raw": {
                "guid": f"PRIVATE-nli-{i}", "premise": f"전제 {i}", "hypothesis": f"가설 {i}",
                "label": i % 3,
            }},
            {"task": "ynat", "upstream_split": "validation", "raw": {
                "guid": f"PRIVATE-topic-{i}", "title": f"제목 {i}", "label": i,
                "url": f"https://example.com/PRIVATE-article/{i}",
            }},
        ])
    examples, _, _ = convert(rows, config)
    path = tmp_path / "dev_public.jsonl"
    write_examples(path, examples)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "bobcat-korean-decisions-v1",
        "files": {"dev_public.jsonl": {"sha256": file_hash(path)}},
    }))
    return freeze(tmp_path, tmp_path / "suite.json", per_task=2)


class Scorer:
    model_name = "unit-test-fixture"
    temperatures = {}

    def __init__(self, fail_first=False):
        self.calls = 0
        self.fail_first = fail_first

    def score(self, state, questions):
        self.calls += 1
        assert "PRIVATE" not in state
        assert all(q.id == f"q{i}" for i, q in enumerate(questions))
        if self.fail_first and self.calls == 1:
            raise RuntimeError("Simulated unavailable model; never invent a score.")
        return [[0.0] * len(question.labels) for question in questions], 100


def test_complete_source_views_and_error_denominators(tmp_path):
    suite = frozen_fixture(tmp_path)
    assert suite["source_groups"] == 4 and suite["questions"] == 8
    assert len({g["group_id"] for g in suite["groups"]}) == 4
    for group in suite["groups"]:
        assert len(group["examples"]) == (3 if group["task"] == "nli" else 1)
    out = tmp_path / "run"
    result = run(suite, Scorer(fail_first=True), out, max_seconds=10)
    failed = len(suite["groups"][0]["examples"])
    assert result["status"] == "completed" and result["failed_questions"] == failed
    summary = json.loads((out / "summary.json").read_text())
    valid = summary["valid_response_metrics"]
    assert summary["accuracy_with_failed_questions_in_denominator"] == pytest.approx(
        valid["accuracy"] * (8 - failed) / 8
    )
    assert summary["independent_completed_input_groups"] == 3
    assert summary["exact_score_ties"] == 8 - failed
    assert not summary["calibration_was_refitted_on_this_data"]
    assert not result["release_gate_passed"]


def test_frozen_suite_cannot_be_changed_and_public_rows_cannot_be_replaced(tmp_path):
    suite = frozen_fixture(tmp_path)
    changed = copy.deepcopy(suite)
    changed["groups"][0]["examples"][0]["target"] = "CORRUPTED"
    with pytest.raises(ValueError, match="unchanged"):
        run(changed, Scorer(), tmp_path / "run")
    (tmp_path / "dev_public.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        freeze(tmp_path, tmp_path / "another-suite.json", per_task=2)


def test_typed_metrics_tie_break_matches_public_response():
    row = {"candidate_ids": ["z", "a"], "logits": [0.0, 0.0], "target": "z"}
    assert scored_row(row)["prediction"] == "a"  # Legacy synthetic behavior.
    typed = scored_row({**row, "tie_break": "request_order"})
    assert typed["prediction"] == "z" and typed["correct"]
    with pytest.raises(ValueError, match="tie-break"):
        scored_row({**row, "tie_break": "unknown"})


def test_paired_comparison_keeps_failed_answers_and_checks_same_evaluation(tmp_path):
    suite = frozen_fixture(tmp_path)
    before, after = tmp_path / "before", tmp_path / "after"
    run(suite, Scorer(fail_first=True), before)
    run(suite, Scorer(), after)
    comparison = compare_runs(suite, before, after, tmp_path / "comparison.json")
    assert comparison["independent_source_components"] == 4
    assert comparison["questions"] == 8
    assert comparison["failed_questions"]["before"] == len(suite["groups"][0]["examples"])
    assert comparison["failed_questions"]["after"] == 0
    assert comparison["paired_context_accuracy_difference"] >= 0
    assert comparison["release_gate_passed"] is False
    same = compare_runs(suite, after, after, tmp_path / "identity.json")
    assert same["paired_context_accuracy_difference"] == 0
    assert same["paired_context_bootstrap_95pct_interval"] == [0, 0]
    with (after / "predictions.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="unchanged"):
        compare_runs(suite, before, after, tmp_path / "changed.json")
