import copy
import json

import pytest

from bobcat.public_decisions import SCHEMA as DATA_SCHEMA
from bobcat.public_decisions import freeze_features, request_record
from bobcat.public_eval import compare_runs, freeze, run, validate
from bobcat.schema import file_hash, json_hash


def data_fixture(root, splits):
    root.mkdir()
    files = {}
    for split_index, split in enumerate(splits):
        source_split = "validation" if split == "dev_public" else "train"
        categories = [f"intent_{i}" for i in range(77)]
        cases = [
            ("klue_sts", {"sentence1": f"{split} 문장 하나", "sentence2": f"{split} 문장 둘",
                          "guid": "PRIVATE-SOURCE",
                          "labels": {"real-label": 1.5, "label": 1.5}}),
            ("banking77", {"text": f"{split} duplicate charge", "category": "intent_76"}),
            ("boolq", {"question": f"{split} Is it open?", "passage": "It is open.",
                       "answer": True}),
        ]
        rows = []
        for task, raw in cases:
            row = request_record(
                task, source_split, split_index, raw, {"secret": "PRIVATE"}, categories,
            )
            row.update(group_id=f"PRIVATE-{split}-{task}", split=split)
            rows.append(row)
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    (root / "manifest.json").write_text(json.dumps({"schema": DATA_SCHEMA, "files": files}))
    return root


class FixtureScorer:
    model_name = "fixture-only"
    temperatures = {"noul": 1.3}
    provenance = {"fixture_only": True}
    fail_noul = False

    def score(self, state, questions):
        assert "PRIVATE" not in json.dumps(state)
        scores = []
        for question in questions:
            assert "PRIVATE" not in question.instructions
            if self.fail_noul and question.kind == "noul":
                raise RuntimeError("fixture failure")
            values = [0.] * len(question.labels)
            if question.kind != "score":
                values[-1] = 3.
            scores.append(values)
        return scores, 100


def test_public_freeze_preserves_mean_and_full_choices_without_gold_in_requests(tmp_path):
    data = data_fixture(tmp_path / "data", ["dev_public"])
    suite = freeze(data, tmp_path / "suite.json", per_task=1)
    validate(suite)
    assert suite["group_count"] == 3 and suite["question_count"] == 3
    assert "PRIVATE" not in json.dumps([g["request"] for g in suite["groups"]])
    assert [len(g["rows"][0]["candidate_ids"]) for g in suite["groups"]
            if g["task"] == "banking77"] == [77]
    again = freeze(data, tmp_path / "repeat.json", per_task=1)
    assert suite == again
    changed = copy.deepcopy(suite)
    changed["groups"][0]["rows"][0]["split"] = "train"
    changed["content_sha256"] = json_hash({
        k: v for k, v in changed.items() if k != "content_sha256"
    })
    with pytest.raises(ValueError, match="alignment"):
        validate(changed)


def test_failed_hard_answers_stay_in_denominator_and_means_do_not_become_accuracy(tmp_path):
    suite = freeze(data_fixture(tmp_path / "data", ["dev_public"]),
                   tmp_path / "suite.json", per_task=1)
    scorer = FixtureScorer()
    scorer.fail_noul = True
    out = tmp_path / "evaluation"
    result = run(suite, scorer, out)
    assert result["status"] == "completed_with_failures" and result["failed_questions"] == 1
    report = json.loads((out / "summary.json").read_text())
    assert report["hard_attempted"] == 2 and report["hard_failed"] == 1
    assert report["hard_accuracy_with_failed_questions_in_denominator"] == 0.5
    assert report["ordinal_mean_attempted"] == 1 and report["ordinal_mean_failed"] == 0
    mean = report["valid_response_metrics"]["ordinal_mean_metrics"]
    assert mean["mae"] == pytest.approx(1)
    assert mean["categorical_nll"] is None and mean["categorical_ece"] is None
    assert report["valid_response_metrics"]["hard_label_questions"] == 1
    assert report["calibration_refitted"] is False


def test_mixed_feature_sampling_preserves_ordinal_order_and_excludes_public_rows(tmp_path):
    data = data_fixture(tmp_path / "data", ["train", "dev_train"])
    plan = freeze_features(
        data, tmp_path / "features.json",
        train_per_task=1, korean_train_per_task=1, dev_per_task=1,
    )
    assert plan["group_count"] == 6 and plan["question_count"] == 6
    for group in plan["groups"]:
        row = group["rows"][0]
        assert "PRIVATE" not in json.dumps(group["request"])
        if row["supervision"] == "score_mean":
            assert row["target"] is None and row["score_target"] == 1.5
            assert row["candidate_ids"] == list("012345")
    path = data / "dev_train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["source_split"] = "test"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="held-out"):
        freeze_features(data, tmp_path / "invalid.json",
                        train_per_task=1, korean_train_per_task=1, dev_per_task=1)


def test_paired_comparison_includes_failures_and_refuses_relabelled_predictions(tmp_path):
    suite = freeze(data_fixture(tmp_path / "data", ["dev_public"]),
                   tmp_path / "suite.json", per_task=1)
    before, after = tmp_path / "before", tmp_path / "after"
    scorer = FixtureScorer()
    scorer.fail_noul = True
    run(suite, scorer, before)
    run(suite, FixtureScorer(), after)
    report = compare_runs(suite, before, after, tmp_path / "comparison.json")
    assert report["hard_context_accuracy"]["difference"] == 0.5
    assert report["hard_context_accuracy"]["components"] == 2
    assert report["ordinal_mean_mae_on_common_valid_components"]["difference"] == 0
    assert report["ordinal_mean_components_missing_either_response"] == 0
    assert report["failed_questions"] == {"before": 1, "after": 0}
    path = after / "predictions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for record in records for row in record["rows"] if row["target"] is not None)
    row["target"] = row["candidate_ids"][0]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    manifest = json.loads((after / "run.json").read_text())
    manifest["predictions_sha256"] = file_hash(path)
    (after / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="annotations"):
        compare_runs(suite, before, after, tmp_path / "invalid-comparison.json")


def test_missing_score_response_has_no_fabricated_paired_error(tmp_path):
    suite = freeze(data_fixture(tmp_path / "data", ["dev_public"]),
                   tmp_path / "suite.json", per_task=1)

    class MissingScore(FixtureScorer):
        def score(self, state, questions):
            if any(q.kind == "score" for q in questions):
                raise RuntimeError("Missing ordinal answer")
            return super().score(state, questions)

    before, after = tmp_path / "before", tmp_path / "after"
    run(suite, MissingScore(), before)
    run(suite, FixtureScorer(), after)
    report = compare_runs(suite, before, after, tmp_path / "comparison.json")
    assert report["ordinal_mean_mae_on_common_valid_components"]["components"] == 0
    assert report["ordinal_mean_mae_on_common_valid_components"]["difference"] is None
    assert report["ordinal_mean_components_missing_either_response"] == 1
    assert report["ordinal_mean_failure_penalty_imputed"] is False
