import json

import pytest

from bobcat.checkpoint_metrics import read_training_completed, read_training_suite_completed
from bobcat.glm_native_train import REVISION
from bobcat.schema import file_hash, json_hash


@pytest.fixture
def published(tmp_path):
    curriculum, predictions = tmp_path / "data", tmp_path / "predictions"
    curriculum.mkdir()
    predictions.mkdir()
    splits = {}
    for split in ("train", "dev_train"):
        rows = []
        for i in range(8):
            inputs = {"input_ids": [1, 2, i + 3], "option_token_ids": [10, 11]}
            rows.append({
                "id": f"{split}-{i}", "group_id": f"group-{split}-{i}", "split": split,
                "task": "native_ko", "language": "ko", "language_origin": "native",
                "kind": "choice", "target_index": i % 2, "score_mean": 0.,
                "supervision": "hard_label", "sampling_loss_weight": 1,
                **inputs, "input_tokens": 3, "input_sha256": json_hash(inputs),
            })
        (curriculum / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
        )
        splits[split] = rows
    manifest = {
        "schema": "bobcat-glm-curriculum-v1", "status": "completed",
        "training_performed": False, "model_source": {"revision": REVISION},
        "curriculum_max_tokens": 128,
        "files": {f"{s}.jsonl": {"sha256": file_hash(curriculum / f"{s}.jsonl")}
                  for s in splits},
        "statistics": {s: {"rows": len(rows)} for s, rows in splits.items()},
    }
    manifest["content_sha256"] = json_hash(manifest)
    (curriculum / "manifest.json").write_text(json.dumps(manifest))
    files = {}
    for rank, gold in enumerate(splits["dev_train"]):
        row = {k: gold[k] for k in (
            "id", "group_id", "task", "language", "kind",
            "target_index", "score_mean", "supervision",
        )}
        row.update(logits=[1., -1.], loss=.1)
        path = predictions / f"baseline-rank-{rank}.json"
        path.write_text(json.dumps([row]))
        files[path.name] = file_hash(path)
    marker = {
        "label": "baseline", "curriculum_sha256": file_hash(curriculum / "manifest.json"),
        "questions": 8, "prompt_tokens": 24, "rng_restored": True,
        "optimizer_updated": False, "final_test": False, "files": files,
    }
    (predictions / "evaluation-baseline-complete.json").write_text(json.dumps(marker))
    return curriculum, predictions


def test_completed_snapshot_gets_provenance_from_hashed_curriculum(published):
    curriculum, predictions = published
    rows, _ = read_training_completed(predictions, "baseline", curriculum)
    assert len(rows) == 8
    assert all(row["language_origin"] == "native" for row in rows)
    assert all(row["input_identity_source"] == "immutable_producer_curriculum" for row in rows)
    assert rows[0]["input_sha256"] == json_hash(
        {"input_ids": [1, 2, 3], "option_token_ids": [10, 11]},
    )


@pytest.mark.parametrize(
    "mutation", ["omit_prediction", "alter_gold", "nonfinite", "wrong_input_hash"],
)
def test_invalid_completed_prediction_never_improves_denominator(published, mutation):
    curriculum, predictions = published
    path = predictions / "baseline-rank-0.json"
    rows = json.loads(path.read_text())
    if mutation == "omit_prediction":
        rows = []
    elif mutation == "alter_gold":
        rows[0]["target_index"] = 1
    elif mutation == "nonfinite":
        rows[0]["logits"][0] = float("nan")
    else:
        rows[0]["input_sha256"] = "different-input"
    path.write_text(json.dumps(rows))
    marker_path = predictions / "evaluation-baseline-complete.json"
    marker = json.loads(marker_path.read_text())
    marker["files"][path.name] = file_hash(path)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        read_training_completed(predictions, "baseline", curriculum)


def test_live_or_calibration_marker_is_not_dev_snapshot(published):
    curriculum, predictions = published
    marker_path = predictions / "evaluation-baseline-complete.json"
    marker = json.loads(marker_path.read_text())
    marker["optimizer_updated"] = True
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        read_training_completed(predictions, "baseline", curriculum)


def expanded_snapshot(published, *, training_overlap=False):
    curriculum, predictions = published
    suite = curriculum.parent / "suite"
    suite.mkdir()
    source = "train" if training_overlap else "dev_train"
    gold = [json.loads(line) for line in (curriculum / f"{source}.jsonl").read_text().splitlines()]
    for row in gold:
        row["split"] = "dev_train"
    (suite / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in gold))
    manifest = {
        "schema": "bobcat-checkpoint-monitor-suite-v1",
        "evaluation_role": "development_monitoring", "source_revision": REVISION,
        "training_or_calibration": False, "rows": len(gold), "max_input_tokens": 128,
        "records_sha256": file_hash(suite / "records.jsonl"),
    }
    manifest["content_sha256"] = json_hash(manifest)
    (suite / "manifest.json").write_text(json.dumps(manifest))
    marker_path = predictions / "evaluation-baseline-complete.json"
    marker = json.loads(marker_path.read_text())
    for rank, row in enumerate(gold):
        prediction = {
            k: row[k] for k in (
                "id", "group_id", "task", "language", "kind", "input_sha256",
                "target_index", "score_mean", "supervision",
            )
        }
        prediction.update(logits=[1., -1.], loss=.1)
        path = predictions / f"baseline-rank-{rank}.json"
        path.write_text(json.dumps([prediction]))
        marker["files"][path.name] = file_hash(path)
    marker["monitoring_suite"] = {
        "manifest_sha256": file_hash(suite / "manifest.json"),
        "records_sha256": manifest["records_sha256"], "components": 8,
        "training_overlap": False, "final_test": False,
    }
    marker_path.write_text(json.dumps(marker))
    return curriculum, predictions, suite


def test_expanded_snapshot_binds_provenance_to_separate_suite(published):
    curriculum, predictions, suite = expanded_snapshot(published)
    rows, marker = read_training_suite_completed(predictions, "baseline", curriculum, suite)
    assert len(rows) == 8
    assert {r["input_identity_source"] for r in rows} == {"immutable_monitoring_suite"}
    assert marker["monitoring_suite"]["training_overlap"] is False
    with pytest.raises(ValueError):
        read_training_completed(predictions, "baseline", curriculum)


def test_false_no_overlap_claim_is_checked_against_actual_training_components(published):
    curriculum, predictions, suite = expanded_snapshot(published, training_overlap=True)
    with pytest.raises(ValueError, match="training component"):
        read_training_suite_completed(predictions, "baseline", curriculum, suite)


@pytest.mark.parametrize("change", ["wrong_suite", "missing_input_hash", "missing_rank"])
def test_expanded_snapshot_rejects_mismatched_or_incomplete_evidence(published, change):
    curriculum, predictions, suite = expanded_snapshot(published)
    marker_path = predictions / "evaluation-baseline-complete.json"
    marker = json.loads(marker_path.read_text())
    if change == "wrong_suite":
        marker["monitoring_suite"]["records_sha256"] = "another-suite"
    elif change == "missing_rank":
        del marker["files"]["baseline-rank-7.json"]
    else:
        path = predictions / "baseline-rank-0.json"
        rows = json.loads(path.read_text())
        del rows[0]["input_sha256"]
        path.write_text(json.dumps(rows))
        marker["files"][path.name] = file_hash(path)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        read_training_suite_completed(predictions, "baseline", curriculum, suite)
