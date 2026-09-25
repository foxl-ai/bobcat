import copy
import json

import pytest
import torch
from test_klue import nli, source

from bobcat.glm_feature_data import extract, freeze, validate_plan
from bobcat.klue import examples_for
from bobcat.schema import file_hash, json_hash, write_examples


@pytest.fixture
def feature_plan(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    files = {}
    for split in ("train", "dev_train"):
        examples = []
        for i in range(3):
            identity = f"PRIVATE-{split}-nli-{i}"
            row = nli(identity, f"{split} 전제 {i}", f"{split} 가설 {i}", i)
            examples.extend(examples_for(row, identity, split, source()))
            identity = f"PRIVATE-{split}-ynat-{i}"
            row = {
                "identity": identity, "task": "ynat", "upstream_split": "train",
                "raw": {"guid": identity, "title": f"{split} 기사 {i}", "label": i},
            }
            examples.extend(examples_for(row, identity, split, source()))
        path = data / f"{split}.jsonl"
        write_examples(path, examples)
        files[path.name] = {"sha256": file_hash(path)}
    (data / "manifest.json").write_text(json.dumps({
        "schema": "bobcat-korean-decisions-v1", "files": files,
    }))
    plan = freeze(data, tmp_path / "plan.json", train_per_task=2, development_per_task=1)
    return plan, data


def test_plan_keeps_components_views_and_gold_out_of_model_requests(feature_plan, tmp_path):
    plan, data = feature_plan
    validate_plan(plan)
    assert plan["group_count"] == 6
    assert plan["question_count"] == 12
    assert len({g["group_id"] for g in plan["groups"]}) == 6
    for group in plan["groups"]:
        assert "PRIVATE" not in json.dumps(group["request"])
        assert len(group["rows"]) == (3 if group["task"] == "nli" else 1)
        assert sum(r["context_weight"] for r in group["rows"]) == 1
    other = freeze(data, tmp_path / "repeat.json", train_per_task=2, development_per_task=1)
    assert other == plan
    changed = copy.deepcopy(plan)
    changed["groups"][0]["rows"][0]["target"] = "missing"
    changed["content_sha256"] = json_hash(
        {k: v for k, v in changed.items() if k != "content_sha256"}
    )
    with pytest.raises(ValueError, match="misaligned"):
        validate_plan(changed)


class FixtureScorer:
    provenance = {"fixture_only": True, "feature_profile": "fixture"}
    last_measurement = {"fixture_only": True}

    def extract_many(self, requests):
        scores = []
        for state, questions in requests:
            assert "PRIVATE" not in json.dumps(state)
            for q in questions:
                scores.append([float(i) for i in range(len(q.labels))])
        return scores, torch.ones(len(scores), 4096), 200


def test_extraction_aligns_native_features_and_separate_loss_labels(feature_plan, tmp_path):
    plan, _ = feature_plan
    out = tmp_path / "extracted"
    result = extract(plan, FixtureScorer(), out, contexts_per_batch=2)
    assert result["status"] == "completed"
    assert result["completed_groups"] == 6
    assert result["completed_questions"] == 12
    rows = []
    for item in result["files"]:
        path = out / item["path"]
        assert file_hash(path) == item["sha256"]
        payload = torch.load(path, weights_only=True)
        rows.extend(payload["rows"])
        for row, target, keep, logits in zip(
            payload["rows"], payload["targets"], payload["candidate_keep"],
            payload["base_logits"], strict=True,
        ):
            assert int(target) == row["candidate_ids"].index(row["target"])
            assert int(keep.sum()) == len(row["candidate_ids"])
            assert torch.isneginf(logits[~keep]).all()
    assert rows == [r for group in plan["groups"] for r in group["rows"]]


def test_partial_failed_features_cannot_be_marked_complete(feature_plan, tmp_path):
    plan, _ = feature_plan

    class FailsAfterOne(FixtureScorer):
        calls = 0

        def extract_many(self, requests):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("wrong hidden position")
            return super().extract_many(requests)

    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="wrong hidden"):
        extract(plan, FailsAfterOne(), out, contexts_per_batch=2)
    result = json.loads((out / "run.json").read_text())
    assert result["status"] == "failed"
    assert result["completed_groups"] == 2
    assert len(result["files"]) == 1
    assert (out / result["files"][0]["path"]).is_file()


def test_changed_input_source_is_rejected_before_freezing(feature_plan, tmp_path):
    _, data = feature_plan
    with (data / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        freeze(data, tmp_path / "changed.json", train_per_task=2, development_per_task=1)
