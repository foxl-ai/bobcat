import copy
import json

import pytest
import torch
from test_checkpoints import MemoryS3
from test_decision_train import assert_equal
from test_glm_feature_data import FixtureScorer
from test_glm_feature_data import feature_plan as feature_plan

from bobcat.checkpoints import S3CheckpointStore, verify_checkpoint
from bobcat.glm_feature_data import extract
from bobcat.glm_features import FEATURE_PROFILE, ResidualFeatureScorer
from bobcat.glm_head_train import FORMAT, load_features, train
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash


@pytest.fixture
def extracted(feature_plan, tmp_path):
    plan, _ = feature_plan
    scorer = FixtureScorer()
    scorer.provenance = {
        "base_repo": "fixture/GLM", "base_revision": "a" * 40,
        "model_source_manifest_content_sha256": "c" * 64, "compiler_sha256": "d" * 64,
        "profile": "fixture",
        "feature_profile": FEATURE_PROFILE, "feature_identity_checked_on_every_request": True,
        "base_weights_updated": False, "option_head_sha256": "f" * 64,
        "fixture_only": True,
    }
    out = tmp_path / "features"
    extract(plan, scorer, out, contexts_per_batch=2)
    return plan, out


def test_feature_training_load_checks_labels_even_with_updated_tensor_checksum(extracted):
    plan, root = extracted
    data = load_features(plan, root)
    assert len(data.groups["train"]) == 4
    assert len(data.groups["dev_train"]) == 2
    manifest = json.loads((root / "run.json").read_text())
    record = manifest["files"][0]
    path = root / record["path"]
    payload = torch.load(path, weights_only=True)
    payload["targets"][0] = (payload["targets"][0] + 1) % len(payload["rows"][0]["candidate_ids"])
    torch.save(payload, path)
    record.update(bytes=path.stat().st_size, sha256=file_hash(path))
    (root / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="misaligned"):
        load_features(plan, root)


def test_head_resume_and_remote_recovery_preserve_optimizer_and_context_cursor(
    extracted, tmp_path,
):
    plan, root = extracted
    torch.set_num_threads(1)
    recipe = {"epochs": 3, "rank": 4, "batch_contexts": 2, "save_every": 1}
    full, stopped, resumed = (tmp_path / name for name in ("full", "stopped", "resumed"))
    train(plan, root, full, **recipe)
    store = S3CheckpointStore("s3://unit-test/runs/head", client=MemoryS3())
    partial = train(plan, root, stopped, **recipe, stop_after_steps=2, remote=store)
    assert partial["status"] == "step_limit"
    recovered = store.restore(tmp_path / "remote-restored")
    assert verify_checkpoint(recovered, expected_format=FORMAT)["step"] == 2
    result = train(plan, root, resumed, **recipe, resume=recovered)
    assert result["status"] == "completed"
    complete = torch.load(full / "last.pt", weights_only=True)
    actual = torch.load(resumed / "last.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "epoch", "cursor", "counters", "step"):
        assert_equal(complete[key], actual[key])
    assert actual["step"] == 6
    assert actual["counters"] == {"contexts": 12, "questions": 24}
    assert actual["provenance"]["base_weights_updated"] is False
    assert actual["model"]["up.weight"].abs().sum() > 0
    assert torch.equal(actual["model"]["up.weight"][7:], torch.zeros_like(
        actual["model"]["up.weight"][7:],
    ))
    with pytest.raises(ValueError, match="identical"):
        train(plan, root, tmp_path / "bad-resume", **{**recipe, "rank": 8}, resume=recovered)


def test_head_objective_gives_each_complete_source_context_weight_one(extracted, tmp_path):
    plan, root = extracted
    data = load_features(plan, root)
    train_indices = [i for group in data.groups["train"] for i in group]
    expected = (
        torch.nn.functional.cross_entropy(
            data.base_logits[train_indices], data.targets[train_indices], reduction="none",
        ).double() * data.weights[train_indices]
    ).sum() / len(data.groups["train"])
    out = tmp_path / "weighted"
    train(plan, root, out, epochs=1, batch_contexts=64, rank=4)
    first = json.loads((out / "training.jsonl").read_text().splitlines()[0])
    assert first["loss"] == pytest.approx(float(expected), abs=1e-7)
    report = json.loads((out / "head-development.json").read_text())
    assert report["independent_contexts"] == 2 and report["questions"] == 4
    assert report["calibration_fitted"] is False


def test_live_residual_scorer_uses_the_trained_profile_and_retains_all_candidates(
    extracted, tmp_path,
):
    plan, root = extracted
    out = tmp_path / "live-head"
    train(plan, root, out, epochs=1, rank=4)
    data = load_features(plan, root)
    indices = [i for group in data.groups["dev_train"] for i in group]
    _, query = parse_request({
        "model": "fixture", "state": "test state",
        "questions": {
            f"q{i}": {
                "type": "choice",
                "criteria": dict.fromkeys(data.rows[i]["candidate_ids"]),
            }
            for i in indices
        },
    })

    class LiveFixture:
        provenance = dict(data.provenance["scorer"])
        limits = {"max_choices": 255}
        last_measurement = {"candidate_log_probability_mass": [-3] * len(indices)}

        def extract(self, state, questions):
            assert state == "test state" and questions == query
            base = [
                data.base_logits[i, :len(data.rows[i]["candidate_ids"])].tolist()
                for i in indices
            ]
            return base, data.hidden[indices], 200

    live = LiveFixture()
    scorer = ResidualFeatureScorer(live, out / "last.pt")
    actual, tokens = scorer.score("test state", query)
    with torch.inference_mode():
        expected = scorer.model(
            data.hidden[indices], data.base_logits[indices], data.candidate_keep[indices],
        )
    assert tokens == 200 and len(actual) == len(indices)
    for i, values in enumerate(actual):
        torch.testing.assert_close(torch.tensor(values), expected[i, :len(values)])
    assert "candidate_log_probability_mass" not in scorer.last_measurement
    assert "base_candidate_log_probability_mass" in scorer.last_measurement
    assert scorer.provenance["base_weights_updated"] is False
    live.provenance["base_revision"] = "b" * 40
    with pytest.raises(ValueError, match="differs"):
        ResidualFeatureScorer(live, out / "last.pt")


def test_mixed_ordinal_mean_training_preserves_targets_and_exact_resume(extracted, tmp_path):
    original, original_root = extracted
    plan = copy.deepcopy(original)
    plan["schema"] = "bobcat-glm-feature-plan-v2"
    for group in plan["groups"]:
        if group["task"] != "ynat":
            continue
        group["request"]["questions"] = {
            "q0": {"type": "score", "instructions": "점수를 매기세요.",
                   "criteria": ["낮음", "보통", "높음"]},
        }
        group["rows"][0].update(
            kind="ordinal", candidate_ids=["0", "1", "2"], target=None,
            supervision="score_mean", score_target=1.0 / 3, family="score_fixture",
        )
    plan["content_sha256"] = json_hash({k: v for k, v in plan.items() if k != "content_sha256"})
    scorer = FixtureScorer()
    scorer.provenance = json.loads((original_root / "run.json").read_text())["scorer_provenance"]
    features = tmp_path / "mixed-features"
    extract(plan, scorer, features)
    data = load_features(plan, features)
    mean_mask = data.targets.eq(-100)
    assert mean_mask.sum() == 3
    assert torch.equal(data.score_targets[mean_mask], torch.full(
        (3,), 1.0 / 3, dtype=torch.float64,
    ))
    recipe = {"epochs": 2, "rank": 4, "batch_contexts": 2, "save_every": 1}
    full, stopped, resumed = (
        tmp_path / name for name in ("mixed-full", "mixed-stop", "mixed-resume")
    )
    train(plan, features, full, **recipe)
    train(plan, features, stopped, **recipe, stop_after_steps=1)
    train(plan, features, resumed, **recipe, resume=stopped / "last.pt")
    complete = torch.load(full / "last.pt", weights_only=True)
    actual = torch.load(resumed / "last.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "epoch", "cursor", "counters", "step"):
        assert_equal(complete[key], actual[key])
    report = json.loads((full / "head-development.json").read_text())
    assert report["hard_label_contexts"] == 1 and report["independent_contexts"] == 2
    assert report["supervision"]["ordinal_mean_metrics"]["questions"] == 1
    assert report["supervision"]["ordinal_mean_metrics"]["categorical_nll"] is None

    manifest = json.loads((features / "run.json").read_text())
    record = manifest["files"][0]
    path = features / record["path"]
    payload = torch.load(path, weights_only=True)
    payload["score_targets"][0] += 0.01
    torch.save(payload, path)
    record.update(bytes=path.stat().st_size, sha256=file_hash(path))
    (features / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="misaligned"):
        load_features(plan, features)
