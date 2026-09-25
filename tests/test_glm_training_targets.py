import copy
import json

import numpy as np
import pytest

from bobcat.glm_training_targets import target_record


def row():
    return {
        "id": "sample", "group_id": "component", "split": "train",
        "source_request_sha256": "source", "input_sha256": "input", "input_tokens": 20,
        "language": "ko", "language_origin": "native", "task": "example", "kind": "choice",
        "candidate_ids": ["거절", "허용"], "option_token_ids": [32, 33],
        "supervision": "hard_label", "target_index": 1, "score_mean": None,
    }


def test_teacher_disagreement_never_overwrites_observed_label():
    original = row()
    before = copy.deepcopy(original)
    output = target_record(
        original, {"id": "sample", "input_sha256": "input", "logits": [-1., -3.],
                   "native_completion_tokens": 0},
        teacher={"source_repo": "zai-org/GLM-5.3-Flash"},
    )
    assert original == before
    assert output["observed_supervision"]["target_index"] == 1
    assert output["teacher_argmax_index"] == 0
    assert output["teacher_matches_observed_label"] is False
    assert output["teacher_distribution_is_ground_truth"] is False
    assert np.isclose(sum(output["teacher_conditional_probabilities"]), 1)
    assert output["training_eligibility"].startswith("requires_")


def test_score_mean_does_not_become_teacher_category_gold():
    source = row()
    source.update(kind="ordinal", supervision="score_mean", target_index=-100, score_mean=.37)
    output = target_record(
        source, {"id": "sample", "input_sha256": "input", "logits": [-1000., -1001.],
                 "native_completion_tokens": 0}, teacher={},
    )
    assert output["teacher_matches_observed_label"] is None
    assert output["observed_supervision"]["score_mean"] == .37
    assert np.isfinite(output["teacher_conditional_logprobs"]).all()


def test_nontraining_or_misaligned_predictions_are_rejected():
    prediction = {"id": "sample", "input_sha256": "input", "logits": [0., 1.],
                  "native_completion_tokens": 0}
    for change in ({"split": "dev_train"}, {"id": "other"}, {"input_sha256": "changed"}):
        value = {**row(), **change}
        with pytest.raises(ValueError, match="complete training question"):
            target_record(value, prediction, teacher={})
    with pytest.raises(ValueError, match="Non-finite"):
        target_record(row(), {**prediction, "logits": [float("nan"), 1.]}, teacher={})


def test_collection_monitor_stops_before_publishing_an_unstable_training_batch(
    tmp_path, monkeypatch,
):
    import bobcat.glm_training_targets as module

    original = row()
    manifest = {"statistics": {"train": {"prompt_tokens": original["input_tokens"]}}}
    monkeypatch.setattr(module, "verify_training_partition", lambda *_: (
        manifest, [original], {"source_revision": "revision"},
    ))

    class Scorer:
        model_info = {"model_path": "/model"}

        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def read(self, _rows, **_kwargs):
            self.calls += 1
            return [{
                **original, "logits": [2. if self.calls == 1 else 3., 0.],
                "native_completion_tokens": 0,
            }], {"native_http_seconds": .2}

    monkeypatch.setattr(module, "CompiledReadout", Scorer)
    out = tmp_path / "targets"
    teacher = {"source_repo": "zai-org/GLM-5.3-Flash", "source_revision": "revision",
               "weights_manifest_sha256": "weights", "jev_outputs_used": False}
    with pytest.raises(ValueError, match="runtime stability monitor"):
        module.collect(tmp_path, tmp_path, out, client=None, model_path="/model",
                       teacher=teacher, seconds=180, batch_size=1, verify_every_batches=128)
    receipt = json.loads((out / "collection.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["runtime_stability_violation"]
    assert receipt["completed_questions"] == 0
    assert receipt["completed_shard_rows"] == 0
    assert receipt["shards"] == []
    assert len(receipt["stability_checks"]) == 1
