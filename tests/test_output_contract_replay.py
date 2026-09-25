"""Artifact boundary tests; fixtures below are not real-model evidence."""

import json

import pytest

from bobcat.output_contract_replay import read_native
from bobcat.schema import file_hash


@pytest.fixture
def captured_fixture(tmp_path, monkeypatch):
    folder, probe = tmp_path / "capture", tmp_path / "probe"
    folder.mkdir()
    probe.mkdir()
    (probe / "manifest.json").write_text('{"fixture":true}')
    branches = [{
        "index": index, "case_id": f"case-{index}", "question_id": "choice",
        "input_sha256": str(index), "option_token_ids": [10, 20],
    } for index in range(16)]
    monkeypatch.setattr("bobcat.output_contract_replay.read_probe", lambda _: (
        {"source_revision": "fixture-revision"}, [], branches,
    ))
    complete = {
        "schema": "bobcat-native-glm-output-probe-result-v1",
        "status": "completed", "world_size": 8,
        "probe_manifest_sha256": file_hash(probe / "manifest.json"),
        "source_revision": "fixture-revision", "model_weights_used": True,
        "generated_text_tokens": 0, "training_updates": 0,
        "parent_adapter_sha256": "fixture-actor", "parent_decode_sha256": "fixture-decode",
        "job_sha256": "fixture-job",
    }
    for rank in range(8):
        rows = [{
            **{k: row[k] for k in (
                "case_id", "question_id", "input_sha256", "option_token_ids"
            )}, "branch_index": row["index"], "logits": [2., -2.],
            "no_tokens_sampled_or_decoded": True,
        } for row in branches[rank::8]]
        prediction = folder / f"predictions-rank-{rank}.jsonl"
        prediction.write_text("".join(json.dumps(row) + "\n" for row in rows))
        status = {
            **complete, "rank": rank, "parent_actor_values_exact": True,
            "actor_values_unchanged": True, "attempted_generation_calls": 0,
            "observed_output_projection_calls": 2, "completed_local_branches": 2,
            "predictions_sha256": file_hash(prediction),
        }
        (folder / f"rank-{rank}.json").write_text(json.dumps(status))

    def seal():
        complete["files"] = {
            p.name: file_hash(p) for p in folder.iterdir() if p.name != "complete.json"
        }
        (folder / "complete.json").write_text(json.dumps(complete))

    seal()
    return folder, probe, seal


def test_complete_capture_requires_eight_exact_non_generating_rank_proofs(captured_fixture):
    folder, probe, _ = captured_fixture
    _, _, branches, predictions = read_native(folder, probe)
    assert len(predictions) == len(branches) == 16


@pytest.mark.parametrize(("field", "value"), [
    ("attempted_generation_calls", 1),
    ("generated_text_tokens", 1),
    ("model_weights_used", False),
    ("actor_values_unchanged", False),
    ("parent_decode_sha256", "another-actor"),
    ("probe_manifest_sha256", "another-corpus"),
    ("job_sha256", "another-job"),
    ("source_revision", "another-model"),
    ("observed_output_projection_calls", 3),
    ("completed_local_branches", 1),
])
def test_rank_provenance_is_checked_even_with_valid_file_hashes(
    captured_fixture, field, value,
):
    folder, probe, seal = captured_fixture
    path = folder / "rank-0.json"
    status = json.loads(path.read_text())
    status[field] = value
    path.write_text(json.dumps(status))
    seal()
    with pytest.raises(ValueError, match="actor/rank proof"):
        read_native(folder, probe)


@pytest.mark.parametrize(("field", "value"), [
    ("no_tokens_sampled_or_decoded", False),
    ("branch_index", 2),
    ("input_sha256", "new-input"),
    ("logits", [True, 0.]),
    ("logits", [float("nan"), 0.]),
    ("logits", ["ordinary prose", 0.]),
    ("option_token_ids", [20, 10]),
])
def test_replay_rejects_semantically_different_or_malformed_captures(
    captured_fixture, field, value,
):
    folder, probe, seal = captured_fixture
    prediction = folder / "predictions-rank-0.jsonl"
    rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    rows[0][field] = value
    prediction.write_text("".join(json.dumps(row) + "\n" for row in rows))
    status_path = folder / "rank-0.json"
    status = json.loads(status_path.read_text())
    status["predictions_sha256"] = file_hash(prediction)
    status_path.write_text(json.dumps(status))
    seal()
    with pytest.raises(ValueError, match="prediction membership"):
        read_native(folder, probe)


def test_missing_rank_is_not_a_zero_violation_success(captured_fixture):
    folder, probe, seal = captured_fixture
    (folder / "rank-7.json").unlink()
    seal()
    with pytest.raises(ValueError, match="omits a rank"):
        read_native(folder, probe)
