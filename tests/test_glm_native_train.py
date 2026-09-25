import pytest
import torch

from bobcat.glm_native_train import (
    checkpoint_layout,
    decision_loss,
    packed_decision_loss,
    question_packs,
    read_monitoring_suite,
    replay_differences,
    tensor_digest,
    validate_record,
    verify_parameter_storage,
    verify_restored_layout,
)
from bobcat.schema import json_hash


def record(mean=False):
    inputs = {"input_ids": [1, 2], "option_token_ids": [10, 11, 12]}
    return {
        **inputs, "input_tokens": 2, "sampling_loss_weight": 1.,
        "input_sha256": json_hash(inputs),
        "supervision": "score_mean" if mean else "hard_label",
        "kind": "ordinal" if mean else "choice", "score_mean": 1.,
        "target_index": -100 if mean else 2,
    }


def test_observed_score_mean_is_not_a_hard_label():
    row = record(mean=True)
    logits = torch.tensor([0., 0., 0.], requires_grad=True)
    validate_record(row, 4)
    assert decision_loss(logits, row).item() == 0
    decision_loss(logits, row).backward()
    assert torch.equal(logits.grad, torch.zeros(3))
    assert decision_loss(torch.tensor([0., 0., 0.]), record()).item() > 1


def test_clipped_or_misaligned_curriculum_is_rejected():
    row = record()
    with pytest.raises(ValueError):
        validate_record(row, 1)
    row["option_token_ids"] = [10, 10, 12]
    with pytest.raises(ValueError):
        validate_record(row, 4)


def test_hash_handles_bfloat_and_scalar_buffer():
    value = torch.tensor(1., dtype=torch.bfloat16)
    assert tensor_digest(value) == tensor_digest(value.clone())
    assert tensor_digest(value) != tensor_digest(value + 1)


def test_restoration_requires_buffers_and_exact_layout_not_parameter_count():
    model = torch.nn.Linear(3, 2, bias=False)
    model.register_buffer("e_score_correction_bias", torch.zeros(2))
    state = model.state_dict()
    expected = checkpoint_layout(state)
    assert len(expected) == len(list(model.parameters())) + 1
    verify_restored_layout(expected, state, set(state))
    with pytest.raises(ValueError, match="e_score_correction_bias"):
        verify_restored_layout(expected, state, {"weight"})
    with pytest.raises(ValueError, match="changed"):
        verify_restored_layout(expected, {**state, "weight": torch.zeros(2, 4)}, set(state))
    with pytest.raises(ValueError, match="unexpected"):
        verify_restored_layout(expected, state, {"weight", "unrelated_buffer"})


def test_storage_check_includes_frozen_shards_and_rejects_resident_cpu_or_meta():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)
    verified = verify_parameter_storage(model, cpu_offload=True)
    assert verified["local_parameter_tensors"] == 4
    assert verified["local_parameter_bytes"] == 44
    with pytest.raises(ValueError, match="requested cuda"):
        verify_parameter_storage(model, cpu_offload=False)
    model[0] = torch.nn.Linear(3, 2, device="meta")
    with pytest.raises(ValueError, match="0.weight"):
        verify_parameter_storage(model, cpu_offload=True)
    with pytest.raises(ValueError, match="explicit boolean"):
        verify_parameter_storage(model, cpu_offload="false")


def test_expanded_monitor_must_remain_disjoint_from_training(tmp_path):
    import json

    from bobcat.glm_native_train import REVISION
    from bobcat.schema import file_hash

    rows = [
        {**record(), "id": f"q{i}", "group_id": f"g{i}", "split": "dev_train"}
        for i in range(8)
    ]
    data = tmp_path / "records.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "schema": "bobcat-checkpoint-monitor-suite-v1",
        "evaluation_role": "development_monitoring", "training_or_calibration": False,
        "source_revision": REVISION, "rows": 8, "max_input_tokens": 4,
        "records_sha256": file_hash(data),
    }
    manifest["content_sha256"] = json_hash(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert len(read_monitoring_suite(tmp_path, [{"group_id": "train-only"}])[1]) == 8
    with pytest.raises(ValueError, match="overlaps"):
        read_monitoring_suite(tmp_path, [{"group_id": "g7"}])


def test_replay_report_distinguishes_small_drift_from_wrong_state():
    old = {"weight": torch.tensor([1., 2.]), "step": 2}
    new = {"weight": old["weight"].clone(), "step": 2}
    new["weight"][0] = torch.nextafter(new["weight"][0], torch.tensor(float("inf")))
    report = replay_differences(old, new)
    assert report["different_tensors"] == 1
    assert 0 < report["max_abs"] < 1e-6
    assert not report["metadata_differences"]
    assert not report["nonfinite"]
    new["step"] = 3
    assert replay_differences(old, new)["metadata_differences"] == ["state/step"]


def test_packed_ranks_preserve_curriculum_and_equal_question_weight():
    rows = [{"input_ids": [index] * (index + 1)} for index in range(8)]
    packs, padded = question_packs(rows, 2, 4)
    assert packs == [rows[::2], rows[1::2]]
    assert padded == 128
    with pytest.raises(ValueError, match="equal"):
        question_packs(rows[:-1], 2, 4)
    with pytest.raises(ValueError, match="limit"):
        question_packs([{"input_ids": [1] * 20000}] * 4, 2, 2)


def test_packed_loss_and_gradients_equal_independent_question_average():
    logits = [torch.tensor([1., -2., .5], requires_grad=True),
              torch.tensor([-.5, .2, .8], requires_grad=True)]
    rows = [record(), record(mean=True)]
    packed = packed_decision_loss(logits, rows)
    independent = (decision_loss(logits[0], rows[0]) + decision_loss(logits[1], rows[1])) / 2
    assert torch.equal(packed, independent)
    packed_grads = torch.autograd.grad(packed, logits, retain_graph=True)
    separate_grads = torch.autograd.grad(independent, logits)
    for actual, expected in zip(packed_grads, separate_grads, strict=True):
        assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="aligned"):
        packed_decision_loss(logits, rows[:1])
