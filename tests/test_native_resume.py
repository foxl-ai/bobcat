import copy

import pytest
import torch
from torch.distributed.checkpoint import load, save
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import get_state_dict

from bobcat.native_resume import (
    apply_learning_rate_transition,
    continuation_cursor,
    evaluation_state,
    materialize_reference_state,
    state_signature,
    verify_materialized_state,
)


def parent():
    return {"world_size": 8, "pack_questions": 1, "source_revision": "model",
            "curriculum_manifest_sha256": "data"}


def marker():
    return {"step": 64, "world_size": 8, "source_revision": "model",
            "curriculum_sha256": "data"}


def test_cursor_preserves_seen_questions_when_global_batch_changes():
    first = continuation_cursor(marker(), parent(), 16384)
    assert first["optimizer_updates"] == 64
    assert first["question_cursor"] == 512
    assert first["previous_questions_per_rank"] == 1
    continued_parent = {**parent(), "pack_questions": 8, "resume_reference": "reference.json"}
    continued_marker = {**marker(), "step": 96, "question_cursor": 2560,
                        "questions_per_rank": 8}
    result = continuation_cursor(continued_marker, continued_parent, 16384)
    assert result["question_cursor"] == 512 + 32 * 64
    assert result["question_cursor"] != result["optimizer_updates"] * 64
    with pytest.raises(ValueError, match="infer"):
        continuation_cursor(marker(), continued_parent, 16384)
    with pytest.raises(ValueError, match="lineage"):
        continuation_cursor({**marker(), "curriculum_sha256": "other"}, parent(), 16384)
    with pytest.raises(ValueError, match="ownership"):
        continuation_cursor({**marker(), "world_size": 4}, parent(), 16384)


def test_independent_dcp_decoder_catches_optimizer_corruption_with_identical_model(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    # Use the same FQN-keyed state contract as the native trainer. The generic
    # optimizer.state_dict() uses integer keys that DCP's format conversion
    # serializes as strings; those are a different, unused checkpoint contract.
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    save(state, checkpoint_id=tmp_path / "checkpoint")
    dcp_to_torch_save(tmp_path / "checkpoint", tmp_path / "independent.pt")
    independent = torch.load(tmp_path / "independent.pt", weights_only=True)
    expected = state_signature(independent)
    restored = copy.deepcopy(state)
    for value in restored["model"].values():
        value.zero_()
    load(restored, checkpoint_id=tmp_path / "checkpoint")
    assert verify_materialized_state(
        restored["model"], restored["optimizer"], expected,
    )["adapter_optimizer_values_exact"]
    next(iter(restored["optimizer"]["state"].values()))["exp_avg"].add_(1)
    with pytest.raises(ValueError, match="optimizer"):
        verify_materialized_state(restored["model"], restored["optimizer"], expected)


def test_development_evaluation_does_not_change_the_next_stochastic_training_gradient():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.Dropout(.5), torch.nn.Linear(8, 2),
    )
    model.train()
    inputs = torch.ones(5, 3)
    rng = torch.get_rng_state()
    expected = torch.autograd.grad(model(inputs).sum(), tuple(model.parameters()))
    torch.set_rng_state(rng)
    with evaluation_state(model, torch.device("cpu")):
        assert not model.training
        model(inputs)
        torch.rand(41)
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    actual = torch.autograd.grad(model(inputs).sum(), tuple(model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    with pytest.raises(RuntimeError):
        with evaluation_state(model, torch.device("cpu")):
            raise RuntimeError("A failed evaluation still restores training mode.")
    assert model.training


def test_reference_materialization_rejects_accidental_base_model_gather():
    with pytest.raises(ValueError, match="large base"):
        materialize_reference_state(
            torch.empty(33 * 1024**2, device="meta"), torch.device("cpu"), keep=True,
        )
    value = {"weight": torch.ones(3), "betas": (.9, .999)}
    actual = materialize_reference_state(value, torch.device("cpu"), keep=True)
    assert state_signature(actual) == state_signature(value)


def test_lower_rate_intervention_preserves_moments_counters_parameters_and_rng():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    weights = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    result = apply_learning_rate_transition(optimizer, previous=1e-4, current=1e-5)
    expected = copy.deepcopy(before)
    expected["param_groups"][0]["lr"] = 1e-5
    assert state_signature(optimizer.state_dict()) == state_signature(expected)
    assert state_signature(model.state_dict()) == state_signature(weights)
    assert torch.equal(rng, torch.get_rng_state())
    assert not result["optimizer_moments_reset"]
    with pytest.raises(ValueError, match="unexpected learning rate"):
        apply_learning_rate_transition(optimizer, previous=1e-4, current=1e-5)
    with pytest.raises(ValueError, match="decrease"):
        apply_learning_rate_transition(optimizer, previous=1e-5, current=1e-4)
