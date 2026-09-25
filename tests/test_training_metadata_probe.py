import pytest
import torch

from bobcat.training_metadata_probe import (
    compare_metadata_backward,
    distributed_metadata_admission,
)


def test_balanced_probe_preserves_parameters_rng_mode_and_gradient_ownership():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))
    model.eval()
    values = torch.randn(4, 3)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    rng = torch.get_rng_state()
    seen = []

    def compute(enabled):
        seen.append(enabled)
        logits = model(values)
        return {"logits": logits, "loss": logits.square().mean()}

    report = compare_metadata_backward(
        model, dict(model.named_parameters()), compute, device=torch.device("cpu"),
    )
    assert seen == [False, True, False, True, True, False]
    assert report["local_numerical_gate_passed"]
    assert not report["distributed_gate_passed"]
    assert report["optimizer_updates"] == 0
    assert sum(not row["warmup"] for row in report["samples"]) == 4
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.training
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
        assert parameter.grad is None


def test_same_logits_with_changed_gradient_fails_admission():
    model = torch.nn.Linear(3, 2)
    values = torch.randn(4, 3)

    def compute(enabled):
        logits = model(values)
        # Forward values are identical but the proposed path has a wrong Jacobian.
        output = logits * 2 - logits.detach() if enabled else logits
        return {"logits": output, "loss": output.square().mean()}

    report = compare_metadata_backward(
        model, dict(model.named_parameters()), compute, device=torch.device("cpu"),
    )
    assert report["readout_exact"] and report["loss_exact"]
    assert report["different_gradient_tensors"]
    assert not report["local_numerical_gate_passed"]


def test_failure_does_not_consume_training_rng_or_leave_probe_gradients():
    model = torch.nn.Linear(3, 2).eval()
    rng = torch.get_rng_state()

    def compute(enabled):
        logits = model(torch.randn(4, 3))
        if enabled:
            raise RuntimeError("candidate failed")
        return {"logits": logits, "loss": logits.square().mean()}

    with pytest.raises(RuntimeError, match="candidate failed"):
        compare_metadata_backward(
            model, dict(model.named_parameters()), compute, device=torch.device("cpu"),
        )
    assert torch.equal(torch.get_rng_state(), rng)
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_probe_refuses_to_discard_pending_training_gradients():
    model = torch.nn.Linear(3, 2)
    model(torch.ones(1, 3)).sum().backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="pending gradient"):
        compare_metadata_backward(
            model, dict(model.named_parameters()), lambda _: None, device=torch.device("cpu"),
        )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.grad, gradients[name])


def distributed_reports():
    return [{
        "rank": rank, "local_numerical_gate_passed": True, "finite": True,
        "readout_exact": True, "loss_exact": True, "different_gradient_tensors": [],
        "adapter_weights_unchanged": True, "rng_restored": True, "optimizer_updates": 0,
        "samples": [{
            "cache_enabled": enabled, "warmup": index < 2,
            "forward_backward_seconds": (100 if index < 2 else 8 if enabled else 10),
            "optimizer_updated": False,
        } for index, enabled in enumerate((False, True, False, True, True, False))],
    } for rank in range(8)]


def test_synchronous_metadata_gate_uses_slowest_rank_and_excludes_warmup():
    reports = distributed_reports()
    result = distributed_metadata_admission(reports)
    assert result["activated"] and result["measured_speedup"] == 1.25
    reports[-1]["samples"][3]["forward_backward_seconds"] = 12
    result = distributed_metadata_admission(reports)
    assert result["all_ranks_numerical_gate_passed"]
    assert not result["activated"] and result["measured_speedup"] == 1


def test_metadata_gate_requires_all_eight_exact_unchanged_rank_results():
    reports = distributed_reports()
    reports[-1]["different_gradient_tensors"] = ["adapter"]
    assert not distributed_metadata_admission(reports)["activated"]
    with pytest.raises(ValueError, match="eight-rank"):
        distributed_metadata_admission(reports[:-1])
    reports[-1]["samples"][0]["optimizer_updated"] = True
    with pytest.raises(ValueError, match="sequence"):
        distributed_metadata_admission(reports)
