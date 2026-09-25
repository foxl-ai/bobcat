import copy

import pytest
import torch

from bobcat.glm_fsdp_adapter_probe import local_copy
from bobcat.glm_native_rl import (
    frozen_reference,
    initialize_adam_state,
    restore_local_optimizer,
    restore_local_parameters,
)


def test_reference_swap_restores_actor_even_when_scoring_fails():
    p = torch.nn.Parameter(torch.tensor([[1., 2.]]))
    parameters = {"attention.lora_A.weight": p}
    original = local_copy(parameters)
    reference = {name: torch.zeros_like(value) for name, value in original.items()}
    identity = id(p)
    with pytest.raises(RuntimeError, match="scoring"):
        with frozen_reference(parameters, reference):
            assert torch.equal(p, reference["attention.lora_A.weight"])
            raise RuntimeError("scoring failure")
    assert id(p) == identity
    assert torch.equal(p, original["attention.lora_A.weight"])


def test_restore_rejects_different_actor_ownership_and_precision():
    p = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="ownership"):
        restore_local_parameters({"a": p}, {"b": torch.zeros(2)})
    with pytest.raises(ValueError, match="precision"):
        restore_local_parameters({"a": p}, {"a": torch.zeros(2, dtype=torch.float64)})


def test_adam_zero_initialization_and_next_update_resume():
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.AdamW([p], lr=.01, foreach=False)
    initialize_adam_state(optimizer)
    assert torch.equal(p, torch.tensor([1., 2.]))
    assert float(optimizer.state[p]["step"]) == 0

    def step():
        optimizer.zero_grad(set_to_none=True)
        p.square().sum().backward()
        optimizer.step()

    step()
    saved_p, saved_opt = p.detach().clone(), local_copy(optimizer.state_dict())
    step()
    expected_p, expected_opt = p.detach().clone(), local_copy(optimizer.state_dict())
    restore_local_parameters({"p": p}, {"p": saved_p})
    restore_local_optimizer(optimizer, saved_opt)
    step()
    assert torch.equal(p, expected_p)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(optimizer.state_dict()["state"][0][key],
                           expected_opt["state"][0][key])


def test_optimizer_resume_keeps_dtensor_ownership(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    if not dist.is_gloo_available():
        pytest.skip("CPU distributed ownership test requires Gloo.")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}",
                            rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        p = torch.nn.Parameter(distribute_tensor(torch.ones(4, 2), mesh, [Shard(0)]))
        optimizer = torch.optim.AdamW([p], lr=.01, foreach=False)
        initialize_adam_state(optimizer)

        def step():
            optimizer.zero_grad(set_to_none=True)
            p.square().sum().backward()
            optimizer.step()

        step()
        state, parameter = local_copy(optimizer.state_dict()), local_copy(p)
        step()
        expected = local_copy(p)
        restore_local_parameters({"p": p}, {"p": parameter})
        restore_local_optimizer(optimizer, copy.deepcopy(state))
        assert isinstance(optimizer.state[p]["exp_avg"], DTensor)
        assert optimizer.state[p]["exp_avg"].placements == p.placements
        step()
        assert torch.equal(local_copy(p), expected)
    finally:
        dist.destroy_process_group()
