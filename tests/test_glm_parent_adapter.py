import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from bobcat.glm_parent_adapter import parent_adapter_bindings


def actor():
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.lora_A = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    model.layer.lora_B = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
    return model


def test_actual_checkpoint_wrapper_names_bind_without_changing_values():
    model = actor()
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert set(parent_adapter_bindings(model, saved)) == set(saved)
    model.layer = checkpoint_wrapper(model.layer)
    assert set(dict(model.named_parameters())) != set(saved)
    bindings = parent_adapter_bindings(model, saved)
    assert set(bindings) == set(dict(model.named_parameters()))
    assert bindings["layer._checkpoint_wrapped_module.lora_A.weight"] is saved[
        "layer.lora_A.weight"
    ]
    with pytest.raises(ValueError, match="membership"):
        parent_adapter_bindings(model, dict(list(saved.items())[:1]))
    bad = {**saved, "layer.lora_A.weight": saved["layer.lora_A.weight"].float()}
    with pytest.raises(ValueError, match="dtype"):
        parent_adapter_bindings(model, bad)


def test_a_named_submodule_is_not_allowed_to_impersonate_checkpoint_wrapper():
    model = nn.Module()
    model._checkpoint_wrapped_module = actor()
    saved = {n: p.detach().clone() for n, p in model._checkpoint_wrapped_module.named_parameters()}
    with pytest.raises(ValueError, match="real checkpoint wrapper"):
        parent_adapter_bindings(model, saved)
