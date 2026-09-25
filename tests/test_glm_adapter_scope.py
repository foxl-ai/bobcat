from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bobcat.glm_adapter_scope import (
    plan_attention_lora_scope,
    validate_checkpoint_adapter_scope,
)
from bobcat.schema import json_hash


class NativeLayout(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleDict()
        for index in range(4):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(12, 8)
            layer.self_attn.o_proj = nn.Linear(8, 12)
            layer.self_attn.indexer = nn.Module()
            layer.self_attn.indexer.q_proj = nn.Linear(12, 6)
            layer.mlp = nn.Linear(12, 12)
            self.model.language_model.layers[str(index)] = layer
        self.lm_head = nn.Linear(12, 31)


def test_suffix_scope_preserves_model_and_excludes_indexer_head_and_mlp():
    model = NativeLayout()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    scope = plan_attention_lora_scope(model, last_layers=2, rank=4)
    assert scope["selected_layers"] == [2, 3]
    assert len(scope["target_modules"]) == 4
    assert scope["trainable_parameters"] == 4 * 4 * (12 + 8)
    assert all(".indexer." not in name and ".mlp." not in name
               for name in scope["target_modules"])
    assert scope["backbone_layers_removed"] == 0
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_default_scope_covers_all_layers():
    scope = plan_attention_lora_scope(NativeLayout())
    assert scope["selected_layers"] == [0, 1, 2, 3]
    assert len(scope["target_modules"]) == 8


@pytest.mark.parametrize("count", [0, -1, 5, True, 1.5])
def test_rejects_invalid_scope_instead_of_silently_training_all_layers(count):
    with pytest.raises(ValueError):
        plan_attention_lora_scope(NativeLayout(), last_layers=count)


def test_rejects_unsupported_or_incomplete_stack():
    with pytest.raises(ValueError):
        plan_attention_lora_scope(SimpleNamespace())
    model = NativeLayout()
    del model.model.language_model.layers["2"]
    with pytest.raises(ValueError):
        plan_attention_lora_scope(model)


def test_rejects_already_wrapped_adapters():
    model = NativeLayout()
    model.model.language_model.layers["0"].self_attn.q_proj.lora_A = nn.Linear(12, 2)
    with pytest.raises(ValueError):
        plan_attention_lora_scope(model, last_layers=1)


def test_checkpoint_scope_must_match_targets_rank_and_layers_before_restore():
    scope = plan_attention_lora_scope(NativeLayout(), last_layers=2, rank=4)
    marker = {"adapter_scope": scope, "adapter_scope_sha256": json_hash(scope)}
    assert validate_checkpoint_adapter_scope(marker, scope)["legacy_full_scope"] is False
    for other in (
        plan_attention_lora_scope(NativeLayout(), last_layers=1, rank=4),
        plan_attention_lora_scope(NativeLayout(), last_layers=2, rank=8),
        {**scope, "target_modules": list(reversed(scope["target_modules"]))},
    ):
        with pytest.raises(ValueError, match="scope differs"):
            validate_checkpoint_adapter_scope(marker, other)


def test_reduced_scope_cannot_silently_accept_legacy_or_incomplete_marker():
    scope = plan_attention_lora_scope(NativeLayout(), last_layers=1)
    with pytest.raises(ValueError, match="unscoped"):
        validate_checkpoint_adapter_scope({}, scope)
    with pytest.raises(ValueError, match="scope differs"):
        validate_checkpoint_adapter_scope({"adapter_scope": scope}, scope)
    with pytest.raises(ValueError, match="scope differs"):
        validate_checkpoint_adapter_scope({
            "adapter_scope": scope, "adapter_scope_sha256": "0" * 64,
        }, scope)
