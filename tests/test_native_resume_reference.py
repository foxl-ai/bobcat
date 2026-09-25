import copy

import pytest
import torch

from bobcat.native_resume_reference import validate_decoded_adapters, validate_decoder_runtime
from bobcat.schema import json_hash


def fixture():
    target = {
        "name": "model.language_model.layers.44.self_attn.q_proj",
        "in_features": 32, "out_features": 64,
    }
    scope = {
        "schema": "bobcat-native-attention-lora-scope-v1",
        "total_language_layers": 45, "selected_layers": [44], "lora_rank": 8,
        "target_modules": [target["name"]], "targets": [target],
        "trainable_parameters": 768,
    }
    marker = {"step": 32, "adapter_scope": scope, "adapter_scope_sha256": json_hash(scope)}
    parent = {"adapter_last_layers": 1, "trainable_parameters": 768}
    state = {
        "model": {
            target["name"] + ".lora_A.weight": torch.ones(8, 32, dtype=torch.bfloat16),
            target["name"] + ".lora_B.weight": torch.ones(64, 8, dtype=torch.bfloat16),
        },
        "optimizer": {"state": {0: {"step": 32}, 1: {"step": 32}}},
    }
    return state, marker, parent


def test_narrow_scope_is_not_mistaken_for_a_missing_full_model_checkpoint():
    args = fixture()
    tensors = validate_decoded_adapters(*args)
    assert len(tensors) == 2 and sum(t.numel() for t in tensors.values()) == 768


@pytest.mark.parametrize("bad", ["scope", "tensor_name", "tensor_shape", "nan", "step", "count"])
def test_narrow_consolidation_rejects_wrong_ownership_or_state(bad):
    state, marker, parent = copy.deepcopy(fixture())
    names = list(state["model"])
    if bad == "scope":
        parent["adapter_last_layers"] = 4
    elif bad == "tensor_name":
        state["model"][names[0].replace(".44.", ".43.")] = state["model"].pop(names[0])
    elif bad == "tensor_shape":
        state["model"][names[0]] = state["model"][names[0]].T
    elif bad == "nan":
        state["model"][names[0]][0, 0] = torch.nan
    elif bad == "step":
        state["optimizer"]["state"][0]["step"] = 31
    else:
        parent["trainable_parameters"] = 769
    with pytest.raises(ValueError):
        validate_decoded_adapters(state, marker, parent)


def test_dcp_decode_uses_matching_python_but_allows_matching_torch_cpu_wheel():
    runtime = {"python": [3, 13], "torch": "2.12.1+cu130"}
    validate_decoder_runtime(runtime, python_version=(3, 13, 15), torch_version="2.12.1")
    for python, torch_release in (((3, 12, 14), "2.12.1"), ((3, 13, 15), "2.11.0")):
        with pytest.raises(ValueError, match="isolated matching CPU environment"):
            validate_decoder_runtime(
                runtime, python_version=python, torch_version=torch_release,
            )
