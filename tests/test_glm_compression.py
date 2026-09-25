import copy
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from bobcat.glm_compression import (
    export_actions,
    export_checkpoint,
    footprint,
    pruned_config,
    validate_selection,
)
from bobcat.schema import file_hash, json_hash


def config():
    return {
        "text_config": {
            "num_hidden_layers": 4,
            "first_k_dense_replace": 1,
            "num_nextn_predict_layers": 1,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "hidden_size": 2,
            "vocab_size": 6,
            "layer_types": [
                "linear_attention",
                "deepseek_sparse_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
            "indexer_types": ["full"] * 4,
            "linear_attn_config": {
                "kda_layers": [0, 2],
                "full_attn_layers": [1, 3],
                "short_conv_kernel_size": 4,
                "head_dim": 2,
            },
        }
    }


def selection(layers=(1, 2, 3)):
    return {
        "schema": "bobcat-expert-selection-v1",
        "measured": True,
        "fitting_partition": "train",
        "observations_sha256": "fixture-observations",
        "source_config_sha256": json_hash(config()),
        "source_checkpoint_sha256": "fixture-original-checkpoint",
        "experts_by_layer": {str(layer): [1, 3] for layer in layers},
    }


def weights():
    tensors = {
        "lm_head.weight": torch.arange(12, dtype=torch.bfloat16).reshape(6, 2),
        "model.language_model.embed_tokens.weight": torch.ones(6, 2, dtype=torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
        "model.visual.fixture.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    for layer in range(5):  # Includes the source MTP layer.
        prefix = f"model.language_model.layers.{layer}."
        tensors[prefix + "input_layernorm.weight"] = torch.ones(2, dtype=torch.bfloat16)
        if layer == 0:
            continue
        tensors[prefix + "mlp.gate.weight"] = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
        tensors[prefix + "mlp.gate.e_score_correction_bias"] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        for expert in range(4):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = prefix + f"mlp.experts.{expert}.{projection}."
                tensors[name + "weight"] = torch.full(
                    (2, 2), layer * 4 + expert, dtype=torch.bfloat16
                )
                tensors[name + "weight_scale_inv"] = torch.full((1, 1), 10.0 + expert)
    return tensors


def test_depth_pruning_remaps_every_hybrid_list_and_preserves_original():
    original = config()
    result = pruned_config(original, [0, 2, 3], experts=2, active=1)
    text = result["text_config"]
    assert text["num_hidden_layers"] == 3
    assert text["linear_attn_config"]["kda_layers"] == [0, 1]
    assert text["linear_attn_config"]["full_attn_layers"] == [2]
    assert text["mlp_layer_types"] == ["dense", "sparse", "sparse"]
    assert text["num_nextn_predict_layers"] == 0
    assert original == config()
    unknown = copy.deepcopy(original)
    unknown["text_config"]["new_per_layer_setting"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="explicit remapping"):
        pruned_config(unknown, [0, 2, 3], experts=2, active=1)


def test_expert_router_scale_and_layer_maps_agree():
    original = weights()
    index = {"weight_map": dict.fromkeys(original, "model.safetensors")}
    actions = export_actions(index, config(), [0, 2, 3], selection((2, 3)))
    by_target = {item["target"]: item for item in actions}
    key = "model.language_model.layers.1.mlp.experts.0.gate_proj.weight"
    assert (
        by_target[key]["source"] == "model.language_model.layers.2.mlp.experts.1.gate_proj.weight"
    )
    assert by_target[key.replace(".weight", ".weight_scale_inv")]["source"].endswith(
        ".experts.1.gate_proj.weight_scale_inv"
    )
    router = by_target["model.language_model.layers.1.mlp.gate.weight"]
    bias = by_target["model.language_model.layers.1.mlp.gate.e_score_correction_bias"]
    assert router["rows"] == bias["rows"] == [1, 3]
    assert not any("visual" in item["target"] or ".layers.4." in item["source"] for item in actions)
    assert by_target["lm_head.weight"]["rows"] is None
    del index["weight_map"]["model.language_model.layers.2.mlp.experts.1.up_proj.weight_scale_inv"]
    with pytest.raises(ValueError, match="dequantization scale"):
        export_actions(index, config(), [0, 2, 3], selection((2, 3)))


def test_storage_reduction_is_not_reported_as_reduced_active_computation():
    layout = {
        "lm_head.weight": {"shape": [6, 2], "dtype": "torch.bfloat16"},
        "model.language_model.embed_tokens.weight": {"shape": [6, 2], "dtype": "torch.bfloat16"},
        "model.language_model.layers.1.mlp.experts.gate_and_up_projs": {
            "shape": [4, 2, 4],
            "dtype": "torch.bfloat16",
        },
        "model.language_model.layers.1.mlp.experts.down_projs": {
            "shape": [4, 2, 2],
            "dtype": "torch.bfloat16",
        },
        "model.language_model.layers.1.mlp.gate.weight": {
            "shape": [4, 2],
            "dtype": "torch.bfloat16",
        },
        "model.language_model.layers.1.mlp.gate.e_score_correction_bias": {
            "shape": [4],
            "dtype": "torch.float32",
        },
    }
    large = footprint(config(), layout, layers=[0, 1, 2, 3], experts=4, active=2)
    small = footprint(config(), layout, layers=[0, 1, 2, 3], experts=2, active=2)
    fewer_active = footprint(config(), layout, layers=[0, 1, 2, 3], experts=2, active=1)
    assert small["state_bytes"] < large["state_bytes"]
    assert small["routed_ffn_parameters_per_token"] == large["routed_ffn_parameters_per_token"]
    assert fewer_active["state_bytes"] == small["state_bytes"]
    assert (
        fewer_active["routed_ffn_parameters_per_token"] * 2
        == small["routed_ffn_parameters_per_token"]
    )
    assert small["measured_latency_ms"] is None
    assert small["removed_source_state_elements_by_component"]["routed_experts"] == 24


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(measured=False),
        lambda s: s.update(fitting_partition="dev_train"),
        lambda s: s.update(source_config_sha256="another-model"),
        lambda s: s["experts_by_layer"].update({"1": [1, 1]}),
        lambda s: s["experts_by_layer"].update({"1": [1, 4]}),
        lambda s: s["experts_by_layer"].pop("2"),
    ],
)
def test_pruning_refuses_unmeasured_or_misaligned_selection(change):
    selected = selection()
    change(selected)
    with pytest.raises(ValueError):
        validate_selection(config(), [0, 1, 2, 3], selected)


def test_real_safetensors_export_preserves_selected_values_and_original_tokens(tmp_path):
    original = weights()
    model_dir = tmp_path / "source"
    model_dir.mkdir()
    save_file(original, str(model_dir / "weights.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(config()))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": dict.fromkeys(original, "weights.safetensors"),
            }
        )
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"):
        (model_dir / name).write_text("exact original fixture bytes")
    source = {
        "repo": "zai-org/GLM-5.3-Flash",
        "revision": "fixture-not-a-real-model",
        "files": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)}
            for p in sorted(model_dir.iterdir())
        ],
    }
    out = tmp_path / "export"
    selected = {**selection((2, 3)), "source_checkpoint_sha256": json_hash(source)}
    result = export_checkpoint(
        model_dir,
        source,
        config(),
        selected,
        out,
        layers=[0, 2, 3],
        active=1,
        max_shard_bytes=2**20,
    )
    assert result["status"] == "weights_exported"
    assert not result["model_loaded"] and not result["release_ready"]
    index = json.loads((out / "model.safetensors.index.json").read_text())
    actions = export_actions(
        {"weight_map": dict.fromkeys(original, "weights.safetensors")},
        config(),
        [0, 2, 3],
        selection((2, 3)),
    )
    assert len(index["weight_map"]) == len(actions)
    for action in actions:
        expected = original[action["source"]]
        if action["rows"] is not None:
            expected = expected[action["rows"]]
        with safe_open(out / index["weight_map"][action["target"]], framework="pt") as handle:
            assert torch.equal(handle.get_tensor(action["target"]), expected)
    assert (out / "tokenizer.json").read_bytes() == (model_dir / "tokenizer.json").read_bytes()
    assert (out / "LICENSE").read_bytes() == (model_dir / "LICENSE").read_bytes()
    # The exporter must not alter or erase the source checkpoints.
    for item in source["files"]:
        assert file_hash(model_dir / item["path"]) == item["sha256"]
