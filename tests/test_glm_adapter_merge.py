import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from bobcat.glm_adapter_merge import (
    block_dequantize,
    merge_projection,
    patch_safetensors,
    pinned_full_scope,
    read_header,
    reference_scope,
    serving_quantization_config,
    tensor_bytes,
    validate_reference,
    verify_router_biases,
)
from bobcat.native_resume import state_signature


def test_merge_uses_native_orientation_and_scale():
    base = torch.zeros(3, 4, dtype=torch.bfloat16)
    a = torch.tensor([[1, 2, 0, -1], [0, 1, 2, 0]], dtype=torch.bfloat16)
    b = torch.tensor([[1, 0], [0, 2], [1, -1]], dtype=torch.bfloat16)
    merged, report = merge_projection(base, a, b)
    x = torch.tensor([[1., 0., 2., 3.]])
    torch.testing.assert_close(x @ merged.float().T, 2 * (x @ a.float().T) @ b.float().T)
    assert report["changed_elements"] > 0
    control, report = merge_projection(base, a, b, control=True)
    assert torch.equal(control, base)
    assert report["changed_elements"] == 0
    assert not report["native_two_gemm_bitwise_equivalence_claimed"]


def test_fp8_block_edges_use_original_scale_multiply_and_bf16_rounding():
    weight = torch.ones(130, 129).to(torch.float8_e4m3fn)
    scale = torch.tensor([[.25, 2.], [4., 8.]])
    actual = block_dequantize(weight, scale)
    assert actual.dtype == torch.bfloat16
    assert actual[127, 127] == .25
    assert actual[127, 128] == 2
    assert actual[128, 127] == 4
    assert actual[129, 128] == 8
    with pytest.raises(ValueError, match="scales"):
        block_dequantize(weight, torch.ones(1, 1))
    with pytest.raises(ValueError, match="scales"):
        block_dequantize(weight, -scale)


def test_patch_preserves_unrelated_fp8_and_integer_tensors(tmp_path):
    sf = pytest.importorskip("safetensors.torch")
    source = tmp_path / "original.safetensors"
    destination = tmp_path / "patched.safetensors"
    original = {
        "expert.weight": torch.arange(24).reshape(4, 6).to(torch.float8_e4m3fn),
        "attention.weight": torch.ones(4, 6).to(torch.float8_e4m3fn),
        "attention.weight_scale_inv": torch.ones(1, 1),
        "counter": torch.tensor([9, 7], dtype=torch.int64),
    }
    sf.save_file(original, source, metadata={"format": "pt"})
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    replacement = torch.full((4, 6), 3.5, dtype=torch.bfloat16)
    result = patch_safetensors(
        source, destination, {"attention.weight": replacement},
        remove=["attention.weight_scale_inv"],
    )
    loaded = sf.load_file(destination)
    assert set(loaded) == set(original) - {"attention.weight_scale_inv"}
    for name in ("expert.weight", "counter"):
        assert tensor_bytes(loaded[name]) == tensor_bytes(original[name])
    assert torch.equal(loaded["attention.weight"], replacement)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert result["unchanged_tensors_verified_exact"] == 2
    assert result["fresh_read_verified"]
    assert read_header(destination)[0]["__metadata__"] == {"format": "pt"}
    with pytest.raises(ValueError, match="new output"):
        patch_safetensors(source, destination, {"attention.weight": replacement})


def test_patch_rejects_shape_change_and_offset_corruption(tmp_path):
    sf = pytest.importorskip("safetensors.torch")
    source = tmp_path / "original.safetensors"
    sf.save_file({"weight": torch.ones(4, 6, dtype=torch.bfloat16)}, source)
    with pytest.raises(ValueError, match="retain shape"):
        patch_safetensors(source, tmp_path / "new.safetensors",
                          {"weight": torch.ones(3, 6, dtype=torch.bfloat16)})
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ValueError, match="cover the file"):
        read_header(source)


def test_serving_precision_config_uses_normalized_fused_names_without_mutating_source():
    original = {"quantization_config": {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"],
    }}
    before = copy.deepcopy(original)
    q = "model.language_model.layers.43.self_attn.q_a_proj.weight"
    kv = q.replace("q_a_proj", "kv_a_proj_with_mqa")
    result = serving_quantization_config(original, [q, kv])
    assert original == before
    assert set(result["quantization_config"]["modules_to_not_convert"]) == {
        "lm_head", "model.layers.43.self_attn.q_a_proj",
        "model.layers.43.self_attn.kv_a_proj_with_mqa",
    }
    with pytest.raises(ValueError, match="both members"):
        serving_quantization_config(original, [q])


def test_real_recovered_adapter_provenance_when_available():
    pytest.importorskip("safetensors")
    root = Path("artifacts/glm-scoped-adapter-export-v1")
    if not root.exists():
        pytest.skip("Optional independently recovered real adapter evidence is not installed.")
    reference = json.loads((root / "reference-000032.json").read_text())
    tensors, scope = validate_reference(reference, root / "adapter-000032.safetensors")
    assert len(tensors) == 32
    assert scope["selected_layers"] == [41, 42, 43, 44]
    bad = copy.deepcopy(reference)
    bad["adapter_artifact"]["scale"] = 1.
    with pytest.raises(ValueError, match="verified"):
        validate_reference(bad, root / "adapter-000032.safetensors")


def test_export_rejects_changed_router_buffer_even_when_adapters_are_valid(tmp_path):
    sf = pytest.importorskip("safetensors.torch")
    biases = {
        f"model.language_model.layers.{i}.mlp.gate.e_score_correction_bias":
        torch.arange(4, dtype=torch.float32) + i
        for i in range(3, 45)
    }
    sf.save_file(biases, tmp_path / "router.safetensors")
    reference = {"full_state_signature": state_signature({"model": biases})}
    mapping = {name: "router.safetensors" for name in biases}
    assert verify_router_biases(reference, tmp_path, mapping) == {
        "bias_tensors": 42, "original_source_values_exact": True,
    }
    biases[next(iter(biases))][0] += 1
    reference["full_state_signature"] = state_signature({"model": biases})
    with pytest.raises(ValueError, match="lose learned state"):
        verify_router_biases(reference, tmp_path, mapping)


def test_real_fullscope96_export_preserves_all_recovered_adapter_values():
    pytest.importorskip("safetensors")
    root = Path("artifacts/glm-fullscope-adapter-export-step96-v1")
    if not root.exists():
        pytest.skip("The independently recovered fullscope96 artifact is not installed.")
    reference = json.loads((root / "reference-000096.json").read_text())
    tensors, scope = validate_reference(
        reference, root / "adapter-000096.safetensors", scope_mode="full45",
    )
    assert len(tensors) == 360
    assert sum(t.numel() for t in tensors.values()) == 17649664
    assert scope["selected_layers"] == list(range(45))
    # Omitting the opt-in must retain the original four-layer contract.
    with pytest.raises(ValueError, match="four-layer"):
        validate_reference(reference, root / "adapter-000096.safetensors")
    wrong = copy.deepcopy(reference)
    wrong["parent_job"]["trainable_parameters"] = 1568768
    with pytest.raises(ValueError, match="partial"):
        reference_scope(wrong, scope_mode="full45")


def test_recomputed_outer_hash_cannot_hide_a_changed_adapter_signature():
    pytest.importorskip("safetensors")
    from bobcat.schema import json_hash

    root = Path("artifacts/glm-fullscope-adapter-export-step96-v1")
    if not root.exists():
        pytest.skip("The independently recovered fullscope96 artifact is not installed.")
    reference = json.loads((root / "reference-000096.json").read_text())
    model = next(
        value for key, value in reference["full_state_signature"]["items"]
        if key == {"type": "str", "value": "model"}
    )
    # An authentic model-state signature must agree with actual tensor bytes,
    # even if someone recomputes the wrapper's JSON digest.
    key, _value = next(
        item for item in model["items"] if "lora_A.weight" in item[0]["value"]
    )
    model["items"] = [
        (k, {"type": "invalid"}) if k == key else (k, v) for k, v in model["items"]
    ]
    reference["full_state_sha256"] = json_hash(reference["full_state_signature"])
    with pytest.raises(ValueError, match="independent full-state"):
        validate_reference(reference, root / "adapter-000096.safetensors", scope_mode="full45")


def test_fullscope_precision_control_covers_all_dsa_layers_and_retains_kda():
    scope = pinned_full_scope()
    converted = [
        t["name"] + ".weight" for t in scope["targets"] if t["layer"] % 4 == 3
    ]
    config = {"quantization_config": {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"],
    }}
    actual = serving_quantization_config(config, converted)
    excluded = actual["quantization_config"]["modules_to_not_convert"]
    assert len(converted) == 44
    assert len(excluded) == 45
    assert "model.layers.3.self_attn.q_a_proj" in excluded
    assert "model.layers.43.self_attn.kv_a_proj_with_mqa" in excluded
    assert "model.layers.44.self_attn.q_proj" not in excluded
