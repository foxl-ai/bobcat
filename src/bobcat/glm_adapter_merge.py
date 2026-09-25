"""Export an explicitly scoped GLM LoRA into a serving checkpoint on the CPU.

Untouched tensors are copied as bytes. Selected FP8 attention projections become
BF16, with a separate zero-adapter control using the same conversion. The default
is the original four-layer experiment; full45 must be explicitly requested.
Neither treatment compresses the backbone or proves serving equivalence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path

import torch

from bobcat.corpus import atomic_json
from bobcat.native_resume import state_signature
from bobcat.schema import file_hash, json_hash

BASE_REPO = "zai-org/GLM-5.3-Flash"
BASE_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
BLOCK = 128
CHUNK = 8 * 1024**2


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def read_header(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors header.")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 64 * 1024**2:
            raise ValueError("Unexpected safetensors header size.")
        header = json.loads(stream.read(size))
    entries = sorted(
        ((name, value) for name, value in header.items() if name != "__metadata__"),
        key=lambda row: row[1]["data_offsets"][0],
    )
    cursor = 0
    for _name, value in entries:
        start, end = value["data_offsets"]
        if (type(start) is not int or type(end) is not int or start != cursor
                or end < start):
            raise ValueError("Safetensors data must be contiguous and non-overlapping.")
        cursor = end
    if 8 + size + cursor != path.stat().st_size:
        raise ValueError("Safetensors offsets do not cover the file exactly.")
    return header, 8 + size, entries


def block_dequantize(weight, scales):
    if (weight.dtype != torch.float8_e4m3fn or weight.ndim != 2
            or scales.dtype != torch.float32
            or tuple(scales.shape) != tuple(math.ceil(n / BLOCK) for n in weight.shape)
            or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all())):
        raise ValueError("Expected original e4m3 weights and positive 128x128 FP32 scales.")
    expanded = scales.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    value = (weight.float() * expanded[:weight.shape[0], :weight.shape[1]]).to(torch.bfloat16)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Dequantization produced a non-finite base weight.")
    return value


def merge_projection(base, a, b, *, scale=2.0, control=False):
    if (base.dtype != torch.bfloat16 or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16 or base.ndim != 2 or a.ndim != 2 or b.ndim != 2
            or a.shape[1] != base.shape[1] or b.shape != (base.shape[0], a.shape[0])
            or not math.isfinite(scale) or scale <= 0
            or any(not bool(torch.isfinite(t).all()) for t in (base, a, b))):
        raise ValueError("Adapter shape, dtype, or finite-value contract changed.")
    delta = (b.float() @ a.float()) * scale
    merged = base.clone() if control else (base.float() + delta).to(torch.bfloat16)
    if not bool(torch.isfinite(merged).all()):
        raise ValueError("Merged weight contains non-finite values.")
    actual = merged.float() - base.float()
    return merged, {
        "elements": base.numel(), "changed_elements": int(torch.count_nonzero(actual)),
        "delta_fp32_l2": float(torch.linalg.vector_norm(delta)),
        "stored_delta_l2": float(torch.linalg.vector_norm(actual)),
        "maximum_stored_delta": float(actual.abs().max()),
        "control": control, "scale": scale,
        "merge_arithmetic": "BF16 base -> FP32 base + scale*(B@A) -> BF16",
        "native_two_gemm_bitwise_equivalence_claimed": False,
    }


def patch_safetensors(source, destination, replacements, *, remove=()):
    """Rewrite only named tensors; verify every other tensor by independent byte hashes."""
    if destination.exists() or destination.is_symlink() or source.is_symlink():
        raise ValueError("Use a new output shard and a regular original source.")
    header, data_start, entries = read_header(source)
    names = {name for name, _value in entries}
    remove = set(remove)
    if (not set(replacements) <= names or not remove <= names
            or set(replacements) & remove):
        raise ValueError("Patch must replace or remove existing, disjoint tensor names.")
    raw_replacements = {}
    output_header = {}
    if "__metadata__" in header:
        output_header["__metadata__"] = header["__metadata__"]
    cursor = 0
    for name, value in entries:
        if name in remove:
            continue
        item = copy.deepcopy(value)
        if name in replacements:
            tensor = replacements[name]
            if tensor.dtype != torch.bfloat16 or list(tensor.shape) != value["shape"]:
                raise ValueError("Replacement must retain shape and use BF16.")
            raw_replacements[name] = tensor_bytes(tensor)
            size = len(raw_replacements[name])
            item["dtype"] = "BF16"
        else:
            size = value["data_offsets"][1] - value["data_offsets"][0]
        item["data_offsets"] = [cursor, cursor + size]
        cursor += size
        output_header[name] = item
    raw_header = json.dumps(output_header, separators=(",", ":"), ensure_ascii=False).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    partial = destination.with_suffix(destination.suffix + ".partial")
    expected = {}
    with source.open("rb") as src, partial.open("xb") as dst:
        dst.write(struct.pack("<Q", len(raw_header)))
        dst.write(raw_header)
        for name, value in entries:
            if name in remove:
                continue
            digest = hashlib.sha256()
            if name in raw_replacements:
                raw = raw_replacements[name]
                dst.write(raw)
                digest.update(raw)
            else:
                start, end = value["data_offsets"]
                src.seek(data_start + start)
                remaining = end - start
                while remaining:
                    raw = src.read(min(CHUNK, remaining))
                    if not raw:
                        raise ValueError("Source tensor was truncated during byte copy.")
                    dst.write(raw)
                    digest.update(raw)
                    remaining -= len(raw)
            expected[name] = digest.hexdigest()
        dst.flush()
        os.fsync(dst.fileno())
    check_header, check_start, check_entries = read_header(partial)
    if check_header != output_header:
        raise ValueError("Rewritten header changed before verification.")
    with partial.open("rb") as stream:
        for name, value in check_entries:
            start, end = value["data_offsets"]
            stream.seek(check_start + start)
            digest = hashlib.sha256()
            remaining = end - start
            while remaining:
                raw = stream.read(min(CHUNK, remaining))
                if not raw:
                    raise ValueError("Output tensor was truncated during verification.")
                digest.update(raw)
                remaining -= len(raw)
            if digest.hexdigest() != expected[name]:
                raise ValueError("Fresh read of a rewritten tensor differs from its source.")
    partial.rename(destination)
    return {
        "path": destination.name, "sha256": file_hash(destination),
        "bytes": destination.stat().st_size,
        "replaced": sorted(replacements), "removed": sorted(remove),
        "unchanged_tensors_verified_exact": len(expected) - len(replacements),
        "tensor_sha256": expected, "fresh_read_verified": True,
    }


def serving_quantization_config(original, converted_names):
    result = copy.deepcopy(original)
    q = result["quantization_config"]
    if (q.get("quant_method") != "fp8" or q.get("fmt") != "e4m3"
            or q.get("activation_scheme") != "dynamic"
            or q.get("weight_block_size") != [BLOCK, BLOCK]
            or not isinstance(q.get("modules_to_not_convert"), list)):
        raise ValueError("The pinned original FP8 configuration changed.")
    converted = {name.removesuffix(".weight").replace("model.language_model.", "model.")
                 for name in converted_names}
    # SGLang fuses these two projections and requires their quantization to agree.
    for name in converted:
        if name.endswith(".q_a_proj"):
            if name.replace(".q_a_proj", ".kv_a_proj_with_mqa") not in converted:
                raise ValueError("Convert both members of the fused Q/KV projection.")
        if name.endswith(".kv_a_proj_with_mqa"):
            if name.replace(".kv_a_proj_with_mqa", ".q_a_proj") not in converted:
                raise ValueError("Convert both members of the fused Q/KV projection.")
    q["modules_to_not_convert"] = sorted(set(q["modules_to_not_convert"]) | converted)
    return result


def pinned_full_scope():
    """The original 180-target GLM layout, including its KDA/DSA differences."""
    targets = []
    for layer in range(45):
        dimensions = (
            {"q_a_proj": (4096, 1536), "q_b_proj": (1536, 16384),
             "kv_a_proj_with_mqa": (4096, 512), "o_proj": (16384, 4096)}
            if layer % 4 == 3 else
            {"q_proj": (4096, 8192), "k_proj": (4096, 8192),
             "v_proj": (4096, 8192), "o_proj": (8192, 4096)}
        )
        for leaf, (inputs, outputs) in dimensions.items():
            targets.append({
                "name": f"model.language_model.layers.{layer}.self_attn.{leaf}",
                "layer": layer, "in_features": inputs, "out_features": outputs,
                "trainable_parameters": 8 * (inputs + outputs),
            })
    return {
        "schema": "bobcat-native-attention-lora-scope-v1",
        "total_language_layers": 45, "selected_layers": list(range(45)),
        "first_trainable_layer": 0, "lora_rank": 8,
        "target_modules": [target["name"] for target in targets],
        "targets": targets,
        "trainable_parameters": sum(target["trainable_parameters"] for target in targets),
        "backbone_layers_removed": 0, "inference_speedup_measured": False,
        "backward_speedup_measured": False,
    }


def reference_scope(reference, *, scope_mode):
    """Keep legacy full-scope provenance distinct from explicit scoped metadata."""
    from bobcat.glm_adapter_scope import validate_checkpoint_adapter_scope

    parent, marker = reference["parent_job"], reference["marker"]
    if scope_mode == "scoped4":
        scope = marker.get("adapter_scope", {})
        if (parent.get("adapter_last_layers") != 4
                or scope.get("selected_layers") != [41, 42, 43, 44]
                or scope.get("trainable_parameters") != 1568768
                or scope.get("lora_rank") != 8
                or json_hash(scope) != marker.get("adapter_scope_sha256")):
            raise ValueError("Require the verified four-layer adapter scope.")
    elif scope_mode == "full45":
        scope = pinned_full_scope()
        if (parent.get("adapter_last_layers") not in (None, 45)
                or parent.get("trainable_parameters") != 17649664):
            raise ValueError("A partial adapter cannot be relabeled as the full45 model.")
        validate_checkpoint_adapter_scope(marker, scope)
    else:
        raise ValueError("Choose the explicit scoped4 or full45 export.")
    return scope


def validate_reference(reference, adapter_path, *, scope_mode="scoped4"):
    from safetensors import safe_open
    from safetensors.torch import load_file

    parent, marker = reference["parent_job"], reference["marker"]
    artifact = reference["adapter_artifact"]
    scope = reference_scope(reference, scope_mode=scope_mode)
    if (reference.get("schema") != "bobcat-native-resume-reference-v1"
            or parent["source_revision"] != BASE_REVISION
            or marker["source_revision"] != BASE_REVISION
            or json_hash(parent) != reference["parent_job_content_sha256"]
            or json_hash(reference["full_state_signature"]) != reference["full_state_sha256"]
            or artifact["scale"] != 2.0 or artifact["decoded_values_exact"] is not True
            or file_hash(adapter_path) != artifact["sha256"]
            or adapter_path.stat().st_size != artifact["bytes"]):
        raise ValueError("Require the verified, explicitly selected original-GLM adapter.")
    tensors = load_file(adapter_path, device="cpu")
    with safe_open(adapter_path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata()
    if (metadata.get("base_repo") != BASE_REPO
            or metadata.get("base_revision") != BASE_REVISION
            or metadata.get("source_complete_sha256") != reference["complete_sha256"]
            or metadata.get("scale") != "2.0"):
        raise ValueError("Adapter file metadata differs from its decoded checkpoint.")
    expected = {
        f"{target['name']}.lora_{part}.weight": (
            [8, target["in_features"]] if part == "A" else [target["out_features"], 8]
        ) for target in scope["targets"] for part in ("A", "B")
    }
    model_signature = next(
        value for key, value in reference["full_state_signature"]["items"]
        if key == {"type": "str", "value": "model"}
    )
    signatures = {key["value"]: value for key, value in model_signature["items"]}
    expected_count = 32 if scope_mode == "scoped4" else 360
    if (len(expected) != expected_count or set(tensors) != set(expected)
            or any(list(t.shape) != expected[name] or t.dtype != torch.bfloat16
                   or state_signature(t) != signatures.get(name)
                   for name, t in tensors.items())):
        raise ValueError("Consolidated adapter differs from the independent full-state decode.")
    return tensors, scope


def verify_router_biases(reference, source_root, mapping):
    """An adapter-only export must not silently omit learned router-buffer changes."""
    from safetensors import safe_open

    model_signature = next(
        value for key, value in reference["full_state_signature"]["items"]
        if key == {"type": "str", "value": "model"}
    )
    biases = {key["value"]: value for key, value in model_signature["items"]
              if key["value"].endswith(".e_score_correction_bias")}
    expected = {
        f"model.language_model.layers.{i}.mlp.gate.e_score_correction_bias"
        for i in range(3, 45)
    }
    if set(biases) != expected:
        raise ValueError("Require all 42 MoE router biases; the first three GLM layers are dense.")
    for name, signature in biases.items():
        with safe_open(source_root / mapping[name], framework="pt", device="cpu") as stream:
            original = stream.get_tensor(name)
        if state_signature(original) != signature:
            raise ValueError("Router bias changed; attention-only export would lose learned state.")
    return {"bias_tensors": len(biases), "original_source_values_exact": True}


def export(source_root, source_manifest, reference_path, adapter_path, out, *,
           control=False, scope_mode="scoped4"):
    from safetensors import safe_open

    started = datetime.now(UTC)
    source = json.loads(source_manifest.read_text())
    reference = json.loads(reference_path.read_text())
    if source["repo"] != BASE_REPO or source["revision"] != BASE_REVISION:
        raise ValueError("This export supports only the user's pinned GLM source.")
    tensors, scope = validate_reference(reference, adapter_path, scope_mode=scope_mode)
    files = {row["path"]: row for row in source["files"]}
    if len(files) != len(source["files"]):
        raise ValueError("Duplicate source manifest paths.")
    for name, row in files.items():
        path = source_root / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or path.stat().st_size != row["bytes"]):
            raise ValueError("Original source layout differs from its manifest.")
    for name in ("config.json", "model.safetensors.index.json"):
        if file_hash(source_root / name) != files[name]["sha256"]:
            raise ValueError("Original model configuration/index changed.")
    config = json.loads((source_root / "config.json").read_text())
    index = json.loads((source_root / "model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    routers = verify_router_biases(reference, source_root, mapping)
    grouped = {}
    for target in scope["targets"]:
        key = target["name"] + ".weight"
        grouped.setdefault(mapping[key], []).append(target)
    expected_shards = 4 if scope_mode == "scoped4" else 45
    if len(grouped) != expected_shards:
        raise ValueError("Original attention shard layout differs from the selected scope.")
    # Rewritten shards plus headroom; untouched source files use hardlinks.
    required = sum(files[name]["bytes"] for name in grouped) + 3 * 1024**3
    if shutil.disk_usage(out.parent).free < required:
        raise ValueError("Not enough disk space for the independently written scope shards.")
    out.mkdir(exist_ok=False)
    result = {
        "schema": ("bobcat-scoped-glm-serving-export-v1" if scope_mode == "scoped4"
                   else "bobcat-fullscope-glm-serving-export-v1"),
        "status": "preparing", "scope_mode": scope_mode,
        "started_at": started.isoformat(), "control": control,
        "source_manifest_sha256": file_hash(source_manifest),
        "reference_sha256": file_hash(reference_path), "adapter_sha256": file_hash(adapter_path),
        "base_repo": BASE_REPO, "base_revision": BASE_REVISION,
        "checkpoint_step": reference["marker"]["step"],
        "frozen_router_verification": routers,
        "adapter_scope_sha256": json_hash(scope),
        "adapter_scope": scope,
        "legacy_full_scope_reference": "adapter_scope" not in reference["marker"],
        "source_code_sha256": file_hash(Path(__file__)),
        "gpu_used": False, "training_modified": False, "source_weights_modified": False,
        "quality_verified": False, "native_forward_equivalence_claimed": False,
        "backbone_compressed": False, "release_model_complete": False,
        "source_verification_scope": (
            "fresh SHA256 for rewritten source shards/config/index; untouched shards use "
            "original manifest hashes and read-only hardlinks, requiring serving preflight SHA256"
        ),
        "shards": [], "projections": [], "files": [], "converted_fp8_projections": [],
    }
    atomic_json(out / "export-progress.json", result)
    try:
        changed = set(grouped) | {"config.json", "model.safetensors.index.json"}
        for name, row in files.items():
            if name not in changed:
                os.link(source_root / name, out / name)
                if (out / name).stat().st_ino != (source_root / name).stat().st_ino:
                    raise ValueError("Untouched source hardlink did not preserve the inode.")
                result["files"].append({**row, "storage": "original_read_only_hardlink"})
        for shard, targets in grouped.items():
            path = source_root / shard
            if file_hash(path) != files[shard]["sha256"]:
                raise ValueError("A target source shard changed.")
            replacement, remove = {}, []
            with safe_open(path, framework="pt", device="cpu") as stream:
                for target in targets:
                    key = target["name"] + ".weight"
                    base = stream.get_tensor(key)
                    if list(base.shape) != [target["out_features"], target["in_features"]]:
                        raise ValueError("Source attention shape differs from the adapter scope.")
                    was_fp8 = base.dtype == torch.float8_e4m3fn
                    if was_fp8:
                        scale_name = key + "_scale_inv"
                        if mapping.get(scale_name) != shard:
                            raise ValueError("FP8 scale is not in the expected source shard.")
                        base = block_dequantize(base, stream.get_tensor(scale_name))
                        remove.append(scale_name)
                        result["converted_fp8_projections"].append(key)
                    elif base.dtype != torch.bfloat16:
                        raise ValueError("Unexpected original attention precision.")
                    value, stats = merge_projection(
                        base, tensors[target["name"] + ".lora_A.weight"],
                        tensors[target["name"] + ".lora_B.weight"], control=control,
                    )
                    if not control or was_fp8:
                        replacement[key] = value
                    result["projections"].append({"name": key, **stats})
            if replacement:
                patched = patch_safetensors(path, out / shard, replacement, remove=remove)
                storage = "rewritten_and_fresh_read_verified"
            else:
                os.link(path, out / shard)
                patched = {
                    "path": shard, "bytes": files[shard]["bytes"],
                    "sha256": files[shard]["sha256"], "replaced": [], "removed": [],
                    "source_fresh_sha256_verified": True, "tensor_sha256": {},
                }
                storage = "original_read_only_hardlink"
            result["shards"].append({**patched, "source_sha256": files[shard]["sha256"]})
            result["files"].append({
                "path": shard, "bytes": patched["bytes"], "sha256": patched["sha256"],
                "storage": storage,
            })
            for key in remove:
                del mapping[key]
            atomic_json(out / "export-progress.json", result)
        converted = result["converted_fp8_projections"]
        expected_converted = {
            target["name"] + ".weight" for target in scope["targets"]
            if target["layer"] % 4 == 3
        }
        if set(converted) != expected_converted or len(converted) != len(expected_converted):
            raise ValueError("FP8 attention conversions differ from the pinned DSA layers.")
        config = serving_quantization_config(config, converted)
        # Safetensors metadata.total_size counts tensor bytes, not JSON/header bytes.
        change = sum(
            (row["out_features"] * row["in_features"]
             - 4 * math.ceil(row["out_features"] / BLOCK)
             * math.ceil(row["in_features"] / BLOCK))
            for row in scope["targets"] if row["name"] + ".weight" in converted
        )
        index["metadata"]["total_size"] += change
        for name, value in (("config.json", config), ("model.safetensors.index.json", index)):
            atomic_json(out / name, value)
            result["files"].append({
                "path": name, "bytes": (out / name).stat().st_size,
                "sha256": file_hash(out / name), "storage": "rewritten_metadata",
            })
        if set(files) != {row["path"] for row in result["files"]}:
            raise ValueError("Export lost a source checkpoint file.")
        result.update(
            status="completed", finished_at=datetime.now(UTC).isoformat(),
            actual_additional_bytes=sum(
                row["bytes"] for row in result["files"]
                if row["storage"] != "original_read_only_hardlink"
            ),
            all_untouched_tensors_in_rewritten_shards_exact=True,
        )
        atomic_json(out / "export-manifest.json", result)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error),
                      finished_at=datetime.now(UTC).isoformat())
        raise
    finally:
        atomic_json(out / "export-progress.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--scope", choices=["scoped4", "full45"], default="scoped4")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = export(args.source_root, args.source_manifest, args.reference, args.adapter, args.out,
                    control=args.control, scope_mode=args.scope)
    print(json.dumps({k: result[k] for k in (
        "status", "control", "checkpoint_step", "actual_additional_bytes",
        "all_untouched_tensors_in_rewritten_shards_exact", "quality_verified",
    )}))


if __name__ == "__main__":
    main()
