"""Auditable GLM-5.3-Flash expert/depth pruning, without changing token identities.

This module plans native parameter footprints and exports selected *original*
HF tensors. It never selects experts from their index or pretends a smaller
adapter is a smaller backbone. Selection requires a separately measured report.
The original vocabulary head is retained: its compact replacement has not
passed Bobcat's full-model numerical tests.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash, json_hash

SCHEMA = "bobcat-glm-compression-v1"
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
EXPERT = re.compile(
    r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float32": 4}


def validate_layers(config: dict, layers: list[int]) -> None:
    text = config["text_config"]
    depth, dense = text["num_hidden_layers"], text["first_k_dense_replace"]
    if (
        not layers
        or any(type(i) is not int for i in layers)
        or layers != sorted(set(layers))
        or layers[-1] >= depth
        or layers[0] < 0
        or layers[:dense] != list(range(dense))
        or len(layers) <= dense
    ):
        raise ValueError(
            "Keep ordered original layers, all initial dense layers, and some MoE layers."
        )
    for field in ("layer_types", "mlp_layer_types", "indexer_types"):
        if len(text[field]) != depth:
            raise ValueError(f"Malformed per-layer configuration: {field}")
    if text["n_group"] != 1 or text["topk_group"] != 1:
        raise ValueError(
            "Grouped routers require a separate expert-group remapping implementation."
        )


def pruned_config(config: dict, layers: list[int], experts: int, active: int) -> dict:
    validate_layers(config, layers)
    source = config["text_config"]
    if (
        type(experts) is not int
        or type(active) is not int
        or not 1 <= active <= experts <= source["n_routed_experts"]
    ):
        raise ValueError("Invalid retained or active expert count.")
    result = copy.deepcopy(config)
    text = result["text_config"]
    old_depth = text["num_hidden_layers"]
    remappable = {"layer_types", "mlp_layer_types", "indexer_types"}
    for key, value in text.items():
        if isinstance(value, list) and len(value) == old_depth and key not in remappable:
            raise ValueError(f"New layer-indexed field needs explicit remapping: {key}")
    for field in remappable:
        text[field] = [source[field][i] for i in layers]
    text["num_hidden_layers"] = len(layers)
    text["n_routed_experts"] = experts
    text["num_experts_per_tok"] = active
    text["num_nextn_predict_layers"] = 0
    text["linear_attn_config"]["kda_layers"] = [
        i for i, kind in enumerate(text["layer_types"]) if kind == "linear_attention"
    ]
    text["linear_attn_config"]["full_attn_layers"] = [
        i for i, kind in enumerate(text["layer_types"]) if kind == "deepseek_sparse_attention"
    ]
    if len(text["linear_attn_config"]["kda_layers"]) + len(
        text["linear_attn_config"]["full_attn_layers"]
    ) != len(layers):
        raise ValueError("An unknown attention layer would lose its hybrid-state configuration.")
    return result


def category(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name == "model.language_model.embed_tokens.weight":
        return "input_embedding"
    if name == "lm_head.weight":
        return "original_vocabulary_head"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.shared_experts." in name:
        return "shared_experts"
    if name.endswith(".mlp.gate.e_score_correction_bias"):
        return "router_buffers"
    if ".mlp.gate." in name:
        return "router"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "dense_mlp"
    if name.startswith("model.language_model."):
        return "norm_mhc"
    raise ValueError(f"Unrecognized native state tensor: {name}")


def footprint(config: dict, layout: dict, *, layers: list[int], experts: int, active: int) -> dict:
    """Exact shape arithmetic, NOT measured VRAM, throughput, or active FLOPs."""
    derived = pruned_config(config, layers, experts, active)
    counts, byte_counts, omitted = Counter(), Counter(), Counter()
    source_experts = config["text_config"]["n_routed_experts"]
    kept = set(layers)
    for name, item in layout.items():
        kind = category(name)
        shape, dtype = list(item["shape"]), item["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError("The native layout contains an unaccounted dtype.")
        match = LAYER.fullmatch(name)
        if kind == "vision" or (match and int(match[1]) not in kept):
            omitted[kind] += math.prod(shape)
            continue
        if kind in ("routed_experts", "router", "router_buffers"):
            if shape[0] != source_experts:
                raise ValueError("Expert and router row layouts do not agree.")
            before = math.prod(shape)
            shape[0] = experts
            omitted[kind] += before - math.prod(shape)
        count = math.prod(shape)
        counts[kind] += count
        byte_counts[kind] += count * DTYPE_BYTES[dtype]
    if counts["input_embedding"] == 0 or counts["original_vocabulary_head"] == 0:
        raise ValueError("Both original embedding and vocabulary head must be accounted for.")
    weights = sum(byte_counts.values())
    params = sum(counts.values()) - counts["router_buffers"]
    return {
        "source_layers": layers,
        "layers": len(layers),
        "retained_experts_per_layer": experts,
        "active_experts_per_token": active,
        "parameters": params,
        "state_bytes": weights,
        "state_gib": weights / 2**30,
        "parameters_by_component": dict(counts),
        "state_bytes_by_component": dict(byte_counts),
        # These two counts describe FFN matrix weights touched by a token.
        # Attention sequence work, embeddings, norms, communication, activation
        # memory, softmax and kernel effects are not FLOP-accounted here.
        "routed_ffn_parameters_per_token": counts["routed_experts"] // experts * active,
        "shared_plus_dense_ffn_parameters_per_token": (
            counts["shared_experts"] + counts["dense_mlp"]
        ),
        "removed_source_state_elements_by_component": dict(omitted),
        "derived_config_sha256": json_hash(derived),
        "original_vocabulary_head_retained": True,
        "vision_removed": True,
        "memory_is_tensor_payload_only": True,
        "measured_latency_ms": None,
        "quality_evaluated": False,
        "weights_exported": False,
    }


def validate_selection(config: dict, layers: list[int], selection: dict) -> dict[int, list[int]]:
    validate_layers(config, layers)
    if (
        selection.get("schema") != "bobcat-expert-selection-v1"
        or selection.get("measured") is not True
        or selection.get("fitting_partition") != "train"
        or not selection.get("observations_sha256")
        or not selection.get("source_checkpoint_sha256")
        or selection.get("source_config_sha256") != json_hash(config)
    ):
        raise ValueError(
            "Pruning needs attributable train-only expert measurements for this config."
        )
    total = config["text_config"]["n_routed_experts"]
    dense = config["text_config"]["first_k_dense_replace"]
    expected = {layer for layer in layers if layer >= dense}
    supplied = {int(key): value for key, value in selection["experts_by_layer"].items()}
    if set(supplied) != expected:
        raise ValueError("Every retained sparse layer needs exactly one expert selection.")
    for ids in supplied.values():
        if (
            not ids
            or any(type(i) is not int for i in ids)
            or ids != sorted(set(ids))
            or ids[0] < 0
            or ids[-1] >= total
        ):
            raise ValueError("Expert selections must be sorted, unique original IDs.")
    if len({len(ids) for ids in supplied.values()}) != 1:
        raise ValueError("The current GLM config requires a uniform retained expert count.")
    return supplied


def export_actions(index: dict, config: dict, layers: list[int], selection: dict) -> list[dict]:
    """Map original FP8 weights AND their scale blocks; slice both router tensors."""
    experts = validate_selection(config, layers, selection)
    layer_map = {old: new for new, old in enumerate(layers)}
    expert_maps = {
        layer: {old: new for new, old in enumerate(ids)} for layer, ids in experts.items()
    }
    depth = config["text_config"]["num_hidden_layers"]
    mtp = config["text_config"]["num_nextn_predict_layers"]
    actions, target_names, inventory = [], set(), defaultdict(set)
    for name, shard in sorted(index["weight_map"].items()):
        if name.startswith("model.visual."):
            continue
        match = LAYER.fullmatch(name)
        rows = None
        if match:
            layer, tail = int(match[1]), match[2]
            if layer >= depth + mtp:
                raise ValueError("Unexpected source layer outside the text/MTP configuration.")
            if layer not in layer_map:
                continue
            expert = EXPERT.fullmatch(tail)
            if expert:
                old_expert = int(expert[1])
                if old_expert >= config["text_config"]["n_routed_experts"]:
                    raise ValueError("Unexpected expert ID in the source index.")
                if old_expert not in expert_maps[layer]:
                    continue
                inventory[layer, old_expert].add(f"{expert[2]}.{expert[3]}")
                tail = f"mlp.experts.{expert_maps[layer][old_expert]}.{expert[2]}.{expert[3]}"
            elif tail.startswith("mlp.experts."):
                raise ValueError(f"Unsupported expert tensor requires an explicit mapping: {tail}")
            elif tail in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias"):
                rows = experts[layer]
            elif tail.startswith("mlp.gate."):
                raise ValueError(
                    "Quantized or extended router needs an explicit row/scale mapping."
                )
            target = f"model.language_model.layers.{layer_map[layer]}.{tail}"
        elif name in (
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
        ):
            target = name
        else:
            raise ValueError(f"Unmapped source tensor: {name}")
        if target in target_names:
            raise ValueError("Two source tensors would overwrite the same destination.")
        target_names.add(target)
        actions.append({"source": name, "target": target, "shard": shard, "rows": rows})
    for layer, ids in experts.items():
        for expert in ids:
            present = inventory[layer, expert]
            required = {
                f"{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")
            }
            if not required <= present:
                raise ValueError("A selected expert is missing a projection.")
            scales = {
                f"{projection}.weight_scale_inv"
                for projection in ("gate_proj", "up_proj", "down_proj")
            }
            if present & scales and not scales <= present:
                raise ValueError("A selected FP8 expert is missing a dequantization scale.")
        for tail in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias"):
            if f"model.language_model.layers.{layer_map[layer]}.{tail}" not in target_names:
                raise ValueError("Pruned router weights and correction bias must travel together.")
    return actions


def export_checkpoint(
    model_dir: Path,
    source: dict,
    config: dict,
    selection: dict,
    out: Path,
    *,
    layers: list[int],
    active: int,
    max_shard_bytes: int = 2 * 2**30,
) -> dict:
    """Bounded CPU export of actual selected weights; no model is instantiated.

    The resulting checkpoint is explicitly a text-only research derivative.
    Loading requires removing the unused vision module from the native factory.
    The complete original source shards are hashed before reading; a failure
    leaves its partial output for diagnosis and never creates a completion manifest.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if out.exists() or source.get("repo") != "zai-org/GLM-5.3-Flash" or max_shard_bytes < 2**20:
        raise ValueError(
            "Use the designated GLM source, a new output path, and a bounded shard size."
        )
    if selection.get("source_checkpoint_sha256") != json_hash(source):
        raise ValueError("The expert observations belong to another source checkpoint.")
    items = {item["path"]: item for item in source["files"]}
    for name in ("config.json", "model.safetensors.index.json"):
        if file_hash(model_dir / name) != items[name]["sha256"]:
            raise ValueError(f"Source metadata failed its pinned checksum: {name}")
    disk_config = json.loads((model_dir / "config.json").read_text())
    if disk_config != config:
        raise ValueError("The selection/config does not describe the original checkpoint.")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    actions = export_actions(index, config, layers, selection)
    retained = len(next(iter(selection["experts_by_layer"].values())))
    derived_config = pruned_config(config, layers, retained, active)
    out.mkdir(parents=True)
    pending, pending_bytes, files, weight_map = {}, 0, [], {}
    source_verified, output_bytes, tensor_count = [], 0, 0
    groups = defaultdict(list)
    for action in actions:
        groups[action["shard"]].append(action)

    def flush():
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f"model-{len(files) + 1:05d}.safetensors"
        path = out / filename
        save_file(pending, str(path), metadata={"format": "pt"})
        for key in pending:
            weight_map[key] = filename
        files.append({"path": filename, "bytes": path.stat().st_size, "sha256": file_hash(path)})
        pending, pending_bytes = {}, 0

    for shard, shard_actions in sorted(groups.items()):
        path = model_dir / shard
        expected = items.get(shard)
        if (
            expected is None
            or path.is_symlink()
            or path.stat().st_size != expected["bytes"]
            or file_hash(path) != expected["sha256"]
        ):
            raise ValueError(f"Source shard failed its pinned checksum: {shard}")
        source_verified.append({"path": shard, "sha256": expected["sha256"]})
        with safe_open(path, framework="pt", device="cpu") as handle:
            for action in shard_actions:
                tensor = handle.get_tensor(action["source"])
                if action["rows"] is not None:
                    if tensor.shape[0] != config["text_config"]["n_routed_experts"]:
                        raise ValueError("Router rows do not match the measured expert vocabulary.")
                    tensor = tensor[action["rows"]]
                # Copy only retained tensors; do not keep a view backed by an entire
                # mmap source shard in the output buffer.
                tensor = tensor.contiguous().clone()
                size = tensor.numel() * tensor.element_size()
                if size > max_shard_bytes:
                    raise ValueError("A single tensor exceeds the explicit output shard size.")
                if pending_bytes + size > max_shard_bytes:
                    flush()
                pending[action["target"]] = tensor
                pending_bytes += size
                output_bytes += size
                tensor_count += 1
        flush()
        atomic_json(
            out / "progress.json",
            {
                "status": "exporting",
                "source_shards_verified": len(source_verified),
                "tensors_written": len(weight_map),
                "weights_tensor_bytes": output_bytes,
            },
        )
    flush()
    atomic_json(out / "config.json", derived_config)
    atomic_json(
        out / "model.safetensors.index.json",
        {
            "metadata": {"total_size": output_bytes},
            "weight_map": weight_map,
        },
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"):
        if name not in items:
            raise ValueError(f"Required original tokenizer artifact missing: {name}")
        path = model_dir / name
        if file_hash(path) != items[name]["sha256"]:
            raise ValueError("Original tokenizer/template changed during export.")
        (out / name).write_bytes(path.read_bytes())
    result = {
        "schema": SCHEMA,
        "status": "weights_exported",
        "created_at": datetime.now(UTC).isoformat(),
        "source_repo": source["repo"],
        "source_revision": source["revision"],
        "source_config_sha256": json_hash(config),
        "selection_sha256": json_hash(selection),
        "source_layers": layers,
        "retained_experts": retained,
        "active_experts": active,
        "source_shards_verified": source_verified,
        "files": files,
        "tensor_count": tensor_count,
        "tensor_payload_bytes": output_bytes,
        "selection": selection,
        "original_vocabulary_head_retained": True,
        "vision_removed": True,
        "mtp_removed": True,
        "tokenizer_replaced": False,
        "output_config_sha256": file_hash(out / "config.json"),
        "output_index_sha256": file_hash(out / "model.safetensors.index.json"),
        "loader_requirement": "Native GLM factory with its unused vision module removed.",
        "model_loaded": False,
        "quality_evaluated": False,
        "training_performed": False,
        "release_ready": False,
    }
    atomic_json(out / "compression-manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new report path.")
    config = json.loads(args.config.read_text())
    layout = json.loads(args.layout.read_text())["layout"]
    full = list(range(config["text_config"]["num_hidden_layers"]))
    # A fixed depth ablation, NOT an importance-based optimum: remove three
    # complete four-layer hybrid blocks, preserving the first dense and last layers.
    reduced = [
        i for i in full if i not in set(range(11, 15)) | set(range(23, 27)) | set(range(31, 35))
    ]
    candidates = [
        (full, 288, 8),
        (full, 144, 8),
        (full, 96, 8),
        (full, 48, 8),
        (full, 24, 8),
        (full, 24, 4),
        (reduced, 24, 8),
        (reduced, 24, 4),
    ]
    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "config_file_sha256": file_hash(args.config),
        "layout_file_sha256": file_hash(args.layout),
        "candidates": [
            footprint(config, layout, layers=layers, experts=experts, active=active)
            for layers, experts, active in candidates
        ],
        "expert_selection_available": False,
        "quality_evaluated": False,
        "warning": "Shape estimates; no candidate weights or runtime measurements created.",
    }
    atomic_json(args.out, report)
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "layers",
                        "retained_experts_per_layer",
                        "active_experts_per_token",
                        "parameters",
                        "state_gib",
                        "routed_ffn_parameters_per_token",
                    )
                }
                for row in report["candidates"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
