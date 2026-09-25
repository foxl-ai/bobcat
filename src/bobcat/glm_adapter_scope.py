"""Plan an explicit suffix of native GLM attention adapters.

This selects training targets only. It never removes backbone layers/experts or
claims an inference speedup, and does not modify a model or a running job.
"""

from __future__ import annotations

from bobcat.glm_adapter_probe import TARGET_LEAVES
from bobcat.schema import json_hash


def plan_attention_lora_scope(model, *, last_layers: int | None = None, rank: int = 8):
    import torch

    if type(rank) is not int or rank < 1:
        raise ValueError("Use a positive integer LoRA rank.")
    prefix = "model.language_model.layers."
    try:
        stack = model.model.language_model.layers
        layers = sorted(int(key) for key in stack.keys())
    except (AttributeError, ValueError) as error:
        raise ValueError("Use the native GLM language-model layer layout.") from error
    if not layers or layers != list(range(len(layers))):
        raise ValueError("Native layers must be contiguous, starting at zero.")
    if last_layers is None:
        last_layers = len(layers)
    if type(last_layers) is not int or not 1 <= last_layers <= len(layers):
        raise ValueError("Select a nonempty suffix within the existing layer stack.")
    if any(hasattr(module, "lora_A") for module in model.modules()):
        raise ValueError("Plan the scope before attaching adapters.")
    selected_layers = layers[-last_layers:]
    candidates = []
    for name, module in model.named_modules():
        if not name.startswith(prefix) or ".self_attn." not in name or ".indexer." in name:
            continue
        layer = int(name.removeprefix(prefix).split(".", 1)[0])
        if (layer not in selected_layers or not isinstance(module, torch.nn.Linear)
                or name.rsplit(".", 1)[-1] not in TARGET_LEAVES):
            continue
        candidates.append({
            "name": name, "layer": layer, "in_features": module.in_features,
            "out_features": module.out_features,
            "trainable_parameters": rank * (module.in_features + module.out_features),
        })
    if {x["layer"] for x in candidates} != set(selected_layers):
        raise ValueError("Every selected native layer must have supported attention targets.")
    return {
        "schema": "bobcat-native-attention-lora-scope-v1",
        "total_language_layers": len(layers),
        "selected_layers": selected_layers,
        "first_trainable_layer": selected_layers[0],
        "lora_rank": rank,
        "target_modules": [x["name"] for x in candidates],
        "targets": candidates,
        "trainable_parameters": sum(x["trainable_parameters"] for x in candidates),
        "backbone_layers_removed": 0,
        "inference_speedup_measured": False,
        "backward_speedup_measured": False,
    }


def plan_pinned_attention_lora_scope(model, *, last_layers: int | None = None):
    """Retain the original full-model layout checks before narrowing training."""
    full = plan_attention_lora_scope(model, rank=8)
    if (full["total_language_layers"] != 45
            or len(full["target_modules"]) != 180
            or full["trainable_parameters"] != 17649664):
        raise ValueError("Pinned GLM attention projection layout changed.")
    return full if last_layers is None else plan_attention_lora_scope(
        model, last_layers=last_layers, rank=8,
    )


def validate_checkpoint_adapter_scope(marker, expected):
    """Reject partial or differently scoped adapter restores before DCP loading.

    Old checkpoints predate explicit scope metadata. Only the exact original
    full scope can use that legacy format; tensor/value checks remain required.
    """
    expected_hash = json_hash(expected)
    if "adapter_scope" not in marker:
        if ("adapter_scope_sha256" in marker
                or expected["selected_layers"] != list(range(45))
                or expected["total_language_layers"] != 45
                or expected["lora_rank"] != 8
                or len(expected["target_modules"]) != 180
                or expected["trainable_parameters"] != 17649664):
            raise ValueError("An unscoped checkpoint cannot restore a reduced adapter scope.")
        return {"legacy_full_scope": True, "scope_sha256": expected_hash}
    if (marker["adapter_scope"] != expected
            or marker.get("adapter_scope_sha256") != expected_hash):
        raise ValueError("Checkpoint adapter scope differs from the requested training scope.")
    return {"legacy_full_scope": False, "scope_sha256": expected_hash}
