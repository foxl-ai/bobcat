"""Restore checkpoint-free GLM buffers after meta-device materialization."""

from __future__ import annotations


def restore_vision_rotary_buffer(model, device) -> dict:
    """Reconstruct only the pinned vision constructor's nonpersistent buffer.

    NeMo's GLM vision init_weights initializes learned modules but does not rerun
    the rotary constructor. The checkpoint cannot restore a nonpersistent
    inv_freq. Never replace a persistent buffer or initialize learned weights.
    """
    import torch

    visual = model.get_submodule("model.visual")
    rotary = visual.rotary_pos_emb
    if (type(rotary).__name__ != "Glm5NextVisionRotaryEmbedding"
            or type(rotary).__module__ !=
            "nemo_automodel.components.models.glm5_next.vision"):
        raise ValueError("Only the pinned GLM vision rotary constructor is supported.")
    if (set(rotary._buffers) != {"inv_freq"}
            or rotary._non_persistent_buffers_set != {"inv_freq"}
            or list(rotary.parameters()) or list(rotary.children())):
        raise ValueError("Refuse to reconstruct learned or persistent rotary state.")
    current = rotary.inv_freq
    if (current is None or hasattr(current, "placements")
            or current.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("Unexpected vision rotary storage or dtype.")
    config = visual.config
    if config.hidden_size % config.num_heads:
        raise ValueError("Vision hidden width must divide into complete heads.")
    head_dim = config.hidden_size // config.num_heads
    if head_dim % 4 or not 4 <= head_dim <= 4096:
        raise ValueError("Unexpected pinned vision rotary dimension.")
    # Execute the verified vendor constructor, including its default theta, on
    # CPU. This also matches ordinary checkpoint-free CPU initialization before
    # the native model casts its buffers to the configured model dtype.
    with torch.device("cpu"):
        reference = type(rotary)(head_dim // 2).inv_freq.to(dtype=current.dtype)
    if reference.shape != current.shape or not bool(torch.isfinite(reference).all()):
        raise ValueError("Derived vision rotary buffer disagrees with model layout.")
    was_finite = None if current.is_meta else bool(torch.isfinite(current).all())
    restored = reference.to(device=device)
    if not torch.equal(reference, restored.detach().cpu()):
        raise ValueError("Derived rotary values changed during device placement.")
    rotary._buffers["inv_freq"] = restored
    return {
        "name": "model.visual.rotary_pos_emb.inv_freq",
        "provenance": "pinned vendor constructor; checkpoint-free nonpersistent buffer",
        "shape": list(restored.shape), "dtype": str(restored.dtype),
        "device": str(restored.device), "previous_storage_finite": was_finite,
        "constructor_reference_exact": True,
        "learned_parameters_modified": False, "persistent_buffers_modified": False,
    }
