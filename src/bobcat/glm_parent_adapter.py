"""Bind canonical saved LoRA names to actual activation-checkpoint wrappers."""

from __future__ import annotations


def parent_adapter_bindings(model, tensors):
    """Return raw native names -> original full parent tensors, without casting.

    Only a real PyTorch CheckpointWrapper can make a name segment transparent.
    Arbitrary prefix stripping, missing rows, collisions and dtype/shape changes
    are rejected before materializing or loading the large original backbone.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    modules = dict(model.named_modules())
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    canonical = {}
    for name, parameter in parameters.items():
        parts, clean = name.split("."), []
        if not any(part in ("lora_A", "lora_B") for part in parts):
            raise ValueError("A trainable parameter lies outside the intended LoRA actor.")
        for index, part in enumerate(parts):
            if part == "_checkpoint_wrapped_module":
                if not isinstance(modules.get(".".join(parts[:index])), CheckpointWrapper):
                    raise ValueError("Only a real checkpoint wrapper may be transparent.")
            else:
                clean.append(part)
        key = ".".join(clean)
        if key in canonical:
            raise ValueError("Canonical adapter names must be one-to-one.")
        canonical[key] = (name, parameter)
    if not canonical or set(canonical) != set(tensors):
        missing = sorted(set(canonical) - set(tensors))[:3]
        extra = sorted(set(tensors) - set(canonical))[:3]
        raise ValueError(f"Parent adapter membership differs; missing={missing}, extra={extra}")
    result = {}
    for key, (name, parameter) in canonical.items():
        tensor = tensors[key]
        if tuple(tensor.shape) != tuple(parameter.shape) or tensor.dtype != parameter.dtype:
            raise ValueError(f"Parent adapter shape or dtype differs: {key}")
        result[name] = tensor
    return result
