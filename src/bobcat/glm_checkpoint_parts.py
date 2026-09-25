"""Bounded initialization of the pinned native GLM from its block-FP8 checkpoint.

The vendor conversion remains the numerical authority. This wrapper invokes it
one complete decoder layer at a time and copies the converted tensors into their
existing model storage. It does not quantize training weights, change the model,
or establish that the full pretrained model fits or trains on a given machine.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def layer_groups(names, layer_count: int) -> tuple[tuple[str, ...], ...]:
    """Give every rank the same ordered shared/decoder-layer load schedule."""
    if type(layer_count) is not int or layer_count < 1:
        raise ValueError("A complete nonempty decoder is required.")
    shared, layers = [], {index: [] for index in range(layer_count)}
    seen = set()
    for name in names:
        if not isinstance(name, str) or name in seen:
            raise ValueError("Native state names must be unique strings.")
        if "lora_" in name or name.endswith("_extra_state"):
            raise ValueError("Pass the base initialization state, without adapters or extra state.")
        seen.add(name)
        match = _LAYER.match(name)
        if match is None:
            shared.append(name)
            continue
        index = int(match.group(1))
        if index not in layers:
            raise ValueError("Partial or MTP decoder state is not supported.")
        layers[index].append(name)
    if any(not names for names in layers.values()):
        raise ValueError("Every decoder layer must be present on every rank.")
    groups = ([tuple(sorted(shared))] if shared else [])
    groups.extend(tuple(sorted(layers[index])) for index in range(layer_count))
    return tuple(groups)


def _local(tensor):
    from torch.distributed.tensor import DTensor

    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _check_copy_layout(target, loaded, *, allow_cuda_to_cpu=False) -> None:
    import torch
    from torch.distributed.tensor import DTensor

    if type(target) is not type(loaded) and (
        isinstance(target, DTensor) or isinstance(loaded, DTensor)
    ):
        raise ValueError("The conversion changed distributed tensor ownership.")
    if target.shape != loaded.shape or target.dtype != loaded.dtype:
        raise ValueError("The conversion changed a native tensor's shape or dtype.")
    if isinstance(target, DTensor) and (
        target.placements != loaded.placements
        or target.device_mesh != loaded.device_mesh
    ):
        raise ValueError("The conversion changed a native tensor's mesh or placement.")
    left, right = _local(target), _local(loaded)
    expected_staging = (
        allow_cuda_to_cpu and left.device.type == "cpu" and right.device.type == "cuda"
        and right.device.index == torch.cuda.current_device()
    )
    if left.shape != right.shape or (left.device != right.device and not expected_staging):
        raise ValueError(
            "The conversion changed a rank's shape or storage device: "
            f"target={tuple(left.shape)}@{left.device}, "
            f"loaded={tuple(right.shape)}@{right.device}."
        )


def bounded_glm_adapter(
    original, *, max_local_layer_bytes: int = 4 * 1024**3,
    allow_expert_cuda_staging: bool = False,
    allow_fp8_cuda_staging: bool = False,
    on_part_loaded=None,
):
    """Wrap the pinned GLM adapter with layerwise DCP initialization.

    Caller must verify the pinned vendor tree and the source checkpoint first.
    The native byte guard bounds a group's stored tensors, not total RSS, CUDA
    memory, DCP workspace, or conversion temporaries. All of those still require
    a measured distributed pilot before attempting a full pretrained update.
    """
    import torch
    from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart
    from nemo_automodel.components.models.glm5_next.state_dict_adapter import (
        Glm5NextStateDictAdapter,
    )

    if type(original) is not Glm5NextStateDictAdapter:
        raise ValueError("This wrapper only accepts the reviewed native GLM adapter.")
    if type(max_local_layer_bytes) is not int or not 0 < max_local_layer_bytes <= 4 * 1024**3:
        raise ValueError("Use a positive per-layer native-byte bound at most 4 GiB.")
    if any(type(value) is not bool for value in (
        allow_expert_cuda_staging, allow_fp8_cuda_staging,
    )):
        raise ValueError("Explicitly select expert and FP8 conversion CUDA staging.")
    if original.backend.dispatcher != "torch" or original.backend.experts not in (
        "torch", "torch_mm",
    ):
        raise ValueError("Only the ordinary grouped expert storage path is reviewed.")

    class LayerwiseGLMAdapter(Glm5NextStateDictAdapter):
        # The ungrouped path still requires large conversion temporaries.
        _supports_low_memory_dcp_load = False

        def iter_checkpoint_load_parts(self, model_state_dict, device_mesh=None):
            if not isinstance(model_state_dict, Mapping):
                raise ValueError("Expected the full native base state mapping.")
            groups = layer_groups(model_state_dict, self.config.text_config.num_hidden_layers)
            for tensor in model_state_dict.values():
                if not isinstance(tensor, torch.Tensor) or tensor.is_meta:
                    raise ValueError("Materialize sharded model storage before checkpoint reading.")
            return self._iter_layers(model_state_dict, groups, device_mesh)

        def _iter_layers(self, state, groups, mesh):
            for names in groups:
                targets = {name: state[name] for name in names}
                local_bytes = sum(_local(value).numel() * value.element_size()
                                  for value in targets.values())
                if local_bytes > max_local_layer_bytes:
                    raise ValueError("A native layer exceeds the frozen local-byte bound.")
                # Conversion mixins keep load-specific view bookkeeping. A fresh
                # adapter prevents one layer's bookkeeping from affecting another.
                converter = Glm5NextStateDictAdapter(
                    self.config, self.moe_config, self.backend, self.dtype,
                )
                checkpoint = converter.to_hf(
                    targets, quantization=True, for_checkpoint_load=True,
                )
                if not checkpoint:
                    raise ValueError("The layer produced no checkpoint destinations.")
                temporary = frozenset(
                    name for name in checkpoint
                    if name.endswith("_scale_inv")
                    or checkpoint[name].dtype == torch.float8_e4m3fn
                )
                # Vendor block dequantization reconstructs a DTensor using
                # from_local. On a CUDA mesh PyTorch moves even an offloaded
                # CPU local shard to CUDA. Permit only these identified FP8
                # matrices, in addition to explicitly permitted EP rebuilding.
                fp8_matrices = frozenset(
                    name for name, value in checkpoint.items()
                    if value.dtype == torch.float8_e4m3fn
                )
                progress = {"attempted": False, "finished": False}

                def finish(
                    converter=converter, checkpoint=checkpoint, targets=targets, mesh=mesh,
                    progress=progress, fp8_matrices=fp8_matrices,
                ):
                    if progress["attempted"]:
                        raise ValueError("A load part may be installed only once.")
                    progress["attempted"] = True
                    with torch.no_grad():
                        converted = converter.from_hf(checkpoint, device_mesh=mesh)
                        if converter.view_loaded_native_keys:
                            raise ValueError(
                                "Unexpected in-place expert conversion on the FP8 path."
                            )
                        if set(converted) != set(targets):
                            raise ValueError(
                                "The layer conversion did not populate every native key."
                            )
                        for name, value in converted.items():
                            # Vendor create_dtensor_from_local moves rebuilt EP
                            # experts to CUDA when available. CPU-offload loading
                            # may explicitly permit that bounded layer staging;
                            # it is not a zero-GPU-memory conversion.
                            try:
                                _check_copy_layout(
                                    targets[name], value,
                                    allow_cuda_to_cpu=(
                                        allow_expert_cuda_staging and ".mlp.experts." in name
                                        or allow_fp8_cuda_staging and name in fp8_matrices
                                    ),
                                )
                            except ValueError as error:
                                raise ValueError(f"{name}: {error}") from error
                        for name, value in converted.items():
                            _local(targets[name]).copy_(_local(value))
                    progress["finished"] = True
                    if on_part_loaded is not None:
                        on_part_loaded(tuple(targets), sum(
                            _local(value).numel() * value.element_size()
                            for value in targets.values()
                        ))

                yield CheckpointLoadPart(
                    checkpoint_tensors=checkpoint,
                    model_keys=frozenset(names),
                    temporary_checkpoint_keys=temporary,
                    finish=finish,
                )
                if not progress["finished"]:
                    raise ValueError(
                        "Finish and release each load part before requesting the next."
                    )
                del checkpoint, converter, targets, finish

    return LayerwiseGLMAdapter(
        original.config, original.moe_config, original.backend, original.dtype,
    )
