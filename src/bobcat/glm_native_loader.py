"""Full pinned GLM initialization reused by the isolated checkpoint evaluator.

Extracted from the finite native trainer; resident GPU mode is a separate
execution configuration and must be measured on hardware with sufficient RAM.
It does not modify the already-running trainer package.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_scope import plan_pinned_attention_lora_scope
from bobcat.glm_checkpoint_parts import bounded_glm_adapter
from bobcat.glm_derived_buffers import restore_vision_rotary_buffer
from bobcat.glm_fsdp_adapter_probe import local_copy
from bobcat.glm_native_train import (
    REVISION,
    checkpoint_layout,
    tensor_digest,
    verify_restored_layout,
)
from bobcat.schema import file_hash, json_hash


def load_native_model(model_dir: Path, source: dict, out: Path, status, *,
                      cpu_offload=True, expert_backend="torch", decision_token_ids=None,
                      adapter_last_layers=None, compact_bank_dir=None,
                      source_manifest_sha256=None, activation_checkpointing=False,
                      source_model_set=None, source_verification=None, parent_adapter_path=None):
    import torch
    import torch.distributed as dist
    from nemo_automodel._transformers.model_init import local_torch_dtype
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.parallelizer import parallelize_model
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

    if expert_backend not in ("torch", "torch_mm"):
        raise ValueError("Use a recorded native expert backend.")
    bank, bank_binding = None, None
    if compact_bank_dir is not None:
        from bobcat.glm_compact_head import load_verified_native_bank

        if decision_token_ids is None or source_manifest_sha256 is None:
            raise ValueError("The compact head needs the original source and identifier binding.")
        bank, bank_binding = load_verified_native_bank(
            compact_bank_dir, source=source, source_manifest_sha256=source_manifest_sha256,
            token_ids=decision_token_ids,
        )
    rank = dist.get_rank()
    device = torch.device("cuda", int(__import__("os").environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    torch.set_num_threads(4)
    torch.manual_seed(202609230001)
    torch.cuda.manual_seed_all(202609230001)
    if (source_model_set is None) != (source_verification is None):
        raise ValueError("The preverified source needs both its model set and portable receipt.")
    if type(activation_checkpointing) is not bool:
        raise ValueError("Activation checkpointing must be an explicit execution setting.")
    args = SimpleNamespace(model_dir=model_dir, out=out,
                           activation_checkpointing=activation_checkpointing)
    if rank == 0 and source_model_set is not None:
        from bobcat.glm_serving_assets import validate_native_source

        status("validating_read_only_cpu_source_proof")
        proof = validate_native_source(
            source_model_set, source_verification, model_dir, source,
        )
        atomic_json(out / "source-verification.json", proof)
    elif rank == 0:
        status("verifying_source_checkpoint")
        source_files = {}
        for item in source["files"]:
            p = args.model_dir / item["path"]
            if not p.is_file() or p.stat().st_size != item["bytes"]:
                raise ValueError(f"Missing original checkpoint file: {item['path']}")
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending = {pool.submit(file_hash, args.model_dir / item["path"]): item
                       for item in source["files"]}
            for future in as_completed(pending):
                item, actual = pending[future], future.result()
                if actual != item["sha256"]:
                    raise ValueError(f"Original checkpoint checksum mismatch: {item['path']}")
                source_files[item["path"]] = actual
                status("verifying_source_checkpoint", checkpoint_files_verified=len(source_files))
        atomic_json(args.out / "source-verification.json", {
            "at": datetime.now(UTC).isoformat(), "revision": REVISION,
            "files": source_files, "bytes": source["total_bytes"],
        })
    # The source digest pass can take minutes on EBS. The process-group timeout
    # is finite but allows verified loading; it is not a perpetual waiter.
    dist.barrier()
    status("constructing_full_pretrained_layout")
    world = init_device_mesh("cuda", (8,), mesh_dim_names=("dp_shard_cp",))
    experts = init_device_mesh("cuda", (2, 4), mesh_dim_names=("ep_shard", "ep"))
    config = Glm5NextConfig.from_pretrained(args.model_dir, local_files_only=True)
    backend = dict(attn="sdpa", linear="torch", rms_norm="torch_fp32", experts=expert_backend,
                   dispatcher="torch", rope_fusion=False, enable_hf_state_dict_adapter=True)
    with torch.device("meta"), local_torch_dtype(torch.bfloat16, "Bobcat native GLM training"):
        model = Glm5NextForConditionalGeneration(config, backend=BackendConfig(**backend))
        base_count = sum(p.numel() for p in model.parameters())
        if base_count != 313890426878:
            raise ValueError(f"The full native GLM parameter count changed: {base_count}")
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, Gate):
                module.bias_update_factor = 0.0
        scope = plan_pinned_attention_lora_scope(model, last_layers=adapter_last_layers)
        targets = scope["target_modules"]
        adapter_count = scope["trainable_parameters"]
        peft = PeftConfig(target_modules=targets, dim=8, alpha=16, dropout=0.,
                          lora_dtype=torch.bfloat16, use_triton=False,
                          use_memory_efficient_lora=False)
        if apply_lora_to_linear_modules(model, peft) != len(targets):
            raise ValueError("Not every intended attention projection received LoRA.")
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != adapter_count:
            raise ValueError("The actual trainable parameter budget changed.")
        if bank is not None:
            from bobcat.glm_compact_head import install_uninitialized_native_decision_projection

            install_uninitialized_native_decision_projection(model, decision_token_ids)
        elif decision_token_ids is not None:
            from bobcat.decision_projection import install_retained_vocabulary_projection

            install_retained_vocabulary_projection(model, decision_token_ids)
    parallelize_model(
        model, world, experts, dp_axis_names=("dp_shard_cp",), ep_axis_name="ep",
        ep_shard_axis_names=("ep_shard",), activation_checkpointing=args.activation_checkpointing,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      output_dtype=torch.bfloat16, cast_forward_inputs=True),
        offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else None,
    )
    if parent_adapter_path is not None:
        from safetensors.torch import load_file

        from bobcat.glm_parent_adapter import parent_adapter_bindings

        parent = load_file(str(parent_adapter_path), device="cpu")
        bindings = parent_adapter_bindings(model, parent)
        atomic_json(out / f"parent-layout-rank-{rank}.json", {
            "canonical_parent_names_bound_to_native_wrappers": True,
            "checked_before_backbone_materialization": True,
            "parent_adapter_sha256": file_hash(parent_adapter_path),
            "parameter_tensors": len(bindings),
            "global_parameters": sum(t.numel() for t in bindings.values()),
            "native_shapes": {name: list(t.shape) for name, t in bindings.items()},
            "dtype_casts": 0,
        })
        del parent, bindings
    atomic_json(out / f"adapter-scope-rank-{rank}.json", {
        "scope": scope, "scope_sha256": json_hash(scope),
    })
    status("materializing_cpu_shards" if cpu_offload else "materializing_gpu_shards",
           base_parameters=base_count,
           adapter_parameters=adapter_count, adapter_scope=scope,
           compact_output_binding=bank_binding,
           materialized_parameter_count=sum(p.numel() for p in model.parameters()))
    Checkpointer.initialize_model_weights(
        model, torch.device("cpu") if cpu_offload else device, peft_init_method="xavier",
    )
    expected_layout = checkpoint_layout(
        ModelState(model, is_peft=True, is_init_step=True).state_dict(),
    )
    router_buffers = {k: v for k, v in expected_layout.items()
                      if k.endswith(".e_score_correction_bias")}
    if (len(expected_layout) - len(router_buffers) != 1635 or len(router_buffers) != 42
            or any(v != {"shape": [288], "dtype": "torch.float32"}
                   for v in router_buffers.values())):
        raise ValueError("Pinned native parameter/persistent-buffer layout changed.")
    atomic_json(out / f"base-layout-rank-{rank}.json", {
        "parameter_tensors": 1635, "router_buffer_tensors": 42,
        "state_tensors": len(expected_layout), "layout": expected_layout,
        "layout_sha256": json_hash(expected_layout),
    })
    loaded = set()

    def loaded_part(names, size):
        if loaded.intersection(names):
            raise ValueError("A native key was restored more than once.")
        loaded.update(names)
        status("loading_original_fp8_parts", loaded_native_keys=len(loaded),
               last_part_native_bytes=size)

    loader_arguments = {
        "max_local_layer_bytes": 4 * 1024**3,
        "allow_expert_cuda_staging": True, "allow_fp8_cuda_staging": True,
        "on_part_loaded": loaded_part,
    }
    if bank is not None:
        from bobcat.glm_compact_head import compact_glm_checkpoint_adapter

        model.state_dict_adapter = compact_glm_checkpoint_adapter(
            model.state_dict_adapter, bank, **loader_arguments,
        )
    else:
        model.state_dict_adapter = bounded_glm_adapter(
            model.state_dict_adapter, **loader_arguments,
        )
    checkpointer = Checkpointer(CheckpointingConfig(
        is_peft=True, save_consolidated=False, model_cache_dir=str(args.model_dir),
        dequantize_base_checkpoint=True, cpu_offload=cpu_offload,
    ), dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=experts)
    checkpointer.load_base_model(
        model, device, str(args.model_dir), str(args.model_dir), load_base_model=True,
    )
    derived_buffer = restore_vision_rotary_buffer(model, device)
    state = ModelState(model, is_peft=True, is_init_step=True).state_dict()
    verify_restored_layout(expected_layout, state, loaded)
    buffer_bytes = 0
    for module in model.modules():
        for name, value in list(module._buffers.items()):
            if value is None:
                continue
            local = value.to_local() if hasattr(value, "to_local") else value
            buffer_bytes += local.numel() * local.element_size()
            if value.is_meta or buffer_bytes > 16 * 1024**2:
                raise ValueError("Unexpected unmaterialized/large persistent buffer.")
            before = tensor_digest(value)
            moved = value.to(device)
            if moved.dtype != value.dtype or tensor_digest(moved) != before:
                raise ValueError("Buffer device placement changed the original checkpoint.")
            if hasattr(value, "placements") and (moved.placements != value.placements
                                                 or moved.device_mesh != value.device_mesh):
                raise ValueError("Buffer placement changed distributed ownership.")
            module._buffers[name] = moved
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if any("lora_" not in n for n in params) or len(params) != 2 * len(targets):
        raise ValueError("Trainable ownership changed during FSDP construction.")
    for name, p in model.named_parameters():
        local = p.to_local() if hasattr(p, "to_local") else p
        if p.is_meta or local.device.type != ("cpu" if cpu_offload else "cuda"):
            raise ValueError(f"A parameter is not on the requested storage device: {name}")
    adapter_modules = [(n, m) for n, m in model.named_modules()
                       if hasattr(m, "lora_A") and hasattr(m, "lora_B")]
    if len(adapter_modules) != len(targets):
        raise ValueError("Could not resolve every wrapped adapter module.")
    for name, p in params.items():
        if ".lora_B." in name and bool(torch.count_nonzero(local_copy(p))):
            raise ValueError("New adapter B must be exactly zero.")
    status("loaded_full_pretrained", pretrained_weights_loaded=True, buffer_bytes=buffer_bytes,
           derived_buffer_restoration=derived_buffer)

    return model, adapter_modules
