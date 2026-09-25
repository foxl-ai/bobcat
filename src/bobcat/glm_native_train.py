"""Finite native pretrained GLM decision training on eight exclusive GPUs.

The base is restored from the original block-FP8 checkpoint into BF16 FSDP
shards, either CPU-offloaded or explicitly resident on larger GPUs.
Only attention LoRA parameters are optimized. There is no
generated target text, no teacher call, and no task-specific classifier head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import verify_source
from bobcat.glm_adapter_scope import (
    plan_pinned_attention_lora_scope,
    validate_checkpoint_adapter_scope,
)
from bobcat.glm_checkpoint_parts import bounded_glm_adapter
from bobcat.glm_derived_buffers import restore_vision_rotary_buffer
from bobcat.glm_fsdp_adapter_probe import local_copy
from bobcat.glm_native_data import packed_rank_batch
from bobcat.native_resume import (
    evaluation_state,
    materialize_reference_state,
    validate_resume_reference,
    verify_materialized_state,
)
from bobcat.schema import file_hash, json_hash
from bobcat.supervision import MEAN_TARGET

REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"


def checkpoint_layout(state):
    """Include persistent buffers, which are not part of named_parameters()."""
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()}


def verify_restored_layout(expected, state, loaded):
    actual = checkpoint_layout(state)
    if set(expected) != loaded or actual != expected:
        changed = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
        raise ValueError(
            "Incomplete native restoration: "
            f"missing={sorted(set(expected) - loaded)}, "
            f"unexpected={sorted(loaded - set(expected))}, "
            f"changed={changed}, "
            f"state_missing={sorted(set(expected) - set(actual))}"
        )


def verify_parameter_storage(model, *, cpu_offload):
    """Check every materialized local shard, including frozen parameters."""
    if type(cpu_offload) is not bool:
        raise ValueError("CPU offload must be an explicit boolean.")
    expected_device = "cpu" if cpu_offload else "cuda"
    count, byte_count = 0, 0
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        if parameter.is_meta or local.is_meta or local.device.type != expected_device:
            raise ValueError(
                f"Parameter storage differs from requested {expected_device} mode: {name}",
            )
        count += 1
        byte_count += local.numel() * local.element_size()
    if not count:
        raise ValueError("No parameter storage was verified.")
    return {"cpu_offload": cpu_offload, "parameter_storage_device": expected_device,
            "local_parameter_tensors": count, "local_parameter_bytes": byte_count}


def validate_record(row, limit):
    ids, options = row["input_ids"], row["option_token_ids"]
    if (not ids or not 2 <= len(options) <= 255 or len(set(options)) != len(options)
            or any(type(v) is not int or v < 0 for v in ids + options)
            or len(ids) > limit or row["input_tokens"] != len(ids)
            or row["sampling_loss_weight"] != 1
            or row["input_sha256"] != json_hash({
                "input_ids": ids, "option_token_ids": options,
            })):
        raise ValueError("Incomplete/misaligned curriculum input.")
    target, mean = row["target_index"], row["score_mean"]
    if row["supervision"] == "score_mean":
        if (row["kind"] != "ordinal" or target != MEAN_TARGET
                or not math.isfinite(mean) or not 0 <= mean <= len(options) - 1):
            raise ValueError("Ordinal means are not synthetic categorical targets.")
    elif row["supervision"] != "hard_label" or not 0 <= target < len(options):
        raise ValueError("Unknown supervised objective.")


def read_curriculum(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest["schema"] != "bobcat-glm-curriculum-v1"
            or manifest["status"] != "completed" or manifest["training_performed"]
            or manifest["model_source"]["revision"] != REVISION
            or manifest["content_sha256"] != json_hash({
                k: v for k, v in manifest.items() if k != "content_sha256"
            })):
        raise ValueError("Require the frozen real-data curriculum.")
    data, groups = {}, set()
    for split in ("train", "dev_train"):
        path = root / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError("Curriculum checksum changed.")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) % 8 or len(rows) != manifest["statistics"][split]["rows"]:
            raise ValueError("Require whole distributed batches.")
        for row in rows:
            validate_record(row, manifest["curriculum_max_tokens"])
            if row["group_id"] in groups or row["split"] != split:
                raise ValueError("A component repeats or crosses the split.")
            groups.add(row["group_id"])
        data[split] = rows
    return manifest, data


def read_monitoring_suite(folder, training_rows):
    from bobcat.glm_native_evaluate import read_suite

    manifest, rows = read_suite(folder)
    training_groups = {row["group_id"] for row in training_rows}
    if any(row["group_id"] in training_groups for row in rows):
        raise ValueError("The monitoring suite overlaps native training components.")
    return manifest, rows


def decision_loss(logits, row):
    import torch
    import torch.nn.functional as functional

    if row["supervision"] == "score_mean":
        levels = torch.arange(len(row["option_token_ids"]), device=logits.device)
        expected = (logits.float().softmax(-1) * levels).sum()
        # Normalize the native scale, without claiming that the mean determines
        # an observed categorical distribution.
        return ((expected - row["score_mean"]) / (len(levels) - 1)).square()
    return functional.cross_entropy(
        logits.float().unsqueeze(0),
        torch.tensor([row["target_index"]], device=logits.device),
    )


def question_packs(rows, world_size, questions_per_rank):
    """Keep curriculum order while giving every data-parallel rank equal question weight."""
    if (type(world_size) is not int or world_size < 1
            or type(questions_per_rank) is not int or questions_per_rank < 1
            or len(rows) != world_size * questions_per_rank):
        raise ValueError("A packed update requires an equal, whole question batch on every rank.")
    packs = [rows[rank::world_size] for rank in range(world_size)]
    real_lengths = [sum(len(row["input_ids"]) for row in pack) for pack in packs]
    padded = (max(real_lengths) + 127) // 128 * 128
    if min(real_lengths) < 1 or padded > 32767:
        raise ValueError("The complete packed update exceeds the native sequence limit.")
    return packs, padded


def packed_decision_loss(logits, rows):
    """Mean over questions, never over tokens or over an artificial shared label set."""
    import torch

    if not rows or len(logits) != len(rows):
        raise ValueError("Each independent question needs exactly one aligned readout.")
    return torch.stack([decision_loss(value, row)
                        for value, row in zip(logits, rows, strict=True)]).mean()


def tensor_digest(tensor):
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    array = local.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy()
    return hashlib.sha256(memoryview(array)).hexdigest()


def replay_differences(expected, actual):
    """Summarize numerical drift without changing either state or its precision."""
    import torch

    result = {"tensors": 0, "different_tensors": 0, "max_abs": 0.,
              "squared_difference": 0., "squared_reference": 0., "nonfinite": False,
              "metadata_differences": []}

    def visit(a, b, name):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                result["metadata_differences"].append(name)
                return
            result["tensors"] += 1
            result["different_tensors"] += int(not torch.equal(a, b))
            left, right = a.double(), b.double()
            difference = left - right
            result["nonfinite"] |= not bool(torch.isfinite(difference).all())
            result["max_abs"] = max(result["max_abs"], float(difference.abs().max()))
            result["squared_difference"] += float(difference.square().sum())
            result["squared_reference"] += float(left.square().sum())
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                result["metadata_differences"].append(name)
            else:
                for key in a:
                    visit(a[key], b[key], f"{name}/{key}")
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                result["metadata_differences"].append(name)
            else:
                for index, (left, right) in enumerate(zip(a, b, strict=True)):
                    visit(left, right, f"{name}/{index}")
        elif a != b:
            result["metadata_differences"].append(name)

    visit(expected, actual, "state")
    result["relative_l2"] = math.sqrt(
        result["squared_difference"] / max(result["squared_reference"], 1e-30),
    )
    return result


def run(args, record):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from nemo_automodel._transformers.model_init import local_torch_dtype
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next import layers
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from nemo_automodel.components.moe.layers import Gate
    from nemo_automodel.components.moe.parallelizer import parallelize_model
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
    from transformers import AutoConfig

    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.95, device)
    torch.set_num_threads(4)
    gpu = torch.cuda.get_device_properties(device)
    if not args.cpu_offload and gpu.total_memory < 130000 * 1024**2:
        raise ValueError("Resident full-base training requires at least 130000 MiB per GPU.")
    record["storage_profile"] = {
        "cpu_offload": args.cpu_offload,
        "parameter_storage_device": "cpu" if args.cpu_offload else "cuda",
        "gpu": gpu.name, "total_memory_bytes": gpu.total_memory,
        "checkpoint_serialization_device": "cpu",
    }
    if args.deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Set the deterministic cuBLAS workspace before CUDA starts.")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    record["deterministic_execution"] = {
        "requested": args.deterministic,
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "precision": "unchanged BF16 parameters and FP32 reductions",
        "custom_kernel_determinism_verified": False,
    }
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    started = time.monotonic()
    path = args.out / f"rank-{rank}.json"

    def status(phase, **fields):
        record.update(phase=phase, **fields)
        record["updated_at"] = datetime.now(UTC).isoformat()
        record["elapsed_seconds"] = time.monotonic() - started
        record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        record["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        atomic_json(path, record)
        if rank == 0:
            print(json.dumps({"phase": phase, **fields}, default=str), flush=True)

    curriculum, data = read_curriculum(args.curriculum)
    record["curriculum_sha256"] = file_hash(args.curriculum / "manifest.json")
    monitoring_rows = None
    if args.evaluation_suite:
        monitoring_manifest, monitoring_rows = read_monitoring_suite(
            args.evaluation_suite, data["train"],
        )
        record["monitoring_suite"] = {
            "manifest_sha256": file_hash(args.evaluation_suite / "manifest.json"),
            "records_sha256": monitoring_manifest["records_sha256"],
            "components": len(monitoring_rows), "training_overlap": False, "final_test": False,
        }
    decision_identifiers = None
    if args.decision_identifiers:
        mapping = json.loads(args.decision_identifiers.read_text())
        decision_identifiers = mapping["token_ids"]
        if (mapping["source_revision"] != REVISION
                or len(decision_identifiers) != 255
                or len(set(decision_identifiers)) != 255
                or any(type(token) is not int or token < 0 for token in decision_identifiers)
                or any(token not in decision_identifiers for rows in data.values()
                       for row in rows for token in row["option_token_ids"])):
            raise ValueError("The immutable native decision identifier mapping is incomplete.")
        record["decision_projection"] = {
            "mapping_sha256": file_hash(args.decision_identifiers),
            "mode": "retained_vocabulary_state_selected_gemm",
            "candidate_rows": len(decision_identifiers),
            "state_layout_reduced": False, "full_weight_all_gather_removed": False,
            "activated": False, "quality_or_speed_improvement_claimed": False,
        }
    global_questions = 8 * args.pack_questions
    resume_reference, resume_cursor = None, None
    start_updates = start_cursor = 0
    if args.resume_reference:
        if not args.deterministic:
            raise ValueError("Native continuation requires the strict deterministic profile.")
        resume_reference = json.loads(args.resume_reference.read_text())
        resume_cursor = validate_resume_reference(
            resume_reference, args.resume_checkpoint,
            curriculum_sha256=record["curriculum_sha256"], source_revision=REVISION,
            train_rows=len(data["train"]), next_pack=args.pack_questions,
            updates=args.max_updates, learning_rate=args.learning_rate,
            resume_learning_rate=args.resume_learning_rate,
        )
        start_updates = resume_cursor["optimizer_updates"]
        start_cursor = resume_cursor["question_cursor"]
        record["resume_reference_sha256"] = file_hash(args.resume_reference)
        record["continuation"] = resume_cursor
    record["starting_optimizer_updates"] = start_updates
    record["starting_question_cursor"] = start_cursor
    record["question_packing"] = {
        "questions_per_rank": args.pack_questions, "global_questions_per_update": global_questions,
        "independent_document_boundaries": True, "loss_weighting": "equal questions",
        "effective_token_count_excludes_padding": True,
        "shared_prefix_reuse_claimed": False,
    }
    if start_cursor + args.max_updates * global_questions > len(data["train"]):
        raise ValueError("The finite update plan exceeds the frozen curriculum.")
    # Check the whole finite plan before spending time loading the base model.
    for offset in range(start_cursor, start_cursor + args.max_updates * global_questions,
                        global_questions):
        question_packs(data["train"][offset:offset + global_questions], 8, args.pack_questions)
    source = curriculum["model_source"]
    if rank == 0 and args.source_verification:
        from bobcat.glm_serving_assets import validate_native_source

        status("checking_preverified_source")
        verification = validate_native_source(
            args.source_model_set, args.source_verification, args.model_dir, source,
        )
        atomic_json(args.out / "source-verification.json", verification)
        status("source_preverification_checked", checkpoint_files_verified=len(source["files"]))
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
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True,
                                       trust_remote_code=False)
    from bobcat.glm_expert_backend import expert_backend_name, inspect_grouped_experts

    selected_experts = expert_backend_name(args.expert_backend)
    backend = dict(attn="sdpa", linear="torch", rms_norm="torch_fp32", experts=selected_experts,
                   dispatcher="torch", rope_fusion=False, enable_hf_state_dict_adapter=True)
    with torch.device("meta"), local_torch_dtype(torch.bfloat16, "Bobcat native GLM training"):
        model = Glm5NextForConditionalGeneration(config, backend=BackendConfig(**backend))
        record["expert_backend"] = inspect_grouped_experts(model, selected_experts)
        base_count = sum(p.numel() for p in model.parameters())
        if base_count != 313890426878:
            raise ValueError(f"The full native GLM parameter count changed: {base_count}")
        model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, Gate):
                module.bias_update_factor = 0.0
        scope = plan_pinned_attention_lora_scope(
            model, last_layers=args.adapter_last_layers,
        )
        targets = scope["target_modules"]
        adapter_count = scope["trainable_parameters"]
        record["adapter_scope"] = scope
        record["adapter_scope_sha256"] = json_hash(scope)
        if resume_reference is not None:
            record["resume_scope_validation"] = validate_checkpoint_adapter_scope(
                resume_reference["marker"], scope,
            )
        peft = PeftConfig(target_modules=targets, dim=8, alpha=16, dropout=0.,
                          lora_dtype=torch.bfloat16, use_triton=False,
                          use_memory_efficient_lora=False)
        if apply_lora_to_linear_modules(model, peft) != len(targets):
            raise ValueError("Not every intended attention projection received LoRA.")
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != adapter_count:
            raise ValueError("The actual trainable parameter budget changed.")
        decision_head = None
        if decision_identifiers is not None:
            from bobcat.decision_projection import install_retained_vocabulary_projection
            decision_head = install_retained_vocabulary_projection(model, decision_identifiers)
    parallelize_model(
        model, world, experts, dp_axis_names=("dp_shard_cp",), ep_axis_name="ep",
        ep_shard_axis_names=("ep_shard",), activation_checkpointing=args.activation_checkpointing,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                      output_dtype=torch.bfloat16, cast_forward_inputs=True),
        offload_policy=CPUOffloadPolicy(pin_memory=True) if args.cpu_offload else None,
    )
    status("materializing_cpu_shards" if args.cpu_offload else "materializing_gpu_shards",
           base_parameters=base_count,
           adapter_parameters=adapter_count)
    Checkpointer.initialize_model_weights(
        model, torch.device("cpu") if args.cpu_offload else device, peft_init_method="xavier",
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
    atomic_json(args.out / f"base-layout-rank-{rank}.json", {
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

    model.state_dict_adapter = bounded_glm_adapter(
        model.state_dict_adapter, max_local_layer_bytes=4 * 1024**3,
        allow_expert_cuda_staging=True, allow_fp8_cuda_staging=True,
        on_part_loaded=loaded_part,
    )
    checkpointer = Checkpointer(CheckpointingConfig(
        is_peft=True, save_consolidated=False, model_cache_dir=str(args.model_dir),
        dequantize_base_checkpoint=True, cpu_offload=args.cpu_offload,
    ), dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=experts)
    checkpointer.load_base_model(
        model, device, str(args.model_dir), str(args.model_dir), load_base_model=True,
    )
    record["derived_buffer_restoration"] = restore_vision_rotary_buffer(model, device)
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
    record["verified_parameter_storage"] = verify_parameter_storage(
        model, cpu_offload=args.cpu_offload,
    )
    adapter_modules = [(n, m) for n, m in model.named_modules()
                       if hasattr(m, "lora_A") and hasattr(m, "lora_B")]
    if len(adapter_modules) != len(targets):
        raise ValueError("Could not resolve every wrapped adapter module.")
    for name, p in params.items():
        if ".lora_B." in name and bool(torch.count_nonzero(local_copy(p))):
            raise ValueError("New adapter B must be exactly zero.")
    status("loaded_full_pretrained", pretrained_weights_loaded=True, buffer_bytes=buffer_bytes)

    def frozen_hashes():
        return {n: tensor_digest(p) for n, p in model.state_dict().items() if "lora_" not in n}

    frozen = frozen_hashes()
    atomic_json(args.out / f"frozen-shards-rank-{rank}.json", frozen)
    optimizer = torch.optim.AdamW(params.values(), lr=args.learning_rate, weight_decay=.01,
                                  foreach=False)
    calls, kernels = Counter(), Counter()
    global_target_norms = torch.zeros(len(adapter_modules), device=device)
    options = StateDictOptions(ignore_frozen_params=True, cpu_offload=True, strict=False)
    packed_boundary_cache = False
    batched_gradient_statistics = False

    def batch_logits(rows, questions_per_rank=1, *, cache_boundaries=None):
        packs, padded = question_packs(rows, 8, questions_per_rank)
        local_rows = packs[rank]
        batch = packed_rank_batch(
            [{"inputs": {"input_ids": row["input_ids"]}} for row in local_rows],
            padded, device=device,
            cache_packed_boundaries=(
                packed_boundary_cache if cache_boundaries is None else cache_boundaries
            ),
        )
        all_logits = model(**batch).logits[0]
        if all_logits.shape[0] != len(local_rows):
            raise ValueError("Native readout did not retain one position per packed question.")
        if decision_head is not None:
            logits = decision_head.select_batch(
                all_logits, [row["option_token_ids"] for row in local_rows],
            )
        else:
            logits = [
                value.index_select(-1, torch.tensor(row["option_token_ids"], device=device)).float()
                for value, row in zip(all_logits, local_rows, strict=True)
            ]
        return logits[0] if questions_per_rank == 1 else logits

    def check_packed_boundary_cache():
        nonlocal packed_boundary_cache
        if not args.cache_packed_boundaries:
            return
        from bobcat.training_metadata_probe import (
            compare_metadata_backward,
            distributed_metadata_admission,
        )

        status("checking_packed_boundary_cache")
        rows = data["train"][start_cursor:start_cursor + global_questions]
        local_rows = rows[rank::8]
        optimizer.zero_grad(set_to_none=True)

        def compute(enabled):
            logits = batch_logits(rows, args.pack_questions, cache_boundaries=enabled)
            values = [logits] if args.pack_questions == 1 else logits
            return {
                "loss": packed_decision_loss(values, local_rows),
                "logits": torch.cat(values),
            }

        report = compare_metadata_backward(model, params, compute, device=device)
        report.update(
            rank=rank, input_sha256=[row["input_sha256"] for row in local_rows],
            source_revision=REVISION, parameter_updates=0,
            curriculum_sha256=record["curriculum_sha256"],
        )
        atomic_json(args.out / f"metadata-admission-rank-{rank}.json", report)
        reports = [None] * 8
        dist.all_gather_object(reports, report)
        decision = distributed_metadata_admission(reports)
        packed_boundary_cache = decision["activated"]
        if rank == 0:
            atomic_json(args.out / "metadata-admission-complete.json", decision)
        record["packed_boundary_cache"] = decision
        status("packed_boundary_cache_checked", packed_boundary_cache=decision)

    def evaluate(label, rows_to_evaluate=None):
        status("development_evaluation", development_evaluation_label=label)
        begin = time.monotonic()
        results = []
        selected_rows = data["dev_train"] if rows_to_evaluate is None else rows_to_evaluate
        with evaluation_state(model, device):
            for offset in range(0, len(selected_rows), 8):
                rows = selected_rows[offset:offset + 8]
                logits = batch_logits(rows)
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite development logits.")
                row = rows[rank]
                results.append({
                    "id": row["id"], "group_id": row["group_id"], "task": row["task"],
                    "input_sha256": row["input_sha256"],
                    "language": row["language"], "kind": row["kind"],
                    "target_index": row["target_index"], "score_mean": row["score_mean"],
                    "supervision": row["supervision"], "logits": logits.cpu().tolist(),
                    "loss": float(decision_loss(logits, row)),
                })
        atomic_json(args.out / f"{label}-rank-{rank}.json", results)
        dist.barrier()
        if rank == 0:
            files = [f"{label}-rank-{r}.json" for r in range(8)]
            atomic_json(args.out / f"evaluation-{label}-complete.json", {
                "label": label, "at": datetime.now(UTC).isoformat(),
                "wall_seconds": time.monotonic() - begin,
                "questions": len(selected_rows),
                "prompt_tokens": sum(row["input_tokens"] for row in selected_rows),
                "curriculum_sha256": record["curriculum_sha256"],
                "monitoring_suite": record.get("monitoring_suite") if rows_to_evaluate else None,
                "rng_restored": True, "optimizer_updated": False, "final_test": False,
                "files": {name: file_hash(args.out / name) for name in files},
            })
        dist.barrier()
        return results

    def check_native_projection():
        """Admit the new GEMM only on the unchanged full-native numerical gate.

        Retaining the original FSDP parameter enables an in-process fallback.
        A failed optimization does not invalidate the verified original path or
        require another expensive base reload.
        """
        if decision_head is None:
            return
        status("checking_decision_projection")
        comparisons = []
        with evaluation_state(model, device):
            for offset in range(0, 32, 8):
                rows = data["dev_train"][offset:offset + 8]
                decision_head.decision_only = False
                reference = batch_logits(rows)
                decision_head.decision_only = True
                selected = batch_logits(rows)
                comparisons.append({
                    "id": rows[rank]["id"],
                    "candidates": len(rows[rank]["option_token_ids"]),
                    "finite": bool(torch.isfinite(selected).all()),
                    "probability_tv": float(
                        (reference.softmax(-1) - selected.softmax(-1)).abs().sum() / 2,
                    ),
                    "argmax_equal": int(reference.argmax()) == int(selected.argmax()),
                    "reference_logits": reference.cpu().tolist(),
                    "selected_logits": selected.cpu().tolist(),
                })
        passed = all(row["finite"] and row["argmax_equal"]
                     and row["probability_tv"] <= 1e-3 for row in comparisons)
        failed = torch.tensor(int(not passed), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        decision_head.decision_only = not bool(failed)
        atomic_json(args.out / f"decision-projection-control-rank-{rank}.json", {
            "comparisons": comparisons, "maximum_allowed_probability_tv": 1e-3,
            "local_gate_passed": passed, "all_ranks_gate_passed": not bool(failed),
            "activated": decision_head.decision_only,
            "fallback": None if decision_head.decision_only else "original_full_vocabulary_gemm",
            "stored_vocabulary_parameter_retained": True,
            "full_parameter_all_gather_removed": False,
            "numerical_tolerance_relaxed": False, "release_gate_passed": False,
        })
        record["decision_projection"]["activated"] = decision_head.decision_only
        status("decision_projection_checked",
               decision_projection_activated=decision_head.decision_only)

    def check_native_packing():
        """Check the proposed native shape before taking a packed optimizer step.

        A fixed-shape document perturbation and a standalone comparison are
        separate controls. This is a finite training-format gate; it does not
        replace the wider cache, serving, language or G6 release evaluation.
        """
        if args.pack_questions == 1:
            return
        rows = data["dev_train"][:global_questions]
        if len(rows) != global_questions:
            raise ValueError("The packing control needs whole independent dev questions.")
        model.eval()
        with torch.no_grad():
            packed = batch_logits(rows, args.pack_questions)
            standalone = [
                batch_logits(rows[offset:offset + 8])
                for offset in range(0, global_questions, 8)
            ]
            perturbed_rows = [dict(row) for row in rows]
            # Perturb each rank's first document without changing its length.
            # These synthetic IDs are an isolation control, not a new training example.
            for index in range(8):
                perturbed_rows[index]["input_ids"] = [1] * len(rows[index]["input_ids"])
            perturbed = batch_logits(perturbed_rows, args.pack_questions)
            def probability_tv(a, b):
                return float((a.softmax(-1) - b.softmax(-1)).abs().sum() / 2)
            standalone_tv = [
                probability_tv(a, b) for a, b in zip(packed, standalone, strict=True)
            ]
            isolated_tv = [
                probability_tv(a, b) for a, b in zip(packed[1:], perturbed[1:], strict=True)
            ]
            argmax_equal = all(int(a.argmax()) == int(b.argmax())
                               for a, b in zip(packed, standalone, strict=True))
            positive_control = not torch.equal(packed[0], perturbed[0])
            finite = all(bool(torch.isfinite(value).all())
                         for value in [*packed, *standalone, *perturbed])
        passed = (finite and argmax_equal and positive_control
                  and max(standalone_tv) <= 1e-3 and max(isolated_tv) <= 1e-3)
        result = {
            "questions_per_rank": args.pack_questions, "global_questions": global_questions,
            "maximum_allowed_probability_tv": 1e-3,
            "standalone_probability_tv": standalone_tv,
            "other_document_probability_tv": isolated_tv,
            "standalone_argmax_equal": argmax_equal,
            "changed_document_positive_control": positive_control,
            "finite": finite, "training_format_gate_passed": passed,
            "release_g6_passed": False, "synthetic_ids_used_only_for_isolation_control": True,
            "packed_logits": [value.cpu().tolist() for value in packed],
            "standalone_logits": [value.cpu().tolist() for value in standalone],
            "perturbed_logits": [value.cpu().tolist() for value in perturbed],
        }
        atomic_json(args.out / f"packing-control-rank-{rank}.json", result)
        failed = torch.tensor(int(not passed), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if bool(failed):
            raise ValueError("The full-pretrained packing format failed its fixed control.")
        status("packing_control_passed", packing_control_passed=True,
               packing_maximum_probability_tv=max(standalone_tv + isolated_tv))

    def save_checkpoint(step):
        status("saving_checkpoint", checkpoint_step=step)
        directory = args.out / f"checkpoint-{step:06d}"
        if (directory / "complete.json").exists():
            existing = json.loads((directory / "complete.json").read_text())
            validate_checkpoint_adapter_scope(existing, scope)
            if existing["step"] != step or any(
                file_hash(directory / name) != digest for name, digest in existing["files"].items()
            ):
                raise ValueError("An existing checkpoint was changed.")
            dist.barrier()
            status("training")
            return directory
        model_state, optim_state = get_state_dict(model, optimizer, options=options)
        if any("lora_" not in n and n in dict(model.named_parameters()) for n in model_state):
            raise ValueError("Adapter checkpoint includes frozen parameters.")
        dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=directory)
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                    "step": step, "curriculum_sha256": record["curriculum_sha256"],
                    "source_revision": REVISION,
                    "adapter_scope_sha256": record["adapter_scope_sha256"]},
                   directory / f"rng-rank-{rank}.pt")
        dist.barrier()
        if rank == 0:
            atomic_json(directory / "complete.json", {
                "step": step, "source_revision": REVISION,
                "producer_runtime": {
                    "python": list(sys.version_info[:2]), "torch": torch.__version__,
                },
                "adapter_scope": scope,
                "adapter_scope_sha256": record["adapter_scope_sha256"],
                "curriculum_sha256": record["curriculum_sha256"], "world_size": 8,
                "questions_per_rank": args.pack_questions,
                "global_questions_per_update": global_questions,
                "question_cursor": start_cursor + (step - start_updates) * global_questions,
                "starting_optimizer_updates": start_updates,
                "starting_question_cursor": start_cursor,
                "updates_this_run": step - start_updates,
                "resume_reference_sha256": record.get("resume_reference_sha256"),
                "files": {str(p.relative_to(directory)): file_hash(p)
                          for p in directory.rglob("*") if p.is_file()},
            })
        dist.barrier()
        status("training")
        return directory

    def restore_checkpoint(directory):
        status("restoring_checkpoint", checkpoint_restore_path=str(directory))
        marker = json.loads((directory / "complete.json").read_text())
        restored_scope = validate_checkpoint_adapter_scope(marker, scope)
        ms, op = get_state_dict(model, optimizer, options=options)
        dcp.load({"model": ms, "optimizer": op}, checkpoint_id=directory)
        set_state_dict(model, optimizer, model_state_dict=ms, optim_state_dict=op,
                       options=options)
        rng = torch.load(directory / f"rng-rank-{rank}.pt", map_location="cpu",
                         weights_only=True)
        if (rng["curriculum_sha256"] != record["curriculum_sha256"]
                or rng["source_revision"] != REVISION or rng["step"] != marker["step"]
                or not restored_scope["legacy_full_scope"]
                and rng.get("adapter_scope_sha256") != restored_scope["scope_sha256"]):
            raise ValueError("Checkpoint references a different curriculum.")
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        if (not torch.equal(rng["cpu"], torch.get_rng_state())
                or not torch.equal(rng["cuda"], torch.cuda.get_rng_state(device))):
            raise ValueError("The saved CPU/CUDA RNG did not restore exactly.")

    def restore_initial_state():
        if resume_reference is None:
            return
        restore_checkpoint(args.resume_checkpoint)
        # Independently decode the source DCP on CPU before this GPU job. Here
        # consolidate adapters only, and compare exact values on rank zero.
        ms, op = get_state_dict(model, optimizer, options=options)
        if any("lora_" not in name and name in dict(model.named_parameters()) for name in ms):
            raise ValueError("Resume verification unexpectedly includes frozen parameters.")
        full_model = materialize_reference_state(ms, device, keep=rank == 0)
        full_optimizer = materialize_reference_state(op, device, keep=rank == 0)
        check = {"adapter_optimizer_values_exact": False}
        if rank == 0:
            try:
                check = verify_materialized_state(
                    full_model, full_optimizer, resume_reference["full_state_signature"],
                )
            except ValueError as error:
                check["error"] = str(error)
            atomic_json(args.out / "initial-continuation-values.json", check)
        failed = torch.tensor(
            int(rank == 0 and not check["adapter_optimizer_values_exact"]), device=device,
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if bool(failed):
            raise ValueError("The initial adapter/optimizer continuation is not exact.")
        del full_model, full_optimizer, ms, op
        status("initial_continuation_restored", initial_checkpoint_values_exact=True,
               optimizer_updates=start_updates, question_cursor=start_cursor,
               old_batch_training_trajectory_preserved=False)
        if args.resume_learning_rate is not None:
            from bobcat.native_resume import apply_learning_rate_transition
            transition = apply_learning_rate_transition(
                optimizer, previous=args.resume_learning_rate, current=args.learning_rate,
            )
            atomic_json(args.out / f"learning-rate-transition-rank-{rank}.json", transition)
            record["learning_rate_transition"] = transition

    def update(step):
        nonlocal batched_gradient_statistics
        model.train()
        optimizer.zero_grad(set_to_none=True)
        cursor = start_cursor + step * global_questions
        rows = data["train"][cursor:cursor + global_questions]
        logits = batch_logits(rows, args.pack_questions)
        local_rows = rows[rank::8]
        values = [logits] if args.pack_questions == 1 else logits
        loss = packed_decision_loss(values, local_rows)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss.")
        loss.backward()
        gradients = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                if p.grad is not None:
                    raise ValueError("Frozen pretrained weights received a gradient.")
                continue
            if p.grad is None:
                raise ValueError(f"Adapter gradient missing: {name}")
            gradients.append(p.grad)
        if args.batch_gradient_statistics:
            from bobcat.gradient_statistics import (
                admit_gradient_statistics,
                compare_gradient_statistics,
                gradient_norm_squared,
            )
            if step == 0:
                status("checking_gradient_statistics")
                comparison = compare_gradient_statistics(gradients, device=device)
                comparison.update(rank=rank, source_revision=REVISION, parameter_updates=0)
                atomic_json(args.out / f"gradient-statistics-admission-rank-{rank}.json",
                            comparison)
                reports = [None] * 8
                dist.all_gather_object(reports, comparison)
                decision = admit_gradient_statistics(reports)
                batched_gradient_statistics = decision["activated"]
                if rank == 0:
                    atomic_json(args.out / "gradient-statistics-admission-complete.json",
                                decision)
                record["batched_gradient_statistics"] = decision
                status("training", batched_gradient_statistics=decision)
            norm2 = gradient_norm_squared(gradients, batched=batched_gradient_statistics)
        else:
            norm2 = 0.
            for gradient in gradients:
                grad = gradient.to_local() if hasattr(gradient, "to_local") else gradient
                if not torch.isfinite(grad).all():
                    raise ValueError("Non-finite gradient.")
                norm2 += float(grad.double().square().sum())
        norm = torch.tensor(norm2, device=device, dtype=torch.float64)
        dist.all_reduce(norm)
        norm = float(norm.sqrt())
        factor = min(1., 1. / max(norm, 1e-12))
        for p in params.values():
            local = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            local.mul_(factor)
        for index, (_, module) in enumerate(adapter_modules):
            for parameter in (module.lora_A.weight, module.lora_B.weight):
                local = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") \
                    else parameter.grad
                global_target_norms[index] += local.float().norm().to(device)
        optimizer.step()
        model.update_moe_gate_bias()
        reduced = loss.detach().clone()
        dist.all_reduce(reduced)
        torch.cuda.synchronize(device)
        _, padded = question_packs(rows, 8, args.pack_questions)
        return {"step": start_updates + step + 1, "update_this_run": step + 1,
                "question_cursor": cursor + global_questions,
                "mean_loss": float(reduced / 8), "gradient_norm": norm,
                "global_prompt_tokens": sum(r["input_tokens"] for r in rows),
                "global_questions": len(rows), "global_padded_tokens": padded * 8,
                "local_logits": values[0].detach().cpu().tolist() if args.pack_questions == 1
                else [value.detach().cpu().tolist() for value in values],
                "local_prediction": int(values[0].argmax()) if args.pack_questions == 1
                else [int(value.argmax()) for value in values]}

    def base_forward(module, inputs):
        from nemo_automodel.components._peft.lora import tp_linear_forward
        return tp_linear_forward(inputs, module.weight, module.bias, mm_for_2d_compile=False)

    def kernel_call(name, impl):
        def call(*a, **kw):
            kernels[name] += 1
            return impl(*a, **kw)
        return call

    with ExitStack() as stack:
        for name in ("_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d"):
            if getattr(layers, name) is None:
                raise ValueError(f"Actual CUDA kernel unavailable: {name}")
            stack.enter_context(patch.object(
                layers, name, kernel_call(name, getattr(layers, name)),
            ))
        for name, module in adapter_modules:
            def counted(_m, _a, _r, key=name):
                calls[key] += 1
            stack.callback(module.register_forward_hook(counted).remove)
        model.eval()
        initial_rows = data["dev_train"][:8]
        with torch.no_grad(), ExitStack() as base:
            for _, module in adapter_modules:
                base.enter_context(patch.object(
                    module, "forward", MethodType(base_forward, module),
                ))
            reference = batch_logits(initial_rows).cpu()
        with torch.no_grad():
            adapted = batch_logits(initial_rows).cpu()
        if not torch.equal(reference, adapted):
            raise ValueError("Zero-adapter native readout changed.")
        status("baseline_evaluation", zero_adapter_identity_exact=True)
        evaluate("baseline")
        if resume_reference is not None:
            restore_initial_state()
            evaluate("resume-start")
        if monitoring_rows is not None:
            evaluate("expanded-start", monitoring_rows)
        check_native_projection()
        if args.pack_questions > 1:
            status("checking_packing")
        check_native_packing()
        check_packed_boundary_cache()
        status("training", optimizer_updates=start_updates, updates_this_run=0)
        logs, consumed_tokens = [], 0
        for step in range(args.max_updates):
            if step >= 4:
                remaining = args.max_seconds - (time.monotonic() - started)
                stop = torch.tensor(
                    int(remaining < args.finalization_reserve_seconds), device=device,
                )
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if bool(stop):
                    break
            before = time.monotonic()
            metrics = update(step)
            metrics["wall_seconds"] = time.monotonic() - before
            consumed_tokens += metrics["global_prompt_tokens"]
            metrics["at"] = datetime.now(UTC).isoformat()
            logs.append(metrics)
            with (args.out / f"updates-rank-{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            status("training", optimizer_updates=start_updates + step + 1,
                   updates_this_run=step + 1,
                   processed_training_tokens=consumed_tokens, last_update=metrics)
            if step == 0:
                first_checkpoint = save_checkpoint(start_updates + 1)
                first_adapter = local_copy(params)
                first_optimizer = local_copy(optimizer.state_dict())
                first_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
            if step == 1:
                expected = local_copy(params)
                expected_opt = local_copy(optimizer.state_dict())
                rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
                restore_checkpoint(first_checkpoint)
                from bobcat.glm_adapter_probe import _same_tree
                restored_exact = (
                    _same_tree(first_adapter, local_copy(params))
                    and _same_tree(first_optimizer, local_copy(optimizer.state_dict()))
                    and torch.equal(first_rng[0], torch.get_rng_state())
                    and torch.equal(first_rng[1], torch.cuda.get_rng_state(device))
                )
                atomic_json(args.out / f"restore-comparison-rank-{rank}.json", {
                    "checkpoint_restored_exact": restored_exact,
                    "adapter_difference": replay_differences(first_adapter, local_copy(params)),
                    "optimizer_difference": replay_differences(
                        first_optimizer, local_copy(optimizer.state_dict()),
                    ),
                    "cpu_rng_exact": torch.equal(first_rng[0], torch.get_rng_state()),
                    "cuda_rng_exact": torch.equal(first_rng[1], torch.cuda.get_rng_state(device)),
                })
                restore_failed = torch.tensor(int(not restored_exact), device=device)
                dist.all_reduce(restore_failed, op=dist.ReduceOp.MAX)
                if bool(restore_failed):
                    raise ValueError("Adapter/optimizer/RNG checkpoint restoration is not exact.")
                status("replay_validation")
                replay = update(1)
                adapter_diff = replay_differences(expected, local_copy(params))
                optimizer_diff = replay_differences(
                    expected_opt, local_copy(optimizer.state_dict()),
                )
                rng_exact = (torch.equal(rng_cpu, torch.get_rng_state())
                             and torch.equal(rng_cuda, torch.cuda.get_rng_state(device)))
                metrics_exact = replay == {
                    k: v for k, v in metrics.items() if k not in ("wall_seconds", "at")
                }
                exact = (metrics_exact and not adapter_diff["different_tensors"]
                         and not optimizer_diff["different_tensors"]
                         and not adapter_diff["metadata_differences"]
                         and not optimizer_diff["metadata_differences"] and rng_exact)
                replay_report = {
                    "checkpoint_restored_exact": restored_exact, "update_replay_exact": exact,
                    "rng_exact_after_replay": rng_exact, "original": metrics, "replay": replay,
                    "adapter_difference": adapter_diff, "optimizer_difference": optimizer_diff,
                    "diagnostic_only": args.diagnose_replay,
                    "quality_gate_passed": False, "numerical_tolerance_relaxed": False,
                }
                atomic_json(args.out / f"replay-comparison-rank-{rank}.json", replay_report)
                invalid = torch.tensor(int(
                    not rng_exact or adapter_diff["nonfinite"] or optimizer_diff["nonfinite"]
                    or bool(adapter_diff["metadata_differences"])
                    or bool(optimizer_diff["metadata_differences"])
                ), device=device)
                dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
                if bool(invalid):
                    raise ValueError("Replay changed RNG/metadata or produced non-finite state.")
                replay_failed = torch.tensor(int(not exact), device=device)
                dist.all_reduce(replay_failed, op=dist.ReduceOp.MAX)
                if bool(replay_failed) and not args.diagnose_replay:
                    raise ValueError("Same-process adapter/optimizer/RNG replay diverged.")
                status("training", adapter_resume_replay_exact=exact,
                       adapter_checkpoint_restore_exact=restored_exact,
                       replay_diagnostic_only=args.diagnose_replay,
                       resume_scope="adapter state in unchanged verified base, same process")
                del expected, expected_opt, first_adapter, first_optimizer, first_rng
            if (step + 1) % args.checkpoint_every == 0:
                save_checkpoint(start_updates + step + 1)
            if args.evaluation_every and (step + 1) % args.evaluation_every == 0:
                evaluate(f"checkpoint-{start_updates + step + 1:06d}")
                status("training")
        completed = start_updates + len(logs)
        checkpoint = save_checkpoint(completed)
        status("final_evaluation", final_checkpoint=str(checkpoint), optimizer_updates=completed)
        evaluate("final")
        if monitoring_rows is not None:
            evaluate("expanded-final", monitoring_rows)
        status("verifying_frozen_base")
        if frozen_hashes() != frozen:
            raise ValueError("A pretrained parameter or persistent buffer changed.")
        dist.all_reduce(global_target_norms)
        if (not bool((global_target_norms > 0).all()) or len(calls) != len(targets)
                or set(kernels) != {"_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d"}):
            raise ValueError("An intended adapter or backend received no training signal.")
        atomic_json(args.out / f"gradient-coverage-rank-{rank}.json", {
            "targets": [n for n, _ in adapter_modules],
            "global_norm_sums": global_target_norms.cpu().tolist(),
            "calls": dict(calls), "kernels": dict(kernels),
        })
        status("completed", status="completed", frozen_base_preserved=True,
               processed_training_tokens=consumed_tokens, optimizer_updates=completed,
               updates_this_run=len(logs),
               question_cursor=start_cursor + len(logs) * global_questions,
               base_parameters=base_count, adapter_parameters=adapter_count,
               quality_gate_passed=False, release_gate_passed=False)


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-model-set", type=Path)
    parser.add_argument("--source-verification", type=Path)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=2048)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--max-seconds", type=int, default=18000)
    parser.add_argument(
        "--finalization-reserve-seconds", type=int, default=900,
        help="Time reserved for final evaluation and frozen-weight verification.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--expert-backend", choices=("torch", "torch_mm"), default="torch")
    parser.add_argument(
        "--resume-learning-rate", type=float,
        help="Exact parent optimizer learning rate before an explicit lower-rate intervention.",
    )
    parser.add_argument("--seed", type=int, default=202609230001)
    parser.add_argument("--pack-questions", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--cache-packed-boundaries", action="store_true",
        help="Try cached native boundaries; require exact eight-rank backward and speed gates.",
    )
    parser.add_argument(
        "--batch-gradient-statistics", action="store_true",
        help="Try batched gradient checks after exact eight-rank value and timing admission.",
    )
    parser.add_argument("--evaluation-every", type=int, default=0)
    parser.add_argument(
        "--evaluation-suite", type=Path,
        help="Frozen disjoint development suite, evaluated at the resumed and final checkpoints.",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-reference", type=Path)
    parser.add_argument("--decision-identifiers", type=Path)
    parser.add_argument(
        "--adapter-last-layers", type=int,
        help="Train only this suffix of native language layers; keep the full backbone.",
    )
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--cpu-offload", action=argparse.BooleanOptionalAction, default=True,
        help="Default: CPU-offloaded shards. Disable explicitly for resident H200/B200/B300.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Require deterministic Torch kernels; unsupported operations must fail.",
    )
    parser.add_argument(
        "--diagnose-replay", action="store_true",
        help="Only a bounded <=8-update study; record drift without claiming a pass.",
    )
    args = parser.parse_args()
    if (int(os.environ.get("WORLD_SIZE", "0")) != 8 or not torch.cuda.is_available()
            or not 4 <= args.max_updates <= 2048 or not 1800 <= args.max_seconds <= 21600
            or args.diagnose_replay and args.max_updates > 8
            or bool(args.resume_checkpoint) != bool(args.resume_reference)
            or args.resume_learning_rate is not None and not args.resume_reference
            or bool(args.source_model_set) != bool(args.source_verification)
            or not 180 <= args.finalization_reserve_seconds <= 900
            or args.evaluation_suite and args.finalization_reserve_seconds < 900
            or args.adapter_last_layers is not None and not 1 <= args.adapter_last_layers <= 45
            or args.evaluation_every not in (0, 16, 32, 64, 128)):
        parser.error("Require a finite, exclusive eight-GPU native training job.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    rank = dist.get_rank()
    record = {
        "schema": "bobcat-glm-native-training-rank-v1", "rank": rank, "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "trainer_sha256": file_hash(Path(__file__)),
        "source_revision": REVISION, "pretrained_weights_loaded": False,
        "quality_gate_passed": False, "activation_checkpointing": args.activation_checkpointing,
        "cpu_offload": args.cpu_offload,
    }
    try:
        if rank == 0:
            if args.out.exists():
                raise ValueError("Preserve prior training artifacts.")
            args.out.mkdir(parents=True)
            record["vendor_verification"] = verify_source(args.source_root, args.git_tree)
        dist.barrier()
        sys.path.insert(0, str(args.source_root.resolve()))
        run(args, record)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:3000])
        raise
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        if args.out.exists():
            atomic_json(args.out / f"rank-{rank}.json", record)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
