"""Finite eight-GPU GLM fixture: EP4, FSDP2, LoRA and DCP resume.

Uses synthetic FP8 weights in the native hybrid GLM architecture. Passing is a
distributed execution prerequisite, not full pretrained training or quality.
CPU offload is the legacy default; explicit GPU residency is also tested.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import TARGET_LEAVES, _same_tree, verify_source
from bobcat.glm_checkpoint_parts import bounded_glm_adapter
from bobcat.glm_derived_buffers import restore_vision_rotary_buffer
from bobcat.schema import file_hash, json_hash


class PhaseProgress:
    """Persist the last entered phase even if an external timeout kills a rank.

    These are host-clock diagnostics, not synchronized GPU latency measurements.
    No CUDA synchronization, RNG operation or numerical tolerance is added.
    """

    def __init__(self, path, record, *, clock=time.monotonic):
        self.path, self.record, self.clock = path, record, clock
        self.current = None
        self.completed = []

    def mark(self, name):
        now = self.clock()
        if self.current is not None:
            self.completed.append({
                "phase": self.current["phase"],
                "host_elapsed_seconds": now - self.current["monotonic_start"],
            })
        self.current = {"phase": name, "monotonic_start": now}
        self.record["phase_progress"] = {
            "current_phase": name,
            "entered_at": datetime.now(UTC).isoformat(),
            "clock": "host_monotonic_not_synchronized_gpu_timing",
            "completed_phases": self.completed,
        }
        atomic_json(self.path, self.record)


def local_copy(value):
    import torch
    from torch.distributed.tensor import DTensor

    if isinstance(value, DTensor):
        return value.to_local().detach().cpu().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: local_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(local_copy(item) for item in value)
    return value


def expected_local(full, sharded):
    """Select a rank's contiguous shard without gathering CPU tensors on NCCL."""
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(sharded, DTensor):
        return full
    slices = [slice(None)] * full.ndim
    shape, offsets = list(full.shape), [0] * full.ndim
    for axis, placement in enumerate(sharded.placements):
        if isinstance(placement, Shard):
            dim = placement.dim
            size, offset = Shard.local_shard_size_and_offset(
                shape[dim], sharded.device_mesh.size(axis),
                sharded.device_mesh.get_local_rank(mesh_dim=axis),
            )
            offsets[dim] += offset
            shape[dim] = size
    for dim in range(full.ndim):
        slices[dim] = slice(offsets[dim], offsets[dim] + shape[dim])
    return full[tuple(slices)]


def probe(config_factory, out, *, cpu_offload=True, expert_backend="torch", progress=None):
    mark = progress if progress is not None else lambda _name: None
    mark("vendor_imports")
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.nn.functional as functional
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
    from safetensors.torch import load_file, save_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

    from bobcat.glm_expert_backend import expert_backend_name

    expert_backend = expert_backend_name(expert_backend)

    mark("distributed_mesh_and_fixture_config")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.10, device)
    torch.manual_seed(202609222125)
    torch.cuda.manual_seed_all(202609222125)
    if dist.get_world_size() != 8 or not all(getattr(layers, name) for name in (
        "_SHORT_CONV_OK", "_CHUNK_KDA_OK", "_KDA_GATE_OK",
    )):
        raise ValueError("Eight exclusive CUDA ranks and the real FLA kernels are required.")
    world = init_device_mesh("cuda", (8,), mesh_dim_names=("dp_shard_cp",))
    expert_mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("ep_shard", "ep"))
    config = config_factory()
    if (
        config.text_config.hidden_size != 64 or config.text_config.num_hidden_layers != 4
        or config.text_config.n_routed_experts != 8 or config.text_config.vocab_size != 96
    ):
        raise ValueError("Only the pinned small synthetic CUDA fixture may run here.")
    config.text_config.mlp_layer_types[-1] = "sparse"
    config.quantization_config = {
        "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
    }
    backend = dict(
        attn="sdpa", linear="torch", rms_norm="torch_fp32", experts=expert_backend,
        dispatcher="torch", rope_fusion=False, enable_hf_state_dict_adapter=True,
    )
    source_dir = out / "synthetic-fp8"
    mark("synthetic_checkpoint_creation_and_barrier")
    if rank == 0:
        source_dir.mkdir()
        original = Glm5NextForConditionalGeneration(config, backend=BackendConfig(**backend))
        original.initialize_weights(torch.device("cpu"), dtype=torch.bfloat16)
        source = original.state_dict_adapter.to_hf(original.state_dict(), quantization=True)
        save_file({name: value.contiguous() for name, value in source.items()},
                  str(source_dir / "model.safetensors"))
        config.save_pretrained(source_dir)
        del original, source
    dist.barrier()
    checkpoint_hash = file_hash(source_dir / "model.safetensors")
    ownership, memory_phases = [], []

    def capture_memory(phase):
        torch.cuda.synchronize(device)
        memory_phases.append({
            "phase": phase,
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })

    def build(with_adapter):
        build_number = len(ownership) + 1
        mark(f"build_{build_number}_meta_construction")
        torch.cuda.reset_peak_memory_stats(device)
        with torch.device("meta"), local_torch_dtype(torch.bfloat16, "Bobcat FSDP fixture"):
            model = Glm5NextForConditionalGeneration(config, backend=BackendConfig(**backend))
            if sum(parameter.numel() for parameter in model.parameters()) != 255044:
                raise ValueError("The synthetic fixture's parameter count changed.")
            model.requires_grad_(False)
            for module in model.modules():
                if isinstance(module, Gate):
                    module.bias_update_factor = 0.0
            names = [
                name for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and ".self_attn." in name
                and ".indexer." not in name and name.rsplit(".", 1)[-1] in TARGET_LEAVES
            ]
            if len(names) != 16:
                raise ValueError("The small fixture's projection layout changed.")
            if with_adapter:
                peft = PeftConfig(
                    target_modules=names, dim=4, alpha=8, dropout=0.0,
                    lora_dtype=torch.bfloat16, use_triton=False,
                    use_memory_efficient_lora=False,
                )
                if apply_lora_to_linear_modules(model, peft) != len(names):
                    raise ValueError("Not every intended projection received its adapter.")
        mark(f"build_{build_number}_parallelize_and_materialize")
        parallelize_model(
            model, world, expert_mesh, dp_axis_names=("dp_shard_cp",),
            ep_axis_name="ep", ep_shard_axis_names=("ep_shard",),
            activation_checkpointing=False, reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16, cast_forward_inputs=True,
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else None,
        )
        Checkpointer.initialize_model_weights(
            model, torch.device("cpu") if cpu_offload else device,
            peft_init_method="xavier" if with_adapter else None,
        )
        model.state_dict_adapter = bounded_glm_adapter(
            model.state_dict_adapter, max_local_layer_bytes=1024**2,
            allow_expert_cuda_staging=True,
            allow_fp8_cuda_staging=True,
        )
        checkpointer = Checkpointer(
            CheckpointingConfig(
                is_peft=with_adapter, save_consolidated=False,
                model_cache_dir=str(source_dir), dequantize_base_checkpoint=True,
                cpu_offload=cpu_offload,
            ),
            dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=expert_mesh,
        )
        mark(f"build_{build_number}_base_checkpoint_load")
        checkpointer.load_base_model(
            model, device, str(source_dir), str(source_dir), load_base_model=True,
        )
        derived_buffer = restore_vision_rotary_buffer(model, device)
        raw_adapter = type(model.state_dict_adapter).__mro__[1](
            model.config, model.model.language_model.moe_config, model.backend,
            torch.bfloat16,
        )
        mark(f"build_{build_number}_reference_shard_verification")
        expected = raw_adapter.from_hf(load_file(str(source_dir / "model.safetensors")))
        state = ModelState(model, is_peft=with_adapter, is_init_step=True).state_dict()
        if set(state) != set(expected):
            raise ValueError("Base loading did not expose every native parameter and buffer.")
        for name, value in state.items():
            if not torch.equal(local_copy(value), expected_local(expected[name], value)):
                raise ValueError(f"Loaded native shard differs from the reference: {name}")
        params = {name: value for name, value in model.named_parameters() if value.requires_grad}
        required = {f"{name}.lora_{letter}.weight" for name in names for letter in "AB"}
        if set(params) != (required if with_adapter else set()):
            raise ValueError("Trainable ownership changed after FSDP/EP construction.")
        for name, value in model.named_parameters():
            local = value.to_local() if hasattr(value, "to_local") else value
            expected_device = "cpu" if cpu_offload else "cuda"
            if local.device.type != expected_device:
                raise ValueError(f"Parameter is not stored on {expected_device}: {name}")
        # FSDP offloads parameters, not the gate/rotary buffers. Meta
        # materialization above deliberately put everything on CPU for loading.
        # Place only the small, exactly verified buffers on the execution device;
        # do not call model.to(cuda), which would defeat parameter offloading.
        mark(f"build_{build_number}_buffer_and_storage_verification")
        buffer_bytes, moved_buffers = 0, []
        for module_name, module in model.named_modules():
            for name, value in list(module._buffers.items()):
                if value is None:
                    continue
                key = f"{module_name}.{name}" if module_name else name
                local = value.to_local() if hasattr(value, "to_local") else value
                buffer_bytes += local.numel() * local.element_size()
                if local.device.type == "meta" or buffer_bytes > 1024**2:
                    raise ValueError("Unmaterialized or unexpectedly large fixture buffers.")
                before = local_copy(value)
                moved = value.to(device=device)
                if moved.dtype != value.dtype or not torch.equal(before, local_copy(moved)):
                    raise ValueError(f"Buffer placement changed a loaded value: {key}")
                if hasattr(value, "placements") and (
                    moved.placements != value.placements
                    or moved.device_mesh != value.device_mesh
                ):
                    raise ValueError(f"Buffer placement changed distributed ownership: {key}")
                module._buffers[name] = moved
                moved_buffers.append(key)
        ownership.append({
            "with_adapter": with_adapter, "base_keys": len(state),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in params.values()),
            "all_loaded_shards_exact": True, "all_parameter_storage_cpu": cpu_offload,
            "all_parameter_storage_cuda": not cpu_offload,
            "execution_buffers_cuda": moved_buffers, "local_buffer_bytes": buffer_bytes,
            "derived_buffer_restoration": derived_buffer,
        })
        capture_memory(f"build_{len(ownership)}_including_synthetic_reference_comparison")
        return model, names, params

    inputs = torch.randint(1, 95, (1, 128), device=device)
    documents = torch.tensor([[1] * 64 + [2] * 64], dtype=torch.int32, device=device)
    calls, kernel_calls = Counter(), Counter()

    def logits(model):
        return model(input_ids=inputs, _packed_seq_ids=documents, logits_to_keep=1).logits[:, -1]

    def traced(name, implementation):
        def call(*args, **kwargs):
            tensors = [value for value in (*args, *kwargs.values())
                       if isinstance(value, torch.Tensor) and value.is_floating_point()]
            if not tensors or any(not value.is_cuda for value in tensors):
                raise ValueError("A claimed FLA call did not receive CUDA floating-point inputs.")
            kernel_calls[name] += 1
            return implementation(*args, **kwargs)
        return call

    def snapshot(model):
        return {
            name: local_copy(value) for name, value in model.state_dict().items()
            if "lora_" not in name
        }

    with ExitStack() as stack:
        for name in ("_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d"):
            stack.enter_context(patch.object(layers, name, traced(name, getattr(layers, name))))
        baseline, _, _ = build(False)
        baseline.eval()
        mark("baseline_forward_including_any_cold_compilation")
        with torch.no_grad():
            reference = logits(baseline).detach().cpu()
        del baseline
        model, targets, params = build(True)
        model.eval()
        mark("zero_adapter_forward_and_identity")
        with torch.no_grad():
            actual = logits(model).detach().cpu()
        if not torch.equal(reference, actual):
            raise ValueError("Distributed zero-adapter identity failed.")
        mark("frozen_snapshot")
        frozen = snapshot(model)

        def instrument(current):
            for name in targets:
                def called(_module, _args, _result, key=name):
                    calls[key] += 1
                stack.callback(current.get_submodule(name).register_forward_hook(called).remove)

        instrument(model)
        optimizer = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=0.0, foreach=False)

        def update(current, opt, label):
            mark(f"{label}_forward")
            current.train()
            opt.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                logits(current).float(), torch.tensor([7], device=device),
            )
            mark(f"{label}_backward_including_any_cold_compilation")
            loss.backward()
            mark(f"{label}_gradient_verification")
            norms = {}
            for name, value in current.named_parameters():
                if value.requires_grad:
                    if value.grad is None:
                        raise ValueError(f"Missing distributed adapter gradient: {name}")
                    grad = local_copy(value.grad)
                    if not torch.isfinite(grad).all():
                        raise ValueError(f"Non-finite distributed gradient: {name}")
                    norms[name] = float(grad.float().norm())
                elif value.grad is not None:
                    raise ValueError("A frozen parameter received a gradient.")
            mark(f"{label}_optimizer_and_frozen_verification")
            opt.step()
            current.update_moe_gate_bias()
            torch.cuda.synchronize(device)
            if not _same_tree(frozen, snapshot(current)):
                raise ValueError("A frozen parameter or router buffer changed.")
            return {"loss": float(loss.detach()), "local_gradient_norms": norms}

        torch.cuda.reset_peak_memory_stats(device)
        first = update(model, optimizer, "first_update")
        mark("adapter_optimizer_state_for_save")
        options = StateDictOptions(ignore_frozen_params=True, cpu_offload=True, strict=False)
        model_state, optim_state = get_state_dict(model, optimizer, options=options)
        if any("lora_" not in key and key in dict(model.named_parameters()) for key in model_state):
            raise ValueError("The adapter checkpoint unexpectedly includes frozen parameters.")
        resume_dir = out / "adapter-step1"
        mark("dcp_save_and_rng_save")
        dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=resume_dir)
        rng_path = out / f"rng-rank-{rank}.pt"
        torch.save({
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
            "base_sha256": checkpoint_hash,
        }, rng_path)
        second = update(model, optimizer, "second_update")
        mark("expected_second_update_snapshot")
        expected_model = local_copy(model.state_dict())
        expected_optimizer = local_copy(optimizer.state_dict())
        expected_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device))
        capture_memory("two_updates_and_adapter_checkpoint")
        del model_state, optim_state, params, model, optimizer
        resumed, _, resumed_params = build(True)
        instrument(resumed)
        resumed_optimizer = torch.optim.AdamW(
            resumed_params.values(), lr=1e-3, weight_decay=0.0, foreach=False,
        )
        mark("resume_dcp_load")
        model_state, optim_state = get_state_dict(resumed, resumed_optimizer, options=options)
        dcp.load({"model": model_state, "optimizer": optim_state}, checkpoint_id=resume_dir)
        set_state_dict(
            resumed, resumed_optimizer, model_state_dict=model_state,
            optim_state_dict=optim_state, options=options,
        )
        mark("resume_rng_restore")
        rng = torch.load(rng_path, map_location="cpu", weights_only=True)
        if rng["base_sha256"] != checkpoint_hash:
            raise ValueError("The resume checkpoint points to a different base.")
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        torch.cuda.reset_peak_memory_stats(device)
        repeated = update(resumed, resumed_optimizer, "resumed_update")
        capture_memory("resumed_update")
        mark("exact_replay_verification")
        if (
            repeated != second
            or not _same_tree(expected_model, local_copy(resumed.state_dict()))
            or not _same_tree(expected_optimizer, local_copy(resumed_optimizer.state_dict()))
            or not torch.equal(expected_rng[0], torch.get_rng_state())
            or not torch.equal(expected_rng[1], torch.cuda.get_rng_state(device))
        ):
            raise ValueError("Distributed model/optimizer/RNG trajectory did not resume exactly.")
        mark("final_gradient_collective_and_kernel_coverage")
        norms = torch.tensor([
            sum(step["local_gradient_norms"][f"{name}.lora_{letter}.weight"]
                for step in (first, second) for letter in "AB")
            for name in targets
        ], device=device, dtype=torch.float32)
        dist.all_reduce(norms)
        required_kernels = {"_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d"}
        if (
            not bool((norms > 0).all()) or set(calls) != set(targets)
            or set(kernel_calls) != required_kernels or not all(kernel_calls.values())
        ):
            raise ValueError("An intended adapter or CUDA kernel received no training signal.")
        mark("probe_checks_complete")
        return {
            "status": "passed", "rank": rank, "world_size": 8,
            "config_sha256": json_hash(config.to_dict()), "backend": backend,
            "mesh": {"expert_parallel": 4, "expert_shards": 2, "non_expert_shards": 8},
            "cpu_offload": cpu_offload, "activation_checkpointing": False,
            "ownership": ownership, "adapter_targets": targets,
            "zero_adapter_identity_exact": True, "frozen_tensors_preserved": True,
            "resume_trajectory_exact": True, "steps": [first, second],
            "global_target_gradient_norm_sums": norms.cpu().tolist(),
            "projection_calls": dict(calls), "cuda_kernel_calls": dict(kernel_calls),
            "base_checkpoint_sha256": checkpoint_hash,
            "memory_phases": memory_phases,
            "peak_allocated_bytes": max(p["peak_allocated_bytes"] for p in memory_phases),
            "peak_reserved_bytes": max(p["peak_reserved_bytes"] for p in memory_phases),
            "pretrained_weights_loaded": False, "full_model_memory_fit_verified": False,
            "language_quality_measured": False, "release_gate_passed": False,
        }


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclusive-gpus", action="store_true")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--expert-backend", choices=("torch", "torch_mm"), default="torch")
    args = parser.parse_args()
    if not args.exclusive_gpus or not torch.cuda.is_available():
        parser.error("Allocate eight exclusive research GPUs for this finite fixture.")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if args.deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            parser.error("Set the deterministic cuBLAS workspace before starting the fixture.")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    started = time.monotonic()
    record = {
        "schema": "bobcat-glm-fsdp-adapter-probe-v1", "rank": rank, "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "probe_sha256": file_hash(Path(__file__)),
        "loader_sha256": file_hash(Path(__file__).with_name("glm_checkpoint_parts.py")),
        "compiler_cache": {
            "explicit_triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "warm_cache_reuse_verified": False,
            "note": "Same host or an existing directory does not establish a cache hit.",
        },
        "release_gate_passed": False,
    }
    try:
        if rank == 0:
            if args.out.exists():
                raise ValueError("Preserve earlier evidence; use a fresh output.")
            args.out.mkdir(parents=True)
            record["source"] = verify_source(args.source_root, args.git_tree)
        dist.barrier()
        atomic_json(args.out / f"rank-{rank}.json", record)
        progress = PhaseProgress(args.out / f"rank-{rank}.json", record)
        progress.mark("source_fixture_import")
        sys.path.insert(0, str(args.source_root.resolve()))
        fixture_path = args.source_root / (
            "tests/functional_tests/context_parallel/run_glm5_next_packed_cp_parity.py"
        )
        record["fixture_sha256"] = file_hash(fixture_path)
        record["versions"] = {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "safetensors", "triton", "flash-linear-attention", "fla-core",
        )}
        spec = importlib.util.spec_from_file_location("bobcat_glm_fsdp_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        record["result"] = probe(
            fixture._config, args.out, cpu_offload=args.cpu_offload,
            expert_backend=args.expert_backend, progress=progress.mark,
        )
        record["status"] = record["result"]["status"]
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        if args.out.exists():
            atomic_json(args.out / f"rank-{rank}.json", record)
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"status": record["status"], "pretrained_weights_loaded": False}))


if __name__ == "__main__":
    main()
