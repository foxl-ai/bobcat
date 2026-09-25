"""Eight-process CPU EP4/shard2 DCP check for the GLM layerwise FP8 loader.

This tests real DTensor ownership, disk reads and exact reconstruction. It does
not execute FSDP hooks, CUDA offload, pretrained loading, or model backward.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import verify_source
from bobcat.glm_checkpoint_parts import bounded_glm_adapter
from bobcat.schema import file_hash


def probe(fixture, checkpoint_dir, out, *, compact_output=False):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from nemo_automodel.components.checkpoint._backports.hf_storage import (
        _HuggingFaceStorageReader,
    )
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    if dist.get_world_size() != 8:
        raise ValueError("The fixed diagnostic requires eight CPU processes.")
    world = init_device_mesh("cpu", (8,), mesh_dim_names=("dp_shard_cp",))
    experts = init_device_mesh("cpu", (2, 4), mesh_dim_names=("ep_shard", "ep"))
    config = fixture.tiny_glm5_next_config()
    config.text_config.torch_dtype = torch.bfloat16
    config.vision_config.torch_dtype = torch.bfloat16
    backend = BackendConfig(
        attn="sdpa", linear="torch", rms_norm="torch_fp32", experts="torch",
        dispatcher="torch", rope_fusion=False, enable_hf_state_dict_adapter=True,
    )
    model = Glm5NextForConditionalGeneration(config, backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=torch.bfloat16)
    expected = model.state_dict_adapter.from_hf(
        load_file(str(checkpoint_dir / "model.safetensors"))
    )
    bank = None
    if compact_output:
        bank = expected["lm_head.weight"][[23, 8, 41, 7, 22, 20, 9, 40, 21]].clone()
        expected["lm_head.weight"] = bank
    targets, placements = {}, {}
    for name, value in expected.items():
        if ".mlp.experts." in name:
            mesh, where = experts, (Shard(1), Shard(0))
        else:
            mesh, where = world, (Shard(0),) if value.ndim else (Replicate(),)
        targets[name] = distribute_tensor(torch.zeros_like(value), mesh, where)
        placements[name] = {
            "mesh": list(mesh.mesh_dim_names), "placement": [str(item) for item in where],
            "global_shape": list(value.shape), "local_shape": list(targets[name].to_local().shape),
        }
    storage = {name: value.to_local().data_ptr() for name, value in targets.items()}
    if compact_output:
        from bobcat.glm_compact_head import compact_glm_checkpoint_adapter
        adapter = compact_glm_checkpoint_adapter(
            model.state_dict_adapter, bank, max_local_layer_bytes=1024**2,
        )
    else:
        adapter = bounded_glm_adapter(model.state_dict_adapter, max_local_layer_bytes=1024**2)
    reader = _HuggingFaceStorageReader(str(checkpoint_dir))
    observations, completed = [], set()
    for part in adapter.iter_checkpoint_load_parts(targets, device_mesh=experts):
        keys = set(part.model_keys)
        if keys & completed:
            raise ValueError("A distributed native tensor was loaded more than once.")
        if compact_output and "lm_head.weight" in part.checkpoint_tensors:
            raise ValueError("Compact loading still requests the full vocabulary output.")
        observations.append({
            "model_keys": sorted(keys), "checkpoint_keys": sorted(part.checkpoint_tensors),
            "temporary_local_bytes": sum(
                (value.to_local() if hasattr(value, "to_local") else value).numel()
                * value.element_size()
                for name, value in part.checkpoint_tensors.items()
                if name in part.temporary_checkpoint_keys
            ),
        })
        dcp.load(part.checkpoint_tensors, storage_reader=reader)
        part.finish()
        for name in sorted(keys):
            actual = targets[name]
            if actual.to_local().data_ptr() != storage[name]:
                raise ValueError("The loader replaced a rank-local tensor's storage.")
            if not torch.equal(actual.full_tensor(), expected[name]):
                raise ValueError(f"Distributed reconstruction mismatch: {name}")
        completed |= keys
        del part
    if completed != set(expected) or len(observations) != 5:
        raise ValueError("Not every native tensor was loaded exactly once.")
    record = {
        "status": "passed", "rank": dist.get_rank(), "world_size": 8,
        "mesh": {"non_expert_shards": 8, "expert_parallel": 4, "expert_shards": 2},
        "all_native_tensors_exact": True, "rank_local_storage_preserved": True,
        "native_tensors": len(expected), "placements": placements,
        "parts": observations, "cuda_executed": False,
        "fsdp_hooks_tested": False, "pretrained_weights_loaded": False,
        "full_model_memory_fit_verified": False, "model_forward_backward_executed": False,
        "compact_output": compact_output,
        "compact_output_shape": list(bank.shape) if compact_output else None,
        "full_vocabulary_output_requested": not compact_output,
    }
    atomic_json(out / f"rank-{dist.get_rank()}.json", record)
    dist.barrier()
    return record


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--git-tree", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compact-output", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.set_num_threads(1)
    started = time.monotonic()
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    record = {
        "schema": "bobcat-glm-bounded-checkpoint-cpu-distributed-v1",
        "status": "running", "started_at": datetime.now(UTC).isoformat(),
        "probe_sha256": file_hash(Path(__file__)),
        "loader_sha256": file_hash(Path(__file__).with_name("glm_checkpoint_parts.py")),
        "compact_loader_sha256": file_hash(Path(__file__).with_name("glm_compact_head.py"))
        if args.compact_output else None,
        "release_gate_passed": False,
    }
    try:
        if rank == 0:
            if args.out.exists():
                raise ValueError("Use a fresh output for the distributed diagnostic.")
            args.out.mkdir(parents=True)
            record["source"] = verify_source(args.source_root, args.git_tree)
            record["checkpoint_sha256"] = file_hash(args.checkpoint_dir / "model.safetensors")
            if record["checkpoint_sha256"] != args.checkpoint_sha256:
                raise ValueError("The synthetic checkpoint identity changed.")
            record["versions"] = {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "safetensors")
            }
            atomic_json(args.out / "result.json", record)
        dist.barrier()
        sys.path.insert(0, str(args.source_root.resolve()))
        fixture_path = args.source_root / "tests/unit_tests/models/glm5_next/conftest.py"
        spec = importlib.util.spec_from_file_location("bobcat_glm_loader_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        result = probe(fixture, args.checkpoint_dir, args.out,
                       compact_output=args.compact_output)
        record["status"] = result["status"]
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
        if args.out.exists():
            atomic_json(args.out / f"rank-{rank}-failure.json", record)
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        if rank == 0 and args.out.exists():
            atomic_json(args.out / "result.json", record)
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({
            "status": record["status"], "cpu_processes": 8,
            "cuda_executed": False, "pretrained_weights_loaded": False,
        }))


if __name__ == "__main__":
    main()
