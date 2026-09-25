"""CPU-only distributed adapter-state materialization probe; no GLM or GPU claim."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.native_resume import materialize_reference_state, state_signature


def main():
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This is a finite two-process CPU probe.")
        if rank == 0:
            args.out.mkdir(exist_ok=False, parents=True)
        dist.barrier()
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("adapter_shard",))
        full = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
        moments = full.float() / 7
        state = {
            "model": {"lora_A.weight": distribute_tensor(full, mesh, [Shard(0)])},
            "optimizer": {
                "exp_avg": distribute_tensor(moments, mesh, [Shard(0)]),
                "step": torch.tensor(64.),
            },
        }
        materialized = materialize_reference_state(state, torch.device("cpu"), keep=rank == 0)
        expected = {"model": {"lora_A.weight": full},
                    "optimizer": {"exp_avg": moments, "step": torch.tensor(64.)}}
        passed = state_signature(materialized) == state_signature(expected) if rank == 0 else (
            materialized["model"]["lora_A.weight"] is None
            and materialized["optimizer"]["exp_avg"] is None
        )
        atomic_json(args.out / f"rank-{rank}.json", {
            "rank": rank, "passed": passed, "world_size": 2, "device": "cpu",
            "gpu_used": False, "full_pretrained_glm_tested": False,
            "checkpoint_continuation_training_tested": False,
        })
        if not passed:
            raise ValueError("CPU DTensor materialization changed complete state values.")
        dist.barrier()
        if rank == 0:
            print(json.dumps({"cpu_dtensor_materialization_passed": True, "gpu_used": False}))
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    main()
