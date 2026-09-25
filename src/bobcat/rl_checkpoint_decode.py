"""Independently decode immutable native RL checkpoints on a matching CPU runtime."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.native_resume import state_signature
from bobcat.native_resume_reference import validate_decoded_adapters, validate_decoder_runtime
from bobcat.schema import file_hash, json_hash


def compare_local_actor(full, local_states, *, updates, loop):
    """Compare separate DCP reconstruction with eight first-axis native shards.

    This is the pinned pilot's layout, not a general resharder. The only accepted
    serialized wrapper spelling is at the known language-layer boundary. The
    producer separately verified real CheckpointWrapper modules before loading.
    """
    import torch

    if len(local_states) != 8 or not full:
        raise ValueError("Require all eight local continuation states.")
    aliases = {}
    for name in full:
        wrapped = re.sub(r"(\.layers\.\d+)\.", r"\1._checkpoint_wrapped_module.", name, count=1)
        for alias in {name, wrapped}:
            if alias in aliases and aliases[alias] != name:
                raise ValueError("Ambiguous canonical actor names.")
            aliases[alias] = name
    for rank, state in enumerate(local_states):
        if state["updates"] != updates or state["loop"] != loop:
            raise ValueError("A local continuation counter differs from the completion marker.")
        canonical = {}
        for name, tensor in state["actor"].items():
            key = aliases.get(name)
            if key is None or key in canonical:
                raise ValueError("Unexpected or duplicated native actor key.")
            canonical[key] = tensor
        if set(canonical) != set(full):
            raise ValueError("Missing native actor shards.")
        for name, tensor in full.items():
            if tensor.ndim < 1 or tensor.shape[0] % 8:
                raise ValueError("The checkpoint is outside the pinned first-axis shard layout.")
            wanted = tensor.chunk(8, dim=0)[rank]
            actual = canonical[name]
            if (actual.shape != wanted.shape or actual.dtype != wanted.dtype
                    or actual.device.type != "cpu" or not torch.equal(actual, wanted)):
                raise ValueError(f"Native actor shard differs from independent DCP: {rank} {name}")
        states = state["optimizer"]["state"]
        if (len(states) != len(full)
                or any(float(item["step"]) != updates for item in states.values())):
            raise ValueError("Local optimizer steps differ from the completed update count.")
    return {"all_eight_local_actor_shards_match_dcp": True,
            "local_optimizer_counters_match": True,
            "new_process_gpu_next_update_verified": False}


def decode(checkpoint: Path, parent_adapter: Path, *, recipe: str, out: Path) -> dict:
    import torch
    from safetensors.torch import load_file, save_file
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    if out.exists():
        raise ValueError("Preserve an existing independent decode.")
    marker_file = checkpoint / "complete.json"
    marker = json.loads(marker_file.read_text())
    receipt = json.loads((checkpoint / "recovery.json").read_text())
    if (marker.get("schema") != "bobcat-native-rl-checkpoint-v1"
            or marker.get("world_size") != 8 or marker.get("recipe_sha256") != recipe
            or marker.get("expert_backend") not in ("torch", "torch_mm")
            or receipt.get("all_members_fresh_get_verified") is not True
            or receipt["marker"]["sha256"] != file_hash(marker_file)
            or set(receipt["members"]) != set(marker["files"])
            or marker["parent_adapter_sha256"] != file_hash(parent_adapter)):
        raise ValueError("Require a recovered checkpoint from the exact parent and recipe.")
    validate_decoder_runtime(marker["producer_runtime"], python_version=sys.version_info,
                             torch_version=torch.__version__)
    for name, digest in marker["files"].items():
        if (Path(name).name != name or name in (".", "..") or "\\" in name
                or receipt["members"][name]["sha256"] != digest
                or receipt["members"][name]["fresh_get_verified"] is not True
                or not receipt["members"][name]["version_id"]
                or (checkpoint / name).is_symlink()
                or file_hash(checkpoint / name) != digest):
            raise ValueError("A recovered checkpoint member changed before CPU decode.")
    updates = marker["actor_optimizer_updates"]
    if type(updates) is not int or updates < 0 or marker["loop"]["cursor"] != updates:
        raise ValueError("Unexpected native study update count.")
    parent = load_file(parent_adapter, device="cpu")
    with tempfile.TemporaryDirectory(prefix="bobcat-rl-dcp-cpu-") as folder:
        consolidated = Path(folder) / "state.pt"
        dcp_to_torch_save(checkpoint, consolidated)
        state = torch.load(consolidated, map_location="cpu", weights_only=True)
        if (set(state) != {"model", "optimizer"}
                or any("lora_" not in name and not name.endswith(".e_score_correction_bias")
                       for name in state["model"])):
            raise ValueError("Unexpected decoded state outside the original adapter checkpoint.")
        # The reusable validator checks all 360 BF16 tensors, parameter count,
        # finite values and all independently reconstructed Adam update counters.
        tensors = validate_decoded_adapters(state, {"step": updates}, {})
        if (set(tensors) != set(parent)
                or any(t.shape != parent[n].shape or t.dtype != parent[n].dtype
                       for n, t in tensors.items())):
            raise ValueError("The decoded actor no longer has the parent layout.")
        locals_ = [torch.load(checkpoint / f"state-rank-{rank}.pt", map_location="cpu",
                              weights_only=True) for rank in range(8)]
        local_proof = compare_local_actor(tensors, locals_, updates=updates, loop=marker["loop"])
        changed = [name for name, tensor in tensors.items()
                   if not torch.equal(tensor, parent[name])]
        if updates == 0 and changed:
            raise ValueError("An arm-start actor differs from the retained parent.")
        full_signature = state_signature(state)
        out.mkdir(parents=True)
        adapter_path = out / "adapter.safetensors"
        save_file({n: t.contiguous().clone() for n, t in tensors.items()}, adapter_path, metadata={
            "format": "bobcat-native-attention-lora-v1", "base_repo": "zai-org/GLM-5.3-Flash",
            "base_revision": marker["source_revision"],
            "source_complete_sha256": file_hash(marker_file),
            "scale": "2.0", "complete_model": "false",
        })
        if state_signature(load_file(adapter_path, device="cpu")) != state_signature(tensors):
            raise ValueError("Portable export differs from the independent decoded actor.")
    result = {
        "schema": "bobcat-native-rl-cpu-decode-v1", "at": datetime.now(UTC).isoformat(),
        "checkpoint_complete_sha256": file_hash(marker_file),
        "recovery_receipt_sha256": file_hash(checkpoint / "recovery.json"),
        "recipe_sha256": recipe, "producer_job_sha256": marker["job_sha256"],
        "loop": marker["loop"], "actor_optimizer_updates": updates,
        "parent_adapter_sha256": marker["parent_adapter_sha256"],
        "adapter": {"file": adapter_path.name, "sha256": file_hash(adapter_path),
                    "bytes": adapter_path.stat().st_size, "tensors": len(tensors),
                    "parameters": sum(t.numel() for t in tensors.values()),
                    "changed_tensors_from_parent": len(changed),
                    "decoded_values_exact": True, "complete_model": False},
        "full_state_signature": full_signature,
        "full_state_sha256": json_hash(full_signature),
        "independent_decoder": "torch.distributed.checkpoint.format_utils.dcp_to_torch_save",
        "decoder_python_version": list(sys.version_info[:3]),
        "decoder_torch_version": torch.__version__, "gpu_used": False,
        "training_performed": False, "new_process_gpu_next_update_verified": False,
        "release_quality_passed": False, **local_proof,
    }
    atomic_json(out / "decode.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parent-adapter", required=True, type=Path)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = decode(args.checkpoint, args.parent_adapter, recipe=args.recipe, out=args.out)
    print(json.dumps({k: result[k] for k in (
        "actor_optimizer_updates", "adapter", "all_eight_local_actor_shards_match_dcp",
        "gpu_used", "new_process_gpu_next_update_verified",
    )}))


if __name__ == "__main__":
    main()
