"""Native checkpoint continuation and side-effect-free development evaluation.

Changing the question batch preserves a verified optimizer state, not the
trajectory of the former batch size. Update count and data cursor stay separate.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

from bobcat.schema import file_hash, json_hash


def state_signature(value):
    """Hash fully materialized values independently of DCP shard serialization."""
    import torch

    if isinstance(value, torch.Tensor):
        if hasattr(value, "placements"):
            raise ValueError("A local DTensor shard is not a full-state fingerprint.")
        array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return {
            "type": "tensor", "dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(memoryview(array)).hexdigest(),
        }
    if isinstance(value, dict):
        return {"type": "dict", "items": [
            [state_signature(key), state_signature(value[key])]
            for key in sorted(value, key=lambda key: (type(key).__name__, str(key)))
        ]}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "items": [state_signature(v) for v in value]}
    if value is None or type(value) in (str, bool, int):
        return {"type": type(value).__name__, "value": value}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value": value}
    raise ValueError(f"Unsupported checkpoint metadata: {type(value).__name__}")


def continuation_cursor(marker, parent_job, train_rows):
    """Legacy checkpoints are usable only with their exact, non-resumed parent job."""
    step = marker.get("step")
    pack = parent_job.get("pack_questions", 1)
    if type(step) is not int or step < 1 or type(pack) is not int or pack not in (1, 2, 4, 8):
        raise ValueError("Invalid native continuation update or batch ownership.")
    if marker.get("world_size") != 8 or parent_job.get("world_size") != 8:
        raise ValueError("Continuation requires the original eight-rank ownership.")
    if (marker.get("source_revision") != parent_job.get("source_revision")
            or marker.get("curriculum_sha256") != parent_job.get("curriculum_manifest_sha256")):
        raise ValueError("Checkpoint and parent job have different data/model lineage.")
    if "question_cursor" in marker:
        cursor = marker["question_cursor"]
    else:
        if parent_job.get("resume_reference"):
            raise ValueError("Do not infer the data cursor after a previous batch transition.")
        cursor = step * 8 * pack
    if type(cursor) is not int or not 0 < cursor <= train_rows or cursor % 8:
        raise ValueError("The checkpoint cursor is outside the frozen curriculum.")
    if marker.get("questions_per_rank", pack) != pack:
        raise ValueError("Checkpoint and parent disagree on question packing.")
    return {"optimizer_updates": step, "question_cursor": cursor,
            "previous_questions_per_rank": pack}


def validate_learning_rate_transition(learning_rate, resume_learning_rate=None):
    """An explicit lower learning rate follows, never bypasses, exact restoration."""
    if (type(learning_rate) not in (int, float) or not math.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("Use a finite positive training learning rate.")
    if resume_learning_rate is None:
        return learning_rate
    if (type(resume_learning_rate) not in (int, float)
            or not math.isfinite(resume_learning_rate)
            or not 0 < learning_rate < resume_learning_rate):
        raise ValueError("A resumed learning-rate intervention must explicitly decrease it.")
    return resume_learning_rate


def apply_learning_rate_transition(optimizer, *, previous, current):
    """Keep every moment, step counter, parameter and RNG; change only group lr."""
    validate_learning_rate_transition(current, previous)
    if (not optimizer.param_groups
            or any(group["lr"] != previous for group in optimizer.param_groups)):
        raise ValueError("The exact restored optimizer has an unexpected learning rate.")
    for group in optimizer.param_groups:
        group["lr"] = current
    return {
        "previous_learning_rate": previous, "learning_rate": current,
        "parameter_groups": len(optimizer.param_groups), "optimizer_moments_reset": False,
        "applied_after_exact_initial_restore": True, "same_training_trajectory_claimed": False,
    }


def validate_resume_reference(reference, root: Path, *, curriculum_sha256, source_revision,
                              train_rows, next_pack, updates, learning_rate,
                              resume_learning_rate=None):
    if reference.get("schema") != "bobcat-native-resume-reference-v1":
        raise ValueError("Use an independently decoded native resume reference.")
    parent_job = reference["parent_job"]
    marker = reference["marker"]
    expected_previous_lr = validate_learning_rate_transition(learning_rate, resume_learning_rate)
    if (marker.get("curriculum_sha256") != curriculum_sha256
            or marker.get("source_revision") != source_revision
            or reference["complete_sha256"] != file_hash(root / "complete.json")
            or json.loads((root / "complete.json").read_text()) != marker
            or parent_job.get("learning_rate") != expected_previous_lr
            or parent_job.get("deterministic") is not True
            or json_hash(parent_job) != reference["parent_job_content_sha256"]
            or json_hash(reference["full_state_signature"]) != reference["full_state_sha256"]):
        raise ValueError("Resume reference does not match the frozen state/optimizer contract.")
    for name, digest in marker["files"].items():
        path = root / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or file_hash(path) != digest):
            raise ValueError("A resumed checkpoint file differs from its completed marker.")
    cursor = continuation_cursor(marker, parent_job, train_rows)
    end = cursor["question_cursor"] + updates * 8 * next_pack
    if end > train_rows or updates < 4 or next_pack not in (1, 2, 4, 8):
        raise ValueError("The continuation would overrun or silently repeat training data.")
    cursor["next_questions_per_rank"] = next_pack
    cursor["batch_changed"] = next_pack != cursor["previous_questions_per_rank"]
    cursor["old_batch_training_trajectory_preserved"] = False
    return cursor


def verify_materialized_state(model_state, optimizer_state, expected_signature):
    actual = state_signature({"model": model_state, "optimizer": optimizer_state})
    if actual != expected_signature:
        raise ValueError("Restored adapter/optimizer values differ from independent DCP decoding.")
    return {"full_state_sha256": json_hash(actual), "adapter_optimizer_values_exact": True}


def materialize_reference_state(value, device, *, keep):
    """All ranks gather small adapter/optimizer DTensors; only rank zero keeps them.

    CPU-offloaded DTensors must move to the NCCL device before all-gather. Do
    not ask FSDP for a consolidated 314B base model just to verify its adapters.
    """
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() > 32 * 1024**2:
            raise ValueError("Reference verification unexpectedly includes a large base tensor.")
        if hasattr(value, "placements"):
            full = value.to(device).full_tensor()
        else:
            full = value
        return full.detach().cpu() if keep else None
    if isinstance(value, dict):
        return {key: materialize_reference_state(item, device, keep=keep)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [materialize_reference_state(item, device, keep=keep) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    return value


@contextmanager
def evaluation_state(model, device):
    """Development evaluation cannot advance either CPU or CUDA training RNG."""
    import torch

    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
