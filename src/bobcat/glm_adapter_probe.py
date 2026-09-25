"""Small CPU correctness probe of pinned AutoModel GLM adapter training.

This imports the vendor's four-layer, 17K-parameter fixture. It does not load
pretrained weights, exercise CUDA kernels, or measure language quality.
Run inside a separate environment satisfying the pinned vendor dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash

REVISION = "2a372db01d5b98aaa01b15747887852f074667ce"
TARGET_LEAVES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa",
}


def verify_source(root: Path, tree_path: Path) -> dict:
    tree = json.loads(tree_path.read_text())
    if tree.get("sha") != REVISION or tree.get("truncated"):
        raise ValueError("Supply the complete pinned vendor Git tree.")
    count, size, skipped_links = 0, 0, []
    for item in tree["tree"]:
        if item["type"] != "blob":
            continue
        if item["mode"] == "120000":
            # Contributor-instruction symlinks are not required by the model.
            skipped_links.append(item["path"])
            continue
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected source path.")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source must be an ordinary file inside the vendor tree.")
        raw = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != item["sha"]:
            raise ValueError(f"Vendor Git blob mismatch: {relative}")
        count += 1
        size += len(raw)
    return {
        "revision": REVISION, "git_tree_sha256": file_hash(tree_path),
        "regular_git_blobs_verified": count, "source_bytes_verified": size,
        "nonexecuted_symlinks_skipped": skipped_links,
    }


def _same_tree(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_tree(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_tree(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def probe(fixture, dtype_name: str, out: Path) -> dict:
    import torch
    import torch.nn.functional as functional
    from nemo_automodel.components._peft.lora import (
        PeftConfig,
        apply_lora_to_linear_modules,
    )
    from nemo_automodel.components.moe.layers import Gate

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(20260922)

    def build():
        model = fixture.tiny_glm5_next_model()
        if dtype != torch.float32:
            model.initialize_weights(torch.device("cpu"), dtype=dtype)
        gates = []
        for name, module in model.named_modules():
            if isinstance(module, Gate):
                module.bias_update_factor = 0.0
                gates.append(name)
        if not gates:
            raise ValueError("The fixture must exercise a real sparse MoE gate.")
        return model, gates

    def attach(model):
        names = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and ".self_attn." in name
            and ".indexer." not in name and name.rsplit(".", 1)[-1] in TARGET_LEAVES
        ]
        if len(names) != 16:
            raise ValueError("The pinned tiny hybrid target layout changed.")
        config = PeftConfig(
            target_modules=names, dim=4, alpha=8, dropout=0.0,
            lora_dtype=dtype, use_triton=False, use_memory_efficient_lora=False,
        )
        if apply_lora_to_linear_modules(model, config) != len(names):
            raise ValueError("Adapter attachment did not match every named target.")
        params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        expected = {f"{name}.lora_{which}.weight" for name in names for which in ("A", "B")}
        if set(params) != expected:
            raise ValueError("Trainable parameters extend beyond the intended adapters.")
        return names, params

    model, gates = build()
    base_parameters = sum(p.numel() for p in model.parameters())
    if base_parameters != 17442:
        raise ValueError("This probe must use the small random vendor fixture.")
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]])
    documents = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.int32)

    def logits(current):
        return current(input_ids=inputs, attention_mask=documents).logits

    model.eval()
    before_attachment = {
        name: value.clone() for name, value in model.state_dict().items()
    }
    with torch.no_grad():
        unfrozen_reference = logits(model).clone()
    # The actual adapter experiment freezes the backbone. Match that execution
    # condition before comparing attachment, and retain the freeze-only change.
    # On the pinned CPU fixture, freezing alone reproduces the old FP32 drift;
    # it must not be attributed to adapter weights or hidden by a looser tolerance.
    model.requires_grad_(False)
    with torch.no_grad():
        original = logits(model).clone()
    freeze_delta = float((unfrozen_reference - original).abs().max())
    freeze_changed = int(torch.count_nonzero(unfrozen_reference != original))
    targets, trainable = attach(model)
    if any(
        not torch.equal(before, model.state_dict()[name])
        for name, before in before_attachment.items()
    ):
        raise ValueError("Adapter attachment changed a base parameter or buffer.")
    with torch.no_grad():
        adapted = logits(model)
    if not torch.equal(original, adapted):
        delta = float((original - adapted).abs().max())
        raise ValueError(
            f"Zero-adapter bitwise identity failed; max absolute logit delta={delta}."
        )

    frozen = {
        name: value.detach().clone() for name, value in model.state_dict().items()
        if name not in trainable
    }
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3, weight_decay=0.0)
    calls = dict.fromkeys(targets, 0)

    def hook(name):
        def called(_module, _inputs, _output):
            calls[name] += 1
        return called

    handles = [model.get_submodule(name).register_forward_hook(hook(name)) for name in targets]

    def update(current, current_optimizer):
        current.train()
        current_optimizer.zero_grad(set_to_none=True)
        values = logits(current)
        # A fixed synthetic target tests gradients, not language quality.
        loss = functional.cross_entropy(values[:, -1].float(), torch.tensor([7]))
        loss.backward()
        norms = {}
        for name, parameter in current.named_parameters():
            if parameter.requires_grad:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Missing or invalid adapter gradient: {name}")
                norms[name] = float(parameter.grad.float().norm())
            elif parameter.grad is not None:
                raise ValueError(f"A frozen parameter received a gradient: {name}")
        current_optimizer.step()
        # Exercise the normal maintenance hook: frozen gate bias must not move.
        current.update_moe_gate_bias()
        for name, before in frozen.items():
            if not torch.equal(before, current.state_dict()[name]):
                raise ValueError(f"Frozen parameter or buffer changed: {name}")
        return {"loss": float(loss.detach()), "gradient_norms": norms}

    first = update(model, optimizer)
    checkpoint = out / f"{dtype_name}-step1.pt"
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(), "dtype": dtype_name, "step": 1,
    }, checkpoint)
    expected_checksum = file_hash(checkpoint)
    second = update(model, optimizer)
    expected_model = {name: value.clone() for name, value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    expected_rng = torch.get_rng_state().clone()
    for handle in handles:
        handle.remove()
    if any(count == 0 for count in calls.values()):
        raise ValueError("A selected module was bypassed during actual forward calls.")

    restored, _ = build()
    _, restored_parameters = attach(restored)
    restored_optimizer = torch.optim.AdamW(
        restored_parameters.values(), lr=1e-3, weight_decay=0.0,
    )
    if file_hash(checkpoint) != expected_checksum:
        raise ValueError("Tiny checkpoint checksum changed before restoration.")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    resumed = update(restored, restored_optimizer)
    if (not _same_tree(expected_model, restored.state_dict())
            or not _same_tree(expected_optimizer, restored_optimizer.state_dict())
            or not torch.equal(expected_rng, torch.get_rng_state())
            or second != resumed):
        raise ValueError("Resumed model, optimizer or gradient trajectory differs.")
    changed = sum(
        not torch.equal(value, payload["model"][name])
        for name, value in expected_model.items() if name in trainable
    )
    if not changed:
        raise ValueError("No adapter changed during the second optimizer step.")
    return {
        "status": "passed", "device": "cpu", "dtype": dtype_name,
        "base_parameters": base_parameters,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "adapter_targets": targets, "target_forward_calls": calls,
        "frozen_gate_names": gates, "gate_bias_update_factor": 0.0,
        "zero_adapter_identity_exact": True,
        "identity_reference": "The same backbone already frozen, before adapter attachment.",
        "unfrozen_to_frozen_max_logit_delta": freeze_delta,
        "unfrozen_to_frozen_changed_logit_elements": freeze_changed,
        "attachment_preserved_base_parameters_and_buffers": True,
        "frozen_parameters_and_buffers_unchanged": True,
        "checkpoint_sha256": expected_checksum, "checkpoint_bytes": checkpoint.stat().st_size,
        "resume_model_optimizer_rng_trajectory_exact": True,
        "adapter_tensors_changed_in_second_update": changed, "updates": [first, second],
        "kv_b_proj_excluded": (
            "The pinned cuDNN absorbed attention path reads .weight directly; "
            "an ordinary forward adapter is not validated for that path."
        ),
        "gpu_kernels_tested": False, "pretrained_weights_loaded": False,
        "language_quality_measured": False, "distributed_training_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh output directory and preserve previous attempts.")
    verification = verify_source(args.source_root, args.git_tree)
    sys.path.insert(0, str(args.source_root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "bobcat_vendor_glm_tiny_fixture",
        args.source_root / "tests/unit_tests/models/glm5_next/conftest.py",
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    import torch

    torch.set_num_threads(2)
    args.out.mkdir(parents=True)
    started = time.monotonic()
    result = {
        "schema": "bobcat-glm-tiny-adapter-probe-v2", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "source": verification,
        "probe_sha256": file_hash(Path(__file__)),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "tokenizers", "safetensors",
        )},
        "profiles": {}, "release_gate_passed": False,
    }
    try:
        for dtype in ("float32", "bfloat16"):
            try:
                result["profiles"][dtype] = probe(fixture, dtype, args.out)
            except Exception as error:
                result["profiles"][dtype] = {
                    "status": "failed", "error_type": type(error).__name__, "error": str(error),
                }
            atomic_json(args.out / "result.json", result)
        result["status"] = (
            "passed" if all(p["status"] == "passed" for p in result["profiles"].values())
            else "failed"
        )
    finally:
        result.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - started)
        atomic_json(args.out / "result.json", result)
    print(json.dumps({"status": result["status"], "profiles": {
        key: value["status"] for key, value in result["profiles"].items()
    }}))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
