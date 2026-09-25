"""Small CUDA/FLA gate before training the real GLM backbone.

Uses the pinned vendor's GPU fixture configuration, with the final feed-forward
layer changed to sparse MoE. Random weights and synthetic targets test execution,
frozen boundaries and resume, not language quality or full-model memory fit.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import time
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import TARGET_LEAVES, _same_tree, verify_source
from bobcat.schema import file_hash, json_hash


def probe(config_factory, out):
    import torch
    import torch.nn.functional as functional
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next import layers
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from nemo_automodel.components.moe.layers import Gate

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
        raise ValueError("Run this CUDA gate on an exclusively owned BF16-capable GPU.")
    if not all(getattr(layers, flag) for flag in (
        "_SHORT_CONV_OK", "_CHUNK_KDA_OK", "_RECURRENT_KDA_OK", "_KDA_GATE_OK",
    )):
        raise ValueError("The real FLA kernels are required; do not silently use Torch fallback.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(0.10, device)
    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)
    config = config_factory()
    config.text_config.mlp_layer_types[-1] = "sparse"
    config_dict = config.to_dict()
    backend = dict(attn="sdpa", linear="torch", rms_norm="torch_fp32", experts="torch",
                   dispatcher="torch", rope_fusion=False, enable_hf_state_dict_adapter=False)

    def build():
        model = Glm5NextForConditionalGeneration(
            config, backend=BackendConfig(**backend),
        ).to(device)
        model.initialize_weights(device, dtype=torch.bfloat16)
        gates = []
        for name, module in model.named_modules():
            if isinstance(module, Gate):
                module.bias_update_factor = 0.0
                gates.append(name)
        if not gates or sum(p.numel() for p in model.parameters()) != 255_044:
            raise ValueError("Keep the real hybrid/MoE fixture small and explicit.")
        return model, gates

    def attach(model):
        names = [name for name, module in model.named_modules()
                 if isinstance(module, torch.nn.Linear) and ".self_attn." in name
                 and ".indexer." not in name and name.rsplit(".", 1)[-1] in TARGET_LEAVES]
        if len(names) != 16:
            raise ValueError("The four-layer attention projection layout changed.")
        peft = PeftConfig(target_modules=names, dim=4, alpha=8, dropout=0.0,
                          lora_dtype=torch.bfloat16, use_triton=False,
                          use_memory_efficient_lora=False)
        if apply_lora_to_linear_modules(model, peft) != len(names):
            raise ValueError("Not every intended attention projection received its adapter.")
        params = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if set(params) != {f"{name}.lora_{letter}.weight" for name in names for letter in "AB"}:
            raise ValueError("Trainable parameters extend beyond the named adapters.")
        return names, params

    model, gates = build()
    base_parameters = sum(p.numel() for p in model.parameters())
    inputs = torch.randint(1, 95, (1, 128), device=device)
    documents = torch.tensor([[1] * 64 + [2] * 64], device=device, dtype=torch.int32)

    def logits(current, ids=inputs):
        # Project only the decision position, retaining the backbone's full causal input.
        return current(input_ids=ids, _packed_seq_ids=documents, logits_to_keep=1).logits[:, -1]

    kernel_calls = Counter()

    def traced(name, implementation):
        def call(*args, **kwargs):
            values = [*args, *kwargs.values()]
            tensors = [x for x in values if isinstance(x, torch.Tensor) and x.is_floating_point()]
            if not tensors or any(not x.is_cuda for x in tensors):
                raise ValueError("A claimed FLA CUDA call received CPU floating-point inputs.")
            kernel_calls[name] += 1
            return implementation(*args, **kwargs)
        return call

    with ExitStack() as stack:
        for name in ("_chunk_kda", "_recurrent_kda", "_fused_kda_gate", "_fla_causal_conv1d"):
            stack.enter_context(patch.object(layers, name, traced(name, getattr(layers, name))))
        model.eval()
        with torch.no_grad():
            unfrozen = logits(model).clone()
        model.requires_grad_(False)
        with torch.no_grad():
            reference = logits(model).clone()
        freeze_delta = float((reference - unfrozen).abs().max())
        frozen = {name: x.detach().cpu().clone() for name, x in model.state_dict().items()}
        targets, params = attach(model)
        if any(not torch.equal(value, model.state_dict()[name].cpu())
               for name, value in frozen.items()):
            raise ValueError("Attaching adapters mutated a frozen tensor.")
        with torch.no_grad():
            initial = logits(model)
            if not torch.isfinite(initial).all() or not torch.equal(initial, reference):
                raise ValueError("Zero-adapter CUDA identity failed; preserve the failure.")
            changed = inputs.clone()
            changed[:, :64] = (changed[:, :64] % 94) + 1
            if not torch.equal(initial, logits(model, changed)):
                raise ValueError("The packed document boundary leaked in the CUDA backend.")
            changed = inputs.clone()
            changed[:, -1] = (changed[:, -1] % 94) + 1
            if torch.equal(initial, logits(model, changed)):
                raise ValueError("Positive input-dependence control failed.")
        target_calls = Counter()

        def hooks(current):
            for name in targets:
                def called(_module, _args, _value, key=name):
                    target_calls[key] += 1
                stack.callback(current.get_submodule(name).register_forward_hook(called).remove)

        hooks(model)
        optimizer = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=0.0)

        def update(current, opt):
            current.train()
            opt.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(
                logits(current).float(), torch.tensor([7], device=device),
            )
            loss.backward()
            norms = {}
            for name, parameter in current.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ValueError(f"Missing/non-finite CUDA adapter gradient: {name}")
                    norms[name] = float(parameter.grad.float().norm())
                elif parameter.grad is not None:
                    raise ValueError(f"Frozen parameter received a gradient: {name}")
            opt.step()
            current.update_moe_gate_bias()
            if any(not torch.equal(value, current.state_dict()[name].cpu())
                   for name, value in frozen.items()):
                raise ValueError("A frozen parameter or router buffer changed.")
            torch.cuda.synchronize(device)
            return {"loss": float(loss.detach()), "gradient_norms": norms}

        torch.cuda.reset_peak_memory_stats(device)
        first = update(model, optimizer)
        checkpoint = out / "cuda-step1.pt"
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
            "step": 1, "config_sha256": json_hash(config_dict),
        }, checkpoint)
        checkpoint_hash = file_hash(checkpoint)
        second = update(model, optimizer)
        expected_model = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        expected_optimizer = optimizer.state_dict()
        expected_cpu_rng = torch.get_rng_state()
        expected_cuda_rng = torch.cuda.get_rng_state(device)
        restored, _ = build()
        restored.requires_grad_(False)
        _, restored_params = attach(restored)
        restored_optimizer = torch.optim.AdamW(restored_params.values(), lr=1e-3, weight_decay=0.0)
        if file_hash(checkpoint) != checkpoint_hash:
            raise ValueError("CUDA checkpoint checksum changed before fresh restoration.")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored.load_state_dict(saved["model"])
        restored_optimizer.load_state_dict(saved["optimizer"])
        hooks(restored)
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        resumed = update(restored, restored_optimizer)
        if (not _same_tree(expected_model, {
                name: value.cpu() for name, value in restored.state_dict().items()})
                or not _same_tree(expected_optimizer, restored_optimizer.state_dict())
                or not torch.equal(expected_cpu_rng, torch.get_rng_state())
                or not torch.equal(expected_cuda_rng, torch.cuda.get_rng_state(device))
                or second != resumed):
            raise ValueError(
                "CUDA model/optimizer/RNG or gradient trajectory did not resume exactly."
            )
        if set(target_calls) != set(targets) or not all(kernel_calls[name] for name in (
            "_chunk_kda", "_fused_kda_gate", "_fla_causal_conv1d",
        )):
            raise ValueError("A named projection or required real CUDA path was not exercised.")
        gradient_targets = {
            name: any(
                step["gradient_norms"][f"{name}.lora_{letter}.weight"] > 0
                for step in (first, second) for letter in ("A", "B")
            )
            for name in targets
        }
        if not all(gradient_targets.values()):
            raise ValueError("A named adapter target received no nonzero training signal.")
        updated = sum(
            not torch.equal(expected_model[name], saved["model"][name]) for name in params
        )
        if not updated:
            raise ValueError("No adapter tensor changed.")
        return {
            "status": "passed", "device": torch.cuda.get_device_name(device),
            "dtype": "bfloat16", "backend": backend, "config": config_dict,
            "base_parameters": base_parameters, "trainable_parameters": sum(
                p.numel() for p in params.values()),
            "adapter_targets": targets, "target_calls": dict(target_calls),
            "adapter_targets_with_nonzero_gradients": gradient_targets,
            "kernel_calls_with_cuda_inputs": dict(kernel_calls),
            "frozen_gates": gates, "router_bias_update_factor": 0.0,
            "zero_adapter_identity_exact": True, "freeze_only_max_logit_delta": freeze_delta,
            "packed_boundary_and_positive_input_controls_passed": True,
            "frozen_tensors_preserved": True, "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": checkpoint.stat().st_size, "resume_trajectory_exact": True,
            "updated_adapter_tensors": updated, "updates": [first, second],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "pretrained_weights_loaded": False, "distributed_training_tested": False,
            "full_model_memory_fit_verified": False, "language_quality_measured": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclusive-gpu", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or not args.exclusive_gpu:
        parser.error("Use a fresh output and an exclusively allocated research GPU.")
    args.out.mkdir(parents=True)
    start = time.monotonic()
    record = {
        "schema": "bobcat-glm-cuda-adapter-probe-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "probe_sha256": file_hash(Path(__file__)),
        "release_gate_passed": False,
    }
    atomic_json(args.out / "result.json", record)
    try:
        record["source"] = verify_source(args.source_root, args.git_tree)
        record["versions"] = {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "tokenizers", "safetensors", "triton",
            "flash-linear-attention", "fla-core", "einops", "numpy",
        )}
        sys.path.insert(0, str(args.source_root.resolve()))
        fixture_path = args.source_root / (
            "tests/functional_tests/context_parallel/run_glm5_next_packed_cp_parity.py"
        )
        record["fixture_sha256"] = file_hash(fixture_path)
        spec = importlib.util.spec_from_file_location("bobcat_vendor_gpu_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        atomic_json(args.out / "result.json", record)
        record["result"] = probe(fixture._config, args.out)
        record["status"] = record["result"]["status"]
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - start)
        atomic_json(args.out / "result.json", record)
    print(json.dumps({"status": record["status"], "language_quality_measured": False}))


if __name__ == "__main__":
    main()
