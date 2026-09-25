"""Compare native expert kernels on isolated GPUs, including input gradients.

Uses random frozen matrices with GLM's real local expert dimensions. This is a
kernel experiment, not pretrained accuracy, distributed correctness or a speed
claim about the full decision service.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash


def main():
    import torch
    import torch.nn.functional as functional
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig
    from nemo_automodel.components.models.glm5_next.model import build_glm5_next_moe_config
    from nemo_automodel.components.moe.experts import GroupedExperts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.set_num_threads(4)
    # Use the pinned vendor config directly, rather than an import-order-dependent
    # AutoConfig registration. This is also exercised by CPU-only preparation.
    config = Glm5NextConfig.from_pretrained(args.model_dir, local_files_only=True)
    moe = build_glm5_next_moe_config(config.text_config, torch.bfloat16)
    if (moe.n_routed_experts, moe.n_activated_experts, moe.dim, moe.moe_inter_dim) != (
        288, 8, 4096, 2048,
    ):
        raise ValueError("Require the original GLM expert dimensions.")
    # Simulate the matrices owned by one EP4 rank. No distributed collectives
    # are being tested by these independent per-GPU experiments.
    moe = dataclasses.replace(moe, n_routed_experts=72)
    if args.metadata_only:
        with torch.device("meta"):
            model = GroupedExperts(moe, BackendConfig(experts="torch"))
        assert all(parameter.is_meta for parameter in model.parameters())
        print(json.dumps({
            "config_sha256": file_hash(args.model_dir / "config.json"),
            "local_experts": 72, "hidden": moe.dim, "intermediate": moe.moe_inter_dim,
            "parameter_shapes": {name: list(p.shape) for name, p in model.named_parameters()},
            "gpu_started": False,
        }))
        return 0
    torch.cuda.set_device(rank)
    torch.manual_seed(202609230100 + rank)
    device = torch.device("cuda", rank)
    with torch.device(device):
        model = GroupedExperts(moe, BackendConfig(experts="torch"))
    model.requires_grad_(False)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0., .01)
    tokens = (128, 256, 512, 1024)[rank % 4]
    x = torch.randn(tokens, 4096, device=device, dtype=torch.bfloat16, requires_grad=True)
    logits = torch.randn(tokens, 72, device=device)
    selected, indices = logits.topk(8, dim=-1)
    weights = selected.softmax(-1).detach().requires_grad_(True)
    mask = torch.ones(tokens, device=device, dtype=torch.bool)
    mask[-17:] = False
    probe = torch.randn(tokens, 4096, device=device, dtype=torch.float32)
    result = {
        "schema": "bobcat-glm-expert-kernel-comparison-v1",
        "config_sha256": file_hash(args.model_dir / "config.json"), "rank": rank,
        "gpu": torch.cuda.get_device_name(rank), "tokens": tokens, "local_experts": 72,
        "hidden": 4096, "intermediate": 2048, "top_k": 8, "precision": "bfloat16",
        "pretrained_weights_used": False, "distributed_correctness_measured": False,
        "end_to_end_service_speed_measured": False, "training_modified": False,
        "warm_repeats": 3, "numerical_contract": {
            "relative_rms_max": .03, "cosine_min": .999,
            "applies_to": "synthetic kernel outputs and input/router-weight gradients only",
        },
    }
    saved = {}
    for mode in ("torch", "torch_mm"):
        model.use_torch_mm = mode == "torch_mm"
        samples = []
        try:
            for repeat in range(4):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                y = model(x, mask, weights, indices)
                gradient = torch.autograd.grad((y.float() * probe).mean(), (x, weights))
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - start)
                if repeat == 0:
                    saved[mode] = [value.detach().float().cpu() for value in (y, *gradient)]
            result[mode] = {"cold_forward_backward_seconds": samples[0],
                            "warm_forward_backward_seconds": samples[1:]}
        except Exception as error:
            result[mode] = {"error_type": type(error).__name__, "error": str(error)[:2000]}
            break
    comparisons = []
    if "torch_mm" in saved:
        for name, a, b in zip(("output", "input_gradient", "routing_weight_gradient"),
                              saved["torch"], saved["torch_mm"], strict=True):
            relative = float((a - b).square().mean().sqrt() / a.square().mean().sqrt().clamp_min(
                1e-12,
            ))
            cosine = float(functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            comparisons.append({
                "tensor": name, "relative_rms": relative, "cosine": cosine,
                "max_abs_error": float((a - b).abs().max()),
                "finite": bool(torch.isfinite(b).all()),
            })
    result["comparisons"] = comparisons
    result["kernel_contract_passed"] = len(comparisons) == 3 and all(
        row["finite"] and row["relative_rms"] <= .03 and row["cosine"] >= .999
        for row in comparisons
    )
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / f"rank-{rank}.json", result)
    print(json.dumps({"rank": rank, "kernel_contract_passed": result["kernel_contract_passed"]}))
    return 0 if result["kernel_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
