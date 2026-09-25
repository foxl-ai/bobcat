"""CPU reference check of native hybrid packing, readout alignment and LoRA gradients.

This uses a small random native GLM fixture. It is not a full-pretrained CUDA
equivalence test, performance benchmark, or evidence of language quality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import TARGET_LEAVES, verify_source
from bobcat.glm_native_data import packed_rank_batch
from bobcat.glm_native_train import packed_decision_loss
from bobcat.schema import file_hash


def probe(fixture):
    import torch
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.moe.layers import Gate

    torch.manual_seed(2026092303)
    torch.set_num_threads(2)
    model = fixture.tiny_glm5_next_model()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, Gate):
            module.bias_update_factor = 0.
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".self_attn." in name
        and ".indexer." not in name and name.rsplit(".", 1)[-1] in TARGET_LEAVES
    ]
    attached = apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=targets, dim=4, alpha=8, dropout=0.,
                          lora_dtype=torch.float32, use_triton=False,
                          use_memory_efficient_lora=False),
    )
    if attached != 16:
        raise ValueError("The verified four-layer hybrid fixture layout changed.")
    # Exercise nonzero adapter A/B gradients instead of only a zero-initialized readout.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.01)
    rows = [
        {"input_ids": [1, 2, 3, 4, 5], "option_token_ids": [7, 8, 9],
         "supervision": "hard_label", "target_index": 2},
        {"input_ids": [11, 12, 13], "option_token_ids": [20, 21, 22, 23],
         "supervision": "score_mean", "score_mean": 1.5},
        {"input_ids": [30, 31, 32, 33], "option_token_ids": [40, 41],
         "supervision": "hard_label", "target_index": 0},
    ]
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if "lora_" not in name}

    def forward(items, length):
        batch = packed_rank_batch(
            [{"inputs": {"input_ids": row["input_ids"]}} for row in items], length,
        )
        output = model(**batch).logits[0]
        return [value[row["option_token_ids"]].float()
                for value, row in zip(output, items, strict=True)]

    model.eval()
    with torch.no_grad():
        packed = forward(rows, 16)
        separate = [forward([row], 16)[0] for row in rows]
        for actual, expected in zip(packed, separate, strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        changed = [{**row, "input_ids": list(row["input_ids"])} for row in rows]
        changed[0]["input_ids"] = [50, 51, 52, 53, 54]
        perturbed = forward(changed, 16)
        if not all(torch.equal(a, b) for a, b in zip(packed[1:], perturbed[1:], strict=True)):
            raise ValueError("Changing one document affected a different packed document.")
        if torch.equal(packed[0], perturbed[0]):
            raise ValueError("The changed-document positive control was unresponsive.")

    model.train()
    model.zero_grad(set_to_none=True)
    packed_loss = packed_decision_loss(forward(rows, 16), rows)
    packed_loss.backward()
    packed_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                    if parameter.requires_grad}
    model.zero_grad(set_to_none=True)
    separate_loss = packed_decision_loss([forward([row], 16)[0] for row in rows], rows)
    separate_loss.backward()
    max_gradient_delta = 0.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, packed_grads[name], rtol=1e-4, atol=1e-6)
            max_gradient_delta = max(
                max_gradient_delta, float((parameter.grad - packed_grads[name]).abs().max()),
            )
        elif parameter.grad is not None:
            raise ValueError("Frozen weights received gradients.")
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()):
        raise ValueError("The reference packing check mutated a frozen weight or buffer.")
    return {
        "status": "passed", "device": "cpu", "dtype": "float32", "questions": len(rows),
        "adapter_modules": attached, "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad),
        "maximum_logit_difference": max(float((a - b).abs().max())
                                        for a, b in zip(packed, separate, strict=True)),
        "maximum_adapter_gradient_difference": max_gradient_delta,
        "packed_mean_loss": float(packed_loss.detach()),
        "independent_mean_loss": float(separate_loss.detach()),
        "cross_document_change_exactly_isolated": True,
        "input_dependence_positive_control": True, "frozen_base_preserved": True,
        "logit_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "gradient_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "full_pretrained_cuda_tested": False, "performance_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Keep prior evidence immutable.")
    source = verify_source(args.source_root, args.git_tree)
    sys.path.insert(0, str(args.source_root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "bobcat_native_packing_fixture",
        args.source_root / "tests/unit_tests/models/glm5_next/conftest.py",
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    record = {"at": datetime.now(UTC).isoformat(), "source": source,
              "probe_sha256": file_hash(Path(__file__))}
    try:
        record.update(probe(fixture))
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.out, record)
        raise
    atomic_json(args.out, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
