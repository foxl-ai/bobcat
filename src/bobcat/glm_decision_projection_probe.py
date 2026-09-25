"""Verify candidate-only native GLM output with a small random CPU fixture.

This changes the output module, not JSON formatting after language generation.
It does not establish full-pretrained CUDA/FSDP compatibility or speed/quality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.decision_projection import (
    install_materialized_decision_projection,
    install_retained_vocabulary_projection,
)
from bobcat.glm_adapter_probe import TARGET_LEAVES, verify_source
from bobcat.glm_native_data import packed_rank_batch
from bobcat.glm_native_train import packed_decision_loss
from bobcat.schema import file_hash


def probe(fixture, *, retain_vocabulary_state=False):
    import torch
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.moe.layers import Gate

    torch.manual_seed(2026092317)
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
        raise ValueError("Pinned native hybrid fixture adapter layout changed.")
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
    identifiers = [23, 8, 41, 7, 22, 20, 9, 40, 21]
    frozen = {name: value.clone() for name, value in model.state_dict().items()
              if "lora_" not in name and not name.startswith("lm_head.")}
    original_head = model.get_output_embeddings()
    original_calls = []
    handle = original_head.register_forward_hook(
        lambda _module, _inputs, output: original_calls.append(tuple(output.shape)),
    )

    def forward(items):
        batch = packed_rank_batch(
            [{"inputs": {"input_ids": row["input_ids"]}} for row in items], 16,
        )
        return model(**batch).logits[0]

    model.train()
    reference_all = forward(rows)
    reference = [logits[row["option_token_ids"]].float()
                 for logits, row in zip(reference_all, rows, strict=True)]
    reference_loss = packed_decision_loss(reference, rows)
    reference_loss.backward()
    reference_gradients = {name: parameter.grad.clone()
                           for name, parameter in model.named_parameters()
                           if parameter.requires_grad}
    if not any(bool(torch.count_nonzero(grad)) for grad in reference_gradients.values()):
        raise ValueError("Adapter gradient positive control failed.")
    model.zero_grad(set_to_none=True)
    if retain_vocabulary_state:
        state_before = {name: value.clone() for name, value in model.state_dict().items()}
        projection = install_retained_vocabulary_projection(model, identifiers)
        if (state_before.keys() != model.state_dict().keys()
                or any(not torch.equal(value, model.state_dict()[name])
                       for name, value in state_before.items())):
            raise ValueError("The distributed integration bridge changed checkpoint state.")
        projection.decision_only = True
    else:
        projection = install_materialized_decision_projection(model, identifiers)
    bank = forward(rows)
    if bank.shape != (3, len(identifiers)):
        raise ValueError("Native forward did not return the reduced candidate bank.")
    actual = projection.select_batch(bank, [row["option_token_ids"] for row in rows])
    actual_loss = packed_decision_loss(actual, rows)
    actual_loss.backward()
    max_gradient_delta = 0.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            expected = reference_gradients[name]
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6)
            max_gradient_delta = max(
                max_gradient_delta, float((parameter.grad - expected).abs().max()),
            )
        elif parameter.grad is not None:
            raise ValueError("Frozen projection/backbone received an unexpected gradient.")
    for a, b in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    if any(a.argmax().item() != b.argmax().item()
           for a, b in zip(actual, reference, strict=True)):
        raise ValueError("Reducing the projection changed a choice.")

    model.eval()
    with torch.no_grad():
        before = projection.select_batch(forward(rows), [r["option_token_ids"] for r in rows])
        changed = [{**r, "input_ids": list(r["input_ids"])} for r in rows]
        changed[0]["input_ids"] = [50, 51, 52, 53, 54]
        after = projection.select_batch(forward(changed),
                                        [r["option_token_ids"] for r in changed])
        if not all(torch.equal(a, b) for a, b in zip(before[1:], after[1:], strict=True)):
            raise ValueError("The new output path mixed independent questions.")
        if torch.equal(before[0], after[0]):
            raise ValueError("The output was unresponsive to changed evidence.")
    if len(original_calls) != 1:
        raise ValueError("A projected forward still executed the full vocabulary head.")
    handle.remove()
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()):
        raise ValueError("Installing the output changed a frozen backbone tensor.")
    return {
        "status": "passed", "device": "cpu", "dtype": "float32",
        "fixture_random_parameters": 17442, "adapter_modules": attached,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "questions": len(rows), "candidate_counts": [len(r["option_token_ids"]) for r in rows],
        "original_vocabulary_output_shape": list(reference_all.shape),
        "candidate_bank_output_shape": list(bank.shape),
        "original_head_calls_after_replacement": 0, "generated_answer_tokens": 0,
        "maximum_logit_difference": max(float((a - b).abs().max().detach())
                                        for a, b in zip(actual, reference, strict=True)),
        "maximum_probability_tv": max(float((
            a.detach().softmax(-1) - b.detach().softmax(-1)).abs().sum() / 2)
            for a, b in zip(actual, reference, strict=True)),
        "maximum_adapter_gradient_difference": max_gradient_delta,
        "reference_loss": float(reference_loss.detach()),
        "candidate_only_loss": float(actual_loss.detach()),
        "frozen_backbone_preserved": True, "cross_question_isolation": True,
        "input_dependence_positive_control": True, "argmax_preserved": True,
        "logit_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "gradient_tolerance": {"rtol": 1e-4, "atol": 1e-6},
        "projection": {
            "mode": "retained_vocabulary_state_selected_gemm",
            "token_ids": identifiers, "state_keys_and_values_preserved": True,
            "full_weight_all_gather_removed": False, "end_to_end_speed_measured": False,
        } if retain_vocabulary_state else projection.provenance(
            original_vocabulary=original_head.out_features,
            source_revision="random-native-fixture"),
        "full_pretrained_cuda_tested": False, "distributed_replacement_tested": False,
        "performance_measured": False, "quality_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retain-vocabulary-state", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve the previous probe evidence.")
    source = verify_source(args.source_root, args.git_tree)
    sys.path.insert(0, str(args.source_root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "bobcat_native_projection_fixture",
        args.source_root / "tests/unit_tests/models/glm5_next/conftest.py",
    )
    record = {"at": datetime.now(UTC).isoformat(), "source": source,
              "probe_sha256": file_hash(Path(__file__)),
              "projection_source_sha256": file_hash(
                  Path(__file__).with_name("decision_projection.py"))}
    try:
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        record.update(probe(fixture, retain_vocabulary_state=args.retain_vocabulary_state))
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.out, record)
        raise
    atomic_json(args.out, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
