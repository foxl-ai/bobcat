"""Check suffix-only adapter ownership and gradient boundaries on native CPU GLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import verify_source
from bobcat.glm_adapter_scope import plan_attention_lora_scope
from bobcat.schema import file_hash


def prefix_replay_control(model, inputs, scope):
    """In-memory CPU control; no deployed prefix cache or split checkpoint claim."""
    import copy

    import torch
    import torch.nn.functional as functional
    from nemo_automodel.components.models.common.utils import compute_lm_head_logits

    stack = model.model.language_model.layers
    captured, layer_calls = {}, dict.fromkeys(range(len(stack)), 0)

    def capture(_module, args, kwargs):
        if args[0].requires_grad:
            raise ValueError("The proposed cached prefix still requires a gradient.")
        captured["hidden"] = args[0].detach().clone()
        captured["kwargs"] = copy.deepcopy(kwargs)

    def observe(index):
        def called(_module, _args, _output):
            layer_calls[index] += 1
        return called

    first = scope["first_trainable_layer"]
    handle = stack[str(first)].register_forward_pre_hook(capture, with_kwargs=True)
    counters = [
        layer.register_forward_hook(observe(int(index))) for index, layer in stack.items()
    ]
    model.train()
    model.zero_grad(set_to_none=True)
    reference = model(**inputs).logits
    handle.remove()
    functional.cross_entropy(reference[0, -1].float()[None], torch.tensor([7])).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    before = dict(layer_calls)
    hidden = captured["hidden"]
    for index in scope["selected_layers"]:
        hidden = stack[str(index)](hidden, **captured["kwargs"])
    hidden = model.model.language_model.norm(hidden.mean(dim=2))
    replayed = compute_lm_head_logits(
        model.get_output_embeddings(), hidden, logits_to_keep=inputs["logits_to_keep"],
    ).logits
    functional.cross_entropy(replayed[0, -1].float()[None], torch.tensor([7])).backward()
    for counter in counters:
        counter.remove()
    replay_calls = {index: layer_calls[index] - before[index] for index in before}
    if replay_calls != {index: int(index >= first) for index in range(len(stack))}:
        raise ValueError("Prefix replay recomputed an excluded layer or skipped a suffix layer.")
    if not torch.equal(reference, replayed):
        raise ValueError("Replaying the full-token prefix hidden state changed the output.")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(gradients[name], parameter.grad):
            raise ValueError(f"Prefix replay changed the adapter gradient: {name}")
    cache = captured["hidden"]
    return {
        "full_token_hidden_shape": list(cache.shape),
        "hidden_cache_bytes": cache.numel() * cache.element_size(),
        "metadata_cache_bytes_included": False,
        "original_prefix_requires_grad": False,
        "replayed_layer_calls": replay_calls,
        "logits_bitwise_equal": True,
        "adapter_gradients_bitwise_equal": True,
        "serialized_checkpoint_tested": False,
        "cuda_or_distributed_replay_tested": False,
        "inference_backbone_removed": False,
    }


def probe(fixture):
    import torch
    import torch.nn.functional as functional
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.moe.layers import Gate

    torch.set_num_threads(2)
    records = []
    for dtype_name in ("float32", "bfloat16"):
        dtype = getattr(torch, dtype_name)
        for suffix in (1, 2, 4):
            torch.manual_seed(2026092321)
            model = fixture.tiny_glm5_next_model()
            if dtype != torch.float32:
                model.initialize_weights(torch.device("cpu"), dtype=dtype)
            if sum(p.numel() for p in model.parameters()) != 17442:
                raise ValueError("Use only the pinned small random native fixture.")
            model.requires_grad_(False)
            for module in model.modules():
                if isinstance(module, Gate):
                    module.bias_update_factor = 0.
            model.eval()
            scope = plan_attention_lora_scope(model, last_layers=suffix, rank=4)
            inputs = {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
                "_packed_seq_ids": torch.ones((1, 6), dtype=torch.int32),
                "logits_to_keep": torch.tensor([5]),
            }
            frozen = {name: value.clone() for name, value in model.state_dict().items()}
            with torch.no_grad():
                original = model(**inputs).logits.clone()
            attached = apply_lora_to_linear_modules(
                model, PeftConfig(
                    target_modules=scope["target_modules"], dim=4, alpha=8,
                    dropout=0., lora_dtype=dtype, use_triton=False,
                    use_memory_efficient_lora=False,
                ),
            )
            trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
            expected = {
                f"{name}.lora_{which}.weight"
                for name in scope["target_modules"] for which in ("A", "B")
            }
            if (attached != len(scope["target_modules"]) or set(trainable) != expected
                    or sum(p.numel() for p in trainable.values()) != scope["trainable_parameters"]):
                raise ValueError("Trainable ownership differs from the explicit suffix plan.")
            with torch.no_grad():
                adapted = model(**inputs).logits
            if not torch.equal(original, adapted):
                raise ValueError("Zero suffix adapters changed the frozen-base output.")

            gradient_layers, calls = {}, dict.fromkeys(scope["target_modules"], 0)

            def layer_hook(index, observed_layers=gradient_layers):
                def observed(_module, _inputs, output):
                    observed_layers[index] = output.requires_grad
                return observed

            def target_hook(name, forward_calls=calls):
                def observed(_module, _inputs, _output):
                    forward_calls[name] += 1
                return observed

            hooks = [
                layer.register_forward_hook(layer_hook(int(index)))
                for index, layer in model.model.language_model.layers.items()
            ] + [
                model.get_submodule(name).register_forward_hook(target_hook(name))
                for name in scope["target_modules"]
            ]
            optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3, weight_decay=0.)
            losses, nonzero_gradients = [], set()
            for _ in range(3):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(**inputs).logits[0, -1].float()
                loss = functional.cross_entropy(logits[None], torch.tensor([7]))
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise ValueError("A selected adapter has no valid gradient.")
                        if bool(torch.count_nonzero(parameter.grad)):
                            nonzero_gradients.add(name)
                    elif parameter.grad is not None:
                        raise ValueError("A frozen base parameter received a gradient.")
                optimizer.step()
                model.update_moe_gate_bias()
                losses.append(float(loss.detach()))
                for name, before in frozen.items():
                    if not torch.equal(before, model.state_dict()[name]):
                        raise ValueError("Suffix training changed a base parameter or buffer.")
            for handle in hooks:
                handle.remove()
            expected_layers = {
                index: index in scope["selected_layers"]
                for index in range(scope["total_language_layers"])
            }
            if (gradient_layers != expected_layers or not nonzero_gradients
                    or any(count != 3 for count in calls.values())):
                raise ValueError("The intended forward/gradient boundary was not exercised.")
            replay = prefix_replay_control(model, inputs, scope)
            for name, before in frozen.items():
                if not torch.equal(before, model.state_dict()[name]):
                    raise ValueError("Prefix replay changed a base parameter or buffer.")
            records.append({
                "dtype": dtype_name, "scope": scope, "zero_adapter_identity_exact": True,
                "frozen_base_and_buffers_preserved": True,
                "observed_layer_output_requires_grad": gradient_layers,
                "target_forward_calls": calls, "gradient_tensors": len(trainable),
                "nonzero_gradient_tensors_across_three_steps": len(nonzero_gradients),
                "synthetic_test_updates": 3, "synthetic_losses": losses,
                "prefix_replay_control": replay,
            })
    return {
        "status": "passed", "device": "cpu", "random_fixture_parameters": 17442,
        "experiments": records, "full_pretrained_weights_tested": False,
        "distributed_checkpoint_tested": False, "gpu_backward_speedup_measured": False,
        "quality_measured": False, "running_gpu_job_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve prior probe evidence.")
    source = verify_source(args.source_root, args.git_tree)
    sys.path.insert(0, str(args.source_root.resolve()))
    spec = importlib.util.spec_from_file_location(
        "bobcat_native_scope_fixture",
        args.source_root / "tests/unit_tests/models/glm5_next/conftest.py",
    )
    record = {
        "at": datetime.now(UTC).isoformat(), "source": source,
        "probe_sha256": file_hash(Path(__file__)),
        "scope_source_sha256": file_hash(Path(__file__).with_name("glm_adapter_scope.py")),
    }
    try:
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        record.update(probe(fixture))
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(args.out, record)
        raise
    atomic_json(args.out, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
