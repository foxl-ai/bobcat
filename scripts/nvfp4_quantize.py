"""NVFP4 (W4A4) quantization of a merged Bobcat checkpoint with llm-compressor.

Blackwell GPUs (RTX PRO 6000, B200/B300) have FP4 tensor cores; NVFP4 stores weights in
4-bit E2M1 with an FP8 scale per 16 values and quantizes activations the same way at run
time, so prefill matmuls run on the FP4 path (vLLM loads the result as a compressed-tensors
checkpoint). Activation scales need calibration: this uses DEV compiled prompts only, never
the calibration or final splits, and writes the ids it used so the accuracy check can also
be reported on the dev rows it did not see.

Quantized: every Linear of the language model (attention, Gated DeltaNet in/out projections,
MLP). Kept in BF16: lm_head (Bobcat's readout rows), the vision tower (unused, text-only
serving), and the Gated DeltaNet a/b projections (48 outputs, a negligible share of compute).
vLLM runs some projections as one fused GEMM with one per-tensor NVFP4 scale: q/k/v and
gate/up (llm-compressor shares their global scale already) and the Gated DeltaNet in_proj_qkv
with in_proj_z (added here; without it vLLM warns that the fused layer's scales differ).

Gemma 4 MoE (Bobcat Flash, model_type gemma4), added 2026-09-26:
  * the 3D expert tensors (gate_up_proj [E, 2I, H], down_proj [E, H, I]) are linearized into
    one gate/up/down Linear per expert before calibration, so every expert is quantized (a
    `targets="Linear"` recipe on the 3D tensors would leave all experts in BF16); calibration
    sends every token through every expert (llm-compressor `moe_calibrate_all_experts`) so each
    expert gets activation statistics, while the forward keeps the routed output;
  * kept in BF16 as well: the routers (`router.proj`: vLLM builds them unquantized), the
    vision tower and `embed_vision` (text-only serving);
  * fused groups vLLM serves as one GEMM: q/k/v (sliding layers), q/k on the full-attention
    layers with attention_k_eq_v (no v_proj; vLLM loads K into the V slot), the dense MLP's
    gate/up, and each expert's gate/up (vLLM's w13). All are covered by llm-compressor's
    groups (a missing v_proj is skipped); the receipt records a check that every group's
    weight global scales are equal and the count of quantized expert weights.

    qvenv/bin/python scripts/nvfp4_quantize.py --model merged/ --dev-compiled \
        eval-dev.compiled.jsonl --out merged-nvfp4/ [--samples 256 --max-tokens 4096]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

SEED = 20260926
GDN_FUSED = ("in_proj_qkv", "in_proj_z")  # vLLM's in_proj_qkvz
IGNORE = ["lm_head", "re:.*visual.*", "re:.*in_proj_a$", "re:.*in_proj_b$"]
# Gemma 4: routers stay BF16 (vLLM's router is an unquantized linear), and so do the vision
# tower and its embedding projection (text-only serving).
GEMMA4_IGNORE = ["re:.*router.*", "re:.*vision_tower.*", "re:.*embed_vision.*",
                 "re:.*audio_tower.*", "re:.*embed_audio.*"]


def fused_scale_check(model, groups) -> dict:
    """For every module holding a fused group (as vLLM fuses them), whether the members'
    NVFP4 weight global scales are equal. Missing (None) members are skipped, as in
    llm-compressor."""
    checked, unequal = 0, []
    for name, module in model.named_modules():
        for group in groups:
            if not all(hasattr(module, member) for member in group):
                continue
            scales = [getattr(getattr(module, m), "weight_global_scale", None) for m in group
                      if getattr(module, m) is not None]
            scales = [s for s in scales if s is not None]
            if len(scales) < 2:
                continue
            checked += 1
            if any(not bool((s.float() == scales[0].float()).all()) for s in scales[1:]):
                unequal.append(f"{name}:{'/'.join(group)}")
    return {"groups_checked": checked, "groups_with_unequal_scales": len(unequal),
            "examples": unequal[:10]}


def quantized_linear_census(model) -> dict:
    """Linear layers with and without a quantization scheme, split by role."""
    import torch

    counts: dict[str, dict[str, int]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        role = ("expert" if ".experts." in name else "router" if ".router" in name
                else "vision" if "vision" in name or "visual" in name
                else "lm_head" if name.endswith("lm_head") else "language")
        state = "quantized" if getattr(module, "quantization_scheme", None) else "bf16"
        counts.setdefault(role, {"quantized": 0, "bf16": 0})[state] += 1
    return counts
KEEP_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "vocab.json", "merges.txt", "preprocessor_config.json",
              "video_preprocessor_config.json", "special_tokens_map.json")


def calibration_rows(path: Path, samples: int, max_tokens: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    rows = [r for r in rows if len(r["input_ids"]) <= max_tokens]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dev-compiled", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--pipeline", default="basic",
                        help="llm-compressor calibration pipeline; basic runs whole-model "
                             "forwards (no graph tracing of the hybrid layers)")
    parser.add_argument("--ignore", action="append", default=[],
                        help="extra module pattern kept in BF16 (repeatable)")
    parser.add_argument("--no-calibrate-all-experts", action="store_true",
                        help="MoE: calibrate each expert only on the tokens routed to it")
    parser.add_argument("--dense-scheme", choices=["NVFP4", "FP8_DYNAMIC"], default="NVFP4",
                        help="MoE: preset for the attention and dense-MLP projections; the "
                             "experts are always NVFP4 (FP8_DYNAMIC: per-channel FP8 weights, "
                             "per-token FP8 activations, no calibration)")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Choose a new output folder.")

    import torch
    import transformers
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.observers import helpers as observer_helpers

    if GDN_FUSED not in observer_helpers.FUSED_LAYER_NAMES:
        observer_helpers.FUSED_LAYER_NAMES.append(GDN_FUSED)

    rows = calibration_rows(args.dev_compiled, args.samples, args.max_tokens)
    dataset = Dataset.from_list([{"input_ids": r["input_ids"],
                                  "attention_mask": [1] * len(r["input_ids"])} for r in rows])
    started = time.time()
    config = json.loads((args.model / "config.json").read_text())
    model_class = getattr(transformers, config["architectures"][0])
    model = model_class.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    # The prompts are already token ids; the tokenizer stands in for the multimodal processor
    # (whose image/video parts need packages the text-only calibration does not use).
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    ignore = list(IGNORE)
    if config.get("model_type") == "gemma4":
        ignore += GEMMA4_IGNORE
    ignore += args.ignore
    moe = {}
    try:
        from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe
    except ImportError:  # an llm-compressor without MoE linearization: dense models only
        get_non_linearized_moes = linearize_moe = None
    if get_non_linearized_moes is not None and get_non_linearized_moes(model):
        moe["fused_expert_modules"] = len(get_non_linearized_moes(model))
        linearize_moe(model)  # one Linear per expert projection, before calibration
        # The freed 3D tensors stay in PyTorch's cache as blocks the many small quantization
        # parameters could not use (out of memory with 45 GB cached but free): release them.
        torch.cuda.empty_cache()
        moe["remaining_fused_expert_modules"] = len(get_non_linearized_moes(model))
        moe["calibrate_all_experts"] = not args.no_calibrate_all_experts
    elif config.get("text_config", config).get("num_experts"):
        raise SystemExit("MoE checkpoint but no expert linearization: experts would stay BF16.")
    loaded = time.time() - started
    if moe and args.dense_scheme != "NVFP4":
        # Experts in NVFP4, the other language-model projections in another preset. The
        # targets must match vLLM's module names as well as the HF ones: vLLM names the text
        # layers `language_model.model.layers.N...` (HF: `model.language_model.layers.N...`),
        # so a pattern with `language_model\.layers` matched nothing there and vLLM read the
        # FP8 tensors as unquantized weights (dev macro 37.5%, 2026-09-26 first attempt).
        scheme = {"NVFP4": [r"re:.*\.experts\.\d+\.(gate|up|down)_proj$"],
                  args.dense_scheme: [r"re:.*layers\.\d+\."
                                      r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"]}
        recipe = QuantizationModifier(scheme=scheme, ignore=ignore)
    else:
        recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=ignore)
    extra = ({"moe_calibrate_all_experts": moe["calibrate_all_experts"]} if moe else {})
    oneshot(model=model, dataset=dataset, recipe=recipe, max_seq_length=args.max_tokens,
            num_calibration_samples=len(rows), pipeline=args.pipeline, processor=tokenizer,
            **extra)
    calibrated = time.time() - started - loaded
    census = quantized_linear_census(model)
    fused = fused_scale_check(model, observer_helpers.FUSED_LAYER_NAMES)
    print(json.dumps({"census": census, "fused_scales": fused}), flush=True)
    if moe and census.get("expert", {}).get("bf16", 0):
        raise SystemExit(f"Some expert projections were not quantized: {census['expert']}")
    model.save_pretrained(args.out, save_compressed=True)
    index = args.out / "model.safetensors.index.json"
    if moe and index.exists():
        names = json.loads(index.read_text())["weight_map"]
        moe["saved_expert_weight_packed"] = sum(
            1 for n in names if ".experts." in n and n.endswith(".weight_packed"))
        moe["saved_expert_bf16_weight"] = sum(
            1 for n in names if ".experts." in n and n.endswith(".weight"))
    for name in KEEP_FILES:
        if (args.model / name).exists():
            shutil.copy(args.model / name, args.out / name)
    receipt = {
        "source_model": str(args.model), "scheme": "NVFP4 (W4A4, group 16, FP8 scales)"
        + ("" if args.dense_scheme == "NVFP4" or not moe else
           f"; attention and dense MLP {args.dense_scheme}"),
        "tool": f"llmcompressor {__import__('llmcompressor').__version__}",
        "calibration": {"split": "product-eval v2 dev (compiled)", "seed": SEED,
                        "samples": len(rows), "max_tokens": args.max_tokens,
                        "pipeline": args.pipeline,
                        "ids": [r["id"] for r in rows]},
        "ignored": ignore,
        "shared_global_scale": [list(group) for group in observer_helpers.FUSED_LAYER_NAMES],
        "fused_scale_check": fused, "linear_census": census, "moe": moe or None,
        "seconds": {"load": round(loaded, 1), "calibrate": round(calibrated, 1),
                    "total": round(time.time() - started, 1)},
    }
    (args.out / "bobcat-nvfp4.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "calibration"}))


if __name__ == "__main__":
    main()
