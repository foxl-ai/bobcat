"""FP8 (W8A8, per-channel weights, dynamic per-token activations) checkpoint of a merged Bobcat
model with llm-compressor, in the compressed-tensors format vLLM loads as it is.

The scheme is llm-compressor's FP8_DYNAMIC preset: every weight of a quantized Linear is stored
as float8_e4m3fn with one float scale per output channel (amax / 448, no calibration data), and
activations are quantized per token at run time. vLLM reads the scheme from config.json
(`quantization_config`, compressed-tensors); the Bobcat Space reads the same tensors into its
own FP8 modules (space_engine.load_fp8_tier) without converting them.

Quantized: every Linear of the language model, including each expert of a Gemma 4 MoE (the 3D
expert tensors are linearized into one gate/up/down Linear per expert first, as vLLM loads
them). Kept in BF16: lm_head (the readout rows), the routers, the vision and audio towers and
their embedding projections (text-only serving), as in infra/aws/nvfp4_quantize.py.

    qvenv/bin/python scripts/fp8_quantize.py --model merged/ --out merged-fp8/
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

IGNORE = ["lm_head", "re:.*visual.*", "re:.*in_proj_a$", "re:.*in_proj_b$", "re:.*router.*",
          "re:.*vision_tower.*", "re:.*embed_vision.*", "re:.*audio_tower.*",
          "re:.*embed_audio.*"]
KEEP_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "preprocessor_config.json", "processor_config.json",
              "special_tokens_map.json", "video_preprocessor_config.json")


def census(model) -> dict:
    """Linear layers with and without a quantization scheme, split by role."""
    import torch

    out: dict[str, dict[str, int]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        role = ("expert" if ".experts." in name else "router" if ".router" in name
                else "lm_head" if name.endswith("lm_head") else
                "vision/audio" if any(k in name for k in ("vision", "audio")) else "dense")
        state = "quantized" if getattr(module, "quantization_scheme", None) else "bf16"
        out.setdefault(role, {}).setdefault(state, 0)
        out[role][state] += 1
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit("Choose a new output folder.")

    import torch
    import transformers
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    started = time.time()
    config = json.loads((args.model / "config.json").read_text())
    model_class = getattr(transformers, config["architectures"][0])
    model = model_class.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    moe = {}
    try:
        from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe
    except ImportError:
        get_non_linearized_moes = linearize_moe = None
    if get_non_linearized_moes is not None and get_non_linearized_moes(model):
        moe["fused_expert_modules"] = len(get_non_linearized_moes(model))
        linearize_moe(model)
        torch.cuda.empty_cache()
        moe["remaining_fused_expert_modules"] = len(get_non_linearized_moes(model))
    elif config.get("text_config", config).get("num_experts"):
        raise SystemExit("MoE checkpoint but no expert linearization: experts would stay BF16.")
    loaded = time.time() - started
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    oneshot(model=model, recipe=recipe)
    counts = census(model)
    print(json.dumps({"census": counts}), flush=True)
    if moe and counts.get("expert", {}).get("bf16", 0):
        raise SystemExit(f"Some expert projections were not quantized: {counts['expert']}")
    model.save_pretrained(args.out, save_compressed=True)
    names = json.loads((args.out / "model.safetensors.index.json").read_text())["weight_map"]
    saved = {"expert_fp8_weights": sum(1 for n in names if ".experts." in n
                                       and n.endswith(".weight")),
             "expert_weight_scales": sum(1 for n in names if ".experts." in n
                                         and n.endswith(".weight_scale")),
             "weight_scales": sum(1 for n in names if n.endswith(".weight_scale"))}
    for name in KEEP_FILES:
        if (args.model / name).exists():
            shutil.copy(args.model / name, args.out / name)
    receipt = {
        "source_model": str(args.model),
        "scheme": "FP8_DYNAMIC: float8_e4m3fn weights with one scale per output channel, "
                  "dynamic per-token FP8 activations (no calibration data)",
        "format": "compressed-tensors (float-quantized)",
        "tool": f"llmcompressor {__import__('llmcompressor').__version__}, "
                f"transformers {transformers.__version__}, torch {torch.__version__}",
        "ignored": IGNORE, "linear_census": counts, "moe": moe or None, "saved": saved,
        "seconds": {"load": round(loaded, 1), "total": round(time.time() - started, 1)},
    }
    (args.out / "bobcat-fp8.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
