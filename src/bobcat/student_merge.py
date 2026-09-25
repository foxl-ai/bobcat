"""Merge a PEFT LoRA adapter into the original checkpoint shards.

For serving engines that load a plain checkpoint. Each targeted weight becomes
`bf16(W + scale * B @ A)` with the product and sum in float32; every other tensor, file and
key name is copied unchanged, so the merged folder loads exactly like the base model. The
merged weights are a different numerical artifact from the unmerged adapter: evaluate them
as their own release rather than inheriting the adapter's results.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash


def module_map(lora_keys, weight_names) -> dict[str, tuple[str, str]]:
    """Checkpoint weight name -> (lora_A key, lora_B key)."""
    result = {}
    for key in lora_keys:
        if not key.endswith(".lora_A.weight"):
            continue
        module = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        candidates = [f"{module}.weight",
                      f"{module.replace('model.', 'model.language_model.', 1)}.weight"]
        found = [c for c in candidates if c in weight_names]
        if len(found) != 1:
            raise ValueError(f"No unique checkpoint weight for adapter module {module}.")
        result[found[0]] = (key, key.replace(".lora_A.", ".lora_B."))
    return result


def merge(model_dir: Path, adapter_dir: Path, out: Path) -> dict:
    import torch
    from safetensors.torch import load_file, save_file

    if out.exists():
        raise ValueError("Merged checkpoints are immutable; choose a new path.")
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank, alpha = config["r"], config["lora_alpha"]
    scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
    lora = load_file(adapter_dir / "adapter_model.safetensors")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    mapping = module_map(lora, set(index["weight_map"]))
    out.mkdir(parents=True)
    merged = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(model_dir / shard)
        for name in tensors:
            if name in mapping:
                a, b = (lora[k].to("cuda" if torch.cuda.is_available() else "cpu",
                                   torch.float32) for k in mapping[name])
                weight = tensors[name]
                delta = (b @ a) * scale
                tensors[name] = (weight.to(delta.device, torch.float32) + delta).to(
                    weight.dtype).cpu()
                merged += 1
        save_file(tensors, out / shard, metadata={"format": "pt"})
    if merged != len(mapping):
        raise ValueError(f"Merged {merged} of {len(mapping)} adapter weights.")
    for path in model_dir.iterdir():
        # The base download receipt describes the base shards, not these.
        if (path.is_file() and not path.name.endswith(".safetensors")
                and path.name != "bobcat-download.json"):
            shutil.copy2(path, out / path.name)
    receipt = {"base": str(model_dir), "adapter_sha256": file_hash(
        adapter_dir / "adapter_model.safetensors"), "rank": rank, "alpha": alpha,
        "scale": scale, "merged_weights": merged, "arithmetic": "float32 product and sum, "
        "stored in the base weight dtype",
        "shards": {s: file_hash(out / s) for s in sorted(set(index["weight_map"].values()))}}
    atomic_json(out / "bobcat-merge.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="the adapter's lora/ folder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = merge(args.model_dir, args.adapter, args.out)
    print(json.dumps({k: receipt[k] for k in ("merged_weights", "scale")}))


if __name__ == "__main__":
    main()
