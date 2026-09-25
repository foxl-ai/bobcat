"""Independently decode a recovered DCP checkpoint before allocating GPU work."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from bobcat.checkpoint_watch import validate_marker
from bobcat.corpus import atomic_json
from bobcat.native_resume import continuation_cursor, state_signature
from bobcat.schema import file_hash, json_hash


def validate_decoder_runtime(runtime, *, python_version, torch_version):
    """DCP's Python metadata is version-sensitive; safetensors export is portable."""
    expected_python = runtime.get("python")
    expected_torch = runtime.get("torch")
    if (expected_python is not None and list(python_version[:2]) != expected_python
            or expected_torch is not None
            and torch_version.split("+", 1)[0] != expected_torch.split("+", 1)[0]):
        raise ValueError(
            "Decode DCP with the producer's Python major/minor and Torch release: "
            f"expected Python {expected_python}, Torch {expected_torch}; "
            f"found Python {list(python_version[:2])}, Torch {torch_version}. "
            "Use an isolated matching CPU environment; do not rewrite checkpoint metadata."
        )


def validate_decoded_adapters(state, marker, parent):
    """Check the complete adapter layout, including explicitly narrowed scopes."""
    import torch

    tensors = {name: tensor for name, tensor in state["model"].items() if "lora_" in name}
    scope = marker.get("adapter_scope")
    if scope is None:
        if parent.get("adapter_last_layers") not in (None, 45):
            raise ValueError("A narrowed adapter needs explicit scope metadata.")
        expected_tensors, expected_elements = 360, 17649664
    else:
        last = parent.get("adapter_last_layers", 45)
        if (scope.get("schema") != "bobcat-native-attention-lora-scope-v1"
                or type(last) is not int or not 1 <= last <= 45
                or scope.get("total_language_layers") != 45
                or scope.get("selected_layers") != list(range(45 - last, 45))
                or scope.get("lora_rank") != 8
                or json_hash(scope) != marker.get("adapter_scope_sha256")):
            raise ValueError("Decoded adapter scope differs from its frozen parent.")
        targets = scope["targets"]
        expected = {
            f"{target['name']}.lora_{part}.weight": (
                (8, target["in_features"]) if part == "A" else (target["out_features"], 8)
            ) for target in targets for part in ("A", "B")
        }
        if (len(expected) != 2 * len(targets)
                or [target["name"] for target in targets] != scope["target_modules"]
                or set(tensors) != set(expected)
                or any(tuple(tensors[name].shape) != shape for name, shape in expected.items())):
            raise ValueError("Decoded adapter names/shapes differ from the training scope.")
        expected_tensors = len(expected)
        expected_elements = scope["trainable_parameters"]
        if parent.get("trainable_parameters") != expected_elements:
            raise ValueError("Parent and checkpoint disagree on the trainable parameter count.")
    if (len(tensors) != expected_tensors
            or sum(t.numel() for t in tensors.values()) != expected_elements
            or any(t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all())
                   for t in tensors.values())):
        raise ValueError("The independently decoded native adapter layout/values changed.")
    optimizer = state["optimizer"]
    if (len(optimizer["state"]) != expected_tensors
            or any(float(item["step"]) != marker["step"] for item in optimizer["state"].values())):
        raise ValueError("Optimizer moments are absent or have inconsistent update counters.")
    return tensors


def build(checkpoint, parent_job_path, train_rows, *, adapter_output=None):
    import torch
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    parent = json.loads(parent_job_path.read_text())
    marker = json.loads((checkpoint / "complete.json").read_text())
    receipt = json.loads((checkpoint / "download-verified.json").read_text())
    validate_decoder_runtime(
        marker.get("producer_runtime", parent.get("runtime", {})),
        python_version=sys.version_info, torch_version=torch.__version__,
    )
    if (receipt.get("verified") is not True
            or receipt["complete_sha256"] != file_hash(checkpoint / "complete.json")
            or receipt["marker"]["key"] != parent["s3_prefix"]
            + f"train/checkpoint-{marker['step']:06d}/complete.json"):
        raise ValueError("Require a version-verified checkpoint from this exact parent run.")
    files = validate_marker(
        marker, step=marker["step"], revision=parent["source_revision"],
        curriculum_sha256=parent["curriculum_manifest_sha256"],
    )
    for name, digest in files.items():
        if file_hash(checkpoint / name) != digest:
            raise ValueError("A recovered source checkpoint changed before independent decoding.")
    cursor = continuation_cursor(marker, parent, train_rows)
    with tempfile.TemporaryDirectory(prefix="bobcat-dcp-reference-") as folder:
        path = Path(folder) / "full-state.pt"
        dcp_to_torch_save(checkpoint, path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if set(state) != {"model", "optimizer"} or not state["model"] or not state["optimizer"]:
            raise ValueError("Expected the adapter and optimizer, not a model-only checkpoint.")
        if any("lora_" not in name and not name.endswith(".e_score_correction_bias")
               for name in state["model"]):
            raise ValueError("Unexpected frozen parameters in the adapter-only reference.")
        signature = state_signature(state)
        tensors = validate_decoded_adapters(state, marker, parent)
        adapter_elements = sum(tensor.numel() for tensor in tensors.values())
        optimizer = state["optimizer"]
        artifact = None
        if adapter_output is not None:
            from safetensors.torch import load_file, save_file

            if adapter_output.exists():
                raise ValueError("Preserve an existing consolidated adapter.")
            save_file({name: value.contiguous().clone() for name, value in tensors.items()},
                      adapter_output, metadata={
                          "format": "bobcat-native-attention-lora-v1",
                          "base_repo": "zai-org/GLM-5.3-Flash",
                          "base_revision": parent["source_revision"],
                          "source_complete_sha256": file_hash(checkpoint / "complete.json"),
                          "scale": "2.0", "complete_model": "false",
                      })
            loaded = load_file(adapter_output, device="cpu")
            if state_signature(loaded) != state_signature(tensors):
                raise ValueError("The safetensors adapter differs from decoded DCP values.")
            artifact = {
                "path": adapter_output.name, "sha256": file_hash(adapter_output),
                "bytes": adapter_output.stat().st_size, "tensor_count": len(tensors),
                "parameters": adapter_elements, "decoded_values_exact": True,
                "scale": 2.0, "complete_model": False,
                "merged_into_base": False, "serving_quality_verified": False,
            }
    result = {
        "schema": "bobcat-native-resume-reference-v1",
        "at": datetime.now(UTC).isoformat(), "parent_job": parent,
        "parent_job_sha256": file_hash(parent_job_path),
        "parent_job_content_sha256": json_hash(parent),
        "marker": marker, "complete_sha256": file_hash(checkpoint / "complete.json"),
        "recovery_receipt_sha256": file_hash(checkpoint / "download-verified.json"),
        "cursor": cursor, "full_state_signature": signature,
        "full_state_sha256": json_hash(signature),
        "independent_decoder": "torch.distributed.checkpoint.format_utils.dcp_to_torch_save",
        "decoder_torch_version": torch.__version__,
        "decoder_python_version": list(sys.version_info[:3]),
        "adapter_tensors": len(tensors), "adapter_elements": adapter_elements,
        "optimizer_parameter_states": len(optimizer["state"]),
        "training_performed": False, "gpu_used": False,
        "new_process_gpu_restoration_verified": False,
    }
    if artifact is not None:
        result["adapter_artifact"] = artifact
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-job", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve each independent checkpoint reference.")
    result = build(args.checkpoint, args.parent_job, args.train_rows,
                   adapter_output=args.adapter_output)
    atomic_json(args.out, result)
    print(json.dumps({key: result[key] for key in (
        "cursor", "full_state_sha256", "adapter_tensors", "adapter_elements",
        "optimizer_parameter_states", "gpu_used", "new_process_gpu_restoration_verified",
    )}))


if __name__ == "__main__":
    main()
