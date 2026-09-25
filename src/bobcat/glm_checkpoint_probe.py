"""Real CPU checkpoint round-trip for bounded native GLM FP8 initialization.

Creates a small synthetic block-FP8 safetensors checkpoint, reads it using the
vendor's DCP reader, and compares every tensor and model logits with the vendor's
whole-state conversion. It does not load pretrained data or measure GPU memory.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import verify_source
from bobcat.glm_checkpoint_parts import bounded_glm_adapter
from bobcat.schema import file_hash


def run_probe(fixture, out):
    import torch
    import torch.distributed.checkpoint as dcp
    from nemo_automodel.components.checkpoint._backports.hf_storage import (
        _HuggingFaceStorageReader,
    )
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(2)
    torch.manual_seed(202609222117)
    config = fixture.tiny_glm5_next_config()
    config.text_config.torch_dtype = torch.bfloat16
    config.vision_config.torch_dtype = torch.bfloat16
    backend = BackendConfig(
        attn="sdpa", linear="torch", rms_norm="torch_fp32", experts="torch",
        dispatcher="torch", rope_fusion=False, enable_hf_state_dict_adapter=True,
    )

    def build():
        model = Glm5NextForConditionalGeneration(config, backend=backend)
        model.initialize_weights(torch.device("cpu"), dtype=torch.bfloat16)
        return model.eval()

    original = build()
    original_state = {name: tensor.clone() for name, tensor in original.state_dict().items()}
    hf = original.state_dict_adapter.to_hf(original_state, quantization=True)
    # Different known block scales make omission or double application detectable.
    scale_count = 0
    for name in sorted(hf):
        if name.endswith("_scale_inv"):
            scale_count += 1
            hf[name].fill_(0.5 if scale_count % 2 else 2.0)
    if scale_count < 10:
        raise ValueError("Exercise dense, sparse-attention and grouped expert FP8 conversions.")
    checkpoint_dir = out / "synthetic-fp8"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model.safetensors"
    save_file({name: value.contiguous() for name, value in hf.items()}, str(checkpoint))
    del hf
    checkpoint_hash = file_hash(checkpoint)
    reference = build()
    converted = reference.state_dict_adapter.from_hf(load_file(str(checkpoint)))
    if set(converted) != set(original_state):
        raise ValueError("The whole-state reference did not reconstruct every native key.")
    reference.load_state_dict(converted, strict=True)
    expected = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    del converted

    candidate = build()
    targets = candidate.state_dict()
    storage_before = {name: tensor.data_ptr() for name, tensor in targets.items()}
    adapter = bounded_glm_adapter(candidate.state_dict_adapter, max_local_layer_bytes=1024**2)
    parts = adapter.iter_checkpoint_load_parts(targets)
    reader = _HuggingFaceStorageReader(str(checkpoint_dir))
    completed, requested, observations = set(), set(), []
    first_callback = None
    for part in parts:
        native = set(part.model_keys)
        keys = set(part.checkpoint_tensors)
        if native & completed or keys & requested:
            raise ValueError("A model/checkpoint tensor was loaded twice.")
        native_bytes = sum(targets[name].numel() * targets[name].element_size() for name in native)
        temporary_bytes = sum(
            part.checkpoint_tensors[name].numel() * part.checkpoint_tensors[name].element_size()
            for name in part.temporary_checkpoint_keys
        )
        observations.append({
            "native_keys": sorted(native), "checkpoint_key_count": len(keys),
            "native_bytes": native_bytes, "temporary_destination_bytes": temporary_bytes,
        })
        dcp.load(part.checkpoint_tensors, storage_reader=reader)
        part.finish()
        if first_callback is None:
            first_callback = part.finish
        for name in native:
            if not torch.equal(targets[name], expected[name]):
                raise ValueError(f"Bounded conversion differs from whole-state conversion: {name}")
            if targets[name].data_ptr() != storage_before[name]:
                raise ValueError("The bounded loader replaced existing model storage.")
        completed |= native
        requested |= keys
        del part
    if completed != set(expected) or len(observations) != 5:
        raise ValueError("Expected one shared group and four complete decoder layers.")
    if file_hash(checkpoint) != checkpoint_hash:
        raise ValueError("The read changed the source checkpoint.")
    try:
        first_callback()
    except ValueError:
        reuse_rejected = True
    else:
        raise ValueError("A retained earlier load callback was allowed to execute again.")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    docs = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.int32)
    with torch.no_grad():
        before = reference(input_ids=ids, attention_mask=docs, logits_to_keep=1).logits
        after = candidate(input_ids=ids, attention_mask=docs, logits_to_keep=1).logits
    if not torch.isfinite(after).all() or not torch.equal(before, after):
        raise ValueError("Whole-state and bounded-loaded models differ in actual CPU forward.")
    incomplete = {name: value for name, value in targets.items()
                  if not name.startswith("model.language_model.layers.2.")}
    try:
        adapter.iter_checkpoint_load_parts(incomplete)
    except ValueError:
        incomplete_rejected = True
    else:
        raise ValueError("The loader accepted a missing complete decoder layer.")
    limited = bounded_glm_adapter(candidate.state_dict_adapter, max_local_layer_bytes=1)
    try:
        next(limited.iter_checkpoint_load_parts(targets))
    except ValueError:
        byte_limit_rejected = True
    else:
        raise ValueError("The native-byte bound was not enforced.")
    early = adapter.iter_checkpoint_load_parts(targets)
    next(early)
    try:
        next(early)
    except ValueError:
        unfinished_rejected = True
    else:
        raise ValueError("The iterator allowed a prior part to remain unfinished.")
    return {
        "status": "passed", "parameters": sum(p.numel() for p in candidate.parameters()),
        "native_tensors": len(expected), "checkpoint_tensors": len(requested),
        "fp8_scale_tensors": scale_count, "load_parts": observations,
        "all_native_tensors_exact": True, "original_storage_preserved": True,
        "actual_cpu_logits_exact": True, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "previous_callback_reuse_rejected": reuse_rejected,
        "missing_layer_rejected": incomplete_rejected,
        "native_byte_limit_rejected": byte_limit_rejected,
        "unfinished_part_rejected": unfinished_rejected,
        "pretrained_weights_loaded": False, "distributed_loading_verified": False,
        "gpu_memory_fit_verified": False, "language_quality_measured": False,
        "temporary_bytes_scope": (
            "DCP destination tensors only; excludes conversion and reader workspace."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--git-tree", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh output to preserve previous evidence.")
    args.out.mkdir(parents=True)
    start = time.monotonic()
    record = {
        "schema": "bobcat-glm-bounded-checkpoint-cpu-probe-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "probe_sha256": file_hash(Path(__file__)),
        "loader_sha256": file_hash(Path(__file__).with_name("glm_checkpoint_parts.py")),
        "release_gate_passed": False,
    }
    atomic_json(args.out / "result.json", record)
    try:
        record["source"] = verify_source(args.source_root, args.git_tree)
        sys.path.insert(0, str(args.source_root.resolve()))
        fixture_path = args.source_root / "tests/unit_tests/models/glm5_next/conftest.py"
        record["fixture_sha256"] = file_hash(fixture_path)
        record["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        }
        spec = importlib.util.spec_from_file_location("bobcat_glm_loader_fixture", fixture_path)
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        record["result"] = run_probe(fixture, args.out)
        record["status"] = record["result"]["status"]
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - start)
        atomic_json(args.out / "result.json", record)
    print(json.dumps({"status": record["status"], "pretrained_weights_loaded": False}))


if __name__ == "__main__":
    main()
