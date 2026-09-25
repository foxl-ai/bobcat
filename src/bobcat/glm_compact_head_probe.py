"""Read a real tiny FP8 checkpoint into a physically compact native output.

CPU only: validates DCP destinations, original backbone restoration, direct
numeric output and input gradients. Full-size CUDA is a separate gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import verify_source
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

    from bobcat.glm_compact_head import (
        compact_glm_checkpoint_adapter,
        install_uninitialized_native_decision_projection,
    )
    from bobcat.glm_native_data import single_rank_batch

    torch.set_num_threads(2)
    torch.manual_seed(2026092323)
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
        return model.eval().requires_grad_(False)

    original = build()
    native = {name: value.clone() for name, value in original.state_dict().items()}
    hf = original.state_dict_adapter.to_hf(native, quantization=True)
    for index, name in enumerate(sorted(n for n in hf if n.endswith("_scale_inv"))):
        hf[name].fill_(.5 if index % 2 else 2.)
    checkpoint = out / "synthetic-fp8"
    checkpoint.mkdir()
    saved = checkpoint / "model.safetensors"
    save_file({name: value.contiguous() for name, value in hf.items()}, str(saved))
    before_file = file_hash(saved)
    reference = build()
    expected = reference.state_dict_adapter.from_hf(load_file(str(saved)))
    reference.load_state_dict(expected, strict=True)
    identifiers = [23, 8, 41, 7, 22, 20, 9, 40, 21]
    bank = expected["lm_head.weight"][identifiers].clone()
    candidate = build()
    embeddings = candidate.get_input_embeddings()
    projection = install_uninitialized_native_decision_projection(candidate, identifiers)
    if candidate.get_input_embeddings() is not embeddings:
        raise ValueError("Output construction changed input embeddings.")
    targets = candidate.state_dict()
    if targets.keys() != expected.keys():
        raise ValueError("The compact readout introduced an untracked native state tensor.")
    pointers = {name: value.data_ptr() for name, value in targets.items()}
    adapter = compact_glm_checkpoint_adapter(
        candidate.state_dict_adapter, bank, max_local_layer_bytes=1024**2,
    )
    reader = _HuggingFaceStorageReader(str(checkpoint))
    completed, requested = set(), set()
    first_finish = None
    parts = []
    for part in adapter.iter_checkpoint_load_parts(targets):
        if completed & part.model_keys or requested & set(part.checkpoint_tensors):
            raise ValueError("Duplicate native tensor or checkpoint destination.")
        if "lm_head.weight" in part.checkpoint_tensors:
            raise ValueError("The compact loader still reads the original full-vocabulary head.")
        dcp.load(part.checkpoint_tensors, storage_reader=reader)
        part.finish()
        first_finish = first_finish or part.finish
        for name in part.model_keys:
            wanted = bank if name == "lm_head.weight" else expected[name]
            if not torch.equal(targets[name], wanted) or targets[name].data_ptr() != pointers[name]:
                raise ValueError("A compact load changed native values or storage ownership.")
        completed.update(part.model_keys)
        requested.update(part.checkpoint_tensors)
        parts.append({
            "native_keys": sorted(part.model_keys),
            "checkpoint_keys": sorted(part.checkpoint_tensors),
        })
    if completed != set(targets) or file_hash(saved) != before_file:
        raise ValueError("Incomplete restore or modified source checkpoint.")
    try:
        first_finish()
    except ValueError:
        pass
    else:
        raise ValueError("A finished compact part was installed twice.")
    rows = [
        {"input_ids": [1, 2, 3, 4, 5], "options": [7, 8, 9]},
        {"input_ids": [11, 12, 13], "options": [20, 21, 22, 23]},
        {"input_ids": [30, 31, 32, 33], "options": [40, 41]},
    ]
    max_tv, max_logit, flipped = 0., 0., 0
    with torch.no_grad():
        for row in rows:
            batch = single_rank_batch({"inputs": {"input_ids": row["input_ids"]}}, 16)
            original_logits = reference(**batch).logits[0, -1, row["options"]].float()
            outputs = candidate(**batch).logits
            if outputs.shape[-1] != len(identifiers):
                raise ValueError("The forward output still contains the original vocabulary.")
            new_logits = projection.select_batch(outputs[0], [row["options"]])[0]
            if not torch.isfinite(new_logits).all():
                raise ValueError("Non-finite candidate output.")
            max_tv = max(max_tv, float((
                original_logits.softmax(-1) - new_logits.softmax(-1)
            ).abs().sum() / 2))
            max_logit = max(max_logit, float((original_logits - new_logits).abs().max()))
            flipped += int(original_logits.argmax() != new_logits.argmax())
    if max_tv > .001 or flipped:
        raise ValueError("Compact native output failed the unchanged numerical gate.")
    # The output remains differentiable with respect to the backbone hidden.
    hidden = torch.randn(3, bank.shape[1], dtype=torch.bfloat16, requires_grad=True)
    full_output = reference.get_output_embeddings()(hidden)[:, identifiers].float()
    full_output.square().mean().backward()
    expected_gradient = hidden.grad.clone()
    hidden.grad = None
    projection(hidden).float().square().mean().backward()
    torch.testing.assert_close(hidden.grad, expected_gradient, rtol=.01, atol=.001)
    return {
        "status": "passed", "device": "cpu", "dtype": "bfloat16",
        "original_parameters": sum(p.numel() for p in reference.parameters()),
        "compact_parameters": sum(p.numel() for p in candidate.parameters()),
        "original_output_shape": list(expected["lm_head.weight"].shape),
        "compact_output_shape": list(bank.shape),
        "native_tensors_restored": len(completed),
        "original_backbone_values_exact": True, "existing_storage_preserved": True,
        "full_vocabulary_output_requested_from_checkpoint": False,
        "full_vocabulary_output_in_candidate_model": False,
        "input_embedding_preserved": True, "questions": len(rows),
        "maximum_probability_tv": max_tv, "maximum_logit_difference": max_logit,
        "argmax_changes": flipped, "hidden_gradient_close": True,
        "hidden_gradient_max_abs": float((hidden.grad - expected_gradient).abs().max()),
        "repeated_finish_rejected": True, "parts": parts,
        "source_checkpoint_sha256": before_file,
        "full_pretrained_weights_tested": False, "distributed_loading_tested": False,
        "gpu_speed_measured": False, "release_gate_passed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--git-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "bobcat-compact-native-output-probe-v1",
        "started_at": datetime.now(UTC).isoformat(),
        "probe_sha256": file_hash(Path(__file__)),
        "implementation_sha256": file_hash(Path(__file__).with_name("glm_compact_head.py")),
    }
    began = time.monotonic()
    try:
        record["source"] = verify_source(args.source_root, args.git_tree)
        sys.path.insert(0, str(args.source_root.resolve()))
        spec = importlib.util.spec_from_file_location(
            "bobcat_compact_head_fixture",
            args.source_root / "tests/unit_tests/models/glm5_next/conftest.py",
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        record.update(run_probe(fixture, args.out))
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - began)
        atomic_json(args.out / "result.json", record)
    print(json.dumps({k: record[k] for k in (
        "status", "original_output_shape", "compact_output_shape",
        "maximum_probability_tv", "hidden_gradient_max_abs",
    )}))


if __name__ == "__main__":
    main()
