"""Export original GLM decision rows and check existing native frozen features.

This is an output-module artifact, not a complete released model. The CPU
feature check reuses recorded hidden vectors; it performs no new backbone run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch

from bobcat.corpus import atomic_json
from bobcat.decision_projection import DecisionProjection
from bobcat.schema import file_hash


def export_and_check(source_path: Path, source_head: Path, features: Path, out: Path):
    from safetensors.torch import load_file, save_file

    source = json.loads(source_path.read_text())
    run = json.loads((features / "run.json").read_text())
    provenance = run["scorer_provenance"]
    expected_hash = provenance["option_head_sha256"]
    if (file_hash(source_head) != expected_hash or run["status"] != "completed"
            or provenance["base_weights_updated"] is not False
            or provenance["feature_identity_checked_on_every_request"] is not True
            or provenance["feature_profile"] != "glm53_last_postnorm_fp32_lm_head_v1"):
        raise ValueError("Use completed original, identity-checked frozen feature evidence.")
    payload = torch.load(source_head, map_location="cpu", weights_only=True)
    shard = next((f for f in source["files"] if f["path"] == payload["verified_shard"]), None)
    weights, ids = payload["weights"], payload["token_ids"]
    if (payload["format"] != "bobcat-glm-option-head-v1" or shard is None
            or shard["sha256"] != payload["verified_shard_sha256"]
            or payload["base_repo"] != source["repo"]
            or payload["base_repo"] != provenance["base_repo"]
            or payload["base_revision"] != source["revision"]
            or payload["base_revision"] != provenance["base_revision"]
            or payload["head_tensor"] != "lm_head.weight"
            or payload["head_shape"] != [154880, 4096]
            or weights.shape != (255, 4096) or weights.dtype != torch.bfloat16
            or len(ids) != 255 or len(set(ids)) != 255
            or any(type(i) is not int or not 0 <= i < 154880 for i in ids)
            or not torch.isfinite(weights).all()):
        raise ValueError("The selected original GLM rows or source identity changed.")
    out.mkdir(parents=True, exist_ok=False)
    projection = DecisionProjection(weights, ids)
    path = out / "decision-head.safetensors"
    save_file(projection.state_dict(), path, metadata={
        "format": "bobcat-original-decision-bank-v1",
        "base_repo": source["repo"], "base_revision": source["revision"],
        "source_head_sha256": expected_hash, "complete_model": "false",
    })
    restored = DecisionProjection(torch.zeros_like(weights), ids)
    restored.load_state_dict(load_file(path))
    if not all(torch.equal(value, restored.state_dict()[name])
               for name, value in projection.state_dict().items()):
        raise ValueError("The safetensors round trip changed a weight or identifier.")

    result = {
        "schema": "bobcat-original-decision-bank-export-v1",
        "at": datetime.now(UTC).isoformat(), "status": "checking_recorded_features",
        "base_repo": source["repo"], "base_revision": source["revision"],
        "source_manifest_sha256": file_hash(source_path),
        "source_head_sha256": expected_hash,
        "verified_original_shard_sha256": shard["sha256"],
        "feature_manifest_sha256": file_hash(features / "run.json"),
        "exporter_sha256": file_hash(Path(__file__)),
        "projection_source_sha256": file_hash(Path(__file__).with_name("decision_projection.py")),
        "artifact": {"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)},
        "projection": projection.provenance(
            original_vocabulary=154880, source_revision=source["revision"]),
        "weights_precision": "bfloat16", "feature_check_compute_precision": "float32",
        "roundtrip_exact": True, "native_relative_logit_absolute_limit": .005,
        "allowed_argmax_changes": 0, "new_backbone_forward_performed": False,
        "full_pretrained_cuda_head_installed": False, "speed_measured": False,
        "quality_claimed": False, "release_model_complete": False,
    }
    atomic_json(out / "manifest.json", result)
    torch.set_num_threads(2)
    restored.float()
    count, flips, max_error, max_tv = 0, 0, 0., 0.
    families, candidate_counts, components, seen = Counter(), Counter(), set(), set()
    try:
        with torch.inference_mode():
            for record in run["files"]:
                name = record["path"]
                file = features / name
                if (Path(name).name != name or name in seen or file.is_symlink()
                        or file.stat().st_size != record["bytes"]
                        or file_hash(file) != record["sha256"]):
                    raise ValueError("A frozen native feature shard changed.")
                seen.add(name)
                data = torch.load(file, map_location="cpu", weights_only=True)
                hidden, base, keep = data["hidden"], data["base_logits"], data["candidate_keep"]
                rows = data["rows"]
                counts = [len(r["candidate_ids"]) for r in rows]
                if (data["plan_sha256"] != run["plan_sha256"]
                        or len(rows) != record["questions"]
                        or hidden.shape != (len(rows), 4096) or hidden.dtype != torch.float32
                        or keep.shape != base.shape or keep.dtype != torch.bool
                        or base.shape[0] != len(rows)
                        or not torch.isfinite(hidden).all()
                        or not torch.isfinite(base[keep]).all()
                        or not torch.equal(keep, torch.arange(base.shape[-1])[None]
                                           < torch.tensor(counts)[:, None])):
                    raise ValueError("The recorded native feature/candidate alignment changed.")
                actual = restored.select_batch(restored(hidden), [ids[:k] for k in counts])
                for i, (value, row, k) in enumerate(zip(actual, rows, counts, strict=True)):
                    reference = base[i, :k]
                    relative_error = float(((value - value[0])
                                            - (reference - reference[0])).abs().max())
                    tv = float((value.softmax(-1) - reference.softmax(-1)).abs().sum() / 2)
                    max_error, max_tv = max(max_error, relative_error), max(max_tv, tv)
                    flips += int(value.argmax() != reference.argmax())
                    count += 1
                    families[row["family"]] += 1
                    candidate_counts[k] += 1
                    components.add(row["group_id"])
        result.update(
            checked_feature_questions=count, checked_source_components=len(components),
            checked_feature_shards=len(seen), candidate_counts=dict(candidate_counts),
            families=dict(families), maximum_relative_logit_error=max_error,
            maximum_probability_tv=max_tv, argmax_changes=flips,
        )
        if count != run["completed_questions"] or max_error > .005 or flips:
            raise ValueError("The candidate-only projection did not preserve recorded decisions.")
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(out / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-head", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = export_and_check(args.source, args.source_head, args.features, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "projection"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
