"""Compile existing, split-frozen judgments for native GLM backbone adaptation.

Uses the identical first-position token compiler as the serving baseline. Gold
and source identities remain separate from model input; no truncation, teacher
query, calibration reuse, or pretrained-weight mutation occurs here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_feature_data import MIXED_PLAN_SCHEMA, validate_plan
from bobcat.glm_readout import PROFILE, GLMCompiler
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash
from bobcat.supervision import target_values

SCHEMA = "bobcat-glm-native-training-data-v1"


def compile_plan(plan: dict, compiler: GLMCompiler) -> dict:
    validate_plan(plan)
    if plan["schema"] != MIXED_PLAN_SCHEMA:
        raise ValueError("Native adaptation starts from the mixed Korean/English supervision plan.")
    records, tokens, groups = [], Counter(), Counter()
    counts, candidate_counts = Counter(), Counter()
    for group in plan["groups"]:
        state, questions = parse_request(group["request"])
        compiled = compiler.compile(state, questions)
        groups[f"{group['split']}/{group['task']}"] += 1
        for row, ids, options in zip(
            group["rows"], compiled.input_ids, compiled.option_token_ids, strict=True,
        ):
            target, mean = target_values(row)
            language = row.get("language")
            if language not in ("ko", "en"):
                raise ValueError("Record the source language instead of inferring it from a label.")
            if len(options) != len(row["candidate_ids"]) or len(set(options)) != len(options):
                raise ValueError("Each offered candidate needs its own verified vocabulary token.")
            inputs = {"input_ids": ids, "option_token_ids": options}
            records.append({
                "inputs": inputs, "input_sha256": json_hash(inputs),
                "supervision": {
                    "target_index": target, "score_mean": mean,
                    "context_weight": row["context_weight"],
                },
                "row": row, "task": group["task"], "split": group["split"],
            })
            name = f"{group['split']}/{language}"
            counts[name] += 1
            tokens[name] += len(ids)
            candidate_counts[f"{group['split']}/{len(options)}"] += 1
    result = {
        "schema": SCHEMA, "source_plan_content_sha256": plan["content_sha256"],
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "compiler_profile": PROFILE, "model_source": compiler.source,
        "generator_sha256": file_hash(Path(__file__)),
        "compiler_sha256": file_hash(Path(__file__).with_name("glm_readout.py")),
        "groups": dict(groups), "questions": dict(counts), "physical_input_tokens": dict(tokens),
        "candidate_counts": dict(candidate_counts), "record_count": len(records),
        "maximum_input_tokens": max(len(row["inputs"]["input_ids"]) for row in records),
        "padding_tokens_counted": False, "physical_prefix_reuse_claimed": False,
        "calibration_and_public_validation_included": False, "teacher_labels_used": False,
        "ordinal_means_are_not_vote_distributions": True, "training_performed": False,
        "records": records,
    }
    result["content_sha256"] = json_hash(result)
    return result


def validate(data: dict) -> None:
    if (
        data.get("schema") != SCHEMA
        or data.get("content_sha256") != json_hash({
            key: value for key, value in data.items() if key != "content_sha256"
        })
        or data.get("record_count") != len(data.get("records", []))
        or not data.get("records")
    ):
        raise ValueError("Use the nonempty frozen native input artifact.")
    identities, split_by_component = set(), {}
    for record in data["records"]:
        row, inputs, loss = record["row"], record["inputs"], record["supervision"]
        target, mean = target_values(row)
        ids, options = inputs["input_ids"], inputs["option_token_ids"]
        if (
            set(inputs) != {"input_ids", "option_token_ids"}
            or record["input_sha256"] != json_hash(inputs)
            or not ids or any(type(i) is not int or i < 0 for i in ids + options)
            or len(options) != len(row["candidate_ids"]) or len(set(options)) != len(options)
            or loss != {"target_index": target, "score_mean": mean,
                        "context_weight": row["context_weight"]}
            or row["id"] in identities
            or record["split"] not in ("train", "dev_train")
            or record["split"] != row["split"]
        ):
            raise ValueError("Compiled inputs and separate supervision are misaligned.")
        component = row["group_id"]
        if component in split_by_component and split_by_component[component] != row["split"]:
            raise ValueError("A native source component crosses the split boundary.")
        split_by_component[component] = row["split"]
        identities.add(row["id"])


def single_rank_batch(record: dict, padded_length: int, *, pad_token_id=0, device="cpu"):
    """Right-pad one branch for EP shape agreement and select its real last position."""
    return packed_rank_batch([record], padded_length, pad_token_id=pad_token_id, device=device)


def packed_rank_batch(
    records: list[dict], padded_length: int, *, pad_token_id=0, device="cpu",
    cache_packed_boundaries=False,
):
    """Pack complete, independent questions using the native GLM document boundary API.

    Every question retains its original tokens and one scored position. Document
    IDs reset KDA/short-convolution state and isolate DSA in the pinned native
    implementation; this helper does not implement those kernels itself.
    Padding has document ID zero and is never scored. Gold and metadata are
    deliberately not accepted as model inputs.
    """
    import torch

    if type(cache_packed_boundaries) is not bool:
        raise ValueError("Packed boundary caching must be an explicit boolean.")
    if not records:
        raise ValueError("Pack at least one complete question.")
    sequences = [record["inputs"]["input_ids"] for record in records]
    if any(not ids or any(type(i) is not int or i < 0 for i in ids) for ids in sequences):
        raise ValueError("Every question must contain valid, nonempty token IDs.")
    real_length = sum(map(len, sequences))
    if type(padded_length) is not int or not real_length <= padded_length <= 32767:
        raise ValueError("The agreed EP length must fit every complete input without truncation.")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("Use a valid nonnegative padding token.")
    tokens, document_ids, scored_positions = [], [], []
    for document_id, ids in enumerate(sequences, start=1):
        tokens.extend(ids)
        document_ids.extend([document_id] * len(ids))
        scored_positions.append(len(tokens) - 1)
    padding = padded_length - real_length
    batch = {
        "input_ids": torch.tensor([tokens + [pad_token_id] * padding],
                                 device=device, dtype=torch.long),
        "_packed_seq_ids": torch.tensor([document_ids + [0] * padding],
                                        device=device, dtype=torch.int32),
        "logits_to_keep": torch.tensor(scored_positions, device=device, dtype=torch.long),
    }
    if cache_packed_boundaries:
        # The pinned native context accepts a populated constructor-visible
        # cache. FSDP rebuilds dataclass kwargs at module boundaries: populating
        # it before the first FSDP call avoids repeating CUDA nonzero + CPU sync
        # in every layer. Boundaries come only from the unchanged CPU token
        # lengths, including the trailing padding run. No mask or token is cut.
        from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext

        boundaries = [0, *(position + 1 for position in scored_positions)]
        if padding:
            boundaries.append(padded_length)
        cpu = torch.tensor(boundaries, dtype=torch.long, device="cpu")
        batch["glm5_next_packed_context"] = Glm5NextPackedContext(
            doc_ids=batch["_packed_seq_ids"], original_seq_len=padded_length,
            _cu_seqlens={0: (cpu.to(device), cpu)},
        )
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve the previous compiled artifact.")
    compiler = GLMCompiler(args.model_dir, json.loads(args.source.read_text()))
    result = compile_plan(json.loads(args.plan.read_text()), compiler)
    validate(result)
    atomic_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("records", "model_source")}, indent=2))


if __name__ == "__main__":
    main()
