"""Prepare original Korean exam questions for evaluation, never for training.

Keep downloaded source files private and unmodified. This adapter emits a
development-monitoring artifact, not a claim of unseen foundation-model data.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_readout import PROFILE, GLMCompiler
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash

REVISION = "d61b3f19e552c576bf5960dd24289763edc36a88"
LETTERS = ("A", "B", "C", "D")
REQUIRED = {"question", "answer", *LETTERS, "Category", "Human Accuracy"}


def original_question(row):
    if (set(row) != REQUIRED or row["answer"] not in ("1", "2", "3", "4")
            or any(not row[key].strip() for key in ("question", *LETTERS))):
        raise ValueError("Keep complete original questions and the publisher's one-based gold.")
    request = {
        "model": "bobcat-glm-reference",
        "state": row["question"],
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "문제를 읽고 가장 적절한 정답 하나를 고르세요.",
                "criteria": {key: row[key] for key in LETTERS},
            },
        },
    }
    return request, int(row["answer"]) - 1


def load_original(folder):
    manifest = json.loads((folder / "source-manifest.json").read_text())
    if (manifest.get("dataset") != "HAERAE-HUB/KMMLU"
            or manifest.get("revision") != REVISION or manifest.get("license") != "cc-by-nd-4.0"
            or manifest.get("model_evaluated") is not False):
        raise ValueError("Use the pinned, unmodified original benchmark source.")
    entries = [item for item in manifest["files"] if item["path"].endswith("-test.csv")]
    if len(entries) != 45 or len({item["path"] for item in entries}) != 45:
        raise ValueError("Preserve all 45 original test subjects.")
    result = []
    for item in entries:
        path = folder / item["path"]
        if (Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts
                or path.is_symlink() or file_hash(path) != item["sha256"]):
            raise ValueError("An original benchmark file changed.")
        with path.open(newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                request, target = original_question(row)
                normalized = unicodedata.normalize("NFC", row["question"])
                result.append({
                    "source_file": item["path"], "source_row": index,
                    "subject": row["Category"], "request": request, "target_index": target,
                    "group_id": "kmmlu:" + json_hash(normalized),
                })
    if len(result) != 35030:
        raise ValueError("The original revision's complete test count changed.")
    return manifest, result


def select_balanced(rows, size, seed):
    """Select by stable input identity, without reading gold or human accuracy."""
    if type(size) is not int or size < 48 or size % 8:
        raise ValueError("Use a subject-spanning multiple of eight.")
    by_subject, duplicate_groups = defaultdict(list), Counter()
    for row in rows:
        by_subject[row["subject"]].append(row)
        duplicate_groups[row["group_id"]] += 1
    for values in by_subject.values():
        values.sort(key=lambda row: json_hash({
            "seed": seed, "request": row["request"], "source_file": row["source_file"],
            "source_row": row["source_row"],
        }))
    chosen, groups, offsets = [], set(), Counter()
    while len(chosen) < size:
        added = 0
        for subject in sorted(by_subject):
            values = by_subject[subject]
            while offsets[subject] < len(values):
                row = values[offsets[subject]]
                offsets[subject] += 1
                if row["group_id"] in groups:
                    continue
                chosen.append(row)
                groups.add(row["group_id"])
                added += 1
                break
            if len(chosen) == size:
                break
        if not added:
            raise ValueError("Not enough distinct source-question groups.")
    return chosen, {
        "method": "round_robin_subjects_then_seeded_input_hash",
        "seed": seed, "gold_or_human_accuracy_used": False,
        "repeated_source_question_groups": sum(count > 1 for count in duplicate_groups.values()),
        "question_groups_selected_once": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--training-curriculum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve a frozen earlier evaluation cohort.")
    source, originals = load_original(args.source)
    selected, sampling = select_balanced(originals, args.size, args.seed)
    model_source = json.loads(args.model_source.read_text())
    compiler = GLMCompiler(args.model_dir, model_source, max_branch_tokens=8192)
    training = [
        json.loads(line) for line in
        (args.training_curriculum / "train.jsonl").read_text().splitlines()
    ]
    trained_inputs = {row["input_sha256"] for row in training}
    records, audit = [], []
    for original in selected:
        state, questions = parse_request(original["request"])
        encoded = compiler.compile(state, questions)
        ids, options = encoded.input_ids[0], encoded.option_token_ids[0]
        input_hash = json_hash({"input_ids": ids, "option_token_ids": options})
        if input_hash in trained_inputs:
            raise ValueError("An evaluated question exactly matches the actual training inputs.")
        identity = f"kmmlu:{original['subject']}:{original['source_row']}"
        records.append({
            "id": identity, "group_id": original["group_id"],
            "observation_id": identity, "split": "dev_train",
            "task": "kmmlu/" + original["subject"], "family": "korean_exam_knowledge",
            "language": "ko", "language_origin": "native", "kind": "choice",
            "supervision": "hard_label", "input_ids": ids, "option_token_ids": options,
            "candidate_ids": list(LETTERS), "input_sha256": input_hash,
            "source_request_sha256": json_hash(original["request"]),
            "target_index": original["target_index"], "score_mean": 0.,
            "context_weight": 1., "source_context_weight": 1., "sampling_loss_weight": 1.,
            "input_tokens": len(ids), "last_input_position": len(ids) - 1,
        })
        audit.append({key: original[key] for key in (
            "source_file", "source_row", "subject", "group_id",
        )})
    args.out.mkdir(parents=True)
    with (args.out / "records.jsonl").open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(args.out / "selection.json", audit)
    manifest = {
        "schema": "bobcat-checkpoint-monitor-suite-v1",
        "at": datetime.now(UTC).isoformat(), "evaluation_role": "development_monitoring",
        "training_or_calibration": False, "source_revision": model_source["revision"],
        "dataset_revision": REVISION, "dataset_license": source["license"],
        "source_manifest_sha256": file_hash(args.source / "source-manifest.json"),
        "model_source_sha256": file_hash(args.model_source),
        "compiler_profile": PROFILE, "compiler_sha256": file_hash(Path(__file__).with_name(
            "glm_readout.py",
        )), "generator_sha256": file_hash(Path(__file__)),
        "records_sha256": file_hash(args.out / "records.jsonl"),
        "selection_sha256": file_hash(args.out / "selection.json"),
        "rows": len(records), "max_input_tokens": max(row["input_tokens"] for row in records),
        "prompt_tokens": sum(row["input_tokens"] for row in records),
        "subjects": dict(Counter(row["subject"] for row in selected)),
        "sampling": sampling, "official_full_benchmark_score": False,
        "raw_data_public_redistribution": False, "original_question_and_choices_edited": False,
        "gold_and_human_accuracy_in_input": False, "input_truncation": False,
        "actual_training_records_checked": len(training), "exact_training_input_overlap": 0,
        "training_curriculum_sha256": file_hash(args.training_curriculum / "manifest.json"),
        "foundation_pretraining_exposure_known": False, "gpu_evaluation_performed": False,
    }
    manifest["content_sha256"] = json_hash(manifest)
    atomic_json(args.out / "manifest.json", manifest)
    from bobcat.glm_native_evaluate import read_suite
    read_suite(args.out)
    print(json.dumps({k: manifest[k] for k in (
        "rows", "max_input_tokens", "prompt_tokens", "exact_training_input_overlap",
        "gpu_evaluation_performed", "content_sha256",
    )}))


if __name__ == "__main__":
    main()
