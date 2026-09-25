"""Freeze and extract real Korean features for a bounded GLM output-head study."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from bobcat.corpus import atomic_json
from bobcat.glm_features import VerifiedFeatureScorer
from bobcat.klue import request_for
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash, read_examples
from bobcat.supervision import target_tensors, target_values

PLAN_SCHEMA = "bobcat-glm-feature-plan-v1"
MIXED_PLAN_SCHEMA = "bobcat-glm-feature-plan-v2"
RUN_SCHEMA = "bobcat-glm-feature-run-v1"


def freeze(data: Path, out: Path, *, train_per_task: int = 2048,
           development_per_task: int = 128, seed: int = 20260922) -> dict:
    if out.exists() or not 1 <= development_per_task <= train_per_task <= 16384:
        raise ValueError("Use a fresh plan and explicit finite per-task sample counts.")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "bobcat-korean-decisions-v1":
        raise ValueError("Use the pinned real Korean decision data.")
    groups, seen_components = [], set()
    for split, per_task in (("train", train_per_task), ("dev_train", development_per_task)):
        path = data / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError("Feature plan source checksum mismatch.")
        by_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for e in read_examples(path):
            m = e.metadata
            if (e.split != split or m.get("source_split") != "train"
                    or m.get("annotation_origin") != "upstream_human"
                    or m.get("source_task") not in {"nli", "ynat"}):
                raise ValueError("Feature fitting only accepts the original training partition.")
            by_task[m["source_task"]][e.group_id][m["independent_observation"]].append(e)
        for task in ("nli", "ynat"):
            ordered = sorted(by_task[task], key=lambda group: json_hash([seed, group]))
            selected = [group for group in ordered if group not in seen_components][:per_task]
            if len(selected) != per_task:
                raise ValueError(f"Insufficient independent components: {split}/{task}")
            for group in selected:
                observations = by_task[task][group]
                observation = min(observations, key=lambda key: json_hash([seed, key]))
                examples = sorted(observations[observation], key=lambda e: e.metadata["view"])
                if (len(examples) != (3 if task == "nli" else 1)
                        or len({e.context for e in examples}) != 1):
                    raise ValueError("The selected source context lost a complete view set.")
                # Choice labels are dynamic at training time. Noul's false/true
                # order is part of the public typed protocol and stays canonical.
                if split == "train":
                    for e in examples:
                        if e.kind == "choice":
                            random.Random(int(json_hash([seed, e.id])[:16], 16)).shuffle(e.choices)
                request = {
                    "model": "bobcat-latest", "state": examples[0].context,
                    "questions": {
                        f"q{i}": request_for(e)["questions"]["decision"]
                        for i, e in enumerate(examples)
                    },
                }
                _, questions = parse_request(request)
                rows = []
                for e, q in zip(examples, questions, strict=True):
                    if e.target not in q.labels:
                        raise ValueError("Source target is not in the typed candidate list.")
                    rows.append({
                        "id": e.id, "group_id": group, "context_id": e.context_id,
                        "family": e.family, "kind": e.kind, "split": split,
                        "target": e.target, "candidate_ids": list(q.labels),
                        "context_weight": 1 / len(examples), "tie_break": "request_order",
                    })
                groups.append({
                    "split": split, "task": task, "group_id": group,
                    "observation_id": observation, "request": request, "rows": rows,
                })
                seen_components.add(group)
    random.Random(seed).shuffle(groups)
    plan = {
        "schema": PLAN_SCHEMA, "seed": seed,
        "dataset_manifest_sha256": file_hash(manifest_path),
        "source_files": {split: manifest["files"][f"{split}.jsonl"]["sha256"]
                         for split in ("train", "dev_train")},
        "generator_sha256": file_hash(Path(__file__)),
        "scope": "Native Korean NLI/topic head adaptation; not a fresh final evaluation.",
        "annotation_origin": "upstream_human", "teacher_labels_used": False,
        "calibration_and_public_validation_included": False,
        "group_count": len(groups), "question_count": sum(len(g["rows"]) for g in groups),
        "groups": groups,
    }
    plan["content_sha256"] = json_hash(plan)
    atomic_json(out, plan)
    return plan


def validate_plan(plan: dict):
    if (plan.get("schema") not in (PLAN_SCHEMA, MIXED_PLAN_SCHEMA)
            or plan.get("content_sha256") != json_hash({
                k: v for k, v in plan.items() if k != "content_sha256"
            }) or not plan.get("groups")
            or plan["group_count"] != len(plan["groups"])
            or plan["question_count"] != sum(len(g["rows"]) for g in plan["groups"])):
        raise ValueError("Use an unchanged, nonempty frozen feature plan.")
    components, identities = set(), set()
    for group in plan["groups"]:
        if group["group_id"] in components or group["split"] not in ("train", "dev_train"):
            raise ValueError("Feature components overlap or contain another partition.")
        components.add(group["group_id"])
        _, questions = parse_request(group["request"])
        if (len(questions) != len(group["rows"]) or not math.isclose(
                sum(row["context_weight"] for row in group["rows"]), 1.0)):
            raise ValueError("Question rows or context weights do not match the request.")
        for row, q in zip(group["rows"], questions, strict=True):
            if (row["id"] in identities or row["split"] != group["split"]
                    or row["group_id"] != group["group_id"]
                    or row["candidate_ids"] != list(q.labels)
                    or row["kind"] != {"choice": "choice", "noul": "boolean",
                                       "score": "ordinal"}[q.kind]
                    or not 0 < row["context_weight"] <= 1):
                raise ValueError("Feature row metadata is misaligned or duplicated.")
            try:
                target_values(row)
                if (plan["schema"] == PLAN_SCHEMA
                        and row.get("supervision", "hard_label") != "hard_label"):
                    raise ValueError("Ordinal means require the mixed-supervision plan.")
            except ValueError as error:
                raise ValueError("Feature row supervision is misaligned.") from error
            identities.add(row["id"])


def extract(plan: dict, scorer: VerifiedFeatureScorer, out: Path, *,
            contexts_per_batch: int = 4, max_seconds: float = 1800, remote=None) -> dict:
    validate_plan(plan)
    if (out.exists() or not 1 <= contexts_per_batch <= 32
            or not 120 <= max_seconds <= 7200):
        raise ValueError("Use a fresh output directory and a bounded extraction run.")
    out.mkdir(parents=True)
    started = time.monotonic()
    manifest = {
        "schema": RUN_SCHEMA, "started_at": datetime.now(UTC).isoformat(),
        "status": "extracting", "plan_sha256": plan["content_sha256"],
        "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
        "scorer_provenance": scorer.provenance, "extractor_sha256": file_hash(Path(__file__)),
        "max_seconds": max_seconds, "contexts_per_batch": contexts_per_batch,
        "planned_groups": plan["group_count"], "planned_questions": plan["question_count"],
        "completed_groups": 0, "completed_questions": 0, "files": [],
        "training_performed": False, "model_quality_measured": False,
        "calibration_and_public_validation_included": False,
    }
    atomic_json(out / "run.json", manifest)
    active_error = None
    try:
        for offset in range(0, len(plan["groups"]), contexts_per_batch):
            if time.monotonic() - started > max_seconds - 90:
                manifest["status"] = "deadline"
                break
            groups = plan["groups"][offset:offset + contexts_per_batch]
            # Gold labels and dataset IDs are not passed to the native scorer.
            requests = [parse_request(g["request"]) for g in groups]
            scores, hidden, tokens = scorer.extract_many(requests)
            rows = [row for group in groups for row in group["rows"]]
            if (len(scores) != len(rows) or hidden.ndim != 2 or len(hidden) != len(rows)
                    or not torch.isfinite(hidden).all()):
                raise ValueError("Feature batch lost or corrupted question representations.")
            width = max(len(r["candidate_ids"]) for r in rows)
            base = torch.full((len(rows), width), float("-inf"))
            keep = torch.zeros(len(rows), width, dtype=torch.bool)
            for i, (row, values) in enumerate(zip(rows, scores, strict=True)):
                count = len(row["candidate_ids"])
                if len(values) != count or any(not math.isfinite(v) for v in values):
                    raise ValueError("Native feature scores omitted a candidate.")
                base[i, :count] = torch.tensor(values)
                keep[i, :count] = True
            payload = {
                "schema": "bobcat-glm-frozen-feature-shard-v1",
                "plan_sha256": plan["content_sha256"],
                "hidden": hidden.detach().cpu().float(), "base_logits": base,
                "candidate_keep": keep,
                "targets": target_tensors(rows)[0],
                "context_weights": torch.tensor([r["context_weight"] for r in rows],
                                                dtype=torch.float64),
                "rows": rows, "logical_input_tokens": tokens,
                "native_measurement": scorer.last_measurement,
            }
            if plan["schema"] == MIXED_PLAN_SCHEMA:
                payload["score_targets"] = target_tensors(rows)[1]
            path = out / f"features-{len(manifest['files']):05d}.pt"
            temporary = path.with_suffix(".partial")
            torch.save(payload, temporary)
            temporary.replace(path)
            record = {
                "path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size,
                "questions": len(rows), "groups": len(groups),
            }
            if remote:
                key = f"{remote.prefix}/objects/{record['sha256']}/{path.name}"
                remote.client.upload_file(str(path), remote.bucket, key, ExtraArgs={
                    "ChecksumAlgorithm": "SHA256", "Metadata": {"sha256": record["sha256"]},
                })
                head = remote.client.head_object(
                    Bucket=remote.bucket, Key=key, ChecksumMode="ENABLED",
                )
                if (head["ContentLength"] != record["bytes"]
                        or head.get("Metadata", {}).get("sha256") != record["sha256"]):
                    raise ValueError("Remote feature shard identity failed.")
                record.update(key=key, version_id=head.get("VersionId"))
            manifest["files"].append(record)
            manifest["completed_groups"] += len(groups)
            manifest["completed_questions"] += len(rows)
            atomic_json(out / "run.json", manifest)
        else:
            manifest["status"] = "completed"
    except BaseException as error:
        active_error = error
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(out / "run.json", manifest)
        if remote:
            digest = file_hash(out / "run.json")
            try:
                remote.client.put_object(
                    Bucket=remote.bucket, Key=f"{remote.prefix}/runs/{digest}.json",
                    Body=(out / "run.json").read_bytes(), Metadata={"sha256": digest},
                    ChecksumAlgorithm="SHA256", IfNoneMatch="*",
                )
            except Exception as error:
                manifest["remote_manifest_error"] = str(error)[:1500]
                atomic_json(out / "run.json", manifest)
                if active_error is None:
                    raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=2048)
    parser.add_argument("--development-per-task", type=int, default=128)
    args = parser.parse_args()
    plan = freeze(
        args.data, args.out, train_per_task=args.train_per_task,
        development_per_task=args.development_per_task,
    )
    print(json.dumps({k: v for k, v in plan.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
