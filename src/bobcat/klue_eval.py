"""Frozen public Korean development evaluation through the typed scorer.

No model, inference engine, or training labels are fetched. The sampler uses
input groups before predictions, and every NLI view stays with its source pair.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bobcat.corpus import atomic_json
from bobcat.klue import request_for
from bobcat.metrics import evaluate_rows
from bobcat.protocol import parse_request, response
from bobcat.schema import Example, file_hash, json_hash, read_examples


def freeze(data: Path, out: Path, per_task: int = 256, seed: int = 20260922) -> dict:
    if out.exists() or not 1 <= per_task <= 1000:
        raise ValueError("Use a new frozen suite with 1–1000 input groups per task.")
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = data / "dev_public.jsonl"
    if (manifest["schema"] != "bobcat-korean-decisions-v1"
            or file_hash(path) != manifest["files"][path.name]["sha256"]):
        raise ValueError("Public development data differs from its manifest.")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for example in read_examples(path):
        metadata = example.metadata
        if (example.split != "dev_public" or metadata["source_split"] != "validation"
                or metadata["source_task"] not in {"nli", "ynat"}
                or metadata["annotation_origin"] != "upstream_human"):
            raise ValueError("Use the attributed original public validation rows.")
        grouped[metadata["source_task"]][example.group_id][
            metadata["independent_observation"]
        ].append(example)
    groups, used = [], set()
    for task in ("nli", "ynat"):
        ordered = sorted(grouped[task], key=lambda group: json_hash([seed, group]))
        selected = [group for group in ordered if group not in used][:per_task]
        if len(selected) != per_task:
            raise ValueError("Insufficient distinct input components for the frozen sample.")
        for group in selected:
            observations = grouped[task][group]
            observation = min(observations, key=lambda key: json_hash([seed, key]))
            examples = sorted(observations[observation], key=lambda e: e.metadata["view"])
            if (len(examples) != (3 if task == "nli" else 1)
                    or len({e.context for e in examples}) != 1):
                raise ValueError("A source observation lost its complete set of views.")
            groups.append({
                "group_id": group, "observation_id": observation, "task": task,
                "examples": [e.to_dict() for e in examples],
            })
            used.add(group)
    random.Random(seed).shuffle(groups)
    result = {
        "schema": "bobcat-klue-development-suite-v1", "seed": seed,
        "source_manifest_sha256": file_hash(manifest_path),
        "source_data_sha256": file_hash(path), "sampler_sha256": file_hash(Path(__file__)),
        "per_task_source_groups": per_task, "source_groups": len(groups),
        "questions": sum(len(g["examples"]) for g in groups),
        "sampling": "Input component hash, then one source observation per component; no labels.",
        "annotation_origin": "upstream_human",
        "scope": "Public task development; not fresh final or unseen pretraining data.",
        "training_use": False, "groups": groups,
    }
    result["content_sha256"] = json_hash(result)
    atomic_json(out, result)
    return result


def run(suite: dict, scorer, out: Path, *, max_seconds: float = 900) -> dict:
    if (suite.get("schema") != "bobcat-klue-development-suite-v1"
            or suite.get("content_sha256") != json_hash({
                key: value for key, value in suite.items() if key != "content_sha256"
            }) or not 0 < max_seconds <= 3600):
        raise ValueError("Use an unchanged frozen suite and a finite runtime.")
    if out.exists():
        raise ValueError("Preserve previous evaluation evidence; use a new directory.")
    out.mkdir(parents=True)
    temperatures = dict(scorer.temperatures)
    manifest = {
        "schema": "bobcat-klue-development-run-v1",
        "started_at": datetime.now(UTC).isoformat(), "status": "running",
        "suite_sha256": suite["content_sha256"], "evaluator_sha256": file_hash(Path(__file__)),
        "model": scorer.model_name, "temperatures": temperatures, "max_seconds": max_seconds,
        "scorer_provenance": getattr(scorer, "provenance", {}),
        "scorer_limits": getattr(scorer, "limits", {}),
        "planned_groups": len(suite["groups"]), "planned_questions": suite["questions"],
        "final_evaluation": False, "release_gate_passed": False,
    }
    atomic_json(out / "run.json", manifest)
    started = time.monotonic()
    rows, failures, attempted, completed_groups = [], [], 0, 0
    with (out / "predictions.jsonl").open("x") as stream:
        for group in suite["groups"]:
            if time.monotonic() - started >= max_seconds:
                break
            examples = [Example.from_dict(e) for e in group["examples"]]
            if (any(e.split != "dev_public" for e in examples)
                    or len({e.context for e in examples}) != 1):
                raise ValueError("Invalid frozen group; do not score another partition.")
            # Build payloads from allowlisted model inputs only. IDs and labels
            # remain outside the request; each opaque question ID is host-owned.
            request = {
                "model": "bobcat-latest", "state": examples[0].context,
                "questions": {
                    f"q{i}": request_for(e)["questions"]["decision"]
                    for i, e in enumerate(examples)
                },
            }
            state, questions = parse_request(request)
            attempted += len(examples)
            group_started = time.monotonic()
            try:
                scores, tokens = scorer.score(state, questions)
                if dict(scorer.temperatures) != temperatures or len(scores) != len(questions):
                    raise ValueError("Evaluation settings changed or a question was omitted.")
                if any(len(values) != len(q.labels) or any(not math.isfinite(x) for x in values)
                       for values, q in zip(scores, questions, strict=True)):
                    raise ValueError("Missing or non-finite candidate scores.")
                typed = response(scorer.model_name, questions, scores, temperatures, tokens)
                native = getattr(scorer, "last_measurement", None)
                result = {
                    "status": "scored", "group_id": group["group_id"],
                    "observation_id": group["observation_id"],
                    "wall_seconds": time.monotonic() - group_started,
                    "native_measurement": native, "response": typed, "rows": [],
                }
                for example, question, logits in zip(examples, questions, scores, strict=True):
                    row = {
                        "id": example.id, "group_id": example.group_id,
                        "family": example.family, "kind": example.kind,
                        "split": example.split, "target": example.target,
                        "candidate_ids": list(question.labels),
                        "logits": [float(x) for x in logits],
                        "temperature": temperatures.get(question.kind, 1.0),
                        "tie_break": "request_order",
                    }
                    result["rows"].append(row)
                rows.extend(result["rows"])
                completed_groups += 1
            except Exception as error:
                result = {
                    "status": "failed", "group_id": group["group_id"],
                    "observation_id": group["observation_id"], "questions": len(examples),
                    "error_type": type(error).__name__, "error": str(error)[:1000],
                    "wall_seconds": time.monotonic() - group_started,
                }
                failures.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    # Metrics consume deployment-scaled scores with a fixed outer T=1. Raw
    # native log probabilities and each fixed temperature remain in predictions.
    scaled = [
        {**row, "logits": [x / row["temperature"] for x in row["logits"]]} for row in rows
    ]
    metrics = evaluate_rows(scaled)
    correct = sum(
        row["candidate_ids"][max(range(len(row["logits"])), key=row["logits"].__getitem__)]
        == row["target"] for row in rows
    )
    summary = {
        "scope": "Public Korean development, not final evaluation.",
        "valid_response_metrics": metrics,
        "accuracy_with_failed_questions_in_denominator": correct / attempted if attempted else None,
        "failed_questions": sum(f["questions"] for f in failures),
        "valid_question_counts_by_family": dict(Counter(r["family"] for r in rows)),
        "independent_completed_input_groups": completed_groups,
        "questions_are_not_independent_input_groups": True,
        "exact_score_ties": sum(r["logits"].count(max(r["logits"])) > 1 for r in rows),
        "tie_break": "first candidate in request order, matching typed response",
        "calibration_was_refitted_on_this_data": False,
    }
    manifest.update(
        status="completed" if attempted == suite["questions"] else "deadline",
        seconds=round(time.monotonic() - started, 3),
        attempted_groups=completed_groups + len(failures), attempted_questions=attempted,
        failed_questions=summary["failed_questions"],
        predictions_sha256=file_hash(out / "predictions.jsonl"),
    )
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "run.json", manifest)
    return manifest


def compare_runs(suite: dict, before: Path, after: Path, out: Path) -> dict:
    """Paired development comparison, retaining failed answers in denominators."""
    if (out.exists() or suite.get("schema") != "bobcat-klue-development-suite-v1"
            or suite.get("content_sha256") != json_hash({
                k: v for k, v in suite.items() if k != "content_sha256"
            })):
        raise ValueError("Use a frozen suite and a new comparison output.")
    expected = {g["group_id"]: g for g in suite["groups"]}
    runs, observations = {}, {}
    for name, root in (("before", before), ("after", after)):
        manifest = json.loads((root / "run.json").read_text())
        path = root / "predictions.jsonl"
        if (manifest.get("schema") != "bobcat-klue-development-run-v1"
                or manifest.get("status") != "completed"
                or manifest.get("suite_sha256") != suite["content_sha256"]
                or manifest.get("attempted_questions") != suite["questions"]
                or file_hash(path) != manifest.get("predictions_sha256")):
            raise ValueError("Paired comparison needs complete, unchanged development attempts.")
        values = {}
        for line in path.read_text().splitlines():
            prediction = json.loads(line)
            group_id = prediction["group_id"]
            if group_id not in expected or group_id in values:
                raise ValueError("Predictions duplicated or replaced a source group.")
            examples = expected[group_id]["examples"]
            if prediction["status"] == "failed":
                if prediction.get("questions") != len(examples):
                    raise ValueError("A failed group's question count changed.")
                values[group_id] = [0.0] * len(examples)
            elif prediction["status"] == "scored":
                rows = prediction["rows"]
                if len(rows) != len(examples):
                    raise ValueError("Scored group lost a question.")
                correct = []
                for row, example in zip(rows, examples, strict=True):
                    if (row["id"] != example["id"] or row["target"] != example["target"]
                            or row["group_id"] != group_id
                            or len(row["logits"]) != len(row["candidate_ids"])
                            or set(row["candidate_ids"]) != {
                                c["id"] for c in example["choices"]
                            } or any(not math.isfinite(v) for v in row["logits"])):
                        raise ValueError("Paired prediction labels or candidate scores changed.")
                    selected = max(range(len(row["logits"])), key=row["logits"].__getitem__)
                    correct.append(float(row["candidate_ids"][selected] == example["target"]))
                values[group_id] = correct
            else:
                raise ValueError("Unknown prediction status.")
        if set(values) != set(expected):
            raise ValueError("The paired comparison omitted a planned source group.")
        runs[name], observations[name] = manifest, values
    if runs["before"]["temperatures"] != runs["after"]["temperatures"]:
        raise ValueError("This study compares fixed identical temperatures.")
    for key in ("base_repo", "base_revision", "profile", "feature_profile", "compiler_sha256"):
        if (runs["before"].get("scorer_provenance", {}).get(key)
                != runs["after"].get("scorer_provenance", {}).get(key)):
            raise ValueError("This head study requires the same base and feature configuration.")
    order = sorted(expected)
    averages = {
        name: np.array([np.mean(observations[name][g]) for g in order])
        for name in ("before", "after")
    }
    delta = averages["after"] - averages["before"]
    rng = np.random.default_rng(20260922)
    draws = rng.integers(0, len(order), size=(2000, len(order)))
    interval = np.quantile(delta[draws].mean(1), [0.025, 0.975]).tolist()
    families = defaultdict(lambda: {"before": [], "after": []})
    for group_id in order:
        for i, example in enumerate(expected[group_id]["examples"]):
            for name in ("before", "after"):
                families[example["family"]][name].append(observations[name][group_id][i])
    result = {
        "schema": "bobcat-klue-paired-development-v1",
        "scope": "Public task development comparison; not fresh final or a release gate.",
        "suite_sha256": suite["content_sha256"],
        "comparison_source_sha256": file_hash(Path(__file__)),
        "run_manifest_sha256": {
            "before": file_hash(before / "run.json"), "after": file_hash(after / "run.json"),
        },
        "independent_source_components": len(order), "questions": suite["questions"],
        "failures_count_as_incorrect": True,
        "failed_questions": {name: run["failed_questions"] for name, run in runs.items()},
        "context_weighted_accuracy": {
            name: float(value.mean()) for name, value in averages.items()
        },
        "paired_context_accuracy_difference": float(delta.mean()),
        "paired_context_bootstrap_95pct_interval": interval,
        "bootstrap": {"unit": "source connected component", "samples": 2000, "seed": 20260922},
        "by_family_accuracy": {
            family: {name: float(np.mean(values)) for name, values in models.items()}
            for family, models in sorted(families.items())
        },
        "calibration_refitted": False, "release_gate_passed": False,
    }
    atomic_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=256)
    args = parser.parse_args()
    suite = freeze(args.data, args.out, args.per_task)
    print(json.dumps({key: value for key, value in suite.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
