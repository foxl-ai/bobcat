"""Public Korean/English development, keeping ordinal means separate from labels."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bobcat.corpus import atomic_json
from bobcat.protocol import parse_request, probabilities, response
from bobcat.public_decisions import KINDS, load_partition
from bobcat.schema import file_hash, json_hash
from bobcat.supervision import MEAN_TARGET, evaluate_supervised, target_values

SCHEMA = "bobcat-public-development-suite-v1"
RUN_SCHEMA = "bobcat-public-development-run-v1"


def freeze(data: Path, out: Path, per_task: int = 128, seed: int = 20260922) -> dict:
    if out.exists() or not 1 <= per_task <= 256:
        raise ValueError("Use a fresh public suite with bounded component counts.")
    records, manifest = load_partition(data, "dev_public")
    tasks = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in records:
        if row["source_split"] not in ("validation", "test"):
            raise ValueError("Public development must preserve held-out upstream partitions.")
        tasks[row["task"]][row["group_id"]][row["observation_id"]].append(row)
    groups, used = [], set()
    for task, components in sorted(tasks.items()):
        ordered = sorted(components, key=lambda group: json_hash([seed, group]))
        chosen = [group for group in ordered if group not in used][:per_task]
        if len(chosen) != per_task:
            raise ValueError(f"Not enough unused public components for {task}.")
        for group_id in chosen:
            observations = components[group_id]
            observation = min(observations, key=lambda key: json_hash([seed, key]))
            selected = sorted(observations[observation], key=lambda row: row["id"])
            if len({json_hash(r["request"]["state"]) for r in selected}) != 1:
                raise ValueError("One source observation has differing states.")
            questions, rows = {}, []
            for i, row in enumerate(selected):
                questions[f"q{i}"] = copy.deepcopy(next(iter(row["request"]["questions"].values())))
                rows.append({key: row[key] for key in (
                    "id", "group_id", "task", "language", "family", "kind", "split",
                    "source_split", "target", "score_target", "supervision", "candidate_ids",
                )} | {"tie_break": "request_order"})
            groups.append({
                "group_id": group_id, "observation_id": observation, "task": task,
                "request": {"model": "bobcat-latest", "state": selected[0]["request"]["state"],
                            "questions": questions},
                "rows": rows,
            })
            used.add(group_id)
    random.Random(seed).shuffle(groups)
    suite = {
        "schema": SCHEMA, "seed": seed, "per_task_components": per_task,
        "source_manifest_sha256": file_hash(data / "manifest.json"),
        "source_data_sha256": manifest["files"]["dev_public.jsonl"]["sha256"],
        "sampler_sha256": file_hash(Path(__file__)),
        "sampling": "Input component hash, then one observation per component; no label selection.",
        "scope": "Public development; not fresh final or unseen pretrained-model data.",
        "upstream_test_is_development_here": True, "training_use": False,
        "group_count": len(groups), "question_count": sum(len(g["rows"]) for g in groups),
        "groups": groups,
    }
    suite["content_sha256"] = json_hash(suite)
    validate(suite)
    atomic_json(out, suite)
    return suite


def validate(suite: dict) -> None:
    if (suite.get("schema") != SCHEMA or suite.get("training_use") is not False
            or suite.get("content_sha256") != json_hash({
                k: v for k, v in suite.items() if k != "content_sha256"
            }) or not suite.get("groups")
            or suite["group_count"] != len(suite["groups"])
            or suite["question_count"] != sum(len(g["rows"]) for g in suite["groups"])):
        raise ValueError("Use an unchanged nonempty frozen public development suite.")
    groups, identities = set(), set()
    for group in suite["groups"]:
        if group["group_id"] in groups:
            raise ValueError("A source component appears more than once.")
        groups.add(group["group_id"])
        _, questions = parse_request(group["request"])
        if len(questions) != len(group["rows"]):
            raise ValueError("Development questions and annotations are misaligned.")
        for row, question in zip(group["rows"], questions, strict=True):
            if (row["split"] != "dev_public" or row["source_split"] not in ("validation", "test")
                    or row["id"] in identities or row["group_id"] != group["group_id"]
                    or row["task"] != group["task"] or row["language"] not in ("ko", "en")
                    or row["candidate_ids"] != list(question.labels)
                    or row["kind"] != KINDS[question.kind]):
                raise ValueError("Development source, candidate or target alignment changed.")
            target_values(row)
            identities.add(row["id"])


def run(suite: dict, scorer, out: Path, *, max_seconds: float = 900) -> dict:
    validate(suite)
    if out.exists() or not 0 < max_seconds <= 3600:
        raise ValueError("Use a fresh output directory and finite development runtime.")
    out.mkdir(parents=True)
    started = time.monotonic()
    temperatures = dict(scorer.temperatures)
    manifest = {
        "schema": RUN_SCHEMA, "status": "running", "started_at": datetime.now(UTC).isoformat(),
        "suite_sha256": suite["content_sha256"], "evaluator_sha256": file_hash(Path(__file__)),
        "supervision_source_sha256": file_hash(Path(__file__).with_name("supervision.py")),
        "model": scorer.model_name, "temperatures": temperatures, "max_seconds": max_seconds,
        "scorer_provenance": getattr(scorer, "provenance", {}),
        "scorer_limits": getattr(scorer, "limits", {}),
        "planned_groups": suite["group_count"], "planned_questions": suite["question_count"],
        "final_evaluation": False, "release_gate_passed": False,
    }
    atomic_json(out / "run.json", manifest)
    rows, attempted_rows, failed = [], [], []
    attempted_groups = 0
    try:
        with (out / "predictions.jsonl").open("x") as stream:
            for group in suite["groups"]:
                if time.monotonic() - started >= max_seconds:
                    break
                state, questions = parse_request(group["request"])
                attempted_groups += 1
                attempted_rows.extend(group["rows"])
                group_started = time.monotonic()
                try:
                    # Only parsed state/questions enter the scorer, never the annotation rows.
                    scores, tokens = scorer.score(state, questions)
                    if (dict(scorer.temperatures) != temperatures or len(scores) != len(questions)
                            or any(len(v) != len(q.labels)
                                   or any(not math.isfinite(x) for x in v)
                                   for v, q in zip(scores, questions, strict=True))):
                        raise ValueError("Scorer settings changed or candidate scores are missing.")
                    typed = response(scorer.model_name, questions, scores, temperatures, tokens)
                    result_rows = [
                        {**row, "logits": list(map(float, logits)),
                         "temperature": temperatures.get(q.kind, 1.0)}
                        for row, q, logits in zip(group["rows"], questions, scores, strict=True)
                    ]
                    rows.extend(result_rows)
                    result = {
                        "status": "scored", "group_id": group["group_id"],
                        "observation_id": group["observation_id"], "rows": result_rows,
                        "response": typed,
                        "native_measurement": getattr(scorer, "last_measurement", None),
                    }
                except Exception as error:
                    result = {
                        "status": "failed", "group_id": group["group_id"],
                        "observation_id": group["observation_id"], "questions": len(group["rows"]),
                        "error_type": type(error).__name__, "error": str(error)[:1000],
                    }
                    failed.append(result)
                result["wall_seconds"] = time.monotonic() - group_started
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        manifest["status"] = (
            "deadline" if len(attempted_rows) != suite["question_count"] else
            "completed_with_failures" if failed else "completed"
        )
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        scaled = [
            {**r, "logits": [v / r["temperature"] for v in r["logits"]]} for r in rows
        ]
        hard_attempted = sum(target_values(r)[0] != MEAN_TARGET for r in attempted_rows)
        hard_valid = [r for r in scaled if target_values(r)[0] != MEAN_TARGET]
        correct = sum(r["candidate_ids"][max(
            range(len(r["logits"])), key=r["logits"].__getitem__,
        )] == r["target"] for r in hard_valid)
        mean_attempted = len(attempted_rows) - hard_attempted
        summary = {
            "scope": suite["scope"], "valid_response_metrics": evaluate_supervised(scaled),
            "hard_accuracy_with_failed_questions_in_denominator": (
                correct / hard_attempted if hard_attempted else None
            ),
            "hard_attempted": hard_attempted, "hard_failed": hard_attempted - len(hard_valid),
            "ordinal_mean_attempted": mean_attempted,
            "ordinal_mean_failed": mean_attempted - (len(rows) - len(hard_valid)),
            "ordinal_mean_error_is_successful_responses_only": True,
            "ordinal_mean_failure_penalty_imputed": False,
            "valid_counts_by_task": dict(Counter(r["task"] for r in rows)),
            "valid_metrics_by_task": {
                task: evaluate_supervised([r for r in scaled if r["task"] == task])
                for task in sorted({r["task"] for r in scaled})
            },
            "valid_metrics_by_language": {
                language: evaluate_supervised([r for r in scaled if r["language"] == language])
                for language in sorted({r["language"] for r in scaled})
            },
            "calibration_refitted": False, "release_gate_passed": False,
        }
        manifest.update(
            attempted_groups=attempted_groups, completed_groups=attempted_groups - len(failed),
            attempted_questions=len(attempted_rows), scored_questions=len(rows),
            failed_questions=sum(r["questions"] for r in failed),
            not_attempted_questions=suite["question_count"] - len(attempted_rows),
            wall_seconds=time.monotonic() - started,
            predictions_sha256=file_hash(out / "predictions.jsonl"),
        )
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "run.json", manifest)
    return manifest


def compare_runs(suite: dict, before: Path, after: Path, out: Path) -> dict:
    """Compare the same source components; never turn a failed Score into a number."""
    validate(suite)
    if out.exists():
        raise ValueError("Preserve prior comparisons.")
    expected = {group["group_id"]: group for group in suite["groups"]}
    runs, values = {}, {}
    for name, root in (("before", before), ("after", after)):
        manifest = json.loads((root / "run.json").read_text())
        path = root / "predictions.jsonl"
        if (manifest.get("schema") != RUN_SCHEMA
                or manifest.get("status") not in ("completed", "completed_with_failures")
                or manifest.get("suite_sha256") != suite["content_sha256"]
                or manifest.get("attempted_questions") != suite["question_count"]
                or manifest.get("attempted_groups") != suite["group_count"]
                or file_hash(path) != manifest.get("predictions_sha256")):
            raise ValueError("Paired comparison needs complete, unchanged development attempts.")
        observed, failed = {}, 0
        for line in path.read_text().splitlines():
            prediction = json.loads(line)
            group_id = prediction["group_id"]
            if group_id not in expected or group_id in observed:
                raise ValueError("A paired source group was duplicated or replaced.")
            group = expected[group_id]
            _, questions = parse_request(group["request"])
            if prediction.get("observation_id") != group["observation_id"]:
                raise ValueError("A paired observation changed.")
            if prediction["status"] == "failed":
                if prediction.get("questions") != len(group["rows"]):
                    raise ValueError("A failed group's question count changed.")
                failed += len(group["rows"])
                observed[group_id] = [
                    None if target_values(row)[0] == MEAN_TARGET else 0.0 for row in group["rows"]
                ]
                continue
            if prediction["status"] != "scored" or len(prediction["rows"]) != len(group["rows"]):
                raise ValueError("A paired result omitted questions.")
            outcomes = []
            for row, gold, question in zip(
                prediction["rows"], group["rows"], questions, strict=True,
            ):
                temperature = manifest["temperatures"].get(question.kind, 1.0)
                if (any(row.get(key) != value for key, value in gold.items())
                        or row.get("temperature") != temperature
                        or len(row["logits"]) != len(gold["candidate_ids"])):
                    raise ValueError("Paired source annotations or score configuration changed.")
                p = probabilities(row["logits"], temperature)
                target, mean = target_values(gold)
                outcomes.append(
                    abs(sum(i * mass for i, mass in enumerate(p)) - mean)
                    if target == MEAN_TARGET else float(int(np.argmax(p)) == target)
                )
            observed[group_id] = outcomes
        if set(observed) != set(expected) or failed != manifest["failed_questions"]:
            raise ValueError("The paired cohort or failure count changed.")
        runs[name], values[name] = manifest, observed
    if runs["before"]["temperatures"] != runs["after"]["temperatures"]:
        raise ValueError("Keep the same frozen temperatures for head comparisons.")
    for key in ("base_repo", "base_revision", "profile", "feature_profile", "compiler_sha256"):
        if (runs["before"].get("scorer_provenance", {}).get(key)
                != runs["after"].get("scorer_provenance", {}).get(key)):
            raise ValueError("Use the same backbone and input/feature configuration.")
    hard, means, mean_total, by_task = [], [], 0, defaultdict(lambda: {"hard": [], "mean": []})
    for group_id, group in expected.items():
        hard_indices = [i for i, row in enumerate(group["rows"])
                        if target_values(row)[0] != MEAN_TARGET]
        mean_indices = [i for i, row in enumerate(group["rows"])
                        if target_values(row)[0] == MEAN_TARGET]
        if hard_indices:
            pair = [float(np.mean([values[name][group_id][i] for i in hard_indices]))
                    for name in ("before", "after")]
            hard.append(pair)
            by_task[group["task"]]["hard"].append(pair)
        if mean_indices:
            mean_total += 1
            if all(values[name][group_id][i] is not None
                   for name in ("before", "after") for i in mean_indices):
                pair = [float(np.mean([values[name][group_id][i] for i in mean_indices]))
                        for name in ("before", "after")]
                means.append(pair)
                by_task[group["task"]]["mean"].append(pair)

    def paired(pairs):
        if not pairs:
            return {"components": 0, "before": None, "after": None, "difference": None,
                    "bootstrap_95pct_interval": None}
        a = np.asarray(pairs)
        delta = a[:, 1] - a[:, 0]
        rng = np.random.default_rng(20260922)
        draws = rng.integers(0, len(a), size=(2000, len(a)))
        return {"components": len(a), "before": float(a[:, 0].mean()),
                "after": float(a[:, 1].mean()), "difference": float(delta.mean()),
                "bootstrap_95pct_interval": np.quantile(
                    delta[draws].mean(1), [0.025, 0.975],
                ).tolist()}

    result = {
        "schema": "bobcat-public-paired-development-v1", "suite_sha256": suite["content_sha256"],
        "scope": suite["scope"], "comparison_source_sha256": file_hash(Path(__file__)),
        "run_manifest_sha256": {"before": file_hash(before / "run.json"),
                                "after": file_hash(after / "run.json")},
        "hard_context_accuracy": paired(hard), "hard_failures_count_as_incorrect": True,
        "ordinal_mean_mae_on_common_valid_components": paired(means),
        "ordinal_mean_total_components": mean_total,
        "ordinal_mean_components_missing_either_response": mean_total - len(means),
        "ordinal_mean_failure_penalty_imputed": False,
        "by_task": {task: {"hard_accuracy": paired(v["hard"]), "mean_mae": paired(v["mean"])}
                    for task, v in sorted(by_task.items())},
        "failed_questions": {name: run["failed_questions"] for name, run in runs.items()},
        "bootstrap": {"unit": "source component", "samples": 2000, "seed": 20260922},
        "calibration_refitted": False, "release_gate_passed": False,
    }
    atomic_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=128)
    args = parser.parse_args()
    result = freeze(args.data, args.out, args.per_task)
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
