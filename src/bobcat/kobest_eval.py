"""Evaluation-only Korean diagnostics from pinned, attributed KoBEST test data."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from bobcat.corpus import atomic_json, normalized_text
from bobcat.protocol import parse_request
from bobcat.public_decisions import KINDS
from bobcat.public_eval import SCHEMA, validate
from bobcat.schema import file_hash, json_hash

TASKS = ("boolq", "copa", "wic", "hellaswag", "sentineg")


def record(task: str, raw: dict, index: int) -> dict:
    """Only source evidence and candidate text enter the model request."""
    if task not in TASKS:
        raise ValueError("Unknown KoBEST task.")

    def text(key):
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid source text: {task}/{key}.")
        return normalized_text(value)

    label = raw["label"]
    if type(label) is not int or label not in range(4 if task == "hellaswag" else 2):
        raise ValueError("Preserve the original zero-based annotation.")
    if task == "boolq":
        state = {"본문": text("paragraph"), "질문": text("question")}
        question = {
            "type": "noul",
            "instructions": "본문을 근거로 주어진 질문의 답이 '예'인지 판단하라.",
            "criteria": {"true": "질문의 답이 예다.", "false": "질문의 답이 아니오다."},
        }
        target = ("no", "yes")[label]
        evidence = [state["본문"]]
        family = "reading_grounding"
    elif task == "copa":
        relation = text("question")
        if relation not in {"원인", "결과"}:
            raise ValueError("Preserve COPA's cause/effect instruction.")
        state = {"상황": text("premise")}
        question = {
            "type": "choice",
            "instructions": f"주어진 상황의 가장 그럴듯한 {relation}에 해당하는 선택지를 고르라.",
            "criteria": {f"선택{i}": text(f"alternative_{i}") for i in (1, 2)},
        }
        target = f"선택{label + 1}"
        evidence = [state["상황"]]
        family = "causal_reasoning"
    elif task == "wic":
        state = {"단어": text("word"), "문장1": text("context_1"), "문장2": text("context_2")}
        question = {
            "type": "noul",
            "instructions": "주어진 단어가 두 문장에서 같은 의미로 사용되었는가?",
            "criteria": {"true": "같은 의미다.", "false": "서로 다른 의미다."},
        }
        target = ("no", "yes")[label]
        evidence = [state["문장1"], state["문장2"]]
        family = "word_sense"
    elif task == "hellaswag":
        state = {"앞선 상황": text("context")}
        question = {
            "type": "choice",
            "instructions": "앞선 상황에서 바로 다음에 일어날 가장 그럴듯한 사건을 고르라.",
            "criteria": {f"선택{i}": text(f"ending_{i}") for i in range(1, 5)},
        }
        target = f"선택{label + 1}"
        evidence = [state["앞선 상황"]]
        family = "commonsense_completion"
    else:
        state = {"문장": text("sentence")}
        question = {
            "type": "choice",
            "instructions": "부정 표현을 포함한 문장 전체의 감정이 긍정인지 부정인지 고르라.",
            "criteria": {"부정": "부정적인 감정이나 평가", "긍정": "긍정적인 감정이나 평가"},
        }
        target = ("부정", "긍정")[label]
        evidence = [state["문장"]]
        family = "sentiment_negation"
    request = {"model": "bobcat-latest", "state": state, "questions": {"q": question}}
    _, parsed = parse_request(request)
    identity = json_hash([task, index, request])
    return {
        "id": "kobest:" + identity, "observation_id": identity,
        "task": "kobest_" + task, "language": "ko", "family": family,
        "kind": KINDS[question["type"]], "split": "dev_public", "source_split": "test",
        "target": target, "score_target": None, "supervision": "hard_label",
        "candidate_ids": list(parsed[0].labels), "request": request,
        "evidence_keys": [json_hash(" ".join(value.split())) for value in evidence],
        "tie_break": "request_order",
    }


def components(rows: list[dict]) -> list[str]:
    """Collapse shared passages/sentences before sampling independent units."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for index, row in enumerate(rows):
        for key in row["evidence_keys"]:
            if key in first:
                left, right = find(index), find(first[key])
                parent[max(left, right)] = min(left, right)
            else:
                first[key] = index
    members = defaultdict(list)
    for index, row in enumerate(rows):
        members[find(index)].append(row["observation_id"])
    identifiers = {
        root: "kobest-component:" + json_hash(sorted(items))
        for root, items in members.items()
    }
    return [identifiers[find(i)] for i in range(len(rows))]


def freeze(config_path: Path, raw_root: Path, out: Path, *,
           per_task: int = 128, seed: int = 20260922) -> dict:
    if out.exists() or not 1 <= per_task <= 256:
        raise ValueError("Use a fresh bounded diagnostic suite.")
    config = json.loads(config_path.read_text())
    if (config.get("schema") != "bobcat-kobest-evaluation-sources-v1"
            or config.get("repo") != "skt/kobest_v1"
            or config.get("training_use") is not False
            or not re.fullmatch("[0-9a-f]{40}", config.get("revision", ""))
            or config.get("license") != "CC-BY-SA-4.0"):
        raise ValueError("Use attributed evaluation-only KoBEST metadata.")
    rows, seen = [], set()
    for item in config["files"]:
        task = item["task"]
        if task not in TASKS or task in seen or item["path"] != f"{task}/test.jsonl":
            raise ValueError("Use exactly the five test files, never training or originated data.")
        path = raw_root / item["path"]
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"Source checksum changed: {item['path']}")
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        if len(entries) != item["rows"]:
            raise ValueError("Source row count changed.")
        rows.extend(record(task, value, i) for i, value in enumerate(entries))
        seen.add(task)
    if seen != set(TASKS):
        raise ValueError("All five Korean tasks are required.")
    grouped = defaultdict(lambda: defaultdict(list))
    for row, group_id in zip(rows, components(rows), strict=True):
        row["group_id"] = group_id
        grouped[row["task"]][group_id].append(row)
    groups, used = [], set()
    for task, candidates in sorted(grouped.items()):
        ordered = sorted(candidates, key=lambda key: json_hash([seed, key]))
        chosen = [key for key in ordered if key not in used][:per_task]
        if len(chosen) != per_task:
            raise ValueError(f"Not enough unused source components: {task}.")
        for group_id in chosen:
            row = min(candidates[group_id], key=lambda r: json_hash([seed, r["observation_id"]]))
            annotation = {key: value for key, value in row.items()
                          if key not in {"request", "evidence_keys", "observation_id"}}
            groups.append({
                "group_id": group_id, "observation_id": row["observation_id"], "task": task,
                "request": row["request"], "rows": [annotation],
            })
            used.add(group_id)
    random.Random(seed).shuffle(groups)
    suite = {
        "schema": SCHEMA, "seed": seed, "per_task_components": per_task,
        "source_manifest_sha256": file_hash(config_path),
        "source_data_sha256": json_hash(config["files"]),
        "sampler_sha256": file_hash(Path(__file__)),
        "sampling": "Shared-evidence components, input-only hash, one observation per component.",
        "scope": (
            "KoBEST Korean external development subset. Excluded from Bobcat decision training; "
            "not a fresh final, full official benchmark, or proof of unseen pretraining data."
        ),
        "upstream_test_is_development_here": True, "training_use": False,
        "group_count": len(groups), "question_count": len(groups),
        "source_rows": len(rows),
        "source_components_by_task": {task: len(value) for task, value in grouped.items()},
        "license": config["license"], "attribution": config["attribution"],
        "revision": config["revision"], "groups": groups,
    }
    suite["content_sha256"] = json_hash(suite)
    validate(suite)
    atomic_json(out, suite)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=128)
    args = parser.parse_args()
    result = freeze(args.sources, args.raw, args.out, per_task=args.per_task)
    print(json.dumps({key: result[key] for key in (
        "group_count", "question_count", "content_sha256", "training_use",
    )}))


if __name__ == "__main__":
    main()
