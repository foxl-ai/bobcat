"""Pinned public judgment data with source labels outside the typed model request."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from bobcat.corpus import atomic_json, normalized_text
from bobcat.klue import group_partition, request_for, text_key
from bobcat.protocol import parse_request, render
from bobcat.schema import Choice, Example, file_hash, json_hash
from bobcat.supervision import target_values

SCHEMA = "bobcat-public-decisions-v1"
SPLITS = ("train", "dev_train", "cal_temperature", "cal_policy", "dev_public")
KINDS = {"choice": "choice", "noul": "boolean", "score": "ordinal"}
STS_LEVELS = [
    "0: 두 문장의 의미가 전혀 다름",
    "1: 주제만 유사하고 전달하는 내용은 다름",
    "2: 일부 내용은 같지만 중요한 의미 차이가 있음",
    "3: 상당한 내용을 공유하지만 중요한 정보가 다름",
    "4: 거의 같은 의미이고 사소한 차이가 있음",
    "5: 두 문장이 같은 의미를 전달함",
]


def checked_file(root: Path, item: dict) -> Path:
    path = root / item["path"]
    if (Path(item["path"]).name != item["path"] or path.is_symlink()
            or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]):
        raise ValueError(f"Public source file identity mismatch: {item['path']}")
    return path


def fetch(sources: Path, out: Path) -> dict:
    config = json.loads(sources.read_text())
    if config.get("schema") != "bobcat-public-decision-sources-v1" or out.exists():
        raise ValueError("Use pinned source metadata and a new download directory.")
    out.mkdir(parents=True)
    for item in [*config["files"], *config["support"]]:
        if (Path(item["path"]).name != item["path"] or not item["url"].startswith("https://")
                or not 0 < item["bytes"] <= 20 * 1024**2):
            raise ValueError("Invalid bounded public download.")
        with urlopen(Request(item["url"], headers={"User-Agent": "Bobcat-research/0.1"}),
                     timeout=30) as response:
            body = response.read(item["bytes"] + 1)
        (out / item["path"]).write_bytes(body)
        checked_file(out, item)
    result = {"source_config_sha256": file_hash(sources), "complete": True,
              "files": len(config["files"]) + len(config["support"])}
    atomic_json(out / "downloads.json", result)
    return result


def request_record(task: str, split: str, index: int, raw: dict,
                   source: dict, categories: list[str]) -> dict:
    """Explicit feature allowlists; outcome, source IDs and labels stay outside."""
    scalar = None
    if task == "klue_sts":
        left, right = normalized_text(raw["sentence1"]), normalized_text(raw["sentence2"])
        state = {"문장1": left, "문장2": right}
        question = {
            "type": "score", "instructions": "두 문장의 의미 유사도를 0~5 척도로 평가하라.",
            "criteria": list(STS_LEVELS),
        }
        target, scalar = None, raw["labels"]["real-label"]
        family, language, texts = "semantic_similarity", "ko", [left, right]
        extra = {
            "mean_source_field": "labels.real-label",
            "upstream_rounded_score": raw["labels"]["label"],
            "vote_histogram_available": False,
        }
    elif task == "banking77":
        text = normalized_text(raw["text"])
        if len(categories) != 77 or len(set(categories)) != 77:
            raise ValueError("Preserve all 77 original Banking77 categories.")
        state = {"customer_message": text}
        question = {
            "type": "choice", "instructions": "Select the banking intent of this customer message.",
            "criteria": {name: name.replace("_", " ") for name in categories},
        }
        target, family, language, texts = raw["category"], "intent_routing", "en", [text]
        extra = {}
    elif task == "boolq":
        passage, text = normalized_text(raw["passage"]), normalized_text(raw["question"])
        if type(raw["answer"]) is not bool:
            raise ValueError("BoolQ supervision must be the original boolean annotation.")
        state = {"passage": passage, "question": text}
        question = {
            "type": "noul",
            "instructions": "Using the passage, is the answer to the question yes?",
            "criteria": {"true": "The answer is yes.", "false": "The answer is no."},
        }
        target = "yes" if raw["answer"] else "no"
        family, language, texts, extra = "reading_grounding", "en", [passage, text], {}
    elif task in ("arc_easy", "arc_challenge"):
        text = normalized_text(raw["question"])
        labels, answers = raw["choices"]["label"], raw["choices"]["text"]
        if len(labels) != len(answers) or len(set(labels)) != len(labels):
            raise ValueError("ARC answer labels must align with all original candidates.")
        state = {"question": text}
        question = {
            "type": "choice", "instructions": "Select the best answer to the question.",
            "criteria": dict(zip(labels, answers, strict=True)),
        }
        target = raw["answerKey"]
        family, language, texts, extra = "science_reasoning", "en", [text], {}
    else:
        raise ValueError("Unsupported public task adapter.")
    if split not in ("train", "validation", "test") or any(not text for text in texts):
        raise ValueError("Invalid original split or empty public input.")
    request = {"model": "bobcat-latest", "state": state, "questions": {"decision": question}}
    _, parsed = parse_request(request)
    identity = f"{task}:{split}:{index}"
    result = {
        "id": identity, "group_id": "", "observation_id": identity, "task": task,
        "family": family, "language": language, "kind": KINDS[question["type"]],
        "source_split": split, "split": "", "request": request,
        "candidate_ids": list(parsed[0].labels), "target": target, "score_target": scalar,
        "supervision": "score_mean" if scalar is not None else "hard_label",
        "source": {**source, "row_index": index, "source_id": raw.get("guid", raw.get("id")),
                   **extra},
        "text_keys": sorted({f"text:{text_key(text)}" for text in texts}),
        "input_sha256": json_hash(request), "context_weight": 1.0,
    }
    if task.startswith("arc_"):
        result["text_keys"].append("arc-id:" + json_hash(raw["id"]))
    target_values(result)
    return result


def read_prior(root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "bobcat-korean-decisions-v1":
        raise ValueError("Use the previously frozen Korean decision manifest.")
    records = []
    for split in SPLITS:
        item = manifest["files"][f"{split}.jsonl"]
        path = root / f"{split}.jsonl"
        if path.is_symlink() or file_hash(path) != item["sha256"]:
            raise ValueError("Prior Korean partition checksum changed.")
        with path.open() as stream:
            for line in stream:
                example = Example.from_dict(json.loads(line))
                if example.split != split:
                    raise ValueError("The prior partition changed.")
                request = request_for(example)
                _, questions = parse_request(request)
                texts = json.loads(example.context)
                if not isinstance(texts, dict) or any(not isinstance(t, str)
                                                      for t in texts.values()):
                    raise ValueError("Unknown prior context format.")
                records.append({
                    "id": example.id, "group_id": example.group_id,
                    "observation_id": example.metadata["independent_observation"],
                    "task": "klue_" + example.metadata["source_task"],
                    "family": example.family, "language": "ko", "kind": example.kind,
                    "source_split": example.metadata["source_split"], "split": split,
                    "request": request, "candidate_ids": list(questions[0].labels),
                    "target": example.target, "score_target": None, "supervision": "hard_label",
                    "source": {**example.metadata, "prior_group_id": example.group_id},
                    "text_keys": sorted({f"text:{text_key(t)}" for t in texts.values()}),
                    "input_sha256": json_hash(request),
                    "context_weight": 1 / (3 if example.metadata["source_task"] == "nli" else 1),
                })
    return records, {
        "manifest_sha256": file_hash(root / "manifest.json"),
        "partition_sha256": {s: manifest["files"][f"{s}.jsonl"]["sha256"] for s in SPLITS},
        "source": manifest.get("source"), "original_partitions_preserved": True,
    }


def partition(new: list[dict], prior: list[dict]) -> tuple[list[dict], dict]:
    """New examples may join a frozen partition, never bridge two frozen partitions."""
    anchors, prior_splits = defaultdict(set), {}
    for record in prior:
        group, split = record["group_id"], record["split"]
        if group in prior_splits and prior_splits[group] != split:
            raise ValueError("Prior components already cross partitions.")
        prior_splits[group] = split
        for key in record["text_keys"]:
            anchors[key].add(group)
    parent, first, keys = list(range(len(new))), {}, []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, record in enumerate(new):
        row_keys = set(record["text_keys"])
        row_keys.update("prior:" + group for key in record["text_keys"] for group in anchors[key])
        keys.append(row_keys)
        for key in sorted(row_keys):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    components = defaultdict(list)
    for i in range(len(new)):
        components[find(i)].append(i)
    kept, removed, remap = [], [], {}
    for indices in components.values():
        combined = set().union(*(keys[i] for i in indices))
        old_groups = {key[6:] for key in combined if key.startswith("prior:")}
        frozen_splits = {prior_splits[group] for group in old_groups}
        public = any(new[i]["source_split"] != "train" for i in indices)
        conflict = (len(frozen_splits) > 1
                    or (public and frozen_splits and frozen_splits != {"dev_public"}))
        if conflict:
            removed.extend({"id": new[i]["id"], "reason": "bridges_frozen_partitions"}
                           for i in indices)
            continue
        group = "public:" + json_hash(sorted(combined))
        split = (next(iter(frozen_splits)) if frozen_splits else
                 "dev_public" if public else group_partition(group))
        existing_targets, seen = defaultdict(set), set()
        for i in indices:
            row = new[i]
            existing_targets[row["input_sha256"]].add(json_hash(
                [row["supervision"], row["target"], row["score_target"]],
            ))
        for i in indices:
            row = copy.deepcopy(new[i])
            reason = None
            if len(existing_targets[row["input_sha256"]]) != 1:
                reason = "conflicting_annotations"
            elif row["source_split"] == "train" and split == "dev_public":
                reason = "shares_public_input_component"
            elif row["input_sha256"] in seen:
                reason = "duplicate_input"
            if reason:
                removed.append({"id": row["id"], "reason": reason})
                continue
            seen.add(row["input_sha256"])
            row.update(group_id=group, split=split)
            kept.append(row)
        # Merging same-partition old components keeps all original source rows.
        for old_group in old_groups:
            remap[old_group] = group
    for old in prior:
        row = copy.deepcopy(old)
        row["group_id"] = remap.get(row["group_id"], row["group_id"])
        kept.append(row)
    identities, partitions, texts = set(), defaultdict(set), defaultdict(set)
    for record in kept:
        if record["id"] in identities:
            raise ValueError("Duplicate adapted row identity.")
        identities.add(record["id"])
        partitions[record["group_id"]].add(record["split"])
        for key in record["text_keys"]:
            texts[key].add(record["split"])
    if any(len(v) > 1 for v in partitions.values()) or any(len(v) > 1 for v in texts.values()):
        raise ValueError("An input component crosses a training/calibration/development boundary.")
    return kept, {
        "new_source_rows": len(new), "prior_questions": len(prior), "questions": len(kept),
        "new_components": len(components), "prior_components_joined": len(remap),
        "removed": removed, "removal_counts": dict(Counter(r["reason"] for r in removed)),
        "text_and_component_split_overlap": 0,
        "near_duplicate_or_pretraining_exposure_excluded": False,
    }


def build(sources: Path, downloads: Path, prior_root: Path, out: Path) -> dict:
    import pyarrow.parquet as pq

    config = json.loads(sources.read_text())
    if config.get("schema") != "bobcat-public-decision-sources-v1" or out.exists():
        raise ValueError("Use the pinned public sources and a fresh output directory.")
    for item in config["support"]:
        checked_file(downloads, item)
    categories = json.loads((downloads / config["banking_categories_file"]).read_text())
    new = []
    for item in config["files"]:
        path = checked_file(downloads, item)
        if item["format"] == "parquet":
            rows = pq.read_table(path).to_pylist()
        elif item["format"] == "csv":
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        else:
            raise ValueError("Use source CSV or Parquet without executing dataset loaders.")
        if len(rows) != item["rows"]:
            raise ValueError("Original public row count changed.")
        source = {**config["sources"][item["source"]], "file_sha256": item["sha256"]}
        new.extend(request_record(item["task"], item["split"], i, raw, source, categories)
                   for i, raw in enumerate(rows))
    prior, prior_manifest = read_prior(prior_root)
    records, audit = partition(new, prior)
    out.mkdir(parents=True)
    files, stats = {}, {}
    for split in SPLITS:
        selected = sorted((r for r in records if r["split"] == split), key=lambda r: r["id"])
        path = out / f"{split}.jsonl"
        with path.open("x") as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        stats[split] = {
            "questions": len(selected), "components": len({r["group_id"] for r in selected}),
            "by_task": dict(Counter(r["task"] for r in selected)),
            "by_language": dict(Counter(r["language"] for r in selected)),
            "by_supervision": dict(Counter(r["supervision"] for r in selected)),
        }
    atomic_json(out / "audit.json", audit)
    result = {
        "schema": SCHEMA, "created_at": datetime.now(UTC).isoformat(),
        "source_config_sha256": file_hash(sources), "source": config,
        "prior": prior_manifest, "adapter_sha256": file_hash(Path(__file__)),
        "files": files, "partitions": stats, "audit_sha256": file_hash(out / "audit.json"),
        "licenses": "per_source; preserve the original licenses and attribution",
        "teacher_or_jev_labels_used": False, "mean_score_histograms_invented": False,
        "scope": "Public task training and development, not new independent final evaluation.",
        "training_performed": False, "release_gate_passed": False,
    }
    atomic_json(out / "manifest.json", result)
    return result


def load_partition(data: Path, split: str) -> tuple[list[dict], dict]:
    if split not in SPLITS:
        raise ValueError("Unknown public data partition.")
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Use the checksummed public decision manifest.")
    path = checked_file(data, {"path": f"{split}.jsonl", **manifest["files"][f"{split}.jsonl"]})
    with path.open() as stream:
        records = [json.loads(line) for line in stream]
    for row in records:
        _, questions = parse_request(row["request"])
        if (row["split"] != split or len(questions) != 1
                or row["candidate_ids"] != list(questions[0].labels)
                or row["kind"] != KINDS[questions[0].kind]):
            raise ValueError("A public request lost source/target alignment.")
        target_values(row)
    return records, manifest


def student_example(row: dict) -> Example:
    """Use the same rendered inputs as StudentScorer; preserve means as means."""
    target_values(row)
    state, questions = parse_request(row["request"])
    if len(questions) != 1 or row["candidate_ids"] != list(questions[0].labels):
        raise ValueError("Student source annotation must match the offered candidates.")
    question = questions[0]
    example = Example(
        id=row["id"], group_id=row["group_id"], family=row["family"], split=row["split"],
        context=render(state), instruction=render(question.instructions),
        choices=[Choice(label, description) for label, description in zip(
            question.labels, question.descriptions, strict=True,
        )],
        target=row["target"], kind=row["kind"], supervision=row["supervision"],
        score_target=row["score_target"],
        metadata={"source": row["source"], "language": row["language"], "task": row["task"],
                  "source_split": row["source_split"], "observation_id": row["observation_id"]},
    )
    example.validate()
    return example


def freeze_features(data: Path, out: Path, *, train_per_task: int = 128,
                    korean_train_per_task: int = 256, dev_per_task: int = 16,
                    seed: int = 20260922) -> dict:
    from bobcat.glm_feature_data import MIXED_PLAN_SCHEMA, validate_plan

    if (out.exists() or not 1 <= dev_per_task <= train_per_task <= 8192
            or not dev_per_task <= korean_train_per_task <= 8192):
        raise ValueError("Use a new plan with explicit bounded sample counts.")
    groups, seen, counts = [], set(), {}
    for split in ("train", "dev_train"):
        records, manifest = load_partition(data, split)
        by_task = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        languages = defaultdict(set)
        for row in records:
            if row["source_split"] != "train":
                raise ValueError("Public held-out annotations cannot enter feature fitting.")
            by_task[row["task"]][row["group_id"]][row["observation_id"]].append(row)
            languages[row["task"]].add(row["language"])
        for task, components in sorted(by_task.items()):
            if len(languages[task]) != 1:
                raise ValueError("Mixed-language tasks require an explicit sampling recipe.")
            maximum = (dev_per_task if split != "train" else
                       korean_train_per_task if languages[task] == {"ko"} else train_per_task)
            available = sorted(components, key=lambda key: json_hash([seed, key]))
            chosen = [key for key in available if key not in seen][:maximum]
            if len(chosen) != maximum:
                raise ValueError(f"Not enough independent components: {split}/{task}")
            counts[f"{split}/{task}"] = len(chosen)
            for group_id in chosen:
                observations = components[group_id]
                observation = min(observations, key=lambda key: json_hash([seed, key]))
                selected = sorted(observations[observation], key=lambda row: row["id"])
                states = {json_hash(row["request"]["state"]) for row in selected}
                if len(states) != 1:
                    raise ValueError("A grouped observation has differing states.")
                questions, rows = {}, []
                for index, row in enumerate(selected):
                    question = copy.deepcopy(next(iter(row["request"]["questions"].values())))
                    if question["type"] == "choice" and split == "train":
                        items = list(question["criteria"].items())
                        random.Random(json_hash([seed, row["id"]])).shuffle(items)
                        question["criteria"] = dict(items)
                    request = {"model": "bobcat-latest", "state": selected[0]["request"]["state"],
                               "questions": {"decision": question}}
                    _, parsed = parse_request(request)
                    questions[f"q{index}"] = question
                    rows.append({key: row[key] for key in (
                        "id", "group_id", "family", "kind", "split", "target",
                        "score_target", "supervision", "language", "task",
                    )} | {
                        "candidate_ids": list(parsed[0].labels),
                        "context_id": json_hash(request["state"]),
                        "context_weight": 1 / len(selected), "tie_break": "request_order",
                    })
                groups.append({
                    "split": split, "task": task, "group_id": group_id,
                    "observation_id": observation, "rows": rows,
                    "request": {"model": "bobcat-latest", "state": selected[0]["request"]["state"],
                                "questions": questions},
                })
                seen.add(group_id)
    random.Random(seed).shuffle(groups)
    plan = {
        "schema": MIXED_PLAN_SCHEMA, "seed": seed,
        "dataset_manifest_sha256": file_hash(data / "manifest.json"),
        "source_files": {split: manifest["files"][f"{split}.jsonl"]["sha256"]
                         for split in ("train", "dev_train")},
        "generator_sha256": file_hash(Path(__file__)),
        "scope": "Public Korean/English mixed judgment study; not a final evaluation.",
        "annotation_origin": "upstream_annotations", "teacher_labels_used": False,
        "calibration_and_public_validation_included": False,
        "ordinal_means_are_not_vote_distributions": True, "component_counts": counts,
        "sampling": {"train_contexts_per_korean_task": korean_train_per_task,
                     "train_contexts_per_other_task": train_per_task,
                     "development_contexts_per_task": dev_per_task},
        "group_count": len(groups), "question_count": sum(len(g["rows"]) for g in groups),
        "groups": groups,
    }
    plan["content_sha256"] = json_hash(plan)
    validate_plan(plan)
    atomic_json(out, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "build", "freeze-features"])
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--downloads", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-per-task", type=int, default=128)
    parser.add_argument("--korean-train-per-task", type=int, default=256)
    parser.add_argument("--dev-per-task", type=int, default=16)
    args = parser.parse_args()
    if args.action == "fetch":
        result = fetch(args.sources, args.out)
    elif args.action == "build":
        result = build(args.sources, args.downloads, args.prior, args.out)
    else:
        result = freeze_features(
            args.data, args.out, train_per_task=args.train_per_task,
            korean_train_per_task=args.korean_train_per_task, dev_per_task=args.dev_per_task,
        )
    print(json.dumps({k: v for k, v in result.items() if k not in ("groups", "source")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
