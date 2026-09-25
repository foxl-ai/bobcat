"""Pinned human-labelled Korean judgment data, with conservative grouped splits.

The public validation data is development data, never a fresh final test.
Labels, IDs, URLs and source metadata are excluded from model inputs. Three
views of one NLI pair share a group and are not three independent observations.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bobcat.corpus import atomic_json, normalized_text
from bobcat.protocol import parse_request
from bobcat.schema import Choice, Example, file_hash, json_hash, write_examples

NLI_LABELS = ["entailment", "neutral", "contradiction"]
TOPICS = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
NLI_CHOICES = [
    Choice("함의", "함의: 전제를 참이라고 가정하면 가설이 반드시 성립합니다."),
    Choice("중립", "중립: 전제만으로는 가설의 참과 거짓을 결정할 수 없습니다."),
    Choice("모순", "모순: 전제를 참이라고 가정하면 가설이 성립할 수 없습니다."),
]
SPLITS = ("train", "dev_train", "cal_temperature", "cal_policy", "dev_public")


def validate_sources(source: dict) -> None:
    if (source.get("schema") != "bobcat-klue-sources-v1" or source.get("repo") != "klue/klue"
            or source.get("language") != "ko" or source.get("license") != "CC-BY-SA-4.0"
            or not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", ""))):
        raise ValueError("Use pinned, attributed KLUE source metadata.")
    if source.get("label_names") != {"nli": NLI_LABELS, "ynat": TOPICS}:
        raise ValueError("The adapter's label mapping does not match the source.")
    seen = set()
    for item in source["files"]:
        pair = (item["task"], item["split"])
        expected = f"{item['task']}/{item['split']}-00000-of-00001.parquet"
        if (pair in seen or pair not in {
                (task, split) for task in ("nli", "ynat") for split in ("train", "validation")
            } or item["path"] != expected
                or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
                or type(item.get("bytes")) is not int or not 0 < item["bytes"] < 20_000_000
                or type(item.get("rows")) is not int or not 0 < item["rows"] < 100_000):
            raise ValueError("Invalid task/split/file; upstream test is not accepted.")
        seen.add(pair)
    if len(seen) != 4:
        raise ValueError("Both original splits for both tasks are required.")


def download(sources_path: Path, out: Path) -> dict:
    source = json.loads(sources_path.read_text())
    validate_sources(source)
    if out.exists():
        raise ValueError("Use a new source directory; preserve previous download evidence.")
    out.mkdir(parents=True)

    def one(item):
        target = out / item["path"]
        target.parent.mkdir(exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{source['repo']}/resolve/"
            f"{source['revision']}/{item['path']}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Bobcat-data-builder"})
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("xb") as stream:
                remaining = item["bytes"] + 1
                while remaining:
                    chunk = response.read(min(1024**2, remaining))
                    if not chunk:
                        break
                    stream.write(chunk)
                    remaining -= len(chunk)
        if partial.stat().st_size != item["bytes"] or file_hash(partial) != item["sha256"]:
            raise ValueError(f"Source checksum failed: {item['path']}")
        partial.replace(target)
        return dict(item)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, item) for item in source["files"]]
        files, errors = [], []
        for future in futures:
            try:
                files.append(future.result())
            except Exception as error:
                errors.append(str(error))
    if errors:
        atomic_json(out / "failed.json", {"errors": errors, "complete": False})
        raise RuntimeError("; ".join(errors))
    receipt = {
        "schema": "bobcat-klue-download-v1", "source_sha256": file_hash(sources_path),
        "revision": source["revision"], "files": files,
    }
    atomic_json(out / "downloads.json", receipt)
    return receipt


def text_key(text: str) -> str:
    return json_hash(" ".join(normalized_text(text).split()))


def row_content(task: str, row: dict) -> tuple[str, list[str]]:
    names = ("premise", "hypothesis") if task == "nli" else ("title",)
    texts = [normalized_text(row[name]) for name in names]
    if any(not text for text in texts):
        raise ValueError("Empty source input.")
    fields = ("전제", "가설") if task == "nli" else ("뉴스 제목",)
    return json.dumps(dict(zip(fields, texts, strict=True)), ensure_ascii=False), texts


def group_partition(group: str) -> str:
    value = int(json_hash(group)[:16], 16) % 100
    if value < 2:
        return "cal_temperature"
    if value < 4:
        return "cal_policy"
    if value < 6:
        return "dev_train"
    return "train"


def examples_for(row: dict, group: str, split: str, source: dict) -> list[Example]:
    task, raw = row["task"], row["raw"]
    context, _ = row_content(task, raw)
    label = raw["label"]
    if type(label) is not int or not 0 <= label < (3 if task == "nli" else 7):
        raise ValueError("Invalid upstream hard label.")
    metadata = {
        "language": "ko", "origin": "native_korean", "source": "klue/klue",
        "source_revision": source["revision"], "license": source["license"],
        "source_task": task, "source_split": row["upstream_split"],
        "source_id": raw["guid"], "source_label": label,
        "annotation_origin": "upstream_human", "teacher": None,
        "independent_observation": row["identity"],
    }
    if task == "ynat":
        views = [(
            "topic", "choice", "뉴스 제목의 주제를 가장 잘 나타내는 분야를 고르세요.",
            [Choice(name, name) for name in TOPICS], TOPICS[label],
        )]
    else:
        views = [
            ("relation", "choice",
             "전제를 참이라고 가정하고, 전제와 가설의 관계를 판단하세요. "
             "외부 지식으로 전제를 교정하지 마세요.",
             NLI_CHOICES, NLI_CHOICES[label].id),
            ("entails", "boolean",
             "전제를 참이라고 가정하면, 가설이 반드시 참이라고 판단할 수 있나요?",
             [Choice("no", "아니요: 가설이 모순되거나 참인지 결정할 근거가 부족합니다."),
              Choice("yes", "예: 전제가 가설의 참을 뒷받침합니다.")],
             "yes" if label == 0 else "no"),
            ("contradicts", "boolean",
             "전제를 참이라고 가정하면, 가설이 반드시 거짓이라고 판단할 수 있나요?",
             [Choice("no", "아니요: 가설이 성립하거나 참인지 결정할 근거가 부족합니다."),
              Choice("yes", "예: 가설이 전제와 모순됩니다.")],
             "yes" if label == 2 else "no"),
        ]
    result = []
    for view, kind, instruction, choices, target in views:
        example = Example(
            id=f"{row['identity']}:{view}", group_id=group, family=f"klue_{task}_{view}",
            split=split, context=context, instruction=instruction, choices=list(choices),
            target=target, kind=kind, metadata=dict(metadata, view=view),
        )
        example.validate()
        result.append(example)
    return result


def request_for(example: Example) -> dict:
    """The only source-to-model boundary; oracle metadata never crosses it."""
    if example.kind == "boolean":
        descriptions = {c.id: c.text for c in example.choices}
        question = {
            "type": "noul", "instructions": example.instruction,
            "criteria": {"true": descriptions["yes"], "false": descriptions["no"]},
        }
    elif example.kind == "choice":
        question = {
            "type": "choice", "instructions": example.instruction,
            "criteria": {c.id: c.text for c in example.choices},
        }
    else:
        raise ValueError("This hard-label adapter does not invent ordinal distributions.")
    request = {
        "model": "bobcat-latest", "state": example.context,
        "questions": {"decision": question},
    }
    parse_request(request)
    return request


def convert(rows: list[dict], source: dict) -> tuple[list[Example], dict, list[dict]]:
    """Connected components preserve shared sentences, URLs and task views."""
    parent = list(range(len(rows)))
    keys_by_row, first, input_labels = [], {}, defaultdict(set)
    identities = set()

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(rows):
        task, raw, split = row["task"], row["raw"], row["upstream_split"]
        if task not in ("nli", "ynat") or split not in ("train", "validation"):
            raise ValueError("Only original train and validation rows are supported.")
        identity = f"klue:{task}:{split}:{raw['guid']}"
        if identity in identities or not raw["guid"]:
            raise ValueError("Source identities must be unique and nonempty.")
        identities.add(identity)
        row["identity"] = identity
        context, texts = row_content(task, raw)
        keys = {f"text:{text_key(text)}" for text in texts}
        if task == "ynat" and raw.get("url"):
            keys.add("url:" + json_hash(raw["url"].split("#")[0].strip()))
        keys_by_row.append(keys)
        row["input_key"] = f"{task}:{json_hash(context)}"
        input_labels[row["input_key"]].add(raw["label"])
        for key in sorted(keys):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    group_for, heldout = {}, set()
    for indices in components.values():
        group = "klue:" + json_hash(sorted(set().union(*(keys_by_row[i] for i in indices))))
        for i in indices:
            group_for[i] = group
        if any(rows[i]["upstream_split"] == "validation" for i in indices):
            heldout.add(group)
    examples, removed, seen_train, exclusions = [], [], set(), {}
    for i, row in enumerate(rows):
        group = group_for[i]
        reason = None
        if len(input_labels[row["input_key"]]) > 1:
            reason = "conflicting_annotations"
        elif row["upstream_split"] == "train" and group in heldout:
            reason = "shares_component_with_public_validation"
        elif row["upstream_split"] == "train" and row["input_key"] in seen_train:
            reason = "duplicate_training_input"
        if reason is not None:
            removed.append({"id": row["identity"], "group_id": group, "reason": reason})
            continue
        if row["upstream_split"] == "validation":
            split = "dev_public"
        else:
            seen_train.add(row["input_key"])
            split = group_partition(group)
        examples.extend(examples_for(row, group, split, source))
        if split != "train":
            for text in row_content(row["task"], row["raw"])[1]:
                exclusions[text_key(text)] = {"text": text}
    partitions = defaultdict(set)
    for example in examples:
        partitions[example.group_id].add(example.split)
    if any(len(values) != 1 for values in partitions.values()):
        raise AssertionError("A context component leaked between partitions.")
    audit = {
        "source_rows": len(rows), "source_components": len(components),
        "largest_source_component": max(map(len, components.values()), default=0),
        "kept_source_rows": len(rows) - len(removed), "removed_rows": removed,
        "removal_counts": dict(Counter(row["reason"] for row in removed)),
        "conflicting_input_count": sum(len(labels) > 1 for labels in input_labels.values()),
        "split_stats": {
            split: {
                "questions": sum(e.split == split for e in examples),
                "source_rows": len({
                    e.metadata["independent_observation"] for e in examples if e.split == split
                }),
                "groups": len({e.group_id for e in examples if e.split == split}),
                "families": dict(Counter(e.family for e in examples if e.split == split)),
            } for split in SPLITS
        },
        "group_overlap_count": 0,
        "split_rule": (
            "All shared NFC/whitespace-normalized input sentences and article URLs form "
            "transitive components. Public validation is dev_public; overlapping training "
            "rows are removed. Remaining component hash modulo 100: [0,2) temperature, "
            "[2,4) policy, [4,6) train-development, [6,100) training."
        ),
        "scope": (
            "Exact normalized inputs/URLs only; paraphrases, near duplicates and historical "
            "language-model pretraining exposure are not excluded. Public validation is "
            "not a fresh final set. Derived NLI views are not independent gold observations."
        ),
    }
    return examples, audit, [exclusions[key] for key in sorted(exclusions)]


def build(sources_path: Path, downloads: Path, out: Path) -> dict:
    import pyarrow.parquet as pq

    source = json.loads(sources_path.read_text())
    validate_sources(source)
    if out.exists():
        raise ValueError("Use a new dataset directory; preserve previous frozen datasets.")
    receipt = json.loads((downloads / "downloads.json").read_text())
    if (receipt.get("schema") != "bobcat-klue-download-v1"
            or receipt["source_sha256"] != file_hash(sources_path)
            or receipt.get("files") != source["files"]):
        raise ValueError("Source receipt does not match the frozen configuration.")
    rows = []
    for item in source["files"]:
        path = downloads / item["path"]
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("Source changed after download.")
        table = pq.read_table(path)
        if table.num_rows != item["rows"]:
            raise ValueError("Upstream row count changed.")
        hf = json.loads(table.schema.metadata[b"huggingface"])
        names = hf["info"]["features"]["label"]["names"]
        if names != source["label_names"][item["task"]]:
            raise ValueError("Parquet label metadata differs from the adapter.")
        rows.extend({
            "task": item["task"], "upstream_split": item["split"], "raw": row,
        } for row in table.to_pylist())
    examples, audit, exclusions = convert(rows, source)
    out.mkdir(parents=True)
    for split in SPLITS:
        write_examples(out / f"{split}.jsonl", [e for e in examples if e.split == split])
    with (out / "evaluation-input-exclusions.jsonl").open("x") as stream:
        for item in exclusions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    atomic_json(out / "split-audit.json", audit)
    report = {
        "schema": "bobcat-korean-decisions-v1", "source": source,
        "source_config_sha256": file_hash(sources_path),
        "adapter_sha256": file_hash(Path(__file__)),
        "download_receipt_sha256": file_hash(downloads / "downloads.json"),
        "files": {
            path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in sorted(out.glob("*.json*"))
        },
        "split_stats": audit["split_stats"], "removed": audit["removal_counts"],
        "input_boundary": "Only context, instruction and offered candidate semantics.",
        "training_candidate_policy": "Exactly requested candidates; disable legacy sentinels.",
        "changes": "NFC text, typed Choice/Noul questions, grouping and split quarantine.",
        "dataset_license": "CC-BY-SA-4.0",
        "evaluation_exclusions_applied_to_existing_language_corpus": False,
        "trained_model_or_quality_result": False,
        "final_evaluation": False,
    }
    atomic_json(out / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fetch", "build"])
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--downloads", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        result = download(args.sources, args.out)
    else:
        if args.downloads is None:
            parser.error("build requires --downloads")
        result = build(args.sources, args.downloads, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
