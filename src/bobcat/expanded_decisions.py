"""Build a larger, attributable judgment corpus without relabelling evaluations.

Raw publisher train/test partitions, shared texts, parallel-language IDs and
prompt groups form connected components before fitting partitions are assigned.
Prior conflicting partitions are quarantined, not silently merged into train.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from array import array
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json, normalized_text
from bobcat.klue import group_partition, text_key
from bobcat.protocol import parse_request
from bobcat.public_decisions import KINDS, SPLITS
from bobcat.schema import file_hash, json_hash
from bobcat.supervision import target_values

SCHEMA = "bobcat-expanded-decisions-v1"
ATTRIBUTES = {
    "helpfulness": ["Not helpful", "Slightly helpful", "Partially helpful",
                    "Mostly helpful", "Perfectly helpful"],
    "correctness": ["Incorrect", "Mostly incorrect", "Partially correct",
                    "Mostly correct", "Correct"],
    "coherence": ["Incoherent", "Mostly incoherent", "Partially coherent",
                  "Mostly coherent", "Coherent"],
    "complexity": ["Basic", "Simple", "Intermediate", "Advanced", "Expert"],
    "verbosity": ["Very little detail", "Little detail", "Moderate detail",
                  "Much detail", "Very much detail"],
}


class Components:
    def __init__(self):
        self.parent = array("q")
        self.size = array("q")
        self.anchor = array("B")
        self.minimum = []
        self.keys = {}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def add(self, keys: list[str], anchor: int = 0) -> int:
        if not keys:
            raise ValueError("Every observation requires an input-based grouping key.")
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        self.anchor.append(anchor)
        self.minimum.append(min(keys))
        for key in keys:
            if key in self.keys:
                left, right = self.find(index), self.find(self.keys[key])
                if left != right:
                    if self.size[left] < self.size[right]:
                        left, right = right, left
                    self.parent[right] = left
                    self.size[left] += self.size[right]
                    self.anchor[left] |= self.anchor[right]
                    self.minimum[left] = min(self.minimum[left], self.minimum[right])
            self.keys[key] = index
        return index

    def assignment(self, index: int, source_split: str) -> tuple[str | None, str, str | None]:
        root = self.find(index)
        bits = self.anchor[root]
        group = "expanded:" + json_hash(self.minimum[root])
        if bits and bits & (bits - 1):
            return None, group, "conflicting_frozen_partitions"
        partition = SPLITS[bits.bit_length() - 1] if bits else group_partition(group)
        if partition == "dev_public" and source_split == "train":
            return None, group, "upstream_train_connected_to_public_holdout"
        return partition, group, None


def record(task: str, upstream: str, index: int, *, state, question,
           language: str, language_origin: str, family: str, source: dict,
           texts: list[str], target=None, mean=None, group_keys=(), view="main",
           votes=None) -> dict:
    request = {"model": "bobcat-latest", "state": state,
               "questions": {"decision": question}}
    _, parsed = parse_request(request)
    observation = f"{task}:{upstream}:{index}"
    row = {
        "id": f"{observation}:{view}", "group_id": "", "observation_id": observation,
        "task": task, "family": family, "language": language,
        "language_origin": language_origin, "kind": KINDS[question["type"]],
        "source_split": upstream, "split": "", "request": request,
        "candidate_ids": list(parsed[0].labels), "target": target, "score_target": mean,
        "supervision": "score_mean" if mean is not None else "hard_label",
        "source": source, "text_keys": sorted({
            *("text:" + text_key(text) for text in texts), *group_keys,
        }),
        "input_sha256": json_hash(request), "context_weight": 1.0,
    }
    if votes is not None:
        row["source"]["individual_ratings"] = votes
        row["source"]["vote_distribution_used_by_current_loss"] = False
    target_values(row)
    return row


def source_map(downloads: Path) -> dict:
    receipt = json.loads((downloads / "receipt.json").read_text())
    if not receipt["complete"]:
        raise ValueError("Finish every pinned source download before conversion.")
    result = {}
    for item in receipt["files"]:
        path = downloads / item["path"]
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("Expanded source file checksum mismatch.")
        result[item["path"]] = {
            key: item[key] for key in ("repo", "revision", "license", "source_path", "sha256")
        }
    return result


def nli_rows(downloads: Path, sources: dict, counters: Counter):
    labels = {"entailment": "함의", "neutral": "중립", "contradiction": "모순"}
    names = [
        ("multinli.train.ko.tsv", "train"), ("snli_1.0_train.ko.tsv", "train"),
        ("xnli.dev.ko.tsv", "validation"), ("xnli.test.ko.tsv", "test"),
    ]
    for name, split in names:
        path = "kornlu--KorNLI--" + name
        with (downloads / path).open(newline="") as stream:
            for index, raw in enumerate(csv.DictReader(stream, delimiter="\t")):
                counters["kornli_raw"] += 1
                premise = normalized_text(raw["sentence1"] or "")
                hypothesis = normalized_text(raw["sentence2"] or "")
                label = raw["gold_label"]
                if not premise or not hypothesis or label not in labels:
                    counters["kornli_invalid"] += 1
                    continue
                yield record(
                    "kornli_" + name.split(".")[0], split, index,
                    state={"전제": premise, "가설": hypothesis},
                    question={"type": "choice",
                              "instructions": "전제만을 근거로 가설과의 논리 관계를 판단하라.",
                              "criteria": {"함의": "가설이 반드시 참이다.",
                                           "중립": "가설이 참인지 거짓인지 확정할 수 없다.",
                                           "모순": "가설이 반드시 거짓이다."}},
                    language="ko",
                    language_origin="machine_translation" if split == "train"
                    else "human_translation",
                    family="nli_grounding", source=dict(sources[path]),
                    texts=[premise, hypothesis], target=labels[label],
                )


def sentiment_rows(downloads: Path, sources: dict, counters: Counter):
    for name, split in (("ratings_train.txt", "train"), ("ratings_test.txt", "test")):
        path = "nsmc--" + name
        with (downloads / path).open(newline="") as stream:
            for index, raw in enumerate(csv.DictReader(stream, delimiter="\t")):
                counters["nsmc_raw"] += 1
                text = normalized_text(raw["document"] or "")
                if not text or raw["label"] not in ("0", "1"):
                    counters["nsmc_invalid"] += 1
                    continue
                yield record(
                    "nsmc", split, index, state={"영화평": text},
                    question={"type": "choice", "instructions": "이 영화평의 감정을 분류하라.",
                              "criteria": {"부정": "영화에 부정적인 평가",
                                           "긍정": "영화에 긍정적인 평가"}},
                    language="ko", language_origin="native", family="sentiment",
                    source={**sources[path], "rating_proxy": True,
                            "neutral_ratings_excluded_upstream": True},
                    texts=[text], target="긍정" if raw["label"] == "1" else "부정",
                )


def jsonl_gzip(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def rating_index(rows, counters: Counter):
    ratings, ambiguous = {}, set()
    for raw in rows:
        key = json_hash([raw["prompt"], raw["response"]])
        votes = {attribute: raw[attribute] for attribute in ATTRIBUTES}
        if key in ratings:
            if ratings[key] == votes:
                counters["helpsteer2_exact_duplicate_rating_rows"] += 1
            else:
                ambiguous.add(key)
            continue
        ratings[key] = votes
    counters["helpsteer2_ambiguous_rating_pairs"] += len(ambiguous)
    # Without annotator IDs, differing duplicate rows cannot be combined as
    # independent votes. Keep neither possible annotation for an ambiguous join.
    for key in ambiguous:
        del ratings[key]
    return ratings, ambiguous


def helpsteer_rows(downloads: Path, sources: dict, counters: Counter):
    vote_file = "helpsteer2--disagreements--disagreements.jsonl.gz"
    ratings, ambiguous = rating_index(jsonl_gzip(downloads / vote_file), counters)
    for split in ("train", "validation"):
        name = f"helpsteer2--{split}.jsonl.gz"
        for index, raw in enumerate(jsonl_gzip(downloads / name)):
            counters["helpsteer2_raw"] += 1
            key = json_hash([raw["prompt"], raw["response"]])
            if key in ambiguous:
                counters["helpsteer2_ambiguous_rating_examples_excluded"] += 1
                continue
            votes = ratings.get(key)
            if votes is None:
                counters["helpsteer2_missing_individual_ratings"] += 1
                continue
            for attribute, levels in ATTRIBUTES.items():
                observed = votes[attribute]
                if not observed or any(type(v) is not int or not 0 <= v <= 4 for v in observed):
                    raise ValueError("Use actual bounded individual ratings.")
                yield record(
                    "helpsteer2", split, index,
                    state={"prompt": raw["prompt"], "response": raw["response"]},
                    question={"type": "score",
                              "instructions": f"Rate the response's {attribute} under this rubric.",
                              "criteria": [f"{i}: {label}" for i, label in enumerate(levels)]},
                    language="en", language_origin="original", family="response_judgment",
                    source={**sources[name],
                            "individual_ratings_sha256": sources[vote_file]["sha256"],
                            "attribute": attribute, "annotation_origin": "human_ratings",
                            "mean_definition": "arithmetic mean of published individual ratings"},
                    texts=[raw["prompt"], raw["response"]], view=attribute,
                    mean=sum(observed) / len(observed), votes=observed,
                )
    for split in ("train", "validation"):
        name = f"helpsteer3--preference--{split}.jsonl.gz"
        for index, raw in enumerate(jsonl_gzip(downloads / name)):
            counters["helpsteer3_raw"] += 1
            natural_language = "english" if raw["domain"].lower() != "multilingual" else (
                raw["language"].lower()
            )
            if natural_language not in ("english", "korean"):
                counters["helpsteer3_other_languages_not_selected"] += 1
                continue
            ko = natural_language == "korean"
            preference = raw["overall_preference"]
            if type(preference) is not int or not -3 <= preference <= 3:
                raise ValueError("Invalid original aggregate preference.")
            text = json.dumps(raw["context"], ensure_ascii=False, sort_keys=True)
            yield record(
                "helpsteer3", split, index,
                state={"conversation": raw["context"],
                       "response1": raw["response1"], "response2": raw["response2"]},
                question={"type": "choice",
                          "instructions": "사용자의 요청에 더 도움이 되는 응답을 고르라." if ko
                          else "Which response is more helpful for the user's request?",
                          "criteria": {"response1": "첫 번째 응답" if ko else "First response",
                                       "tie": "두 응답이 비슷함" if ko else "About the same",
                                       "response2": "두 번째 응답" if ko else "Second response"}},
                language="ko" if ko else "en",
                language_origin="publisher_multilingual" if ko else "original",
                family="response_preference",
                source={**sources[name], "annotation_origin": "human_preference",
                        "original_overall_preference": preference,
                        "domain": raw["domain"],
                        "annotator_reasoning_excluded_from_input": True},
                texts=[text, raw["response1"], raw["response2"]],
                target="response1" if preference < 0 else "response2" if preference > 0 else "tie",
                votes=[item["score"] for item in raw["individual_preference"]],
            )


def massive_rows(downloads: Path, counters: Counter):
    import pyarrow.parquet as pq

    receipt = json.loads((downloads / "receipt.json").read_text())
    if any(item["status"] != "fulfilled" for item in receipt["files"]):
        raise ValueError("Incomplete MASSIVE download.")
    for wrapped in receipt["files"]:
        item = wrapped["value"]
        path = downloads / item["path"]
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("MASSIVE source identity changed.")
        table = pq.read_table(path)
        labels = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent"][
            "names"
        ]
        if len(labels) != 60 or len(set(labels)) != 60:
            raise ValueError("Keep all 60 source intents.")
        for index, raw in enumerate(table.to_pylist()):
            counters["massive_raw"] += 1
            split = {"train": "train", "dev": "validation", "test": "test"}[raw["partition"]]
            ko = raw["locale"] == "ko-KR"
            yield record(
                "massive_" + raw["locale"], split, index,
                state={"utterance": raw["utt"]},
                question={"type": "choice",
                          "instructions": ("이 요청의 의도를 고르라." if ko
                                           else "Select the intent."),
                          "criteria": {label: label.replace("_", " ") for label in labels}},
                language="ko" if ko else "en", language_origin="human_localization" if ko
                else "original",
                family="intent_routing", source={
                    "repo": receipt["repo"], "revision": receipt["revision"],
                    "original_revision": receipt["original_revision"],
                    "license": receipt["license"],
                    "source_path": item["source_path"], "sha256": item["sha256"],
                    "annotation_origin": "publisher_annotated",
                },
                texts=[raw["utt"]], target=labels[raw["intent"]],
                group_keys=["massive-parallel-id:" + str(raw["id"])],
            )


def all_rows(prior: Path, downloads: Path, sources: dict, massive: Path, counters: Counter):
    for split in SPLITS:
        with (prior / f"{split}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                row["prior_split"] = row["split"]
                row["text_keys"] = [*row["text_keys"], "prior-component:" + row["group_id"]]
                yield row
    yield from nli_rows(downloads, sources, counters)
    yield from sentiment_rows(downloads, sources, counters)
    yield from helpsteer_rows(downloads, sources, counters)
    yield from massive_rows(massive, counters)


def build(prior: Path, downloads: Path, massive: Path, out: Path):
    if out.exists():
        raise ValueError("Preserve older corpora and failed builds.")
    out.mkdir(parents=True)
    start = time.monotonic()
    sources = source_map(downloads)
    old = json.loads((prior / "manifest.json").read_text())
    for name, item in old["files"].items():
        if name.endswith(".jsonl") and file_hash(prior / name) != item["sha256"]:
            raise ValueError("Prior split files changed.")
    components, counters, identities = Components(), Counter(), set()
    input_gold, conflicting_inputs = {}, set()
    progress = {"schema": SCHEMA, "status": "grouping",
                "started_at": datetime.now(UTC).isoformat(), "training_performed": False}
    for index, row in enumerate(all_rows(prior, downloads, sources, massive, counters)):
        if row["id"] in identities:
            raise ValueError("Duplicate expanded example identity.")
        identities.add(row["id"])
        signature = json_hash([
            row["request"]["state"], row["request"]["questions"], row["supervision"],
        ])
        gold = json_hash([row["target"], row["score_target"]])
        if signature in input_gold and input_gold[signature] != gold:
            conflicting_inputs.add(signature)
        input_gold[signature] = gold
        anchor = 0
        if "prior_split" in row:
            anchor = 1 << SPLITS.index(row["prior_split"])
        elif row["source_split"] != "train":
            anchor = 1 << SPLITS.index("dev_public")
        components.add(row["text_keys"], anchor)
        if index % 10000 == 0:
            atomic_json(out / "progress.json", {
                **progress, "source_rows": index + 1, "wall_seconds": time.monotonic() - start,
            })
    source_rows = len(components.parent)
    del identities, input_gold
    progress["status"] = "writing_partitions"
    handles = {split: (out / f"{split}.jsonl").open("w") for split in SPLITS}
    excluded = (out / "quarantine.jsonl").open("w")
    counts, task_counts, languages, origin_counts = Counter(), Counter(), Counter(), Counter()
    groups, observations = {}, {}
    seen_inputs, rejected = {}, Counter()
    try:
        for index, row in enumerate(all_rows(prior, downloads, sources, massive, Counter())):
            split, group, reason = components.assignment(index, row["source_split"])
            if split is not None:
                # Duplicate judgments keep provenance in quarantine. Different
                # supplied instructions or labels remain separate observations.
                signature = json_hash([
                    row["request"]["state"], row["request"]["questions"], row["supervision"],
                ])
                gold = json_hash([row["target"], row["score_target"]])
                if signature in conflicting_inputs:
                    reason = "conflicting_annotation"
                    split = None
                elif signature in seen_inputs:
                    reason = "duplicate_input"
                    split = None
                else:
                    seen_inputs[signature] = gold
            if split is None:
                rejected[reason] += 1
                excluded.write(json.dumps({"id": row["id"], "group_id": group, "reason": reason})
                               + "\n")
                continue
            row["split"], row["group_id"] = split, group
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
            task_counts[f"{split}/{row['task']}"] += 1
            languages[f"{split}/{row['language']}"] += 1
            origin_counts[f"{split}/{row.get('language_origin', 'prior_source')}"] += 1
            groups.setdefault(split, set()).add(group)
            observations.setdefault(split, set()).add(row["observation_id"])
            if index % 10000 == 0:
                atomic_json(out / "progress.json", {
                    **progress, "source_rows": source_rows, "rows_written": sum(counts.values()),
                    "rows_examined": index + 1, "wall_seconds": time.monotonic() - start,
                })
    finally:
        for handle in [*handles.values(), excluded]:
            handle.close()
    if index + 1 != source_rows:
        raise ValueError("The two deterministic source passes diverged.")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
             for path in sorted(out.glob("*.jsonl"))}
    result = {
        **progress, "status": "completed", "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": time.monotonic() - start, "source_rows": source_rows,
        "source_audit": dict(counters), "partitions": dict(counts),
        "task_counts": dict(task_counts), "language_counts": dict(languages),
        "language_origins": dict(origin_counts), "quarantine": dict(rejected),
        "components": {key: len(value) for key, value in groups.items()},
        "observations": {key: len(value) for key, value in observations.items()},
        "files": files, "prior_manifest_sha256": file_hash(prior / "manifest.json"),
        "download_receipt_sha256": file_hash(downloads / "receipt.json"),
        "massive_receipt_sha256": file_hash(massive / "receipt.json"),
        "generator_sha256": file_hash(Path(__file__)), "sources": sources,
        "near_duplicate_audit_complete": False, "fresh_final_evaluation": False,
        "teacher_gold_used": False, "original_source_files_modified": False,
        "scope": "Expanded public adaptation corpus. Public test remains development.",
    }
    result["content_sha256"] = json_hash(result)
    atomic_json(out / "manifest.json", result)
    atomic_json(out / "progress.json", {
        **progress, "status": "completed", "partitions": dict(counts),
        "wall_seconds": time.monotonic() - start,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--massive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.prior, args.downloads, args.massive, args.out)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("sources", "files")}, indent=2))


if __name__ == "__main__":
    main()
