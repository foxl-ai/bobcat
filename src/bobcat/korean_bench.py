"""External Korean benchmarks, with English controls, as typed decisions (evaluation only).

Public test/validation splits at pinned revisions become one Choice/Noul request each, with
the benchmark's own options as candidates:
  Korean   KoBEST (all five test sets), KMMLU test, HAE-RAE Bench 1.1, CLIcK
  English  MMLU test, a HellaSwag validation sample (controls for KMMLU / KoBEST HellaSwag)
Every fetched file is verified against the Hub (LFS SHA256 or git blob SHA-1). Items whose
text appears in the student training mixture are dropped and counted. Nothing here is used
for training, selection or calibration; upstream test sets are development data for Bobcat.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path

from bobcat.corpus import atomic_json, normalized_text
from bobcat.kobest_eval import TASKS as KOBEST_TASKS
from bobcat.kobest_eval import components as kobest_components
from bobcat.kobest_eval import record as kobest_record
from bobcat.product_eval import text_key
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash

SCHEMA = "bobcat-korean-benchmarks-v1"
MODEL = "bobcat-latest"
SOURCES = {
    "kobest": {"repo": "skt/kobest_v1", "revision": "a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
               "patterns": [f"{t}/test.jsonl" for t in KOBEST_TASKS],
               "license": "CC-BY-SA-4.0", "language": "ko"},
    "kmmlu": {"repo": "HAERAE-HUB/KMMLU", "revision": "d61b3f19e552c576bf5960dd24289763edc36a88",
              "patterns": ["data/*-test.csv"], "license": "CC-BY-ND-4.0", "language": "ko"},
    "haerae": {"repo": "HAERAE-HUB/HAE_RAE_BENCH_1.1",
               "revision": "b480e81024913f27783d2b05d2f0b10089db19ad",
               "patterns": ["data/*.parquet"], "license": "CC-BY-NC-ND-4.0", "language": "ko"},
    "click": {"repo": "EunsuKim/CLIcK", "revision": "d61627859645b5e6edc03fd9f835735d8226fa4e",
              "patterns": ["Dataset/*.json"], "license": "none declared on the dataset card",
              "language": "ko"},
    "mmlu": {"repo": "cais/mmlu", "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
             "patterns": ["all/test-00000-of-00001.parquet"], "license": "MIT", "language": "en"},
    "hellaswag_en": {"repo": "Rowan/hellaswag",
                     "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                     "patterns": ["data/validation-00000-of-00001.parquet"],
                     "license": "none declared on the dataset card (MIT upstream)",
                     "language": "en"},
}
LETTERS = "ABCDEFGH"
HELLASWAG_SAMPLE = 2000


def fetch(name: str, root: Path) -> dict:
    """Download one pinned source and verify every file against the Hub listing."""
    source = SOURCES[name]
    repo, revision = source["repo"], source["revision"]
    api = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        meta = json.loads(response.read())
    if meta.get("sha") != revision:
        raise ValueError(f"{repo}: the Hub returned a different revision.")
    files = {}
    for sibling in meta["siblings"]:
        path = sibling["rfilename"]
        if ".ipynb_checkpoints" in path or not any(fnmatch(path, p) for p in source["patterns"]):
            continue
        url = (f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
               + urllib.parse.quote(path))
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        lfs = sibling.get("lfs")
        if lfs:
            ok = len(data) == lfs["size"] and hashlib.sha256(data).hexdigest() == lfs["sha256"]
        else:
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            ok = blob == sibling.get("blobId")
        if not ok:
            raise ValueError(f"Checksum failed: {repo}/{path}")
        target = root / name / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[path] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if not files:
        raise ValueError(f"{repo}: no files matched.")
    return {"repo": repo, "revision": revision, "license": source["license"], "files": files}


def text(value) -> str:
    if not isinstance(value, str) or not normalized_text(value):
        raise ValueError("Empty benchmark text.")
    return normalized_text(value)


def choice_row(bench: str, family: str, key, state: dict, instructions: str,
               options: list[str], answer: int, texts: list[str], group: str | None = None):
    if not 2 <= len(options) <= len(LETTERS) or not 0 <= answer < len(options):
        raise ValueError(f"{bench}: invalid options or answer.")
    criteria = {LETTERS[i]: text(option) for i, option in enumerate(options)}
    request = {"model": MODEL, "state": state,
               "questions": {"q": {"type": "choice", "instructions": instructions,
                                   "criteria": criteria}}}
    _, parsed = parse_request(request)
    identity = json_hash([bench, family, key, request])
    return {"id": f"{bench}:{identity[:24]}", "task": bench, "family": family, "kind": "choice",
            "language": SOURCES[bench]["language"], "supervision": "hard_label",
            "target": LETTERS[answer], "score_target": None,
            "candidate_ids": list(parsed[0].labels), "request": request,
            "group_id": f"{bench}:" + (group or identity), "text_keys": sorted(
                {text_key(t) for t in texts if t})}


def kobest_rows(root: Path) -> list[dict]:
    rows = []
    for task in KOBEST_TASKS:
        path = root / "kobest" / task / "test.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        rows += [kobest_record(task, value, i) for i, value in enumerate(entries)]
    for row, group in zip(rows, kobest_components(rows), strict=True):
        state = row["request"]["state"]
        row.update(group_id=group, supervision="hard_label", family=row["task"],
                   task="kobest", text_keys=sorted(text_key(v) for v in state.values()))
        for key in ("evidence_keys", "observation_id", "split", "source_split", "tie_break"):
            row.pop(key, None)
    return rows


def kmmlu_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "kmmlu" / "data").glob("*-test.csv")):
        subject = path.name.removesuffix("-test.csv")
        for index, item in enumerate(csv.DictReader(io.StringIO(path.read_text()))):
            question = text(item["question"])
            rows.append(choice_row(
                "kmmlu", subject, index, {"문제": question}, "다음 문제의 정답을 고르라.",
                [item[x] for x in "ABCD"], int(item["answer"]) - 1, [question]))
    return rows


HAERAE_GENERATION = ("lyrics_denoising", "proverbs_denoising")  # free text, not choices


def haerae_options(listed: str) -> list[str]:
    """The options column is a Python list literal, or " | "-separated for the CSAT tasks."""
    try:
        return [str(option) for option in ast.literal_eval(listed)]
    except (ValueError, SyntaxError):
        return [option.strip() for option in listed.split("|")]


def haerae_rows(root: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for path in sorted((root / "haerae" / "data").glob("*.parquet")):
        task = path.name.split("-00000-")[0]
        if task in HAERAE_GENERATION:
            continue
        for index, item in enumerate(pq.read_table(path).to_pylist()):
            query = text(item["query"])
            body = query.split("### 선택지")[0].strip()
            passage = body.split("### 지문:")[1].split("### 질문")[0] if "### 지문:" in body \
                else None
            options = haerae_options(item["options"])
            answer = LETTERS.index(item["answer"].strip().strip("()"))
            rows.append(choice_row(
                "haerae", task, index, {"문제": body}, "문제에 대한 정답을 선택지에서 고르라.",
                options, answer, [body],
                group=json_hash(normalized_text(passage)) if passage else None))
    return rows


def click_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "click" / "Dataset").rglob("*.json")):
        family = "/".join(path.relative_to(root / "click" / "Dataset").parts[:2])
        for item in json.loads(path.read_text()):
            question = text(item["question"])
            paragraph = normalized_text(item.get("paragraph") or "")
            choices = [text(c) for c in item["choices"]]
            matches = [i for i, c in enumerate(choices) if c == text(item["answer"])]
            if len(matches) != 1:
                raise ValueError(f"CLIcK {item['id']}: the answer must match one choice.")
            state = {"지문": paragraph, "문제": question} if paragraph else {"문제": question}
            rows.append(choice_row(
                "click", family, item["id"], state, "문제의 정답을 고르라.", choices,
                matches[0], [question, paragraph],
                group=json_hash(paragraph) if paragraph else None))
    return rows


def mmlu_rows(root: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    table = pq.read_table(root / "mmlu" / "all" / "test-00000-of-00001.parquet")
    for index, item in enumerate(table.to_pylist()):
        question = text(item["question"])
        rows.append(choice_row(
            "mmlu", item["subject"], index, {"question": question},
            "Choose the correct answer to the question.", list(item["choices"]),
            int(item["answer"]), [question]))
    return rows


def hellaswag_rows(root: Path, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(root / "hellaswag_en" / "data" / "validation-00000-of-00001.parquet")
    items = sorted(table.to_pylist(), key=lambda r: json_hash([seed, r["ind"]]))
    rows = []
    for item in items[:HELLASWAG_SAMPLE]:
        context = text(f"{item['activity_label']}: {item['ctx']}")
        rows.append(choice_row(
            "hellaswag_en", item["split_type"], item["ind"], {"context": context},
            "Choose the most plausible continuation of the context.", list(item["endings"]),
            int(item["label"]), [context], group=item["source_id"]))
    return rows


def training_keys(path: Path) -> set[str]:
    keys = set()
    for line in path.open():
        keys.update(k for k in json.loads(line).get("text_keys", []) if k.startswith("text:"))
    return keys


def build(args) -> dict:
    if args.out.exists():
        raise ValueError("Benchmark sets are immutable; choose a new path.")
    receipts = {}
    for name in SOURCES:
        receipt = args.raw / name / "bobcat-receipt.json"
        if not receipt.exists():
            atomic_json(receipt, fetch(name, args.raw))
        receipts[name] = json.loads(receipt.read_text())
    rows = (kobest_rows(args.raw) + kmmlu_rows(args.raw) + haerae_rows(args.raw)
            + click_rows(args.raw) + mmlu_rows(args.raw) + hellaswag_rows(args.raw, args.seed))
    forbidden = training_keys(args.train_mixture)
    kept, dropped = [], Counter()
    for row in rows:
        if set(row["text_keys"]) & forbidden:
            dropped[row["task"]] += 1
            continue
        kept.append(row)
    if len({r["id"] for r in kept}) != len(kept):
        raise ValueError("Duplicate benchmark row IDs.")
    args.out.mkdir(parents=True)
    with (args.out / "rows.jsonl").open("x") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = defaultdict(Counter)
    for row in kept:
        counts[row["task"]][row["family"]] += 1
    manifest = {
        "schema": SCHEMA, "seed": args.seed,
        "sources": {name: {k: v for k, v in SOURCES[name].items() if k != "patterns"}
                    for name in SOURCES},
        "receipts": receipts, "rows": len(kept),
        "rows_by_benchmark": {k: sum(v.values()) for k, v in sorted(counts.items())},
        "families": {k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
        "dropped_for_training_text_overlap": dict(dropped),
        "excluded_generation_tasks": {"haerae": list(HAERAE_GENERATION)},
        "training_mixture_sha256": file_hash(args.train_mixture),
        "rows_sha256": file_hash(args.out / "rows.jsonl"),
        "use": "evaluation only; upstream test sets are Bobcat development data",
        "pretraining_contamination": "not measured for the public base model",
    }
    atomic_json(args.out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--train-mixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092506)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({k: manifest[k] for k in ("rows", "rows_by_benchmark",
                                               "dropped_for_training_text_overlap")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
