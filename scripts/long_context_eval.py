"""Does Bobcat keep its answers when the state grows with irrelevant text?

`build` takes a seeded, task-stratified sample of development questions and, for each, writes
the unchanged question plus versions whose state is padded to about N compiled tokens with
passages from OTHER development components: half before and half after the original state,
under two neutral keys, so the relevant content and the question are untouched and in the
middle. Padding comes only from search and citation passages (encyclopedic KLUE text), never
from the injection, routing, classification or tool-call tasks, so it carries no
instructions and no competing labels. `score` runs in infra/aws/vllm_bench.py
(`--workloads longctx`); `report` turns the logits into accuracy per level with a paired
cluster bootstrap against the unpadded level.

    python scripts/long_context_eval.py build --dev-rows dev.jsonl --compiler-model compiler \
        --levels 8192,16384,30720,61440 --count 300 --out longctx
    python scripts/long_context_eval.py report --dev-rows dev.jsonl \
        --logits longctx-logits --out long-context-report.json

`build --write-rows` also writes each level's padded dev rows (the dev row with its padded
state; `long_context_level` added), so another model can compile the identical request text
(`compile`) and an HTTP client can send it. `--exclude-ids` (a manifest's `ids` or a JSON
list) draws a sample disjoint from an earlier one. Neither option changes the compiled rows.

    python scripts/long_context_eval.py compile --rows-dir longctx --compiler-model other \
        --out longctx-other
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from bobcat.schema import file_hash, json_hash

SALT = "bobcat-long-context-v1"
PAD_KEYS = ("기타 자료 1", "기타 자료 2")
POOL_TASKS = ("product_search", "product_citation")
MIN_PASSAGE_CHARS = 200


def stratified(rows: list[dict], count: int, salt: str = SALT) -> list[dict]:
    """Round-robin over tasks, each task in a fixed hash order (as bobcat.stress does)."""
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    queues = [sorted(items, key=lambda r: json_hash([salt, r["id"]]))
              for _, items in sorted(by_task.items())]
    chosen = []
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    return chosen


def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def passage_pool(rows: list[dict], salt: str = SALT) -> list[tuple[str, str]]:
    """(group_id, passage) from search and citation states, deduplicated, hash-ordered."""
    seen, pool = set(), []
    for row in rows:
        if row["task"] not in POOL_TASKS:
            continue
        for text in strings(row["request"]["state"]):
            if len(text) >= MIN_PASSAGE_CHARS and text not in seen:
                seen.add(text)
                pool.append((row["group_id"], text))
    return sorted(pool, key=lambda item: json_hash([salt, item[1]]))


def padding_text(row: dict, pool: list[tuple[str, str]], chars: int, salt: str = SALT) -> str:
    """At least `chars` characters of other components' passages, from a per-row offset."""
    own = set(strings(row["request"]["state"]))
    usable = [text for group, text in pool if group != row["group_id"] and text not in own]
    if not usable:
        raise ValueError("No padding passages outside the row's own component.")
    start = int(json_hash([salt, row["id"]]), 16) % len(usable)
    parts, total, index = [], 0, start
    while total < chars:
        text = usable[index % len(usable)]
        parts.append(text)
        total += len(text) + 2
        index += 1
        if index - start > 10 * len(usable):
            break
    return "\n\n".join(parts)[:chars]


def padded_state(state: dict, text: str) -> dict:
    """The original state, unchanged and in order, between two halves of `text`."""
    if not isinstance(state, dict) or any(key in state for key in PAD_KEYS):
        raise ValueError("Padding needs a dict state without the padding keys.")
    half = len(text) // 2
    return {PAD_KEYS[0]: text[:half], **state, PAD_KEYS[1]: text[half:]}


def fit(row: dict, pool, target: int, length, salt: str = SALT, rounds: int = 6):
    """(padded state, compiled length) with length <= target and as close as a few secant
    steps get; `length(state)` compiles the row's question over `state`."""
    state = row["request"]["state"]
    base = length(state)
    if base >= target:
        return state, base
    chars = (target - base) * 2  # first guess; corrected from the measured slope
    best = (state, base)
    for _ in range(rounds):
        candidate = padded_state(state, padding_text(row, pool, max(1, chars), salt))
        got = length(candidate)
        if got <= target and got > best[1]:
            best = (candidate, got)
        if target - 64 <= got <= target:
            break
        slope = max(1e-6, (got - base) / chars)
        chars = max(1, int(chars + (target - 32 - got) / slope))
    return best


def build(args) -> None:
    from tokenizers import Tokenizer

    from bobcat.protocol import RequestLimitError, parse_request
    from bobcat.student_readout import StudentCompiler, identifier_scheme

    receipt = json.loads((args.compiler_model / "bobcat-download.json").read_text())
    tokenizer_path = args.compiler_model / "tokenizer.json"
    reserved = {t["id"] for t in json.loads(tokenizer_path.read_text()).get("added_tokens", [])}
    identifiers = identifier_scheme(Tokenizer.from_file(str(tokenizer_path)), reserved,
                                    json.loads(args.identifiers.read_text())["identifiers"])
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    # Room for the fitting steps to overshoot; the kept version never exceeds its level.
    compiler = StudentCompiler(args.compiler_model, receipt["files"], identifiers,
                               max_branch_tokens=2 * max(levels) + 64, piecewise=True)
    rows = [json.loads(line) for line in args.dev_rows.open()]
    excluded = set(excluded_ids(args.exclude_ids)) if args.exclude_ids else set()
    chosen = stratified([r for r in rows if r["id"] not in excluded], args.count, args.salt)
    pool = passage_pool(rows, args.salt)
    args.out.mkdir(parents=True, exist_ok=False)
    sinks = {level: (args.out / f"level-{level}.compiled.jsonl").open("w")
             for level in [0, *levels]}
    row_sinks = ({level: (args.out / f"level-{level}.rows.jsonl").open("w")
                  for level in [0, *levels]} if args.write_rows else {})
    achieved = defaultdict(list)
    for row in chosen:
        state, (question,) = parse_request(row["request"])

        def length(value, question=question):
            try:
                return len(compiler.compile(value, question)[0])
            except RequestLimitError:
                return math.inf

        for level in [0, *levels]:
            if level:
                padded, _ = fit(row, pool, level, length, args.salt)
            else:
                padded = state
            sequence, options = compiler.compile(padded, question)
            achieved[level].append(len(sequence))
            sinks[level].write(json.dumps({"id": row["id"], "level": level, "task": row["task"],
                                           "input_ids": sequence, "option_ids": options}) + "\n")
            if level in row_sinks:
                row_sinks[level].write(json.dumps(padded_row(row, padded, level),
                                                  ensure_ascii=False) + "\n")
    for sink in [*sinks.values(), *row_sinks.values()]:
        sink.close()
    manifest = {"schema": "bobcat-long-context-v1", "salt": args.salt, "count": len(chosen),
                "ids": [r["id"] for r in chosen], "pool_passages": len(pool),
                "pool_tasks": list(POOL_TASKS), "pad_keys": list(PAD_KEYS),
                "levels": {str(k): {"min": min(v), "median": statistics.median(v),
                                    "max": max(v)} for k, v in achieved.items()}}
    if excluded:
        manifest["excluded_ids"] = len(excluded)
    if row_sinks:
        manifest["rows_files"] = {str(k): file_hash(args.out / f"level-{k}.rows.jsonl")
                                  for k in row_sinks}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                       indent=1) + "\n")
    print(json.dumps(manifest["levels"]))


def excluded_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text())
    return value["ids"] if isinstance(value, dict) else value


def padded_row(row: dict, state, level: int) -> dict:
    """The dev row with the padded state: same ID, question, candidates and target."""
    return {**row, "request": {**row["request"], "state": state}, "long_context_level": level}


def compile_rows(args) -> None:
    """Compile every level-*.rows.jsonl of a build with another model's compiler, so that
    model scores the identical request text (vllm_bench --workloads longctx reads the out)."""
    from tokenizers import Tokenizer

    from bobcat.protocol import RequestLimitError, parse_request
    from bobcat.student_readout import StudentCompiler, identifier_scheme

    receipt = json.loads((args.compiler_model / "bobcat-download.json").read_text())
    tokenizer_path = args.compiler_model / "tokenizer.json"
    reserved = {t["id"] for t in json.loads(tokenizer_path.read_text()).get("added_tokens", [])}
    identifiers = identifier_scheme(Tokenizer.from_file(str(tokenizer_path)), reserved,
                                    json.loads(args.identifiers.read_text())["identifiers"])
    compiler = StudentCompiler(args.compiler_model, receipt["files"], identifiers,
                               max_branch_tokens=args.max_tokens, piecewise=True)
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for path in sorted(args.rows_dir.glob("level-*.rows.jsonl")):
        name = path.name.replace(".rows.jsonl", "")
        lengths, refused = [], []
        with (args.out / f"{name}.compiled.jsonl").open("w") as sink:
            for line in path.open():
                row = json.loads(line)
                state, (question,) = parse_request(row["request"])
                try:
                    sequence, options = compiler.compile(state, question)
                except RequestLimitError:
                    refused.append(row["id"])  # never truncated; reported
                    continue
                lengths.append(len(sequence))
                sink.write(json.dumps({"id": row["id"], "level": row.get("long_context_level", 0),
                                       "task": row["task"], "input_ids": sequence,
                                       "option_ids": options}) + "\n")
        summary[name] = {"rows": len(lengths), "refused": refused,
                         "tokens": {"min": min(lengths), "median": statistics.median(lengths),
                                    "max": max(lengths)} if lengths else None,
                         "rows_sha256": file_hash(path)}
    (args.out / "manifest.json").write_text(json.dumps(
        {"schema": "bobcat-long-context-compile-v1", "compiler": receipt["repo"],
         "revision": receipt["revision"], "max_tokens": args.max_tokens, "levels": summary},
        indent=1) + "\n")
    print(json.dumps({k: (v["rows"], len(v["refused"])) for k, v in summary.items()}))


def softmax(values, temperature):
    scaled = [v / temperature for v in values]
    top = max(scaled)
    exps = [math.exp(v - top) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def accuracy_table(dev: dict, logits: dict, temperature: float) -> dict:
    """{level: {id: (correct, probability on the gold candidate)}}."""
    table = defaultdict(dict)
    for (row_id, level), values in logits.items():
        row = dev[row_id]
        gold = row["candidate_ids"].index(row["target"])
        probs = softmax(values, temperature)
        table[level][row_id] = (max(range(len(values)), key=values.__getitem__) == gold,
                                probs[gold])
    return table


def paired_bootstrap(dev, base: dict, other: dict, draws: int = 10000, seed: int = 20260925):
    """95% interval of accuracy(other) - accuracy(base), resampling components."""
    clusters = defaultdict(list)
    for row_id in base:
        if row_id in other:
            clusters[dev[row_id]["group_id"]].append(row_id)
    keys = sorted(clusters)
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        sample = [row_id for key in (rng.choice(keys) for _ in keys) for row_id in clusters[key]]
        diffs.append(statistics.fmean(other[i][0] - base[i][0] for i in sample))
    diffs.sort()
    return [diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]]


def report(args) -> None:
    dev = {r["id"]: r for r in map(json.loads, args.dev_rows.open())}
    logits = {}
    for path in sorted(args.logits.glob("level-*.logits.jsonl")):
        for line in path.open():
            item = json.loads(line)
            logits[(item["id"], item["level"])] = item["logits"]
    table = accuracy_table(dev, logits, args.temperature)
    base = table.get(0, {})
    result = {"temperature": args.temperature, "levels": {}}
    for level in sorted(table):
        rows = table[level]
        by_task = defaultdict(list)
        for row_id, (correct, _) in rows.items():
            by_task[dev[row_id]["task"]].append(correct)
        entry = {"questions": len(rows),
                 "accuracy": statistics.fmean(c for c, _ in rows.values()),
                 "task_macro": statistics.fmean(statistics.fmean(v) for v in by_task.values()),
                 "mean_gold_probability": statistics.fmean(p for _, p in rows.values()),
                 "task_accuracy": {k: statistics.fmean(v) for k, v in sorted(by_task.items())}}
        if level and base:
            entry["minus_unpadded_95ci"] = paired_bootstrap(dev, base, rows)
            entry["minus_unpadded"] = entry["accuracy"] - statistics.fmean(
                base[i][0] for i in rows if i in base)
        result["levels"][str(level)] = entry
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: (round(v["accuracy"], 4), v.get("minus_unpadded_95ci"))
                      for k, v in result["levels"].items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dev-rows", type=Path, required=True)
    b.add_argument("--compiler-model", type=Path, required=True)
    b.add_argument("--identifiers", type=Path,
                   default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    b.add_argument("--levels", default="8192,16384,30720,61440")
    b.add_argument("--count", type=int, default=300)
    b.add_argument("--salt", default=SALT)
    b.add_argument("--exclude-ids", type=Path,
                   help="a build manifest (its `ids`) or a JSON list of dev IDs to leave out")
    b.add_argument("--write-rows", action="store_true",
                   help="also write level-*.rows.jsonl (padded dev rows)")
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compile")
    c.add_argument("--rows-dir", type=Path, required=True)
    c.add_argument("--compiler-model", type=Path, required=True)
    c.add_argument("--identifiers", type=Path,
                   default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    c.add_argument("--max-tokens", type=int, default=32768)
    c.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--dev-rows", type=Path, required=True)
    r.add_argument("--logits", type=Path, required=True)
    r.add_argument("--temperature", type=float, default=1.1489)
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"build": build, "compile": compile_rows, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
