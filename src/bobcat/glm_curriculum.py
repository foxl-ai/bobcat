"""Freeze a component-first, token-balanced curriculum from compiled real data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.schema import file_hash, json_hash

# Token budgets, not row proportions. Translation receives 10%, native/localized
# Korean another 50%. No task receives more than 15% of intended prompt tokens.
MIXTURE = {
    "klue_ynat": .15, "nsmc": .15, "massive_ko-KR": .10,
    "klue_sts": .05, "klue_nli": .05,
    "kornli_multinli": .05, "kornli_snli_1": .05,
    "helpsteer2": .10, "helpsteer3": .10, "massive_en-US": .08,
    "banking77": .06, "boolq": .04, "arc_easy": .01, "arc_challenge": .01,
}


def select_components(pools, count, seed, weights=None):
    """Choose a component uniformly, then one of its observations uniformly.

    The least-served token stratum is selected next. Components cannot appear
    twice in one curriculum, including components occurring in several tasks.
    Loss weights are therefore one, not another inverse component size.
    """
    weights = weights or MIXTURE
    if count < 1 or set(pools) != set(weights) or abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Provide all positive mixture strata and a finite row count.")
    if any(w <= 0 for w in weights.values()):
        raise ValueError("Every stratum needs a positive token budget.")
    rng = random.Random(seed)
    queues = {task: list(sorted(groups)) for task, groups in pools.items()}
    for queue in queues.values():
        rng.shuffle(queue)
    used, tokens, selected = set(), Counter(), []
    for _ in range(count):
        task = min(weights, key=lambda key: tokens[key] / weights[key])
        queue = queues[task]
        while queue and queue[-1] in used:
            queue.pop()
        if not queue:
            raise ValueError(f"Component pool exhausted; do not silently resample: {task}")
        component = queue.pop()
        row = rng.choice(pools[task][component])
        if row["input_tokens"] < 1:
            raise ValueError("Token budgets require complete positive-length prompts.")
        used.add(component)
        tokens[task] += row["input_tokens"]
        selected.append(row)
    return selected


def build(source: Path, out: Path, *, train_rows=16384, dev_rows=128,
          max_tokens=4096, seed=202609230001):
    import pyarrow.parquet as pq

    if out.exists() or max_tokens < 128 or train_rows % 8 or dev_rows % 8:
        raise ValueError("Use a fresh directory and whole eight-rank batches.")
    out.mkdir(parents=True)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["schema"] != "bobcat-glm-training-corpus-v1"
            or manifest["status"] != "completed" or manifest["truncation"]
            or manifest["shortlisting"] or manifest["training_performed"]):
        raise ValueError("Use the complete, unmodified compiled real-data manifest.")
    pools = {split: defaultdict(lambda: defaultdict(list)) for split in ("train", "dev_train")}
    blocked, component_split, inventory = set(), {}, []
    columns = ["group_id", "task", "split", "input_tokens", "language", "language_origin"]
    for number, item in enumerate(manifest["files"]):
        path = source / item["path"]
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]):
            raise ValueError(f"Source shard changed: {item['path']}")
        inventory.append(item)
        for offset, row in enumerate(pq.read_table(path, columns=columns).to_pylist()):
            split, group, task = row["split"], row["group_id"], row["task"]
            if split not in pools or task not in MIXTURE:
                raise ValueError("Unknown task or non-training partition.")
            if component_split.setdefault(group, split) != split:
                raise ValueError("A component crosses train/development.")
            if row["input_tokens"] > max_tokens:
                blocked.add(group)
            pools[split][task][group].append({
                "file": number, "offset": offset, "input_tokens": row["input_tokens"],
                "group_id": group, "task": task,
            })
    for split in pools.values():
        for task in split.values():
            for group in blocked:
                task.pop(group, None)
    schedules = {
        split: select_components(pools[split], count, seed + index)
        for index, (split, count) in enumerate((("train", train_rows), ("dev_train", dev_rows)))
    }
    # FSDP/EP ranks agree on tensor shapes. Sort a bounded window by complete
    # prompt length, then shuffle whole eight-rank batches. No rows are lost and
    # the token mixture is unchanged; padding does not count as training data.
    for index, (split, schedule) in enumerate(schedules.items()):
        rng, batches = random.Random(seed + 100 + index), []
        for start in range(0, len(schedule), 256):
            window = sorted(schedule[start:start + 256], key=lambda r: r["input_tokens"])
            chunks = [window[k:k + 8] for k in range(0, len(window), 8)]
            rng.shuffle(chunks)
            batches.extend(chunks)
        schedules[split] = [row for batch in batches for row in batch]
    wanted = defaultdict(list)
    for split, schedule in schedules.items():
        for position, row in enumerate(schedule):
            wanted[row["file"]].append((split, position, row))
    records = {split: [None] * len(schedule) for split, schedule in schedules.items()}
    for number, requests in wanted.items():
        rows = pq.read_table(source / inventory[number]["path"]).to_pylist()
        for split, position, expected in requests:
            row = rows[expected["offset"]]
            inputs = {key: row[key] for key in ("input_ids", "option_token_ids")}
            if (row["group_id"] != expected["group_id"] or row["split"] != split
                    or row["input_tokens"] != expected["input_tokens"]
                    or row["input_tokens"] != len(row["input_ids"])
                    or row["input_sha256"] != json_hash(inputs)):
                raise ValueError("Selected row and compiled token identity differ.")
            row["sampling_loss_weight"] = 1.0
            records[split][position] = row
    files, statistics = {}, {}
    for split, rows in records.items():
        task_tokens, language_tokens, origins, kinds, candidates = (
            Counter(), Counter(), Counter(), Counter(), Counter()
        )
        with (out / f"{split}.jsonl").open("x") as stream:
            for row in rows:
                if row is None:
                    raise ValueError("A scheduled observation was not materialized.")
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                n = row["input_tokens"]
                task_tokens[row["task"]] += n
                language_tokens[row["language"]] += n
                origins[row["language_origin"]] += n
                kinds[row["kind"]] += 1
                candidates[len(row["option_token_ids"])] += 1
        path = out / f"{split}.jsonl"
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        statistics[split] = {
            "rows": len(rows), "components": len({r["group_id"] for r in rows}),
            "prompt_tokens": sum(language_tokens.values()), "task_tokens": dict(task_tokens),
            "language_tokens": dict(language_tokens), "origin_tokens": dict(origins),
            "kinds": dict(kinds),
            "candidate_counts": {str(k): v for k, v in candidates.items()},
            "maximum_input_tokens": max(r["input_tokens"] for r in rows),
        }
    train = statistics["train"]
    total = train["prompt_tokens"]
    if (abs(train["language_tokens"]["ko"] / total - .6) > .03
            or train["origin_tokens"].get("machine_translation", 0) / total > .2
            or max(train["task_tokens"].values()) / total > .2):
        raise ValueError("The actual selected token mixture misses the frozen bounds.")
    result = {
        "schema": "bobcat-glm-curriculum-v1", "status": "completed",
        "at": datetime.now(UTC).isoformat(), "seed": seed, "mixture_token_targets": MIXTURE,
        "curriculum_max_tokens": max_tokens, "excluded_long_components": len(blocked),
        "truncated_prompts": 0, "source_manifest_sha256": file_hash(manifest_path),
        "model_source": manifest["model_source"], "compiler_profile": manifest["compiler_profile"],
        "source_shards_verified": len(inventory), "statistics": statistics, "files": files,
        "sampling": "component without replacement, uniform observation, token deficit stratum",
        "length_bucketing": "256-row windows, sorted lengths, shuffled eight-rank batches",
        "training_loss_weight": 1.0, "inverse_component_weight_applied_again": False,
        "training_performed": False, "source_sha256": file_hash(Path(__file__)),
    }
    result["content_sha256"] = json_hash(result)
    atomic_json(out / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=16384)
    parser.add_argument("--dev-rows", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    result = build(args.source, args.out, train_rows=args.train_rows,
                   dev_rows=args.dev_rows, max_tokens=args.max_tokens)
    print(json.dumps(result["statistics"], indent=2))
