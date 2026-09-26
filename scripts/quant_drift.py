"""Accuracy drift of quantized serving checkpoints on the dev split, paired by row.

Each `--pred NAME=PATH` is a jsonl of {"id", "logits"} over a dev row's candidates
(`vllm_bench.py --dev-predictions`, or the stored evaluation-path logits). Rows whose ids are
listed in `--exclude` (the calibration ids an NVFP4 receipt records) are left out of every
comparison, so a quantized model is scored only on dev rows its calibration did not see.
For each prediction set: task accuracy, task macro, mean gold probability at the calibration
temperature, argmax agreement with `--base`, and the 95% interval of the accuracy difference
against `--base` from a bootstrap over components (group_id).

Near ties: a row is a near tie when the reference's top two probabilities (at the
calibration temperature) differ by less than `--near-tie-margin`; argmax agreement is also
reported on those rows, where a small numerical drift can flip an answer. `--pair
NAME=A:B` compares two prediction files of the same checkpoint (for example the cached and
the uncached pass of `vllm_bench.py --dev-cache-check`, B the reference) on the same rows.

    python scripts/quant_drift.py --dev-rows dev.jsonl --base fp8 \
        --pred fp8=fp8.dev.jsonl --pred nvfp4=nvfp4.dev.jsonl --pred bf16=stored.jsonl \
        --pair nvfp4_cache=nvfp4.cached.jsonl:nvfp4.uncached.jsonl \
        --exclude merged-nvfp4/bobcat-nvfp4.json --out drift.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.long_context_eval import paired_bootstrap, softmax


def load_predictions(path: Path) -> dict[str, list[float]]:
    return {r["id"]: r["logits"] for r in map(json.loads, path.open())}


def excluded_ids(paths: list[Path]) -> set[str]:
    ids = set()
    for path in paths:
        data = json.loads(path.read_text())
        ids.update(data["calibration"]["ids"] if "calibration" in data else data)
    return ids


def score(dev: dict, predictions: dict, rows: list[str], temperature: float) -> dict:
    """{id: (correct, gold probability)} for the given rows."""
    table = {}
    for row_id in rows:
        values, row = predictions[row_id], dev[row_id]
        gold = row["candidate_ids"].index(row["target"])
        table[row_id] = (max(range(len(values)), key=values.__getitem__) == gold,
                         softmax(values, temperature)[gold])
    return table


def argmax(values) -> int:
    return max(range(len(values)), key=values.__getitem__)


def agreement(ref: dict, other: dict, rows: list[str], temperature: float,
              margin: float) -> dict:
    """Argmax agreement of `other` with `ref`, on all rows and on the reference's near ties,
    and the largest / mean per-row probability gap."""
    same, near, gaps = [], [], []
    for row_id in rows:
        p, q = softmax(ref[row_id], temperature), softmax(other[row_id], temperature)
        agree = argmax(p) == argmax(q)
        same.append(agree)
        top = sorted(p, reverse=True)
        if len(top) > 1 and top[0] - top[1] < margin:
            near.append(agree)
        gaps.append(max(abs(a - b) for a, b in zip(p, q, strict=True)))
    return {"rows": len(rows), "argmax_agreement": statistics.fmean(same),
            "disagreements": len(same) - sum(same), "near_tie_margin": margin,
            "near_tie_rows": len(near),
            "near_tie_agreement": statistics.fmean(near) if near else None,
            "max_probability_gap": max(gaps), "mean_probability_gap": statistics.fmean(gaps)}


def compare_pair(dev: dict, a: dict, b: dict, exclude: set[str], temperature: float,
                 margin: float) -> dict:
    rows = sorted((set(a) & set(b)) - exclude)
    table_a, table_b = score(dev, a, rows, temperature), score(dev, b, rows, temperature)
    return {**agreement(b, a, rows, temperature, margin),
            "accuracy": statistics.fmean(c for c, _ in table_a.values()),
            "reference_accuracy": statistics.fmean(c for c, _ in table_b.values()),
            "minus_reference_95ci": paired_bootstrap(dev, table_b, table_a)}


def compare(dev: dict, predictions: dict[str, dict], base: str, exclude: set[str],
            temperature: float, margin: float = 0.1) -> dict:
    rows = sorted(set.intersection(*(set(p) for p in predictions.values())) - exclude)
    tables = {name: score(dev, p, rows, temperature) for name, p in predictions.items()}
    result = {"rows": len(rows), "excluded": len(exclude), "base": base,
              "temperature": temperature, "sets": {}}
    for name, table in tables.items():
        by_task = defaultdict(list)
        for row_id, (correct, _) in table.items():
            by_task[dev[row_id]["task"]].append(correct)
        by_language = defaultdict(list)
        for row_id, (correct, _) in table.items():
            by_language[dev[row_id]["language"]].append(correct)
        entry = {"accuracy": statistics.fmean(c for c, _ in table.values()),
                 "task_macro": statistics.fmean(statistics.fmean(v) for v in by_task.values()),
                 "mean_gold_probability": statistics.fmean(p for _, p in table.values()),
                 "task_accuracy": {k: statistics.fmean(v) for k, v in sorted(by_task.items())},
                 "language_accuracy": {k: statistics.fmean(v)
                                       for k, v in sorted(by_language.items())}}
        if name != base:
            ref, mine = predictions[base], predictions[name]
            entry["argmax_agreement_with_base"] = statistics.fmean(
                max(range(len(mine[i])), key=mine[i].__getitem__)
                == max(range(len(ref[i])), key=ref[i].__getitem__) for i in rows)
            entry["agreement_with_base"] = agreement(ref, mine, rows, temperature, margin)
            entry["minus_base"] = entry["accuracy"] - statistics.fmean(
                c for c, _ in tables[base].values())
            entry["minus_base_95ci"] = paired_bootstrap(dev, tables[base], table)
        result["sets"][name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--base", required=True, help="prediction set the others are paired with")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="receipt with calibration ids, or a JSON list of ids")
    parser.add_argument("--temperature", type=float, default=1.1489)
    parser.add_argument("--near-tie-margin", type=float, default=0.1)
    parser.add_argument("--pair", action="append", default=[], metavar="NAME=A:B",
                        help="agreement of prediction file A with reference B (same rows)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dev = {r["id"]: r for r in map(json.loads, args.dev_rows.open())}
    predictions = {}
    for pair in args.pred:
        name, _, path = pair.partition("=")
        predictions[name] = load_predictions(Path(path))
    if args.base not in predictions:
        raise SystemExit(f"--base {args.base} is not one of the --pred names")
    exclude = excluded_ids(args.exclude)
    result = compare(dev, predictions, args.base, exclude, args.temperature,
                     args.near_tie_margin)
    result["pairs"] = {}
    for spec in args.pair:
        name, _, paths = spec.partition("=")
        first, _, second = paths.partition(":")
        if not (name and first and second):
            raise SystemExit(f"--pair takes NAME=A:B, not {spec!r}")
        result["pairs"][name] = compare_pair(dev, load_predictions(Path(first)),
                                             load_predictions(Path(second)), exclude,
                                             args.temperature, args.near_tie_margin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: (round(v["accuracy"], 4), round(v["task_macro"], 4),
                          v.get("minus_base_95ci")) for k, v in result["sets"].items()}))
    print(json.dumps({k: (round(v["argmax_agreement"], 4), v["near_tie_rows"],
                          v["near_tie_agreement"]) for k, v in result["pairs"].items()}))


if __name__ == "__main__":
    main()
