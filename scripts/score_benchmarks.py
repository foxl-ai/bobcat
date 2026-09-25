"""Score the external benchmark set (`korean_bench`) for several arms.

Inputs: the benchmark rows, their compiled inputs, and per-arm logits written by
`student_train --eval-only`. Rows that failed to compile or have no logits count as wrong.
Reports per benchmark and sub-task: accuracy, chance rate, NLL and 10-bin ECE raw and at
each arm's calibration temperature (fitted on the product-eval calibration split, never on
these benchmarks), and a paired cluster-bootstrap difference between the first two arms.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def softmax(values, temperature):
    z = np.asarray(values, dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    return z / z.sum()


def ece(confidence, correct, bins=10):
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def arm_rows(rows, compiled, logits, temperature):
    """Per-row (correct, nll, top probability) at T=1 and at `temperature`."""
    out = {}
    for row in rows:
        record, scored = compiled.get(row["id"]), logits.get(row["id"])
        if record is None or scored is None:
            out[row["id"]] = None
            continue
        entry = {}
        for name, t in (("raw", 1.0), ("calibrated", temperature)):
            p = softmax(scored["logits"], t)
            target = record["target"]
            entry[name] = (int(np.argmax(p)) == target, float(-np.log(max(p[target], 1e-12))),
                           float(p.max()))
        out[row["id"]] = entry
    return out


def summarize(rows, scored):
    correct = [bool(scored[r["id"]] and scored[r["id"]]["raw"][0]) for r in rows]
    present = [scored[r["id"]] for r in rows if scored[r["id"]]]
    result = {"n": len(rows), "failed": len(rows) - len(present),
              "accuracy": float(np.mean(correct)) if rows else None,
              "chance": float(np.mean([1 / len(r["candidate_ids"]) for r in rows]))}
    for name in ("raw", "calibrated"):
        if present:
            result[f"nll_{name}"] = float(np.mean([s[name][1] for s in present]))
            result[f"ece_{name}"] = ece([s[name][2] for s in present],
                                        [s[name][0] for s in present])
    return result


def paired(rows, a, b, samples=1000, seed=7):
    groups = defaultdict(list)
    for row in rows:
        ok_a = bool(a[row["id"]] and a[row["id"]]["raw"][0])
        ok_b = bool(b[row["id"]] and b[row["id"]]["raw"][0])
        groups[row["group_id"]].append(ok_a - ok_b)
    sums = np.array([sum(v) for v in groups.values()], dtype=float)
    counts = np.array([len(v) for v in groups.values()], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(samples, len(sums)))
    boots = np.sort(sums[picks].sum(1) / counts[picks].sum(1))
    return [float(sums.sum() / counts.sum()), float(boots[int(0.025 * samples)]),
            float(boots[int(0.975 * samples) - 1])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True,
                        help="name=logits.jsonl:temperature (first two are compared)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.open()]
    compiled = {r["id"]: r for r in map(json.loads, args.compiled.open())}
    arms = {}
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        path, temperature = rest.rsplit(":", 1)
        logits = {r["id"]: r for r in map(json.loads, Path(path).open())}
        arms[name] = arm_rows(rows, compiled, logits, float(temperature))
    by_task, by_family = defaultdict(list), defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
        by_family[row["task"], row["family"]].append(row)
    names = list(arms)
    result = {"arms": {n: s for n, s in zip(names, args.arm, strict=True)}, "benchmarks": {}}
    for task, items in sorted(by_task.items()):
        entry = {"language": items[0]["language"],
                 **{n: summarize(items, arms[n]) for n in names}, "families": {}}
        if len(names) >= 2:
            entry[f"{names[0]}_minus_{names[1]}"] = paired(items, arms[names[0]], arms[names[1]])
        for (t, family), members in sorted(by_family.items()):
            if t == task:
                entry["families"][family] = {n: summarize(members, arms[n])["accuracy"]
                                             for n in names} | {"n": len(members)}
        if len(entry["families"]) > 1:
            entry["family_macro"] = {n: float(np.mean([f[n] for f in entry["families"].values()]))
                                     for n in names}
        result["benchmarks"][task] = entry
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for task, entry in result["benchmarks"].items():
        cells = " | ".join(f"{n} {entry[n]['accuracy']:.1%}" for n in names)
        delta = entry.get(f"{names[0]}_minus_{names[1]}") if len(names) >= 2 else None
        tail = f" | Δ {delta[0]*100:+.1f} [{delta[1]*100:+.1f}, {delta[2]*100:+.1f}]" if delta \
            else ""
        print(f"{task} ({entry['language']}, n={entry[names[0]]['n']}, chance "
              f"{entry[names[0]]['chance']:.1%}) | {cells}{tail}")


if __name__ == "__main__":
    main()
