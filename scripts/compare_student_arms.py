"""Compare trained student arms against the same-input zero-shot baseline (PLAN step 3).

Reads each arm's scored dev rows, calibration result and monitor metrics. Reports task
macro accuracy (failures count as wrong), per-task accuracy, the held-out tool-call task,
candidate-count families, NLL/ECE raw and after the calibration-fitted temperature, and a
paired component-bootstrap difference against the baseline.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

TASKS = ("product_search", "product_citation", "product_tool_call", "product_injection",
         "product_routing", "product_classification")


def rows(arm: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, (arm / "scored-dev" / "rows.jsonl").open())}


def macro(table: dict, ids, tasks=TASKS) -> float:
    groups = defaultdict(list)
    for i in ids:
        if table[i]["task"] in tasks:
            groups[table[i]["task"]].append(table[i]["correct"])
    return sum(sum(v) / len(v) for v in groups.values()) / len(groups)


def paired(a: dict, b: dict, seed: int = 7, samples: int = 1000, tasks=TASKS):
    common = sorted(set(a) & set(b))
    groups = defaultdict(list)
    for i in common:
        if a[i]["task"] in tasks:
            groups[a[i]["group_id"]].append(i)
    keys = list(groups)
    rng = random.Random(seed)

    def diff(sample):
        chosen = [i for g in sample for i in groups[g]]
        return macro(a, chosen, tasks) - macro(b, chosen, tasks)

    boots = sorted(diff([rng.choice(keys) for _ in keys]) for _ in range(samples))
    return diff(keys), boots[int(0.025 * samples)], boots[int(0.975 * samples) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", default="zero_shot_piecewise")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    base = rows(args.root / args.baseline)
    table = []
    for name in [args.baseline, *args.arms]:
        folder = args.root / name
        if not (folder / "scored-dev" / "rows.jsonl").exists():
            continue
        r = rows(folder)
        calibration = json.loads((folder / "calibration.json").read_text())
        monitor = json.loads((folder / "monitor.json").read_text())
        summary = json.loads((folder / "scored-dev" / "summary.json").read_text())
        families = summary["families"]
        entry = {
            "arm": name, "macro": macro(r, r), "macro_trained_tasks": macro(
                r, r, tuple(t for t in TASKS if t != "product_tool_call")),
            "tasks": {t.removeprefix("product_"): summary["tasks"][t]["accuracy"]
                      for t in TASKS if t in summary["tasks"]},
            "title_k": {f.rsplit("_", 1)[1]: families[f]["accuracy"] for f in families
                        if f.startswith("document_title_k")},
            "nll": calibration["dev"]["raw"]["nll"],
            "ece": calibration["dev"]["raw"]["ece_10_equal_width_bins"],
            "temperature": calibration["temperature"],
            "nll_calibrated": calibration["dev"]["calibrated"]["nll"],
            "ece_calibrated": calibration["dev"]["calibrated"]["ece_10_equal_width_bins"],
            "monitor_accuracy": monitor["monitor_hard_label_accuracy"],
            "monitor_score_nmae": monitor["monitor_score_normalized_mae"],
            "counterfactual_all_correct": summary["counterfactual_all_correct"],
        }
        if name != args.baseline:
            entry["vs_baseline"] = paired(r, base)
            entry["vs_baseline_trained_tasks"] = paired(
                r, base, tasks=tuple(t for t in TASKS if t != "product_tool_call"))
            entry["tool_call_vs_baseline"] = paired(r, base, tasks=("product_tool_call",))
        table.append(entry)
    head = ("| arm | macro | Δ vs zero-shot [95%] | tool-call (held out) | NLL | ECE | "
            "T | ECE@T | monitor | K255 |")
    print(head)
    print("|" + "---|" * 10)
    for e in table:
        delta = e.get("vs_baseline")
        text = f"{delta[0]*100:+.2f} [{delta[1]*100:+.2f}, {delta[2]*100:+.2f}]" if delta else "—"
        print(f"| {e['arm']} | {e['macro']:.1%} | {text} | {e['tasks'].get('tool_call', 0):.1%} | "
              f"{e['nll']:.3f} | {e['ece']:.3f} | {e['temperature']:.2f} | "
              f"{e['ece_calibrated']:.3f} | {e['monitor_accuracy']:.1%} | "
              f"{e['title_k'].get('k255', 0):.1%} |")
    if args.json:
        args.json.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
