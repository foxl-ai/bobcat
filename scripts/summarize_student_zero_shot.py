"""Tabulate student zero-shot summaries (PLAN step 2) as Markdown and JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ("search", "citation", "tool_call", "injection", "routing", "classification")


def load(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        for path in sorted(root.glob("*/summary.json")):
            summary = json.loads(path.read_text())
            tasks = summary["tasks"]
            rows.append({
                "repo": summary["repo"], "revision": summary["revision"],
                "macro": summary["task_macro_accuracy_including_failures"],
                "accuracy": summary["overall"]["accuracy"],
                "nll": summary["overall"]["nll"],
                "ece": summary["overall"]["ece_10_equal_width_bins"],
                "counterfactual_all_correct": summary["counterfactual_all_correct"],
                "failed_rows": summary["failed_rows"], "scored_rows": summary["scored_rows"],
                "median_seconds": summary["seconds_per_row_median"],
                "input_tokens": summary["input_tokens"]["total"],
                "identifiers_shared_with_glm_prefix": summary.get(
                    "identifiers_shared_with_glm_prefix", 255),
                "gpus": summary["environment"].get("visible_gpus", 1),
                "gpu": summary["environment"]["gpu"],
                "tasks": {name: tasks.get(f"product_{name}", {}).get(
                    "accuracy_including_failures") for name in TASKS},
                "title_k": {f.rsplit("_", 1)[1]: v["accuracy"]
                            for f, v in summary["families"].items()
                            if f.startswith("document_title_k")},
            })
    return sorted(rows, key=lambda r: -r["macro"])


def markdown(rows: list[dict]) -> str:
    head = ("| student | macro | " + " | ".join(TASKS) +
            " | NLL | ECE | 반사실 | s/행 | 식별자 공유 |")
    lines = [head, "|" + "---|" * (len(TASKS) + 7)]
    for r in rows:
        cells = [f"{r['macro']:.1%}", *(f"{r['tasks'][t]:.1%}" for t in TASKS),
                 f"{r['nll']:.3f}", f"{r['ece']:.3f}", f"{r['counterfactual_all_correct']:.1%}",
                 f"{r['median_seconds']:.3f}", str(r["identifiers_shared_with_glm_prefix"])]
        lines.append(f"| `{r['repo']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = load(args.roots)
    print(markdown(rows))
    for r in rows:
        print(f"- {r['repo']}: title K " + ", ".join(
            f"{k}={v:.1%}" for k, v in sorted(r["title_k"].items(), key=lambda kv: int(kv[0][1:]))))
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
