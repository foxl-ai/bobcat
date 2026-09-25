"""Fit one scalar temperature on the product-eval calibration split; apply it to dev.

Inputs are scored rows (`student_readout score` output). The calibration split plays the
`cal_temperature` role: it alone fits T; dev is only transformed. Final stays sealed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bobcat.metrics import basic_metrics, fit_temperature, scored_row


def load(path: Path, split: str) -> list[dict]:
    rows = []
    for line in path.open():
        row = json.loads(line)
        rows.append({"id": row["id"], "task": row["task"], "group_id": row["group_id"],
                     "candidate_ids": row["candidate_ids"], "target": row["target"],
                     "supervision": "hard_label", "logits": row["logits"],
                     "tie_break": "request_order", "split": split})
    return rows


def report(rows: list[dict], temperature: float) -> dict:
    scored = [scored_row(row, temperature) for row in rows]
    tasks = defaultdict(list)
    for row in scored:
        tasks[row["task"]].append(row["correct"])
    metrics = {k: v for k, v in basic_metrics(scored).items() if k != "calibration_bins"}
    metrics["task_macro_accuracy"] = sum(sum(v) / len(v) for v in tasks.values()) / len(tasks)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    calibration = load(args.calibration, "cal_temperature")
    dev = load(args.dev, "dev")
    temperature = fit_temperature(calibration)
    result = {
        "temperature": temperature, "fitted_on": str(args.calibration),
        "calibration": {"raw": report(calibration, 1.0), "fitted": report(calibration,
                                                                           temperature)},
        "dev": {"raw": report(dev, 1.0), "calibrated": report(dev, temperature)},
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"temperature": round(temperature, 4),
                      "dev_ece": [round(result["dev"][k]["ece_10_equal_width_bins"], 4)
                                  for k in ("raw", "calibrated")],
                      "dev_nll": [round(result["dev"][k]["nll"], 4) for k in ("raw",
                                                                              "calibrated")]}))


if __name__ == "__main__":
    main()
