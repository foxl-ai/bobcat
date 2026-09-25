"""Report the one sealed-final evaluation of a frozen release (PLAN step 4).

Inputs are `student_readout score --release-manifest` outputs on the final split for the
release and for the same-input zero-shot baseline. Each model keeps the temperature the
frozen manifest records (fitted on calibration before final was opened); nothing is refit
on final. Reports task macro accuracy with failures counted wrong, per-task accuracy,
NLL/ECE raw and at the frozen temperature, counterfactual consistency and the paired
component-bootstrap difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_student import load, report
from compare_student_arms import TASKS, paired


def arm(folder: Path, temperature: float) -> dict:
    summary = json.loads((folder / "summary.json").read_text())
    rows = load(folder / "rows.jsonl", "final")
    return {
        "scored_rows": summary["scored_rows"], "failed_rows": summary["failed_rows"],
        "task_macro_accuracy_including_failures":
            summary["task_macro_accuracy_including_failures"],
        "tasks": {t.removeprefix("product_"): summary["tasks"][t]["accuracy_including_failures"]
                  for t in TASKS if t in summary["tasks"]},
        "counterfactual_all_correct": summary["counterfactual_all_correct"],
        "temperature": temperature, "raw": report(rows, 1.0),
        "calibrated": report(rows, temperature),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True, help="scored-final folder")
    parser.add_argument("--baseline", type=Path, required=True, help="scored-final folder")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.release_manifest.read_text())
    if frozen.get("status") != "frozen":
        raise ValueError("Score final only for a frozen release manifest.")
    release = arm(args.release, frozen["calibration"]["temperature"])
    baseline = arm(args.baseline, frozen["final_evaluation"]["baseline_temperature"])
    rows = {name: {r["id"]: r for r in map(json.loads, (folder / "rows.jsonl").open())}
            for name, folder in (("release", args.release), ("baseline", args.baseline))}
    result = {
        "schema": "bobcat-final-evaluation-v1", "release": frozen["name"],
        "release_manifest": str(args.release_manifest),
        "split": "product-eval v2 final", "data_sha256": frozen["final_evaluation"]["data_sha256"],
        "evaluated_once": True, "settings_changed_after_final": False,
        "release_metrics": release, "baseline_metrics": baseline,
        "release_minus_baseline": paired(rows["release"], rows["baseline"]),
        "release_minus_baseline_tool_call": paired(rows["release"], rows["baseline"],
                                                   tasks=("product_tool_call",)),
        "note": "Scored rows exclude over-limit failures; paired differences use scored rows, "
                "the headline macro counts failures as wrong.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("release_minus_baseline",
                                             "release_minus_baseline_tool_call")}))
    print(json.dumps({"release": release["task_macro_accuracy_including_failures"],
                      "baseline": baseline["task_macro_accuracy_including_failures"],
                      "release_tasks": release["tasks"]}))


if __name__ == "__main__":
    main()
