"""Bobcat Flash analysis helpers (2026-09-26).

  monitor  accuracy on held-out monitor rows (gold where it exists) by task, language and
           source, plus agreement with the teacher (argmax match, mean KL(teacher||student)
           at the teacher temperature) on every row, gold or teacher-only.
  cascade  offline Flash -> Bobcat 1 routing on dev: questions whose input features are out
           of the Flash training distribution (script, length, candidate count) or whose
           calibrated Flash confidence is below a threshold go to Bobcat 1. Thresholds are
           chosen on calibration and applied to dev; reports coverage, accuracy and the
           share of prefill compute that stays on Flash.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

T_TEACHER = 1.1488982760285609


def softmax(values, temperature=1.0):
    peak = max(values)
    exps = [math.exp((v - peak) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def read(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def monitor(pairs, out: Path) -> dict:
    report = {}
    for logits_path, compiled_path, teacher_path in pairs:
        student = {r["id"]: r["logits"] for r in read(logits_path)}
        teacher = ({r["id"]: r["logits"] for r in read(teacher_path)}
                   if teacher_path and teacher_path.exists() else {})
        groups = defaultdict(lambda: {"gold_rows": 0, "gold_correct": 0, "teacher_rows": 0,
                                      "teacher_agree": 0, "kl_sum": 0.0,
                                      "teacher_gold_correct": 0})
        for row in read(compiled_path):
            z = student.get(row["id"])
            if z is None:
                continue
            keys = ["all", f"language:{row['language']}", f"task:{row['task']}",
                    f"source:{row.get('source')}"]
            t = teacher.get(row["id"])
            for key in keys:
                g = groups[key]
                if row["supervision"] == "hard_label" and row.get("target") is not None:
                    g["gold_rows"] += 1
                    g["gold_correct"] += argmax(z) == row["target"]
                    if t is not None and len(t) == len(z):
                        g["teacher_gold_correct"] += argmax(t) == row["target"]
                if t is not None and len(t) == len(z):
                    p = softmax(t, T_TEACHER)
                    q = softmax(z)
                    g["teacher_rows"] += 1
                    g["teacher_agree"] += argmax(t) == argmax(z)
                    g["kl_sum"] += sum(pi * (math.log(pi + 1e-12) - math.log(qi + 1e-12))
                                       for pi, qi in zip(p, q, strict=True))
        summary = {}
        for key, g in sorted(groups.items()):
            summary[key] = {
                "gold_rows": g["gold_rows"],
                "accuracy": g["gold_correct"] / g["gold_rows"] if g["gold_rows"] else None,
                "teacher_accuracy": (g["teacher_gold_correct"] / g["gold_rows"]
                                     if g["gold_rows"] and teacher else None),
                "teacher_rows": g["teacher_rows"],
                "teacher_argmax_agreement": (g["teacher_agree"] / g["teacher_rows"]
                                             if g["teacher_rows"] else None),
                "mean_kl_teacher_student": (g["kl_sum"] / g["teacher_rows"]
                                            if g["teacher_rows"] else None)}
        report[logits_path.stem.replace("logits-", "")] = summary
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v.get("all") for k, v in report.items()}))
    return report


HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")
LATIN = re.compile(r"[A-Za-z]")


def input_features(request_row: dict) -> dict:
    """Features known before any forward pass."""
    text = json.dumps(request_row["request"], ensure_ascii=False)
    letters = [c for c in text if c.isalpha()]
    known = sum(1 for c in letters if HANGUL.match(c) or LATIN.match(c))
    return {"known_script_share": known / max(1, len(letters)), "chars": len(text),
            "candidates": len(request_row["candidate_ids"])}


def cascade(flash_dev: Path, flash_cal: Path, big_dev: Path, rows_dev: Path, rows_cal: Path,
            t_flash: float, t_big: float, out: Path, max_candidates: int,
            flash_cost: float) -> dict:
    """Offline routing study; `flash_cost` = Flash prefill cost relative to Bobcat 1 per token."""
    def load_scored(path):
        return {r["id"]: r for r in read(path)}

    fdev, fcal = load_scored(flash_dev), load_scored(flash_cal)
    # Bobcat 1 evaluation-path logits ({id, logits}); targets come from the scored Flash rows.
    bdev = {}
    for r in read(big_dev):
        if r["id"] in fdev:
            bdev[r["id"]] = {**fdev[r["id"]], "logits": r["logits"]}
    meta = {r["id"]: r for r in read(rows_dev)}
    meta.update({r["id"]: r for r in read(rows_cal)})

    def ood(row_id):
        f = input_features(meta[row_id])
        return f["known_script_share"] < 0.5 or f["candidates"] > max_candidates

    def conf(row, temperature):
        return max(softmax(row["logits"], temperature))

    def correct(row):
        return argmax(row["logits"]) == row["candidate_ids"].index(row["target"])

    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    cal_rows = [r for r in fcal.values()]
    table = []
    for threshold in thresholds:
        kept = [r for r in cal_rows if not ood(r["id"]) and conf(r, t_flash) >= threshold]
        table.append({"threshold": threshold, "coverage": len(kept) / len(cal_rows),
                      "flash_accuracy_on_kept": (sum(correct(r) for r in kept) / len(kept)
                                                 if kept else None)})
    results = []
    for threshold in thresholds:
        tasks = defaultdict(list)
        kept = 0
        for row_id, row in fdev.items():
            use_flash = not ood(row_id) and conf(row, t_flash) >= threshold
            source = row if use_flash else bdev.get(row_id, row)
            kept += use_flash
            tasks[row["task"]].append(correct(source))
        macro = sum(sum(v) / len(v) for v in tasks.values()) / len(tasks)
        coverage = kept / len(fdev)
        results.append({"threshold": threshold, "flash_coverage": coverage,
                        "task_macro": macro,
                        "relative_compute": coverage * flash_cost + (1 - coverage) * (
                            1 + flash_cost)})
    flash_only = defaultdict(list)
    big_only = defaultdict(list)
    for row_id, row in fdev.items():
        flash_only[row["task"]].append(correct(row))
        if row_id in bdev:
            big_only[row["task"]].append(correct(bdev[row_id]))
    report = {"calibration_table": table, "dev_cascade": results,
              "flash_only_macro": sum(sum(v) / len(v) for v in flash_only.values())
              / len(flash_only),
              "bobcat1_only_macro": sum(sum(v) / len(v) for v in big_only.values())
              / max(1, len(big_only)),
              "ood_rule": f"known-script share < 0.5 or candidates > {max_candidates}",
              "note": "relative_compute counts a routed question as Flash + Bobcat 1 prefill; "
                      "thresholds listed for calibration, dev shown for each"}
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("monitor")
    m.add_argument("--logits", type=Path, action="append", required=True)
    m.add_argument("--compiled", type=Path, action="append", required=True)
    m.add_argument("--teacher", type=Path, action="append", default=[])
    m.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("cascade")
    for name in ("flash-dev", "flash-cal", "big-dev", "rows-dev", "rows-cal", "out"):
        c.add_argument(f"--{name}", type=Path, required=True)
    c.add_argument("--t-flash", type=float, required=True)
    c.add_argument("--t-big", type=float, default=T_TEACHER)
    c.add_argument("--max-candidates", type=int, default=64)
    c.add_argument("--flash-cost", type=float, default=0.15)
    args = parser.parse_args()
    if args.command == "monitor":
        teachers = args.teacher + [None] * (len(args.logits) - len(args.teacher))
        monitor(list(zip(args.logits, args.compiled, teachers, strict=True)), args.out)
    else:
        cascade(args.flash_dev, args.flash_cal, args.big_dev, args.rows_dev, args.rows_cal,
                args.t_flash, args.t_big, args.out, args.max_candidates, args.flash_cost)


if __name__ == "__main__":
    main()
