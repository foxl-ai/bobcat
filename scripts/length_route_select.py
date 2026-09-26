"""Choose the route server's length threshold T on DEV only (pre-registered rule).

For every candidate T, each question is routed exactly as `bobcat.route_server` does, from
saved engine logits of the two served builds: out of range (more than 64 candidates, or
mostly neither Hangul nor Latin letters) -> Bobcat; Flash-compiled length > T -> Bobcat;
Flash calibrated top probability < 0.8 -> Bobcat; otherwise Flash. The same
`RoutePolicy.out_of_range` code decides the first two, over the padded request itself.

Levels: the padded dev questions of a `scripts/long_context_eval.py build --write-rows` folder
(one set of 300 questions at several state lengths; accuracy), plus optionally the whole dev
split unpadded (task macro). Missing logits count as wrong. Rule (fixed before any logits
existed, .aws-local/flashlc-20260926-preregistration.json): T qualifies when, at every
selection level, routed quality >= Bobcat-alone quality - margin (point estimates); the
largest qualifying T is chosen (most questions answered by Flash). A holdout folder is scored
the same way and only reported.

    python scripts/length_route_select.py --dev-rows dev.jsonl \
        --set canonical=longctx:flash-logits:bobcat-logits \
        --holdout holdout=longctx-h:flash-logits-h:bobcat-logits-h \
        --full-dev flash-dev.jsonl:bobcat-dev.jsonl:flash-eval-dev.compiled.jsonl \
        --levels 0,1024,2048,3072,4096,6144,8192 --out selection.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.long_context_eval import paired_bootstrap

FLASH_T = 0.8911980656311957
BOBCAT_T = 1.200767737961694
CANDIDATES = (1024, 2048, 3072, 4096, 6144, 8192, None)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def top_probability(logits, temperature):
    peak = max(logits)
    weights = [math.exp((v - peak) / temperature) for v in logits]
    return max(weights) / sum(weights)


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def load_logits(path: Path) -> dict[str, list[float]]:
    return {r["id"]: r["logits"] for r in read_jsonl(path)} if path.exists() else {}


def load_lengths(path: Path) -> dict[str, int]:
    out = {}
    for line in path.open():
        row = json.loads(line)
        out[row["id"]] = row.get("tokens") or len(row["input_ids"])
    return out


class Level:
    """One set of questions at one state length: rows (padded requests), Flash lengths and
    both models' logits."""

    def __init__(self, name, rows, lengths, flash, bobcat, metric):
        self.name, self.rows, self.lengths = name, rows, lengths
        self.flash, self.bobcat, self.metric = flash, bobcat, metric
        self.features = {}
        from bobcat.protocol import parse_request
        from bobcat.route_server import RoutePolicy

        base = RoutePolicy(max_flash_tokens=None)
        for row in rows:
            state, (question,) = parse_request(row["request"])
            self.features[row["id"]] = base.out_of_range(state, question)

    def route(self, row, limit):
        """(model, reason) for one question under length threshold `limit`."""
        rid = row["id"]
        reason = self.features[rid]
        if reason is None and limit is not None and self.lengths.get(rid, 0) > limit:
            reason = "length"
        if reason is not None:
            return ("bobcat", reason) if rid in self.bobcat else ("flash", "bobcat_limit")
        values = self.flash.get(rid)
        if values is not None and top_probability(values, FLASH_T) < 0.8 and rid in self.bobcat:
            return "bobcat", "low_confidence"
        return "flash", "in_range"

    def outcome(self, limit, mode="auto"):
        """{id: (correct, 0.0)}, Flash share and reasons for a mode."""
        table, reasons = {}, Counter()
        flash_answers = 0
        for row in self.rows:
            rid = row["id"]
            model, reason = (("flash", "forced") if mode == "flash" else
                             ("bobcat", "forced") if mode == "bobcat" else self.route(row, limit))
            reasons[reason] += 1
            values = (self.flash if model == "flash" else self.bobcat).get(rid)
            flash_answers += model == "flash"
            gold = row["candidate_ids"].index(row["target"])
            table[rid] = (values is not None and argmax(values) == gold, 0.0)
        return table, flash_answers / len(self.rows), dict(reasons)

    def quality(self, table):
        by_task = defaultdict(list)
        for row in self.rows:
            by_task[row["task"]].append(table[row["id"]][0])
        macro = statistics.fmean(statistics.fmean(v) for v in by_task.values())
        accuracy = statistics.fmean(c for c, _ in table.values())
        return {"accuracy": accuracy, "task_macro": macro,
                "task_accuracy": {k: statistics.fmean(v) for k, v in sorted(by_task.items())},
                "headline": macro if self.metric == "task_macro" else accuracy}


def evaluate(levels: list[Level], candidates, margin: float, dev_by_id: dict) -> dict:
    out = {}
    for limit in candidates:
        key = str(limit) if limit is not None else "none"
        per_level, ok = {}, True
        for level in levels:
            routed, share, reasons = level.outcome(limit)
            bobcat, _, _ = level.outcome(limit, "bobcat")
            flash, _, _ = level.outcome(limit, "flash")
            q_routed, q_bobcat, q_flash = (level.quality(t) for t in (routed, bobcat, flash))
            gap = q_routed["headline"] - q_bobcat["headline"]
            meets = gap >= -margin / 100 - 1e-12
            ok &= meets
            per_level[level.name] = {
                "metric": level.metric, "rows": len(level.rows),
                "routed": q_routed["headline"], "bobcat": q_bobcat["headline"],
                "flash": q_flash["headline"], "routed_minus_bobcat": gap,
                "meets": meets, "flash_share": share, "reasons": reasons,
                "routed_task_accuracy": q_routed["task_accuracy"],
                "routed_minus_bobcat_accuracy_ci95": paired_bootstrap(
                    dev_by_id, bobcat, routed, draws=2000)}
        out[key] = {"T": limit, "qualifies": ok, "levels": per_level,
                    "mean_flash_share": statistics.fmean(v["flash_share"]
                                                         for v in per_level.values()),
                    "mean_routed": statistics.fmean(v["routed"] for v in per_level.values())}
    return out


def choose(table: dict, candidates) -> dict:
    qualifying = [c for c in candidates if table[str(c) if c is not None else "none"]
                  ["qualifies"]]
    if not qualifying:
        return {"T": candidates[0], "qualified": False,
                "note": "no candidate met the constraint; the smallest candidate is used"}
    # `None` (no length rule) is the largest threshold.
    best = max(qualifying, key=lambda c: math.inf if c is None else c)
    return {"T": best, "qualified": True, "qualifying": qualifying}


def build_levels(spec: str, level_names, dev_by_id) -> list[Level]:
    name, folder, flash_dir, bobcat_dir = spec.split("=", 1)[0], *spec.split("=", 1)[1].split(":")
    folder, flash_dir, bobcat_dir = Path(folder), Path(flash_dir), Path(bobcat_dir)
    levels = []
    for level in level_names:
        rows_path = folder / f"level-{level}.rows.jsonl"
        if not rows_path.exists():
            continue
        rows = read_jsonl(rows_path)
        lengths = load_lengths(folder / f"level-{level}.compiled.jsonl")
        levels.append(Level(f"{name}:{level}", rows, lengths,
                            load_logits(flash_dir / f"level-{level}.logits.jsonl"),
                            load_logits(bobcat_dir / f"level-{level}.logits.jsonl"),
                            "accuracy"))
    return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--set", required=True,
                        help="name=build-folder:flash-logits-folder:bobcat-logits-folder")
    parser.add_argument("--holdout", help="the same form; reported only")
    parser.add_argument("--full-dev",
                        help="flash-predictions:bobcat-predictions:flash-compiled (unpadded dev)")
    parser.add_argument("--levels", default="0,1024,2048,3072,4096,6144,8192")
    parser.add_argument("--candidates", default="1024,2048,3072,4096,6144,8192,none")
    parser.add_argument("--margin-pt", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dev = read_jsonl(args.dev_rows)
    if any(r.get("split") not in (None, "dev") for r in dev):
        raise SystemExit("Only the dev split is used here.")
    dev_by_id = {r["id"]: r for r in dev}
    names = [int(x) for x in args.levels.split(",")]
    candidates = [None if c == "none" else int(c) for c in args.candidates.split(",")]
    levels = build_levels(args.set, names, dev_by_id)
    if args.full_dev:
        flash_path, bobcat_path, compiled = (Path(p) for p in args.full_dev.split(":"))
        levels.append(Level("dev_full:0", dev, load_lengths(compiled), load_logits(flash_path),
                            load_logits(bobcat_path), "task_macro"))
    table = evaluate(levels, candidates, args.margin_pt, dev_by_id)
    choice = choose(table, candidates)
    result = {"schema": "bobcat-length-route-selection-v1", "flash_temperature": FLASH_T,
              "bobcat_temperature": BOBCAT_T, "confidence_threshold": 0.8,
              "margin_pt": args.margin_pt, "levels": [lv.name for lv in levels],
              "missing_logits": {lv.name: {"flash": sum(r["id"] not in lv.flash for r in lv.rows),
                                           "bobcat": sum(r["id"] not in lv.bobcat
                                                         for r in lv.rows)} for lv in levels},
              "candidates": table, "choice": choice}
    if args.holdout:
        held = build_levels(args.holdout, names, dev_by_id)
        chosen = {r["id"] for lv in levels if not lv.name.startswith("dev_full")
                  for r in lv.rows}
        overlap = {r["id"] for lv in held for r in lv.rows} & chosen
        if overlap:
            raise SystemExit(f"The holdout shares {len(overlap)} questions with the selection set.")
        result["holdout"] = evaluate(held, [choice["T"]], args.margin_pt, dev_by_id)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    brief = {k: (v["qualifies"], round(v["mean_flash_share"], 3),
                 {n: round(x["routed_minus_bobcat"] * 100, 2) for n, x in v["levels"].items()})
             for k, v in table.items()}
    print(json.dumps({"choice": choice, "table": brief}, indent=1))


if __name__ == "__main__":
    main()
