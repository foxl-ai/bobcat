"""Product-eval dev through a running `bobcat.route_server`, over HTTP, one request at a time.

Every dev row is one single-question request. For each `--modes` entry the whole dev split is
sent once: `auto` (the server's routing), and, when the server runs with
`--allow-route-override`, `flash` / `bobcat` (one model for every question, the same engines
and settings). Records per request: status, client and engine time, the model that answered
(`x-bobcat-route`) and why (`x-bobcat-route-reasons`), and the answer's probabilities in the
row's candidate order (written as log-probabilities, so `scripts/quant_drift.py` can pair the
runs). Summary per mode: task accuracy and macro (failed requests count as wrong), the share
Flash answered, latency by route; paired component-bootstrap intervals between modes; and the
agreement of the served out-of-range rule with the offline one (`bobcat.flash_report`).
Nothing here trains, calibrates or selects. The final split is never read.

    python scripts/route_dev_eval.py --url http://127.0.0.1:8100/v1/systemone \
        --dev-rows dev.jsonl --modes auto,flash,bobcat --out runs/route-dev
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from scripts.long_context_eval import paired_bootstrap


def post(url: str, body: bytes, headers: dict) -> tuple[int, dict, dict, float]:
    request = urllib.request.Request(url, data=body, method="POST", headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as reply:
            status, payload, got = reply.status, json.loads(reply.read()), dict(reply.headers)
    except urllib.error.HTTPError as error:
        status, payload, got = error.code, {}, dict(error.headers)
    return status, payload, {k.lower(): v for k, v in got.items()}, (
        time.perf_counter() - started) * 1000


def distribution(row: dict, payload: dict) -> list[float] | None:
    """The answer's probabilities in the row's candidate order, or None."""
    answers = payload.get("answers") or {}
    (qid,) = row["request"]["questions"]
    item = answers.get(qid)
    if not item:
        return None
    if item["type"] == "noul":
        by_label = {"no": 1 - item["noul"], "yes": item["noul"]}
    else:
        by_label = {str(k): v for k, v in item["probabilities"].items()}
    try:
        return [by_label[c] for c in row["candidate_ids"]]
    except KeyError:
        return None


def correct(row: dict, probs: list[float] | None) -> bool:
    if probs is None:
        return False
    pick = max(range(len(probs)), key=probs.__getitem__)
    return row["candidate_ids"][pick] == row["target"]


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))] if values else None


def latency(records) -> dict:
    client = [r["client_ms"] for r in records]
    engine = [r["engine_ms"] for r in records if r["engine_ms"] is not None]
    return {"requests": len(records),
            "client_p50_ms": percentile(client, 0.5), "client_p95_ms": percentile(client, 0.95),
            "client_mean_ms": statistics.fmean(client) if client else None,
            "engine_p50_ms": percentile(engine, 0.5), "engine_p95_ms": percentile(engine, 0.95)}


def summarize(rows: dict, records: list[dict]) -> dict:
    by_task = defaultdict(list)
    for record in records:
        by_task[rows[record["id"]]["task"]].append(record["correct"])
    routes = defaultdict(list)
    for record in records:
        routes[record["route"] or "failed"].append(record)
    reasons = defaultdict(int)
    for record in records:
        reasons[record["reason"] or "none"] += 1
    return {
        "requests": len(records), "failed": sum(r["status"] != 200 for r in records),
        "accuracy": statistics.fmean(r["correct"] for r in records),
        "task_macro": statistics.fmean(statistics.fmean(v) for v in by_task.values()),
        "task_accuracy": {k: statistics.fmean(v) for k, v in sorted(by_task.items())},
        "flash_share": len(routes.get("flash", [])) / len(records),
        "route_counts": {k: len(v) for k, v in sorted(routes.items())},
        "reason_counts": dict(sorted(reasons.items())),
        "latency_all": latency(records),
        "latency_by_route": {k: latency(v) for k, v in sorted(routes.items())},
        "latency_by_reason": {k: latency([r for r in records if (r["reason"] or "none") == k])
                              for k in sorted(reasons)},
        "accuracy_by_route": {k: statistics.fmean(r["correct"] for r in v)
                              for k, v in sorted(routes.items())}}


def macro_bootstrap(rows: dict, first: dict, second: dict, draws: int = 2000,
                    seed: int = 20260926) -> list[float]:
    """95% interval of task macro(first) - task macro(second), resampling components."""
    import random

    clusters = defaultdict(list)
    for row_id in first:
        if row_id in second:
            clusters[rows[row_id]["group_id"]].append(row_id)
    keys = sorted(clusters)
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        tasks = defaultdict(lambda: [0, 0, 0])  # first correct, second correct, rows
        for key in (rng.choice(keys) for _ in keys):
            for row_id in clusters[key]:
                cell = tasks[rows[row_id]["task"]]
                cell[0] += first[row_id][0]
                cell[1] += second[row_id][0]
                cell[2] += 1
        diffs.append(statistics.fmean((a - b) / n for a, b, n in tasks.values()))
    diffs.sort()
    return [diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]]


def offline_rule_agreement(rows: list[dict], max_candidates: int = 64) -> dict:
    """Rows the served out-of-range rule sends to Bobcat vs the offline study's rule."""
    from bobcat.flash_report import input_features
    from bobcat.protocol import parse_request
    from bobcat.route_server import RoutePolicy

    policy = RoutePolicy(max_candidates=max_candidates)
    served = offline = both = 0
    for row in rows:
        state, (question,) = parse_request(row["request"])
        mine = policy.out_of_range(state, question) is not None
        feats = input_features(row)
        theirs = feats["known_script_share"] < 0.5 or feats["candidates"] > max_candidates
        served += mine
        offline += theirs
        both += mine and theirs
    return {"rows": len(rows), "served_rule_out_of_range": served,
            "offline_rule_out_of_range": offline, "both": both,
            "same_rows": served == offline == both}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dev-rows", type=Path, required=True)
    parser.add_argument("--modes", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    secret = os.environ.get("BOBCAT_EDGE_SECRET", "")
    rows = [json.loads(line) for line in args.dev_rows.open()]
    if any(r.get("split") not in (None, "dev") for r in rows):
        raise SystemExit("Only the dev split is evaluated here.")
    rows = rows[:args.limit] if args.limit else rows
    by_id = {r["id"]: r for r in rows}
    args.out.mkdir(parents=True, exist_ok=True)
    warm = {"model": "bobcat-latest", "state": "warm-up",
            "questions": {"q": {"type": "noul", "instructions": "Is this a warm-up?"}}}
    base_headers = {"content-type": "application/json", "x-bobcat-edge-secret": secret}
    summary = {"dev_rows": str(args.dev_rows), "rows": len(rows), "url": args.url,
               "modes": {}, "offline_rule": offline_rule_agreement(rows)}
    tables = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        headers = dict(base_headers)
        if mode != "auto":
            headers["x-bobcat-route-mode"] = mode
        for _ in range(3):  # untimed warm-up of both engines' kernels
            post(args.url, json.dumps(warm).encode(), headers)
        records = []
        with (args.out / f"{mode}.responses.jsonl").open("w") as sink, \
                (args.out / f"{mode}.predictions.jsonl").open("w") as preds:
            for row in rows:
                body = json.dumps({**row["request"], "model": "bobcat-latest"},
                                  ensure_ascii=False).encode()
                status, payload, got, ms = post(args.url, body, headers)
                probs = distribution(row, payload) if status == 200 else None
                engine = got.get("x-bobcat-engine-ms")
                record = {"id": row["id"], "status": status, "client_ms": round(ms, 2),
                          "engine_ms": float(engine) if engine else None,
                          "route": got.get("x-bobcat-route"),
                          "reason": got.get("x-bobcat-route-reasons"),
                          "processed_tokens": int(got.get("x-bobcat-processed-tokens", 0)),
                          "probabilities": probs, "correct": correct(row, probs)}
                records.append(record)
                sink.write(json.dumps(record) + "\n")
                if probs is not None:
                    preds.write(json.dumps({"id": row["id"], "logits": [
                        math.log(max(p, 1e-300)) for p in probs]}) + "\n")
        summary["modes"][mode] = summarize(by_id, records)
        tables[mode] = {r["id"]: (r["correct"], 0.0) for r in records}
        print(json.dumps({mode: {k: summary["modes"][mode][k] for k in
                                 ("task_macro", "flash_share", "failed", "latency_all")}}),
              flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    names = list(tables)
    summary["paired"] = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            diff = statistics.fmean(tables[first][k][0] - tables[second][k][0]
                                    for k in tables[first])
            summary["paired"][f"{first}_minus_{second}"] = {
                "accuracy_difference": diff,
                "accuracy_ci95": paired_bootstrap(by_id, tables[second], tables[first]),
                "task_macro_difference": (summary["modes"][first]["task_macro"]
                                          - summary["modes"][second]["task_macro"]),
                "task_macro_ci95": macro_bootstrap(by_id, tables[first], tables[second])}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["paired"]))


if __name__ == "__main__":
    main()
