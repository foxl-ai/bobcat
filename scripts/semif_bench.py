"""Bobcat on the SemIf (formerly openjev) benchmark bundle, with SemIf's own metrics.

SemIf (TheoLeeCJ/SemIf-OpenJev, MIT) publishes fixtures, row selections and evaluators for
typed decisions from open models, plus the Jev figure TypeSafe released for 102 aligned rows
of its public workflow cases. This script turns SemIf's rows into Bobcat /v1/systemone
requests, runs them against a Bobcat server, and writes predictions in SemIf's row format
(`id`, `option_ids`, `probabilities`) so SemIf's unmodified evaluators score them.

Request mapping (one request per identical state, as an API user would send it):
  choice    SemIf rows with described options -> Choice {option id: description},
            question -> instructions (authored144, perturbations108, WANLI256)
  every     Every lab rows -> Noul (Every tested Jev with Nouls); P(yes) = noul
  typesafe  the 102 TypeSafe rows -> the original TypeSafe question (Noul or Choice with
            its own criteria) on the original JSON document; P(true) = noul
  shape     shape777 systems fixture -> Noul with the yes/no descriptions as criteria

Jev is never called; its published figures come from the public records SemIf aligned.

Commands:
  requests    --kind KIND --rows ROWS.jsonl [--payload-dir DIR] --out requests.jsonl
  run         --requests requests.jsonl --url URL --out responses.jsonl [--concurrency N]
  predictions --requests requests.jsonl --responses responses.jsonl --out predictions.jsonl
  shape       --requests requests.jsonl --url URL --out timing.json [--repeats N]
              [--responses replies.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_QUESTIONS = 128
KINDS = ("choice", "every", "typesafe", "shape")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_payload(path: Path) -> dict:
    """SemIf's reader for the `__VIEWER_DATA__(...)` case snapshots."""
    text = path.read_text().strip()
    prefix = "__VIEWER_DATA__("
    if not text.startswith(prefix):
        raise ValueError(f"Unexpected wrapper in {path}")
    payload, _ = json.JSONDecoder().raw_decode(text[len(prefix):])
    return payload


def state_key(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True,
                                                            ensure_ascii=False)


def question_for(row: dict, kind: str, payloads: dict | None) -> tuple[object, dict, list[str]]:
    """(state, Bobcat question, option ids in SemIf's order) for one SemIf row."""
    option_ids = [option["id"] for option in row["options"]]
    if kind == "choice":
        criteria = {option["id"]: option["description"] for option in row["options"]}
        return row["state"], {"type": "choice", "instructions": row["question"],
                              "criteria": criteria}, option_ids
    if kind == "every":
        if option_ids != ["yes", "no"]:
            raise ValueError(f"Every rows are yes/no: {row['id']}")
        return row["state"], {"type": "noul", "instructions": row["question"]}, option_ids
    if kind == "shape":
        if option_ids != ["yes", "no"]:
            raise ValueError(f"shape777 rows are yes/no: {row['id']}")
        described = {option["id"]: option["description"] for option in row["options"]}
        return row["state"], {"type": "noul", "instructions": row["question"],
                              "criteria": {"true": described["yes"],
                                           "false": described["no"]}}, option_ids
    if kind == "typesafe":
        upstream = row["provenance"]
        evaluation = payloads[upstream["workflow"]]["eval"]
        original = evaluation["questions"][upstream["question_index"]]
        if original["type"] != upstream["primitive"]:
            raise ValueError(f"Primitive changed for {row['id']}")
        question = {"type": original["type"], "instructions": original["instructions"]}
        if original.get("criteria") is not None:
            question["criteria"] = original["criteria"]
        document = evaluation["documents"][upstream["document_index"]]
        return document, question, option_ids
    raise ValueError(f"Unknown kind {kind}")


def build_requests(rows: list[dict], kind: str, payloads: dict | None = None) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in rows:
        state, question, option_ids = question_for(row, kind, payloads)
        group = groups.setdefault(state_key(state), {"state": state, "rows": []})
        group["rows"].append((row["id"], question, option_ids))
    requests = []
    for group in groups.values():
        for start in range(0, len(group["rows"]), MAX_QUESTIONS):
            part = group["rows"][start:start + MAX_QUESTIONS]
            questions = {f"q{i}": question for i, (_, question, _) in enumerate(part)}
            requests.append({
                "request_id": f"r{len(requests)}", "kind": kind,
                "rows": {f"q{i}": {"id": row_id, "option_ids": option_ids}
                         for i, (row_id, _, option_ids) in enumerate(part)},
                "request": {"model": "bobcat-1", "state": group["state"],
                            "questions": questions},
            })
    return requests


def distribution(answer: dict, kind: str, option_ids: list[str]) -> list[float]:
    """A Bobcat answer as probabilities in SemIf's option order."""
    if answer["type"] == "noul":
        p = float(answer["noul"])
        yes = "true" if kind == "typesafe" else "yes"
        values = [p if key == yes else 1 - p for key in option_ids]
    else:
        values = [float(answer["probabilities"][key]) for key in option_ids]
    total = sum(values)
    return [value / total for value in values]


def predictions(requests: list[dict], responses: list[dict]) -> list[dict]:
    by_id = {response["request_id"]: response for response in responses}
    rows = []
    for request in requests:
        response = by_id.get(request["request_id"])
        answers = (response or {}).get("answers") or {}
        for qid, meta in request["rows"].items():
            answer = answers.get(qid) if response and response["status"] == 200 else None
            if answer is None:
                # SemIf's evaluators count a missing prediction as wrong; do not invent one.
                rows.append({"id": meta["id"], "status": "missing",
                             "http_status": (response or {}).get("status")})
                continue
            probabilities = distribution(answer, request["kind"], meta["option_ids"])
            rows.append({"id": meta["id"], "option_ids": meta["option_ids"],
                         "probabilities": probabilities,
                         # SemIf's calibrate.py fits a temperature on option logits; the
                         # served distribution's log-probabilities are its logits up to a
                         # constant, so a fitted T here is a factor on top of Bobcat's own.
                         "option_logits": [math.log(max(p, 1e-12)) for p in probabilities],
                         "status": "distribution", "request_id": request["request_id"],
                         "request_ms": response["client_ms"],
                         "model": "bobcat-1 (served FP8, one L40S)"})
    return rows


def post(url: str, body: bytes, secret: str, timeout: float = 600) -> tuple[int, dict, float]:
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "content-type": "application/json", "x-bobcat-edge-secret": secret})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            status, payload = reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        status, payload = error.code, {}
    return status, payload, (time.perf_counter() - started) * 1000


def send(url: str, secret: str, request: dict) -> dict:
    body = json.dumps(request["request"], ensure_ascii=False).encode()
    attempt = 0
    while True:
        attempt += 1
        status, payload, ms = post(url, body, secret)
        if status != 529 or attempt == 3:
            break
        time.sleep(1)
    return {"request_id": request["request_id"], "status": status, "attempts": attempt,
            "client_ms": round(ms, 1), "usage": payload.get("usage"),
            "answers": payload.get("answers", {})}


def run(requests: list[dict], url: str, secret: str, concurrency: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda request: send(url, secret, request), requests))


def health(url: str) -> dict:
    try:
        with urllib.request.urlopen(url.rsplit("/v1/", 1)[0] + "/health", timeout=10) as reply:
            return json.loads(reply.read())
    except OSError as error:
        return {"error": type(error).__name__}


def shape(requests: list[dict], url: str, secret: str, repeats: int) -> tuple[dict, dict]:
    """Wall time for every state's request at once, and one request after another.

    Returns the timing record and the replies of the first pass of each mode, so the
    decisions themselves can be compared across modes and with SemIf's predictions.
    """
    decisions = sum(len(request["rows"]) for request in requests)
    send(url, secret, requests[0])  # warm-up, not timed
    result = {"states": len(requests), "decisions": decisions, "parallel": [],
              "sequential": [], "health_before": health(url)}
    kept: dict[str, list[dict]] = {}
    for _ in range(repeats):
        # The server is shared; record other in-flight work seen just before each pass.
        inflight = health(url).get("inflight")
        tick = time.perf_counter()
        replies = run(requests, url, secret, concurrency=len(requests))
        seconds = time.perf_counter() - tick
        kept.setdefault("parallel", replies)
        result["parallel"].append({"seconds": seconds, "decisions_per_second": decisions / seconds,
                                   "inflight_before": inflight,
                                   "per_state_ms": [r["client_ms"] for r in replies],
                                   "failed": sum(r["status"] != 200 for r in replies)})
        inflight = health(url).get("inflight")
        tick = time.perf_counter()
        replies = [send(url, secret, request) for request in requests]
        seconds = time.perf_counter() - tick
        kept.setdefault("sequential", replies)
        result["sequential"].append({"seconds": seconds,
                                     "decisions_per_second": decisions / seconds,
                                     "inflight_before": inflight,
                                     "per_state_ms": [r["client_ms"] for r in replies],
                                     "failed": sum(r["status"] != 200 for r in replies)})
    result["health_after"] = health(url)
    return result, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("requests")
    r.add_argument("--kind", choices=KINDS, required=True)
    r.add_argument("--rows", type=Path, required=True)
    r.add_argument("--payload-dir", type=Path, help="typesafe-<workflow>-cases.js snapshots")
    r.add_argument("--out", type=Path, required=True)
    x = sub.add_parser("run")
    x.add_argument("--requests", type=Path, required=True)
    x.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("--concurrency", type=int, default=4)
    p = sub.add_parser("predictions")
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("shape")
    s.add_argument("--requests", type=Path, required=True)
    s.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--responses", type=Path,
                   help="write first-pass replies to <stem>-parallel/-sequential.jsonl")
    args = parser.parse_args()
    if args.command == "requests":
        payloads = None
        if args.kind == "typesafe":
            payloads = {path.name[len("typesafe-"):-len("-cases.js")]: parse_payload(path)
                        for path in args.payload_dir.glob("typesafe-*-cases.js")}
        built = build_requests(read_jsonl(args.rows), args.kind, payloads)
        args.out.write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in built))
        print(json.dumps({"requests": len(built),
                          "questions": sum(len(b["rows"]) for b in built)}))
    elif args.command == "run":
        replies = run(read_jsonl(args.requests), args.url, os.environ["BOBCAT_EDGE_SECRET"],
                      args.concurrency)
        args.out.write_text("".join(json.dumps(reply) + "\n" for reply in replies))
        print(json.dumps({"requests": len(replies),
                          "failed": sum(reply["status"] != 200 for reply in replies)}))
    elif args.command == "predictions":
        rows = predictions(read_jsonl(args.requests), read_jsonl(args.responses))
        args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(json.dumps({"rows": len(rows),
                          "missing": sum(row["status"] == "missing" for row in rows)}))
    else:
        result, kept = shape(read_jsonl(args.requests), args.url,
                             os.environ["BOBCAT_EDGE_SECRET"], args.repeats)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        if args.responses:
            for mode, replies in kept.items():
                path = args.responses.with_name(f"{args.responses.stem}-{mode}.jsonl")
                path.write_text("".join(json.dumps(reply) + "\n" for reply in replies))
        print(json.dumps({k: v for k, v in result.items() if k in ("states", "decisions")}
                         | {"parallel_s": [round(x["seconds"], 2) for x in result["parallel"]],
                            "sequential_s": [round(x["seconds"], 2)
                                             for x in result["sequential"]]}))


if __name__ == "__main__":
    main()
