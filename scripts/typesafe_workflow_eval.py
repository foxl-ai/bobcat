"""Bobcat on TypeSafe's published workflow-eval examples (https://evals.typesafe.ai).

TypeSafe publishes four workflow evals. For each it releases five example cases with the
state documents, the exact System One questions each workflow step asks, the per-question
answers of Claude Opus 5, GPT Sol and Jev (`typesafe:v13_snowy_elephant`), and reference
answers from GPT-6 Astra and Claude Fable 5.1 at high thinking. This sends the same states
and questions to a Bobcat endpoint and scores every model the same way against the
reference consensus: the mean of the reference probabilities (or the shared value where a
reference published no probabilities).

Rules (AGENTS.md): Jev is never called; its published answers are only compared against.
Nothing here trains, tunes, calibrates or prompts Bobcat differently: the served model
answers each request once. A failed request counts as a wrong answer.

Commands:
  fetch --out DIR                         download the case files; record URL and SHA256
  build --cases DIR --out requests.jsonl  one request per (case, workflow step)
  run --requests F --url URL --out F      POST each request (edge secret from the env)
  score --cases DIR --responses F --out report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

SITE = "https://evals.typesafe.ai/"
WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing",
             "customer_service"]
PUBLISHED = ["opus", "sol", "typesafe"]
PRICE_PER_MTOK = 0.042
AGENT = {"User-Agent": "Mozilla/5.0 (bobcat-eval)"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=60) as r:
        return r.read()


def published_points(page: str) -> list[dict]:
    """TypeSafe's plotted full-eval results: 'model · mode · 67.8% · $0.0004 · 0.4 s'.
    The first plot on a page is that page's own; later plots are other workflows."""
    points = {}
    for title in re.findall(r"<title>([^<]+)</title>", page):
        parts = [p.strip() for p in title.split("·")]
        if len(parts) == 5 and parts[2].endswith("%"):
            points.setdefault((parts[0], parts[1]), {
                "model": parts[0], "mode": parts[1], "accuracy": float(parts[2][:-1]),
                "usd": float(parts[3].lstrip("$")), "seconds": float(parts[4].rstrip(" s"))})
    return list(points.values())


def fetch(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {},
               "published_full_eval": {"all": published_points(get(SITE).decode())}}
    for workflow in WORKFLOWS:
        page = get(f"{SITE}{workflow}.html").decode()
        receipt["published_full_eval"][workflow] = published_points(page)
        cases = re.search(r'data-cases="([^"]+)"', page).group(1)
        body = get(SITE + cases)
        (out / f"{workflow}-cases.js").write_bytes(body)
        receipt["files"][workflow] = {"url": SITE + cases,
                                      "sha256": hashlib.sha256(body).hexdigest()}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


def load(cases: Path, workflow: str) -> dict:
    text = (cases / f"{workflow}-cases.js").read_text()
    return json.loads(text[text.index("(") + 1:text.rindex(")")])["eval"]


def layout(case: dict) -> tuple[dict, set]:
    """(step, question) -> (document, question index) from every model's run, and the
    questions whose index differs between models (built from a model's own earlier
    answers, so the reference's version is unknown)."""
    index, dynamic = {}, set()
    for model in case["models"].values():
        for node in model["nodes"]:
            if not node["ran"]:
                continue
            for qid, position in node["questions"].items():
                key, value = (node["node"], qid), (node["doc"], position)
                if key in index and index[key] != value:
                    dynamic.add(key)
                index.setdefault(key, value)
    return index, dynamic


def build(cases: Path, out: Path) -> None:
    rows = []
    for workflow in WORKFLOWS:
        data = load(cases, workflow)
        for case_id, case in data["cases"].items():
            index, dynamic = layout(case)
            steps = {}
            for (step, qid), (doc, position) in index.items():
                if (step, qid) not in dynamic:
                    steps.setdefault((step, doc), {})[qid] = data["questions"][position]
            for (step, doc), questions in sorted(steps.items()):
                rows.append({"workflow": workflow, "case": case_id, "step": step,
                             "request": {"model": "bobcat-latest",
                                         "state": data["documents"][doc],
                                         "questions": questions}})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"requests": len(rows),
                      "questions": sum(len(r["request"]["questions"]) for r in rows)}))


def post(url: str, body: bytes, secret: str) -> tuple[int, dict, dict, float]:
    request = urllib.request.Request(url, data=body, method="POST", headers={
        "content-type": "application/json", "x-bobcat-edge-secret": secret})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as r:
            status, payload, headers = r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as error:
        status, payload, headers = error.code, {}, dict(error.headers)
    return status, payload, headers, (time.perf_counter() - started) * 1000


def run(requests: Path, url: str, out: Path) -> None:
    secret = os.environ["BOBCAT_EDGE_SECRET"]
    rows = [json.loads(line) for line in requests.read_text().splitlines()]
    warm = {"model": "bobcat-latest", "state": "warm-up",
            "questions": {"q": {"type": "noul", "instructions": "Is this a warm-up?"}}}
    post(url, json.dumps(warm).encode(), secret)
    with out.open("x") as sink:
        for row in rows:
            body = json.dumps(row["request"], ensure_ascii=False).encode()
            attempts = 0
            while True:
                attempts += 1
                status, payload, headers, ms = post(url, body, secret)
                if status != 529 or attempts == 3:
                    break
                time.sleep(1)
            sink.write(json.dumps({
                "workflow": row["workflow"], "case": row["case"], "step": row["step"],
                "status": status, "attempts": attempts, "client_ms": round(ms, 1),
                "engine_ms": float(headers.get("x-bobcat-engine-ms", "nan")),
                "processed_tokens": int(headers.get("x-bobcat-processed-tokens", 0)),
                "route": headers.get("x-bobcat-route"),  # bobcat.route_server only
                "usage": payload.get("usage"), "answers": payload.get("answers", {}),
            }) + "\n")
            print(row["workflow"], row["case"], row["step"], status, f"{ms:.0f} ms",
                  flush=True)


def distribution(answer: dict | None) -> dict | None:
    """A published or Bobcat answer as {label: probability}."""
    if not answer:
        return None
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return {str(k): v for k, v in answer["probabilities"].items()}


def modal(dist: dict | None) -> str | None:
    if not dist:
        return None
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 1e-9:
        return None  # a tie has no modal answer
    return ranked[0][0]


def consensus(sets: list[dict]) -> tuple[str | None, dict | None]:
    """Mean of the references' probabilities; with values only, the shared value."""
    if all(s.get("probabilities") for s in sets):
        labels = set().union(*(s["probabilities"] for s in sets))
        mean = {k: sum(s["probabilities"].get(k, 0.0) for s in sets) / len(sets)
                for k in labels}
        return modal(mean), mean
    values = {str(s["value"]).lower() if isinstance(s["value"], bool) else str(s["value"])
              for s in sets}
    return (values.pop() if len(values) == 1 else None), None


def items(cases: Path, bobcat: dict) -> list[dict]:
    rows = []
    for workflow in WORKFLOWS:
        data = load(cases, workflow)
        for case_id, case in data["cases"].items():
            index, dynamic = layout(case)
            published = {}
            for key, model in case["models"].items():
                for node in model["nodes"]:
                    if node["ran"]:
                        for qid, answer in node["answers"].items():
                            published[(key, node["node"], qid)] = answer
            for step, questions in case["reference_answers"].items():
                for qid, ref in questions.items():
                    target, mean = consensus(ref["sets"])
                    row = {"workflow": workflow, "case": case_id, "step": step, "qid": qid,
                           "type": ref["type"], "target": target,
                           "probabilistic": mean is not None,
                           "dynamic": (step, qid) in dynamic, "answers": {}}
                    for key in PUBLISHED:
                        row["answers"][key] = distribution(published.get((key, step, qid)))
                    reply = bobcat.get((workflow, case_id, step))
                    if reply is not None and (step, qid) in index and not row["dynamic"]:
                        answer = reply["answers"].get(qid) if reply["status"] == 200 else None
                        # Bobcat was asked; no answer (a failed request) scores as wrong.
                        row["answers"]["bobcat"] = distribution(answer) or {}
                    row["mean"] = mean
                    rows.append(row)
    return rows


def agree(row: dict, model: str) -> float:
    dist = row["answers"].get(model)
    if dist is None:
        raise KeyError(model)
    return float(modal(dist) == row["target"]) if dist else 0.0


def summary(rows: list[dict], models: list[str]) -> dict:
    result = {}
    for model in models:
        values = [agree(r, model) for r in rows]
        mass = [(r["answers"][model] or {}).get(r["target"], 0.0) for r in rows]
        tv = [0.5 * sum(abs((r["answers"][model] or {}).get(k, 0.0) - p)
                        for k, p in r["mean"].items()) for r in rows if r["probabilistic"]]
        result[model] = {"agreement": sum(values) / len(values),
                         "mass_on_consensus": sum(mass) / len(mass),
                         "total_variation": sum(tv) / len(tv) if tv else None,
                         "n": len(values)}
    return result


def macro(rows: list[dict], model: str) -> float:
    per = []
    for workflow in WORKFLOWS:
        subset = [agree(r, model) for r in rows if r["workflow"] == workflow]
        if subset:
            per.append(sum(subset) / len(subset))
    return sum(per) / len(per)


def bootstrap(rows: list[dict], a: str, b: str, *, draws: int = 10000, seed: int = 20260925):
    """Workflow-stratified case bootstrap of the macro difference a - b."""
    rng = random.Random(seed)
    by_case = {}
    for r in rows:
        by_case.setdefault((r["workflow"], r["case"]), []).append(r)
    strata = {w: [k for k in by_case if k[0] == w] for w in WORKFLOWS}
    diffs = []
    for _ in range(draws):
        sample = [r for keys in strata.values() if keys
                  for k in (rng.choice(keys) for _ in keys) for r in by_case[k]]
        diffs.append(macro(sample, a) - macro(sample, b))
    diffs.sort()
    return [diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]]


def score(cases: Path, responses: Path, out: Path) -> None:
    replies = [json.loads(line) for line in responses.read_text().splitlines()]
    bobcat = {(r["workflow"], r["case"], r["step"]): r for r in replies}
    rows = items(cases, bobcat)
    usable = [r for r in rows if r["target"] is not None and not r["dynamic"]]
    models = PUBLISHED + ["bobcat"]
    common = [r for r in usable if all(r["answers"].get(m) is not None for m in models)]
    report = {
        "items": {"reference": len(rows), "disputed_or_tied": sum(r["target"] is None
                                                                   for r in rows),
                  "dynamic_question": sum(r["dynamic"] for r in rows),
                  "common_to_all_models": len(common)},
        "common": {"all": summary(common, models),
                   "macro": {m: macro(common, m) for m in models},
                   "by_workflow": {w: summary([r for r in common if r["workflow"] == w],
                                              models) for w in WORKFLOWS},
                   "by_type": {t: summary([r for r in common if r["type"] == t], models)
                               for t in ("noul", "choice", "score")
                               if any(r["type"] == t for r in common)},
                   "bobcat_minus_jev_macro_95ci": bootstrap(common, "bobcat", "typesafe"),
                   "bobcat_minus_opus_macro_95ci": bootstrap(common, "bobcat", "opus"),
                   "bobcat_minus_sol_macro_95ci": bootstrap(common, "bobcat", "sol")},
        "sensitivity": sensitivity(cases, common, models),
        "bobcat_all_runnable": summary([r for r in usable if "bobcat" in r["answers"]],
                                       ["bobcat"]),
        "requests": {"count": len(replies),
                     "failed": sum(r["status"] != 200 for r in replies),
                     "retried": sum(r["attempts"] > 1 for r in replies)},
        "cost_time_per_case": cost_time(cases, replies),
    }
    receipt = cases / "receipt.json"
    if receipt.exists():
        report["source"] = json.loads(receipt.read_text())
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["common"]["macro"], indent=1))


def sensitivity(cases: Path, common: list[dict], models: list[str]) -> dict:
    """TypeSafe chose the five examples per workflow by how Opus, Sol and Jev behaved (one
    case where each alone differs, one all miss, one all agree); Bobcat played no part in
    the choice. Re-score without the case chosen for Jev's disagreement, and on the two
    cases per workflow chosen without singling out any model."""
    labels = {}
    for workflow in WORKFLOWS:
        for example in load(cases, workflow)["examples"]:
            labels[(workflow, example["case_id"])] = example["label"]
    subsets = {
        "without_jev_differs_case": [r for r in common if not labels.get(
            (r["workflow"], r["case"]), "").startswith("TypeSafe differs")],
        "neutral_cases_only": [r for r in common if labels.get(
            (r["workflow"], r["case"])) in ("All three miss the reference", "All three agree")],
    }
    return {name: {"all": summary(rows, models), "macro": {m: macro(rows, m) for m in models},
                   "bobcat_minus_jev_macro_95ci": bootstrap(rows, "bobcat", "typesafe")}
            for name, rows in subsets.items()}


def cost_time(cases: Path, replies: list[dict]) -> dict:
    result = {}
    for workflow in WORKFLOWS:
        data = load(cases, workflow)
        per = {}
        for case_id, case in data["cases"].items():
            mine = [r for r in replies if r["workflow"] == workflow and r["case"] == case_id]
            tokens = sum((r["usage"] or {}).get("input_tokens", 0) for r in mine)
            per[case_id] = {
                "bobcat": {"usd": tokens * PRICE_PER_MTOK / 1e6,
                           "sequential_client_seconds": sum(r["client_ms"] for r in mine) / 1000,
                           "requests": len(mine), "billed_tokens": tokens},
                **{m: {"usd": (case["models"][m].get("cost") or {}).get("usd"),
                       "seconds": case["models"][m].get("seconds")} for m in PUBLISHED},
            }
        result[workflow] = per
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch").add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--cases", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--requests", type=Path, required=True)
    r.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--cases", type=Path, required=True)
    s.add_argument("--responses", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.out)
    elif args.command == "build":
        build(args.cases, args.out)
    elif args.command == "run":
        run(args.requests, args.url, args.out)
    else:
        score(args.cases, args.responses, args.out)


if __name__ == "__main__":
    main()
