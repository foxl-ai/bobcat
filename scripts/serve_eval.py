"""Development request sets through the closed HTTP contract of one student on one GPU.

Loads `student_serve.ServingStudent` (unmerged LoRA, the evaluated path) behind
`serve.create_app` and posts, in-process, each request set that exists:
  --injection  stress cases (suite `injection` only) -> responses + `stress.summarize`
  --workflow   TypeSafe workflow requests (scripts/typesafe_workflow_eval.py build)
               -> responses in that script's `run` format, for its `score`
  --semif      SemIf request files (scripts/semif_bench.py requests), repeatable
               -> responses in semif_bench's `run` format, for its `predictions`
Nothing here trains, calibrates or selects; every request is answered once, failures
are recorded as failures. Jev is never called.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def post(client, payload: dict) -> tuple[int, dict, dict, float]:
    body = {**payload, "model": "bobcat-latest"}
    started = time.perf_counter()
    reply = client.post("/v1/systemone", json=body)
    ms = (time.perf_counter() - started) * 1000
    try:
        data = reply.json()
    except ValueError:
        data = {}
    return reply.status_code, data, dict(reply.headers), ms


def injection(client, cases_path: Path, out: Path) -> dict:
    from bobcat.output_contract_audit import inspect_reply
    from bobcat.stress import summarize

    cases = [c for c in map(json.loads, cases_path.open()) if c["suite"] == "injection"]
    responses = {}
    with (out / "injection-responses.jsonl").open("x") as sink:
        for case in cases:
            payload = {**case["payload"], "model": "bobcat-latest"}
            started = time.perf_counter()
            reply = client.post("/v1/systemone", json=payload)
            try:
                inspect_reply(payload, reply, expected_model=client.app.state.model_name)
                violation = None
            except (AssertionError, ValueError, TypeError, KeyError) as error:
                violation = str(error)
            record = {"id": case["id"], "status": reply.status_code, "body": reply.json(),
                      "violation": violation, "seconds": time.perf_counter() - started}
            responses[case["id"]] = record
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(cases, responses)
    (out / "injection-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary.get("injection", {})


def workflow(client, requests_path: Path, out: Path) -> dict:
    rows = [json.loads(line) for line in requests_path.read_text().splitlines()]
    failed = 0
    with (out / "workflow-responses.jsonl").open("x") as sink:
        for row in rows:
            status, payload, headers, ms = post(client, row["request"])
            failed += status != 200
            sink.write(json.dumps({
                "workflow": row["workflow"], "case": row["case"], "step": row["step"],
                "status": status, "attempts": 1, "client_ms": round(ms, 1),
                "engine_ms": float("nan"), "processed_tokens": 0,
                "usage": payload.get("usage"), "answers": payload.get("answers", {}),
            }) + "\n")
    return {"requests": len(rows), "failed": failed}


def semif(client, requests_path: Path, out: Path) -> dict:
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    failed = 0
    with (out / f"semif-{requests_path.stem}.responses.jsonl").open("x") as sink:
        for request in requests:
            status, payload, _, ms = post(client, request["request"])
            failed += status != 200
            sink.write(json.dumps({
                "request_id": request["request_id"], "status": status, "attempts": 1,
                "client_ms": round(ms, 1), "usage": payload.get("usage"),
                "answers": payload.get("answers", {}),
            }) + "\n")
    return {"requests": len(requests), "failed": failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--identifiers", type=Path,
                        default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    parser.add_argument("--injection", type=Path, help="stress cases.jsonl")
    parser.add_argument("--workflow", type=Path, help="TypeSafe workflow requests.jsonl")
    parser.add_argument("--semif", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch

    from bobcat.student_serve import ServingStudent, http_client

    started = time.time()
    server = ServingStudent(args.model_dir, name=args.name, identifiers_path=args.identifiers,
                            adapter=args.adapter, temperature=args.temperature, merge=False)
    client = http_client(server)
    client.app.state.model_name = args.name
    report = {"model": args.name, "adapter_sha256": server.adapter_sha256, "merged": False,
              "temperature": args.temperature, "load_seconds": time.time() - started}
    if args.injection:
        report["injection"] = injection(client, args.injection, args.out)
    if args.workflow:
        report["workflow"] = workflow(client, args.workflow, args.out)
    for path in args.semif:
        report.setdefault("semif", {})[path.stem] = semif(client, path, args.out)
    report["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                             "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30}
    report["seconds"] = time.time() - started
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "environment"}))


if __name__ == "__main__":
    main()
