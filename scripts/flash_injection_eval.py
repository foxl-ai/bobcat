"""Stress-suite injection cases against a Bobcat HTTP endpoint (Bobcat Flash, 2026-09-26).

Posts every `suite == "injection"` case of the 2026-09-25 stress suite (`cases.jsonl`) to a
running server, checks each reply with the independent wire audit
(`output_contract_audit.inspect_reply`) and summarizes with `bobcat.stress.summarize`, the same
measures the Bobcat reports use (attack success, accuracy by variant). Each case is
sent once; nothing is tuned. The edge secret comes from BOBCAT_EDGE_SECRET.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main():
    import httpx

    from bobcat.output_contract_audit import inspect_reply
    from bobcat.stress import summarize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", required=True, help="the server's own model name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [c for c in map(json.loads, args.cases.open()) if c["suite"] == "injection"]
    headers = {"x-bobcat-edge-secret": os.environ.get("BOBCAT_EDGE_SECRET", "")}
    responses = {}
    args.out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.url, headers=headers, timeout=120) as client, \
            (args.out / "injection-responses.jsonl").open("w") as sink:
        for case in cases:
            payload = {**case["payload"], "model": "bobcat-latest"}
            started = time.perf_counter()
            reply = client.post("/v1/systemone", json=payload)
            try:
                inspect_reply(payload, reply, expected_model=args.model_name)
                violation = None
            except (AssertionError, ValueError, TypeError, KeyError) as error:
                violation = str(error)
            record = {"id": case["id"], "status": reply.status_code, "body": reply.json(),
                      "violation": violation, "seconds": time.perf_counter() - started}
            responses[case["id"]] = record
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(cases, responses)
    (args.out / "injection-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.get("injection", {}).items()
                      if k in ("attack_success_all", "accuracy_by_variant")}))


if __name__ == "__main__":
    main()
