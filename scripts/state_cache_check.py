"""Token IDs with and without the state cache on real requests and a real tokenizer (CPU).

Compiles every question of the given request files three ways: the stock `StudentCompiler`
(state tokenized per question), `bobcat.api_server.cache_state_encoding` (state once per
request) and `bobcat.flash_server.CachedCompiler`; counts sequences that differ (expected 0)
and times the compile of each request.

    python scripts/state_cache_check.py --compiler-model fbase/ \
        --requests typesafe/requests.jsonl --dev-rows dev.jsonl --out state-cache-check.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compiler-model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, action="append", default=[],
                        help="jsonl with a 'request' payload per line (repeatable)")
    parser.add_argument("--dev-rows", type=Path)
    parser.add_argument("--identifiers", type=Path,
                        default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from bobcat.api_server import load_compiler
    from bobcat.flash_server import build_compiler
    from bobcat.protocol import parse_request

    stock = load_compiler(args.compiler_model, args.identifiers, args.max_tokens,
                          state_cache=False)
    cached = load_compiler(args.compiler_model, args.identifiers, args.max_tokens)
    flash = build_compiler(args.compiler_model, args.identifiers, args.max_tokens, cached=True)
    payloads = []
    for path in args.requests:
        payloads += [json.loads(line)["request"] for line in path.open() if line.strip()]
    if args.dev_rows:
        payloads += [json.loads(line)["request"] for line in args.dev_rows.open()]
    result = {"requests": len(payloads), "questions": 0, "differ_stock_vs_cached": 0,
              "differ_stock_vs_flash_server": 0, "seconds": {"stock": 0.0, "cached": 0.0},
              "multi_question_requests": 0}
    for payload in payloads:
        state, questions = parse_request(payload)
        result["questions"] += len(questions)
        result["multi_question_requests"] += len(questions) > 1
        tick = time.perf_counter()
        a = [stock.compile(state, q) for q in questions]
        result["seconds"]["stock"] += time.perf_counter() - tick
        tick = time.perf_counter()
        b = [cached.compile(state, q) for q in questions]
        result["seconds"]["cached"] += time.perf_counter() - tick
        c = [flash.compile(state, q) for q in questions]
        result["differ_stock_vs_cached"] += sum(x != y for x, y in zip(a, b, strict=True))
        result["differ_stock_vs_flash_server"] += sum(x != y for x, y in zip(a, c, strict=True))
    memo = cached._data
    result["state_cache"] = {"hits": memo.hits, "misses": memo.misses, "tokens": memo.tokens}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
