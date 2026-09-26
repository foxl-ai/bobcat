"""Bobcat Flash on the TypeSafe-compatible HTTP contract (2026-09-26).

Same wire contract, compiler, readout and closed serializer as `bobcat.api_server`
(`create_api` and, for `--engine vllm`, `VLLMEngine` are reused unchanged). Two additions:
  * `CachedCompiler`: the state's token IDs are computed once per request instead of once per
    question (the IDs are identical; `api_server` re-encodes the state for every question,
    which dominates a small model's time on many-question requests);
  * `--engine packed`: `flash_packed.PackedEngine`, all questions of a request in one
    block-masked sequence over the shared state (pure full-attention checkpoints only).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections import OrderedDict
from pathlib import Path

from bobcat.student_readout import StudentCompiler


class CachedCompiler(StudentCompiler):
    """StudentCompiler whose data encodings are memoised by their exact JSON text."""

    def __init__(self, *args, cache_size: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._cache_size = cache_size

    def _data(self, value) -> list[int]:
        key = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return list(hit)
        ids = super()._data(value)
        self._cache[key] = tuple(ids)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ids


def build_compiler(model_dir: Path, identifiers_path: Path, max_tokens: int, *, cached=True):
    from tokenizers import Tokenizer

    from bobcat.student_readout import identifier_scheme

    receipt = json.loads((model_dir / "bobcat-download.json").read_text())
    tokenizer = model_dir / "tokenizer.json"
    reserved = {t["id"] for t in json.loads(tokenizer.read_text()).get("added_tokens", [])}
    identifiers = identifier_scheme(Tokenizer.from_file(str(tokenizer)), reserved,
                                    json.loads(identifiers_path.read_text())["identifiers"])
    kind = CachedCompiler if cached else StudentCompiler
    return kind(model_dir, receipt["files"], identifiers, max_branch_tokens=max_tokens,
                piecewise=True)


def main():
    import uvicorn

    from bobcat.api_server import create_api, parse_engine_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("vllm", "packed"), default="vllm")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--compiler-model", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--name", default="bobcat-flash-1.1")
    parser.add_argument("--release-date", default="2026-09-26")
    parser.add_argument("--identifiers", type=Path,
                        default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32832)
    parser.add_argument("--quantization", choices=["none", "fp8"], default="none")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--schedule", default="default")
    parser.add_argument("--window-ms", type=float, default=2.0)
    parser.add_argument("--no-compile-cache", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    secret = os.environ.get("BOBCAT_EDGE_SECRET")
    if not secret and not args.local:
        raise SystemExit("Set BOBCAT_EDGE_SECRET, or pass --local for development.")
    compiler = build_compiler(args.compiler_model, args.identifiers, args.max_model_len - 64,
                              cached=not args.no_compile_cache)
    if args.engine == "packed":
        from bobcat.flash_packed import PackedEngine, PackedScorer

        engine = PackedEngine(PackedScorer(args.model), window_ms=args.window_ms)
    else:
        from bobcat.api_server import VLLMEngine

        options = dict(identifier_ids=compiler.identifier_ids,
                       quantization=None if args.quantization == "none" else args.quantization,
                       max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                       engine_kwargs=parse_engine_args(args.engine_arg))
        if "schedule" in inspect.signature(VLLMEngine).parameters:
            options["schedule"] = args.schedule
        engine = VLLMEngine(args.model, **options)
    if not args.no_compile_cache:
        engine.name += ", cached state encoding"
    description = ("Bobcat Flash typed decision model (Choice/Noul/Score). Korean and English. "
                   "Output tokens are never generated.")
    models = [{"name": args.name, "description": description,
               "release_date": args.release_date}]
    app = create_api(engine, compiler, model_name=args.name,
                     aliases={"bobcat-latest", "bobcat-flash-latest", "jev-latest",
                              "jev-preview"},
                     temperature=args.temperature, models=models, edge_secret=secret,
                     max_inflight=args.max_inflight)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
