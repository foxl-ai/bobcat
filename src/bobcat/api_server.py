"""Public Bobcat decision API on the TypeSafe System One wire contract (PLAN step 4).

Endpoints use the paths and shapes of TypeSafe's published OpenAPI 0.2.0, so the official
SDKs work unchanged with `TYPESAFE_BASE_URL` pointed at this server:
  POST /v1/systemone   state + typed questions -> closed Choice/Noul/Score answers
  GET  /v1/models      {"models": [{"name", "description", "release_date"}]}
  GET  /health
The engine returns only the offered identifiers' logits at the first answer position. The
host builds every answer from the request's own names (`protocol.response`) and checks the
reply (`protocol.validate_response`) before sending it; nothing generated reaches a reply.

Deployment: behind an authenticating edge, which checks API keys, rate-limits and
deducts credits from `usage.input_tokens`. The origin accepts only requests carrying the
edge's shared secret (`BOBCAT_EDGE_SECRET`) and refuses to start without one unless
`--local`.

Billed tokens are the request's own content as the student tokenizer counts it: the state
once, plus each question's instructions and candidates. Bobcat's fixed template and the
per-question copies of the state are not billed; processed tokens are reported in the
`x-bobcat-processed-tokens` header.

The state is tokenized once per request (`StateCache`, ported from `bobcat.flash_server`):
every question's sequence still holds its own full copy of the state, with the same token
IDs the per-question encoding gives; only the repeated tokenizer calls are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hmac
import json
import os
import time
import uuid
from array import array
from collections import OrderedDict
from pathlib import Path

from bobcat.protocol import (
    CONFIDENCE_PROFILE,
    RequestLimitError,
    answer,
    decode_request,
    parse_request,
    probabilities,
    response,
    validate_response,
)

MAX_BODY_BYTES = 2 * 1024 * 1024
PARENT_FIRST_TOKENS = 1024  # shared prefix length from which the parent question runs first
# How the questions of one request reach the engine (--schedule). "default" is the rule the
# server has always used; "rule" prefills the shared prefix alone first ("warm") only when
# the state the other questions would otherwise recompute, (n - 1) * shared prefix, is at
# least RULE_RECOMPUTE_TOKENS, and sends every question at once otherwise. It pays off with
# the engine argument prefix_match_unit=16, which lets the warm-up end exactly at the prefix.
SCHEDULES = ("default", "all", "parent", "warm", "rule")
RULE_RECOMPUTE_TOKENS = 16384
DEFAULT_MAX_MODEL_LEN = 16448
# Arguments VLLMEngine sets itself; --engine-arg must not silently replace them.
ENGINE_OWNED = frozenset({"model", "dtype", "quantization", "max_model_len",
                          "gpu_memory_utilization", "max_num_seqs", "max_logprobs",
                          "logprobs_mode", "limit_mm_per_prompt"})
# Token IDs the state cache keeps across requests (uint32: about 16 MB). One state is at most
# a compiled question's length (tens of thousands of tokens), so every state fits.
STATE_CACHE_TOKENS = 4_000_000


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class StateCache:
    """Exact-text memo in front of a compiler's data encoder.

    `StudentCompiler.compile` encodes the state for every question; with this in front, the
    first question of a request encodes it and the others (and `billed_tokens`) reuse those
    IDs. The key is the exact JSON text the encoder tokenizes, so a hit returns the IDs the
    encoder would return: identical sequences, fewer tokenizer calls. A later request with the
    byte-identical state (workflow steps over one document) also hits; nothing but the text's
    own token IDs is stored, least recently used entries go first once `max_tokens` IDs are
    held, and a value longer than that is encoded every time rather than cached."""

    def __init__(self, encode, max_tokens: int = STATE_CACHE_TOKENS):
        self.encode, self.max_tokens = encode, max_tokens
        self.entries: OrderedDict[str, array] = OrderedDict()
        self.tokens = self.hits = self.misses = 0

    def __call__(self, value) -> list[int]:
        key = _dumps(value)
        hit = self.entries.get(key)
        if hit is not None:
            self.entries.move_to_end(key)
            self.hits += 1
            return hit.tolist()  # a fresh list: callers may extend it
        ids = self.encode(value)
        self.misses += 1
        if len(ids) <= self.max_tokens:
            self.entries[key] = array("I", ids)
            self.tokens += len(ids)
            while self.tokens > self.max_tokens:
                _, dropped = self.entries.popitem(last=False)
                self.tokens -= len(dropped)
        return ids


def cache_state_encoding(compiler, max_tokens: int = STATE_CACHE_TOKENS):
    """Put a `StateCache` in front of `compiler._data` (idempotent) and return the compiler.
    The compiler's own methods call `self._data`, which now resolves to the cache."""
    if not isinstance(compiler.__dict__.get("_data"), StateCache):
        compiler._data = StateCache(compiler._data, max_tokens)
    return compiler


def load_compiler(model_dir: Path, identifiers_path: Path, max_branch_tokens: int, *,
                  state_cache: bool = True):
    """The pinned piecewise student compiler for a served checkpoint's tokenizer folder."""
    from tokenizers import Tokenizer

    from bobcat.student_readout import StudentCompiler, identifier_scheme

    receipt = json.loads((model_dir / "bobcat-download.json").read_text())
    tokenizer = model_dir / "tokenizer.json"
    reserved = {t["id"] for t in json.loads(tokenizer.read_text()).get("added_tokens", [])}
    identifiers = identifier_scheme(Tokenizer.from_file(str(tokenizer)), reserved,
                                    json.loads(identifiers_path.read_text())["identifiers"])
    compiler = StudentCompiler(model_dir, receipt["files"], identifiers,
                               max_branch_tokens=max_branch_tokens, piecewise=True)
    return cache_state_encoding(compiler) if state_cache else compiler


def build_response(model: str, questions, scores, temperatures, input_tokens: int) -> dict:
    """`protocol.response`, also for one temperature per question (a list, as a routed
    request answers each question with the calibration of the model that scored it)."""
    if isinstance(temperatures, dict):
        return response(model, questions, scores, temperatures, input_tokens)
    if not (len(questions) == len(scores) == len(temperatures)):
        raise ValueError("Every question needs one score vector and one temperature.")
    if type(input_tokens) is not int or input_tokens < 0:
        raise ValueError("Input token usage must be a nonnegative integer.")
    return {"model": model, "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "answers": {q.id: answer(q, probabilities(logits, t))
                        for q, logits, t in zip(questions, scores, temperatures, strict=True)}}


def warmup_prefix(sequences, block_size: int | None) -> list[int] | None:
    """The block-aligned shared prefix to prefill once before the questions, or None.

    With vLLM's "align" Mamba cache mode the hybrid layers' recurrent state is cached only at
    the end of a prefill chunk that falls on a cache-block boundary, so a question sent in
    full leaves at most a clipped chunk end cached and its siblings recompute most of the
    state. Prefilling exactly the first (L // B) * B shared tokens first ends that prefill on
    a boundary; every question then starts from a cache hit of that length."""
    if not block_size or len(sequences) < 2:
        return None
    shared = common_prefix(sequences)
    aligned = (shared // block_size) * block_size
    return list(sequences[0][:aligned]) if aligned >= block_size else None


def plan_schedule(schedule: str, count: int, prefix: int,
                  recompute_tokens: int = RULE_RECOMPUTE_TOKENS) -> str:
    """"all" (every question at once), "parent" (question 1 alone, then the rest from its
    cached prefix) or "warm" (the block-aligned shared prefix alone, then every question) for a
    request of `count` questions sharing `prefix` tokens."""
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule {schedule!r}; choose one of {', '.join(SCHEDULES)}.")
    if count < 2:
        return "all"
    if schedule == "default":
        return "parent" if prefix >= PARENT_FIRST_TOKENS else "all"
    if schedule == "rule":
        return "warm" if (count - 1) * prefix >= recompute_tokens else "all"
    return schedule


def parse_engine_args(pairs: list[str] | None) -> dict:
    """`KEY=VALUE` pairs for extra vLLM engine arguments (for example
    `max_num_batched_tokens=8192`, `kv_cache_dtype=fp8`, `enforce_eager=true`). A value is
    read as JSON when it parses (numbers, true/false, null, quoted strings) and as a plain
    string otherwise. With no pairs the engine is configured exactly as before."""
    result = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key.isidentifier():
            raise ValueError(f"Engine arguments are KEY=VALUE; got {pair!r}.")
        if key in ENGINE_OWNED:
            raise ValueError(f"{key} has its own option; it is not an --engine-arg.")
        if key in result:
            raise ValueError(f"Engine argument {key} is given twice.")
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            result[key] = raw
    return result


class VLLMEngine:
    """Continuous batching and automatic prefix caching through vLLM's async engine."""

    def __init__(self, model: Path, *, identifier_ids: list[int], quantization: str | None = None,
                 gpu_memory_utilization: float = 0.88,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN, max_num_seqs: int = 256,
                 engine_kwargs: dict | None = None, prefix_warmup: bool = False,
                 schedule: str = "default", recompute_tokens: int = RULE_RECOMPUTE_TOKENS):
        self.schedule = "warm" if prefix_warmup else schedule
        self.recompute_tokens = recompute_tokens
        plan_schedule(self.schedule, 1, 0)  # refuse an unknown name before loading the model
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM

        self.SamplingParams = SamplingParams
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(model), dtype="bfloat16", quantization=quantization,
            max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True, max_logprobs=-1, logprobs_mode="raw_logits", seed=0,
            # One Mamba state block per running sequence must fit beside the weights
            # (H100 80 GB FP8: 763 blocks; L40S 48 GB FP8: 202).
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 0, "video": 0}, **(engine_kwargs or {})))
        # A pre-quantized checkpoint (e.g. NVFP4 compressed-tensors) names its own format.
        config = Path(model) / "config.json"
        packed = (json.loads(config.read_text()).get("quantization_config", {}).get("format")
                  if config.exists() else None)
        self.name = f"vllm ({quantization or packed or 'bf16'})"
        self.identifier_ids = list(identifier_ids)
        self.block_size = None
        if self.schedule != "default":
            self.name += f", {self.schedule} schedule"
        if self.schedule in ("warm", "rule"):
            cache = self.engine.vllm_config.cache_config
            if getattr(cache, "mamba_cache_mode", "align") != "align":
                raise ValueError("prefix_warmup assumes vLLM's align Mamba cache mode.")
            # prefix_match_unit (an engine argument) lets align mode cache the recurrent
            # state at a prompt's last unit boundary inside a block: warm to that unit.
            self.block_size = int(getattr(cache, "prefix_match_unit", None) or cache.block_size)

    WIDTH = 64

    def _chunks(self, options):
        """Requests of exactly WIDTH identifier ids covering `options` in order: one gather
        shape (vLLM JIT-compiles per length) and no full-vocabulary path (it ran out of
        memory). Extra chunks of a long candidate list reuse the cached prompt."""
        parts = []
        for start in range(0, len(options), self.WIDTH):
            real = list(options[start:start + self.WIDTH])
            taken = set(real)
            pad = [t for t in self.identifier_ids if t not in taken][:self.WIDTH - len(real)]
            parts.append((real, real + pad))
        return parts

    async def _request(self, sequence, real, ids):
        # The engine samples one token that is discarded; only raw logits are read.
        params = self.SamplingParams(max_tokens=1, temperature=0.0, detokenize=False,
                                     logprobs=self.WIDTH, logprob_token_ids=ids)
        final = None
        async for output in self.engine.generate({"prompt_token_ids": list(sequence)}, params,
                                                 request_id=uuid.uuid4().hex):
            final = output
        table = final.outputs[0].logprobs[0]
        return [float(table[token].logprob) for token in real]

    async def _one(self, sequence, options):
        parts = self._chunks(options)
        first = await self._request(sequence, *parts[0])
        rest = await asyncio.gather(*(self._request(sequence, *p) for p in parts[1:]))
        return first + [value for chunk in rest for value in chunk]

    async def _warm(self, prefix_ids):
        params = self.SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
        async for _ in self.engine.generate({"prompt_token_ids": prefix_ids}, params,
                                            request_id=uuid.uuid4().hex):
            pass

    async def logits(self, sequences, option_ids, prefix: int):
        plan = plan_schedule(self.schedule, len(sequences), prefix, self.recompute_tokens)
        warm = warmup_prefix(sequences, self.block_size) if plan == "warm" else None
        if warm is not None:
            await self._warm(warm)
        elif plan == "parent":
            first = await self._one(sequences[0], option_ids[0])
            rest = await asyncio.gather(*(self._one(s, o) for s, o in zip(
                sequences[1:], option_ids[1:], strict=True)))
            return [first, *rest]
        return list(await asyncio.gather(*(self._one(s, o) for s, o in zip(
            sequences, option_ids, strict=True))))


class TransformersEngine:
    """The research serving path (`student_serve.ServingStudent`), one request at a time."""

    def __init__(self, server):
        self.server, self.lock, self.name = server, asyncio.Lock(), "transformers"

    async def logits(self, sequences, option_ids, prefix: int):
        async with self.lock:
            values, _ = await asyncio.to_thread(self.server.logits, sequences, option_ids,
                                                shared=prefix >= PARENT_FIRST_TOKENS)
        return values


def billed_tokens(compiler, state, sequences) -> int:
    template = len(compiler.before) + len(compiler.between) + len(compiler.after)
    state_tokens = len(compiler._data(state))
    return state_tokens + sum(len(s) - template - state_tokens for s in sequences)


def common_prefix(sequences) -> int:
    first, length = sequences[0], min(map(len, sequences))
    for index in range(length):
        if any(s[index] != first[index] for s in sequences[1:]):
            return index
    return length


def validation_error(message: str, loc=("body",)) -> dict:
    # FastAPI/TypeSafe HTTPValidationError shape.
    return {"detail": [{"loc": list(loc), "msg": message, "type": "value_error"}]}


def create_api(engine, compiler, *, model_name: str, aliases: set[str], temperature: float,
               models: list[dict], edge_secret: str | None, max_inflight: int = 256):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Bobcat", version="1.0", docs_url=None, redoc_url=None,
                  openapi_url=None)
    temperatures = {kind: temperature for kind in ("choice", "noul", "score")}
    accepted = aliases | {model_name}
    state = {"inflight": 0}
    decide = getattr(engine, "decide", None)  # bobcat.route_server.RoutedEngine

    def reply(status, body, request_id, extra=None):
        headers = {"x-request-id": request_id, **(extra or {})}
        return JSONResponse(status_code=status, content=body, headers=headers)

    def authorized(request) -> bool:
        if edge_secret is None:
            return True
        given = request.headers.get("x-bobcat-edge-secret", "")
        return hmac.compare_digest(given.encode(), edge_secret.encode())

    async def systemone(request: Request):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        if not authorized(request):
            return reply(401, {"detail": "Missing or invalid API key."}, request_id)
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                return reply(413, {"detail": "Request body exceeds 2 MiB."}, request_id)
            chunks.append(chunk)
        try:
            payload = decode_request(b"".join(chunks))
            state_value, questions = parse_request(payload)
            if payload["model"] not in accepted:
                return reply(422, validation_error(
                    f"Unknown model {payload['model']!r}; see GET /v1/models.",
                    ("body", "model")), request_id)
            contract = copy.deepcopy(questions)
            compiled = [compiler.compile(state_value, q) for q in questions]
        except RequestLimitError as error:
            return reply(422, validation_error(
                f"{error} Inputs are never truncated."), request_id)
        except ValueError as error:
            return reply(422, validation_error(str(error)), request_id)
        if state["inflight"] >= max_inflight:
            return reply(529, {"detail": "Bobcat is temporarily overloaded."}, request_id,
                         {"retry-after": "1"})
        sequences = [c[0] for c in compiled]
        options = [c[1] for c in compiled]
        state["inflight"] += 1
        started = time.perf_counter()
        try:
            billed = billed_tokens(compiler, state_value, sequences)
            if decide is not None:
                # A routing engine returns numbers only (scores, one temperature per
                # question, processed tokens, header values); the host still builds and
                # checks the closed reply from the request's own names.
                scores, per_question, processed, extra = await decide(
                    state_value, questions, sequences, options, request.headers)
                result = build_response(model_name, contract, scores, per_question, billed)
            else:
                scores = await engine.logits(sequences, options, common_prefix(sequences))
                result = response(model_name, contract, scores, temperatures, billed)
                processed, extra = sum(map(len, sequences)), {}
            validate_response(result, contract, model=model_name)
        except Exception:
            # Never forward engine text, tracebacks or partial answers.
            return reply(500, {"detail": "The decision could not be produced."}, request_id)
        finally:
            state["inflight"] -= 1
        return reply(200, result, request_id, {
            "x-bobcat-processed-tokens": str(processed),
            "x-bobcat-engine-ms": f"{(time.perf_counter() - started) * 1000:.1f}",
            "x-bobcat-confidence-method": CONFIDENCE_PROFILE,
            **extra,
        })

    async def list_models(request: Request):
        if not authorized(request):
            return reply(401, {"detail": "Missing or invalid API key."}, uuid.uuid4().hex)
        return {"models": models}

    # Request is imported lazily; bind the concrete type so FastAPI injects it.
    systemone.__annotations__["request"] = Request
    list_models.__annotations__["request"] = Request
    app.add_api_route("/v1/systemone", systemone, methods=["POST"])
    app.add_api_route("/v1/models", list_models, methods=["GET"])

    @app.get("/health")
    async def health():
        return {"model": model_name, "engine": engine.name, "inflight": state["inflight"]}

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--model", type=Path, required=True, help="checkpoint to serve")
    parser.add_argument("--compiler-model", type=Path, required=True,
                        help="folder with the pinned tokenizer and bobcat-download.json")
    parser.add_argument("--adapter", type=Path, help="LoRA folder (transformers engine)")
    parser.add_argument("--quantization", choices=["none", "fp8"], default="none")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--name", default="bobcat-1")
    parser.add_argument("--release-date", default="2026-09-25")
    parser.add_argument("--identifiers", type=Path,
                        default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN,
                        help="engine context; a compiled question may use this minus 64")
    parser.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE",
                        help="extra vLLM engine argument (repeatable), e.g. "
                             "max_num_batched_tokens=8192")
    parser.add_argument("--prefix-warmup", action="store_true",
                        help="prefill the block-aligned shared prefix once before the "
                             "questions (vLLM engine; off by default; same as --schedule warm)")
    parser.add_argument("--schedule", choices=SCHEDULES, default="default",
                        help="how a request's questions reach the vLLM engine: default "
                             f"(question 1 first when the shared prefix is >= "
                             f"{PARENT_FIRST_TOKENS} tokens), all, parent, warm, or rule "
                             f"(warm when (n - 1) x prefix >= {RULE_RECOMPUTE_TOKENS} tokens, "
                             "else all; use with --engine-arg prefix_match_unit=16)")
    parser.add_argument("--schedule-recompute-tokens", type=int, default=RULE_RECOMPUTE_TOKENS,
                        help="threshold of --schedule rule")
    parser.add_argument("--no-state-cache", action="store_true",
                        help="tokenize the state once per question, as before 2026-09-26 "
                             "(same token IDs; for comparisons)")
    parser.add_argument("--local", action="store_true", help="no edge secret (development)")
    args = parser.parse_args()
    engine_kwargs = parse_engine_args(args.engine_arg)
    secret = os.environ.get("BOBCAT_EDGE_SECRET")
    if not secret and not args.local:
        raise SystemExit("Set BOBCAT_EDGE_SECRET, or pass --local for development.")

    compiler = load_compiler(args.compiler_model, args.identifiers, args.max_model_len - 64,
                             state_cache=not args.no_state_cache)
    if args.engine == "vllm":
        engine = VLLMEngine(args.model, identifier_ids=compiler.identifier_ids,
                            quantization=None if args.quantization == "none"
                            else args.quantization, max_num_seqs=args.max_num_seqs,
                            max_model_len=args.max_model_len, engine_kwargs=engine_kwargs,
                            prefix_warmup=args.prefix_warmup, schedule=args.schedule,
                            recompute_tokens=args.schedule_recompute_tokens)
    else:
        from bobcat.student_serve import ServingStudent

        engine = TransformersEngine(ServingStudent(
            args.model, name=args.name, identifiers_path=args.identifiers,
            adapter=args.adapter, temperature=args.temperature))
    description = ("Bobcat typed decision model (Choice/Noul/Score). Korean and English. "
                   "Output tokens are never generated.")
    models = [{"name": name, "description": description, "release_date": args.release_date}
              for name in (args.name, "bobcat-latest")]
    app = create_api(engine, compiler, model_name=args.name,
                     aliases={"bobcat-latest", "jev-latest", "jev-preview"},
                     temperature=args.temperature, models=models, edge_secret=secret,
                     max_inflight=args.max_inflight)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
