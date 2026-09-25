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
from pathlib import Path

from bobcat.protocol import (
    CONFIDENCE_PROFILE,
    RequestLimitError,
    decode_request,
    parse_request,
    response,
    validate_response,
)

MAX_BODY_BYTES = 2 * 1024 * 1024
PARENT_FIRST_TOKENS = 1024  # shared prefix length from which the parent question runs first


class VLLMEngine:
    """Continuous batching and automatic prefix caching through vLLM's async engine."""

    def __init__(self, model: Path, *, identifier_ids: list[int], quantization: str | None = None,
                 gpu_memory_utilization: float = 0.88, max_model_len: int = 16448,
                 max_num_seqs: int = 256):
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
            limit_mm_per_prompt={"image": 0, "video": 0}))
        self.name = f"vllm ({quantization or 'bf16'})"
        self.identifier_ids = list(identifier_ids)

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

    async def logits(self, sequences, option_ids, prefix: int):
        if len(sequences) > 1 and prefix >= PARENT_FIRST_TOKENS:
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
            scores = await engine.logits(sequences, options, common_prefix(sequences))
            result = response(model_name, contract, scores, temperatures,
                              billed_tokens(compiler, state_value, sequences))
            validate_response(result, contract, model=model_name)
        except Exception:
            # Never forward engine text, tracebacks or partial answers.
            return reply(500, {"detail": "The decision could not be produced."}, request_id)
        finally:
            state["inflight"] -= 1
        return reply(200, result, request_id, {
            "x-bobcat-processed-tokens": str(sum(map(len, sequences))),
            "x-bobcat-engine-ms": f"{(time.perf_counter() - started) * 1000:.1f}",
            "x-bobcat-confidence-method": CONFIDENCE_PROFILE,
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
    parser.add_argument("--local", action="store_true", help="no edge secret (development)")
    args = parser.parse_args()
    secret = os.environ.get("BOBCAT_EDGE_SECRET")
    if not secret and not args.local:
        raise SystemExit("Set BOBCAT_EDGE_SECRET, or pass --local for development.")

    from tokenizers import Tokenizer

    from bobcat.student_readout import StudentCompiler, identifier_scheme

    receipt = json.loads((args.compiler_model / "bobcat-download.json").read_text())
    tokenizer = args.compiler_model / "tokenizer.json"
    reserved = {t["id"] for t in json.loads(tokenizer.read_text()).get("added_tokens", [])}
    identifiers = identifier_scheme(Tokenizer.from_file(str(tokenizer)), reserved,
                                    json.loads(args.identifiers.read_text())["identifiers"])
    compiler = StudentCompiler(args.compiler_model, receipt["files"], identifiers,
                               max_branch_tokens=16384, piecewise=True)
    if args.engine == "vllm":
        engine = VLLMEngine(args.model, identifier_ids=compiler.identifier_ids,
                            quantization=None if args.quantization == "none"
                            else args.quantization, max_num_seqs=args.max_num_seqs)
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
