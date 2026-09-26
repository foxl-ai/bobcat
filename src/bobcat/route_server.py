"""One Bobcat server that answers with Bobcat Flash first and hands questions to Bobcat (27B).

Same wire contract and closed replies as `bobcat.api_server` (`create_api` is reused; the host
builds and validates every reply from the request's own names). Each question is routed on
its own, from its own input and its own Flash scores only:
  1. out of range (decided from input features before any forward pass): more candidates
     than Flash's range (64), or letters that are mostly neither Hangul nor Latin (Flash was
     trained and evaluated on Korean and English only) -> Bobcat, sent at once, beside Flash;
  2. otherwise Flash scores it; when its calibrated top probability (softmax at Flash's
     calibration temperature) is below the threshold (0.8, chosen on the calibration split;
     see the Bobcat Flash 1.1 model card) -> Bobcat scores it again, from its own compile.
The answer carries the probabilities of the model that scored it at that model's calibration
temperature. Questions never see each other: a question's route, sequence and scores do not
depend on the other questions of the request (tests/test_route_server.py).

Both engines live in this process on one GPU (vLLM, each with its own share of GPU memory).
Headers: `x-bobcat-route` lists, in question order, `flash` or `bobcat` (the model whose
scores answer it); `x-bobcat-route-reasons` gives `in_range`, `low_confidence`, `candidates`,
`script` or `bobcat_limit` (Bobcat could not compile it; Flash's answer stands). With
`--allow-route-override` (measurement only), `x-bobcat-route-mode: flash|bobcat` forces one
model for a request.

    python -m bobcat.route_server --flash-model fnvfp4/ --flash-compiler-model fbase/ \
        --flash-temperature 0.8912 --big-model b11-nvfp4/ --big-compiler-model b11-compiler/ \
        --big-temperature 1.2008 --schedule all --engine-arg max_num_batched_tokens=16384
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from dataclasses import dataclass
from pathlib import Path

from bobcat.api_server import common_prefix
from bobcat.protocol import RequestLimitError

THRESHOLD = 0.8
MAX_CANDIDATES = 64
MIN_KNOWN_SCRIPT_SHARE = 0.5
MODES = ("auto", "flash", "bobcat")


def _strings(value):
    """Every string a JSON value holds, object keys included (the request's own text)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _known(char: str) -> bool:
    # The character classes of the offline rule (bobcat.flash_report.input_features).
    return "가" <= char <= "힣" or "ㄱ" <= char <= "ㆎ" or "a" <= char <= "z" or "A" <= char <= "Z"


def known_script_share(state, question) -> float:
    """Share of letters (str.isalpha) that are Hangul or ASCII Latin in the text the model
    reads for this question: the state, the instructions and the candidates' names and
    descriptions. The offline study counted the whole one-question request JSON; this leaves
    out the protocol's own keys and the question ID. No letters counts as fully known."""
    letters = known = 0
    for text in [*_strings(state), *_strings(question.instructions),
                 *_strings(question.criteria)]:
        for char in text:
            if char.isalpha():
                letters += 1
                known += _known(char)
    return known / letters if letters else 1.0


@dataclass(frozen=True)
class RoutePolicy:
    threshold: float = THRESHOLD
    max_candidates: int = MAX_CANDIDATES
    min_known_script_share: float = MIN_KNOWN_SCRIPT_SHARE

    def out_of_range(self, state, question) -> str | None:
        """The reason a question goes straight to Bobcat, or None (known before any pass)."""
        if len(question.labels) > self.max_candidates:
            return "candidates"
        if known_script_share(state, question) < self.min_known_script_share:
            return "script"
        return None


def top_probability(logits, temperature: float) -> float:
    peak = max(logits)
    weights = [math.exp((v - peak) / temperature) for v in logits]
    return max(weights) / sum(weights)


@dataclass
class Tier:
    """A model behind the router: engine (`logits(sequences, option_ids, prefix)`), its
    compiler and its calibration temperature."""
    name: str
    engine: object
    compiler: object
    temperature: float


class RoutedEngine:
    """Engine for `api_server.create_api`: `decide` returns numbers only (scores, one
    temperature per question, processed tokens, headers); the host builds the reply."""

    def __init__(self, flash: Tier, big: Tier, policy: RoutePolicy | None = None, *,
                 allow_override: bool = False):
        self.flash, self.big, self.policy = flash, big, policy or RoutePolicy()
        self.allow_override = allow_override
        self.name = (f"route (flash: {getattr(flash.engine, 'name', flash.name)}; "
                     f"{big.name}: {getattr(big.engine, 'name', big.name)}; "
                     f"confidence < {self.policy.threshold})")

    def mode(self, headers) -> str:
        if not self.allow_override or headers is None:
            return "auto"
        value = headers.get("x-bobcat-route-mode", "auto")
        if value not in MODES:
            raise ValueError(f"Unknown route mode {value!r}.")
        return value

    async def _score(self, tier: Tier, sequences, options):
        if not sequences:
            return []
        return await tier.engine.logits(sequences, options, common_prefix(sequences))

    async def decide(self, state, questions, sequences, options, headers=None):
        """`sequences`/`options` are the Flash compiles of `questions` (from create_api)."""
        mode = self.mode(headers)
        count = len(questions)
        routes, reasons = ["flash"] * count, ["in_range"] * count
        scores, temperatures = [None] * count, [self.flash.temperature] * count
        big_compiled = {}

        def compile_big(index) -> bool:
            try:
                big_compiled[index] = self.big.compiler.compile(state, questions[index])
                return True
            except RequestLimitError:
                return False

        direct, on_flash = [], []
        for index, question in enumerate(questions):
            reason = (None if mode == "flash" else "forced" if mode == "bobcat"
                      else self.policy.out_of_range(state, question))
            if reason is not None and compile_big(index):
                direct.append(index)
                reasons[index] = reason
            else:
                on_flash.append(index)
                if reason is not None:
                    reasons[index] = "bobcat_limit"

        async def big(indices):
            return await self._score(self.big, [big_compiled[i][0] for i in indices],
                                     [big_compiled[i][1] for i in indices])

        # Out-of-range questions start on Bobcat at once, beside the Flash pass.
        direct_task = asyncio.ensure_future(big(direct)) if direct else None
        second = []
        try:
            flash_scores = await self._score(self.flash, [sequences[i] for i in on_flash],
                                             [options[i] for i in on_flash])
            for index, values in zip(on_flash, flash_scores, strict=True):
                scores[index] = values
                if (mode == "auto" and reasons[index] == "in_range"
                        and top_probability(values, self.flash.temperature)
                        < self.policy.threshold):
                    if compile_big(index):
                        second.append(index)
                        reasons[index] = "low_confidence"
                    else:
                        reasons[index] = "bobcat_limit"
            second_scores = await big(second) if second else []
            direct_scores = await direct_task if direct_task is not None else []
        finally:
            if direct_task is not None and not direct_task.done():
                direct_task.cancel()
        for index, values in [*zip(direct, direct_scores, strict=True),
                              *zip(second, second_scores, strict=True)]:
            scores[index], temperatures[index], routes[index] = (
                values, self.big.temperature, self.big.name)
        processed = (sum(len(sequences[i]) for i in on_flash)
                     + sum(len(big_compiled[i][0]) for i in [*direct, *second]))
        return scores, temperatures, processed, {
            "x-bobcat-route": ",".join("flash" if r == "flash" else "bobcat" for r in routes),
            "x-bobcat-route-reasons": ",".join(reasons)}


def main():
    import uvicorn

    from bobcat.api_server import (
        DEFAULT_MAX_MODEL_LEN,
        SCHEDULES,
        VLLMEngine,
        create_api,
        load_compiler,
        parse_engine_args,
    )

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for tier, share in (("flash", 0.36), ("big", 0.52)):
        parser.add_argument(f"--{tier}-model", type=Path, required=True)
        parser.add_argument(f"--{tier}-compiler-model", type=Path, required=True)
        parser.add_argument(f"--{tier}-temperature", type=float, required=True)
        parser.add_argument(f"--{tier}-quantization", choices=["none", "fp8"], default="none")
        parser.add_argument(f"--{tier}-gpu-memory-utilization", type=float, default=share,
                            help="this engine's share of the GPU (both engines share one)")
        parser.add_argument(f"--{tier}-engine-arg", action="append", default=[],
                            metavar="KEY=VALUE", help=f"vLLM engine argument for the {tier} "
                                                      "engine only (repeatable)")
    parser.add_argument("--big-name", default="bobcat-1.1")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--min-known-script-share", type=float, default=MIN_KNOWN_SCRIPT_SHARE)
    parser.add_argument("--name", default="bobcat-flash-1.1")
    parser.add_argument("--release-date", default="2026-09-26")
    parser.add_argument("--identifiers", type=Path,
                        default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-inflight", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE",
                        help="vLLM engine argument for both engines (repeatable)")
    parser.add_argument("--schedule", choices=SCHEDULES, default="default")
    parser.add_argument("--allow-route-override", action="store_true",
                        help="honour x-bobcat-route-mode (measurement only)")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    secret = os.environ.get("BOBCAT_EDGE_SECRET")
    if not secret and not args.local:
        raise SystemExit("Set BOBCAT_EDGE_SECRET, or pass --local for development.")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold is a probability.")
    shared = parse_engine_args(args.engine_arg)
    tiers = {}
    for tier in ("flash", "big"):  # Flash first: the smaller share is taken before the larger
        compiler = load_compiler(getattr(args, f"{tier}_compiler_model"), args.identifiers,
                                 args.max_model_len - 64)
        quantization = getattr(args, f"{tier}_quantization")
        engine = VLLMEngine(
            getattr(args, f"{tier}_model"), identifier_ids=compiler.identifier_ids,
            quantization=None if quantization == "none" else quantization,
            gpu_memory_utilization=getattr(args, f"{tier}_gpu_memory_utilization"),
            max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
            engine_kwargs={**shared, **parse_engine_args(getattr(args, f"{tier}_engine_arg"))},
            schedule=args.schedule)
        tiers[tier] = Tier("flash" if tier == "flash" else args.big_name, engine, compiler,
                           getattr(args, f"{tier}_temperature"))
    policy = RoutePolicy(args.threshold, args.max_candidates, args.min_known_script_share)
    routed = RoutedEngine(tiers["flash"], tiers["big"], policy,
                          allow_override=args.allow_route_override)
    description = ("Bobcat typed decision model (Choice/Noul/Score): Bobcat Flash answers, "
                   "Bobcat answers what Flash is not confident about. Korean and English. "
                   "Output tokens are never generated.")
    models = [{"name": args.name, "description": description,
               "release_date": args.release_date}]
    app = create_api(routed, tiers["flash"].compiler, model_name=args.name,
                     aliases={"bobcat-latest", "bobcat-flash-latest", "jev-latest",
                              "jev-preview"},
                     temperature=args.flash_temperature, models=models, edge_secret=secret,
                     max_inflight=args.max_inflight)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
