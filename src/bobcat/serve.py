"""Local System One endpoint for an explicitly identified Bobcat checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import threading
from pathlib import Path

import torch

from bobcat.batching import EncodedDataset
from bobcat.checkpoints import CHECKPOINT_FORMATS, verify_checkpoint
from bobcat.model import DecisionModel, ModelConfig
from bobcat.protocol import (
    CONFIDENCE_PROFILE,
    RequestLimitError,
    decode_request,
    parse_request,
    render,
    response,
    validate_response,
)
from bobcat.schema import Choice, Example, file_hash, json_hash
from bobcat.tokenization import ScratchTokenizer


class StudentScorer:
    def __init__(self, checkpoint_path: Path, tokenizer_path: Path, *, device: str,
                 allow_unvalidated: bool = False):
        checkpoint_path = checkpoint_path.resolve(strict=True)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("format") in CHECKPOINT_FORMATS:
            verify_checkpoint(checkpoint_path)
        if not allow_unvalidated and not payload.get("release_gate_passed", False):
            raise ValueError("This checkpoint has not passed its release gate.")
        self.device = torch.device(device)
        self.tokenizer = ScratchTokenizer(tokenizer_path)
        provenance = payload.get("provenance", {})
        if provenance.get("tokenizer_sha256") != self.tokenizer.digest:
            raise ValueError("Checkpoint/tokenizer provenance mismatch.")
        self.model = DecisionModel(ModelConfig(**payload["config"]))
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()
        self.model_name = payload.get("model_name") or (
            f"bobcat-scratch-research-{file_hash(checkpoint_path)[:12]}"
        )
        encoding_source = {
            name: file_hash(Path(__file__).with_name(name))
            for name in ("serve.py", "protocol.py", "batching.py", "tokenization.py")
        }
        self.provenance = {
            "profile": "bobcat_student_schema_v1", "compiler_sha256": json_hash(encoding_source),
            "checkpoint_sha256": file_hash(checkpoint_path),
            "checkpoint_format": payload.get("format", "legacy_research_fixture"),
            "tokenizer_sha256": self.tokenizer.digest,
            "tokenizer_encoding_profile": self.tokenizer.encoding_profile,
            "model_config": self.model.config.to_dict(), "input_source_sha256": encoding_source,
            "model_source_sha256": file_hash(Path(__file__).with_name("model.py")),
            "weight_origin": provenance.get("weight_origin", provenance.get("initialization")),
            "decision_reader_training": provenance.get("decision_reader"),
            "language_parent": provenance.get("language_parent"),
            "training_counters": payload.get("counters"),
            "precision": "bf16_autocast" if self.device.type == "cuda" else "fp32",
            "device": str(self.device),
        }
        self.release_gate_passed = bool(payload.get("release_gate_passed", False))
        self.temperatures = payload.get("temperatures", {})
        self.limits = {
            "max_context_tokens": self.model.config.max_context_tokens,
            "max_schema_tokens": self.model.config.max_schema_tokens,
            "max_joint_tokens": (
                self.model.config.max_joint_tokens
                if self.model.config.architecture == "joint" else None
            ),
        }
        self.lock = threading.Lock()

    @torch.inference_mode()
    def score(self, state, questions):
        examples = [
            Example(
                id=str(i), group_id="request", family="runtime", split="inference",
                context=render(state),
                instruction=render(q.instructions) if q.instructions is not None else "",
                choices=[Choice(str(j), text) for j, text in enumerate(q.descriptions)],
                target="0", kind={"noul": "boolean", "score": "ordinal"}.get(q.kind, "choice"),
            ) for i, q in enumerate(questions)
        ]
        try:
            encoded = EncodedDataset(
                examples, self.tokenizer, self.model.config, include_sentinels=False,
            )
        except ValueError as error:
            raise RequestLimitError(str(error)) from error
        batch = encoded.collate(list(range(len(examples)))).to(self.device)
        with self.lock, torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda",
        ):
            logits = self.model(batch.model_inputs()).float().cpu()
        return [row[:len(q.labels)].tolist() for row, q in zip(logits, questions, strict=True)], (
            batch.nonpadding_tokens
        )


def create_app(scorer):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    if getattr(scorer, "readout_mode", None) == "one_token":
        raise ValueError("The decision HTTP endpoint requires a non-generating numeric backend.")
    model_name = scorer.model_name
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("A fixed server model identity is required.")
    model_name.encode("utf-8")
    app = FastAPI(title="Bobcat decisions", version="0.2.0")
    # This is the research HTTP body limit, not a claim about model token capacity.
    max_body_bytes = 2 * 1024 * 1024

    def failed_decision():
        return JSONResponse(status_code=500, content={
            "error": {"code": "decision_backend_failure",
                      "message": "The decision could not be produced."},
        })

    @app.exception_handler(Exception)
    async def backend_error(_request, _error):
        # Do not echo model text, tracebacks, upstream bodies or attacker content.
        # A failed judgment is an error, never a fabricated fallback decision.
        return failed_decision()

    @app.get("/health")
    def health():
        return {"model": model_name, "release_gate_passed": scorer.release_gate_passed,
                "confidence_method": CONFIDENCE_PROFILE,
                "numerically_identical_to_jev": False,
                "max_body_bytes": max_body_bytes,
                "token_limits": getattr(scorer, "limits", {})}

    def evaluate(payload: dict):
        try:
            state, questions = parse_request(payload)
            if payload["model"] not in {
                "bobcat-latest", "jev-latest", model_name,
            }:
                raise ValueError("Unknown model alias.")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        # The backend receives a separate mutable view. Its behavior cannot
        # replace the request-owned labels/rubrics used for serialization.
        contract = copy.deepcopy(questions)
        try:
            scores, tokens = scorer.score(state, questions)
            # A broken backend score vector is a server failure, not a valid decision.
            result = response(model_name, contract, scores, scorer.temperatures, tokens)
            validate_response(result, contract, model=model_name)
        except RequestLimitError as error:
            raise HTTPException(status_code=422, detail=(
                "Request exceeds this model's verified token or candidate limits; "
                "inputs are never truncated."
            )) from error
        except Exception:
            # Even an upstream HTTPException must not forward a model's raw body.
            return failed_decision()
        return JSONResponse(result, headers={
            "X-Bobcat-Confidence-Method": CONFIDENCE_PROFILE,
            "X-Bobcat-Release-Validated": str(scorer.release_gate_passed).lower(),
        })

    async def systemone(request: Request):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_body_bytes:
                raise HTTPException(status_code=413, detail="Request exceeds 2 MiB.")
            chunks.append(chunk)
        try:
            payload = decode_request(b"".join(chunks))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return await run_in_threadpool(evaluate, payload)

    # Request is a local optional dependency; bind its concrete type for FastAPI.
    systemone.__annotations__["request"] = Request
    app.add_api_route("/v1/systemone", systemone, methods=["POST"])

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--allow-unvalidated", action="store_true")
    args = parser.parse_args()
    scorer = StudentScorer(args.checkpoint, args.tokenizer, device=args.device,
                           allow_unvalidated=args.allow_unvalidated)
    print(json.dumps({
        "model": scorer.model_name, "release_gate_passed": scorer.release_gate_passed,
    }))
    uvicorn.run(create_app(scorer), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
