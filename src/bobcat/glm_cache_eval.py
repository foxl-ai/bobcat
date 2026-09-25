"""Bounded cache/batching comparisons on an exclusively owned native GLM server.

Cache salts are transport metadata, never prompt tokens. Reported cache hits and
HTTP timings do not by themselves establish physical GPU prefix reuse.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from bobcat.corpus import atomic_json
from bobcat.glm_readout import GLMCompiler, SGLangScorer
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash

SCHEMA = "bobcat-glm-cache-cases-v1"
CONDITIONS = (
    "isolated_batch", "shared_batch", "isolated_sequential", "shared_sequential",
    "warm_shared_batch", "warm_shared_sequential",
)


def build_cases() -> dict:
    """Synthetic inventory checks exercise serving, not general model quality."""
    cases = []
    for language in ("ko", "en"):
        for question_count in (1, 8, 32):
            for filler_count in (0, 128):
                records, questions = [], {}
                for index in range(question_count):
                    identifier = f"item-{index:03d}"
                    records.append({
                        "item": identifier, "stock": index % 5, "required": (index + 2) % 5,
                    })
                    instruction = (
                        f"{identifier}의 stock이 required 이상인가?"
                        if language == "ko"
                        else f"Is stock greater than or equal to required for {identifier}?"
                    )
                    questions[f"q{index}"] = {"type": "noul", "instructions": instruction}
                filler = (
                    "이 문단은 보관용 설명이며 재고 수량을 변경하지 않는다."
                    if language == "ko"
                    else "This paragraph is archival context and does not change stock quantities."
                )
                request = {
                    "model": "bobcat-latest",
                    "state": {"inventory": records, "notes": [filler] * filler_count},
                    "questions": questions,
                }
                cases.append({
                    "id": f"{language}-q{question_count}-notes{filler_count}",
                    "language": language, "request": request,
                    "priming_question": {
                        "type": "noul", "instructions": (
                            "state에 inventory라는 목록이 있는가?" if language == "ko"
                            else "Does the state contain an inventory list?"
                        ),
                    },
                })
    result = {"schema": SCHEMA, "purpose": "serving_consistency_not_quality", "cases": cases}
    result["content_sha256"] = json_hash(result)
    return result


def validate_cases(suite, compiler):
    if (suite.get("schema") != SCHEMA or not 1 <= len(suite.get("cases", [])) <= 24
            or suite.get("content_sha256") != json_hash({
                key: value for key, value in suite.items() if key != "content_sha256"
            })):
        raise ValueError("Use the bounded, checksummed serving suite.")
    seen, compiled = set(), []
    for case in suite["cases"]:
        if case["id"] in seen:
            raise ValueError("Serving case IDs must be unique.")
        seen.add(case["id"])
        state, questions = parse_request(case["request"])
        if len(questions) > 32:
            raise ValueError("The initial serving study supports at most 32 questions.")
        together = compiler.compile(state, questions)
        separate = [compiler.compile(state, [q]) for q in questions]
        if together.input_ids != [item.input_ids[0] for item in separate]:
            raise ValueError("Batch and standalone model inputs differ.")
        if len({tuple(ids) for ids in together.input_ids}) != len(questions):
            raise ValueError("Use distinct question suffixes, not repeated full prompts.")
        _, primer_questions = parse_request({
            "model": case["request"]["model"], "state": state,
            "questions": {"primer": case["priming_question"]},
        })
        primer = compiler.compile(state, primer_questions)
        prefix = together.shared_prefix_tokens
        if (primer.input_ids[0] in together.input_ids
                or primer.input_ids[0][:prefix] != together.input_ids[0][:prefix]):
            raise ValueError("Priming must reuse the state but introduce a distinct question.")
        compiled.append({
            "case_id": case["id"], "language": case["language"], "questions": len(questions),
            "prompt_sha256": [json_hash(ids) for ids in together.input_ids],
            "priming_prompt_sha256": json_hash(primer.input_ids[0]),
            "priming_prompt_tokens": len(primer.input_ids[0]),
            "shared_prefix_tokens": together.shared_prefix_tokens,
            "logical_input_tokens": together.logical_input_tokens,
            "native_prompt_tokens": sum(map(len, together.input_ids)),
        })
    return compiled


class SaltedTransport:
    """Add a native cache namespace without changing the compiled model input."""

    def __init__(self, client, deadline, namespace):
        self.client, self.deadline, self.namespace = client, deadline, namespace
        self.shared, self.counter = False, 0

    def get(self, path):
        remaining = self.deadline - time.monotonic()
        if path != "/get_model_info" or remaining <= 0:
            raise TimeoutError("Serving study deadline or unexpected native endpoint.")
        return self.client.get(path, timeout=min(60, remaining))

    def post(self, path, *, json):
        if path != "/generate" or self.deadline <= time.monotonic():
            raise TimeoutError("Serving study deadline or unexpected native endpoint.")
        payload = copy.deepcopy(json)
        if "cache_salt" in payload:
            raise ValueError("The study owns the native cache namespace.")
        self.counter += 1
        payload["cache_salt"] = (
            [self.namespace] * len(payload["input_ids"]) if self.shared else
            [f"{self.namespace}:{self.counter}:{index}"
             for index in range(len(payload["input_ids"]))]
        )
        return self.client.post(
            path, json=payload, timeout=min(60, self.deadline - time.monotonic()),
        )


def distribution_changes(reference, observed):
    if len(reference) != len(observed):
        raise ValueError("Compare the same questions and all their options.")
    changes = []
    for before, after in zip(reference, observed, strict=True):
        if len(before) != len(after) or not before:
            raise ValueError("Compare the same candidate sets.")
        probabilities = []
        for row in (before, after):
            if not all(math.isfinite(x) for x in row):
                raise ValueError("Non-finite native scores are a failed comparison.")
            maximum = max(row)
            mass = [math.exp(value - maximum) for value in row]
            probabilities.append([value / sum(mass) for value in mass])
        p, q = probabilities
        changes.append({
            "tv": sum(abs(a - b) for a, b in zip(p, q, strict=True)) / 2,
            "argmax_changed": max(range(len(p)), key=p.__getitem__)
            != max(range(len(q)), key=q.__getitem__),
            "top_probability_threshold_crossings": {
                str(t): (max(p) >= t) != (max(q) >= t) for t in (0.8, 0.9, 0.95)
            },
        })
    return changes


def run(suite, compiler, client, upstream_model_path, out, *,
        max_seconds=1200, repeats=2, dedicated_server=False):
    if (not dedicated_server or out.exists() or not 30 <= max_seconds <= 3600
            or not 1 <= repeats <= 3):
        raise ValueError("Use a dedicated server, fresh output and bounded study.")
    # Complete input checks before clearing any server cache.
    compiled = validate_cases(suite, compiler)
    start = time.monotonic()
    deadline = start + max_seconds
    namespace = "bobcat-cache-" + json_hash({
        "out": str(out.resolve()), "started_ns": time.time_ns(),
    })[:24]
    transport = SaltedTransport(client, deadline, namespace)
    scorer = SGLangScorer(compiler, "", upstream_model_path, client=transport)
    out.mkdir(parents=True)
    record = {
        "schema": "bobcat-glm-cache-run-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "max_seconds": max_seconds,
        "suite_sha256": suite["content_sha256"], "evaluator_sha256": file_hash(Path(__file__)),
        "scorer_provenance": scorer.provenance, "compiled_cases": compiled,
        "repeats": repeats, "temperature": 1.0, "dedicated_server": True,
        "physical_prefix_reuse_verified": False, "gpu_kernel_profile_collected": False,
        "calibration_refitted": False, "release_gate_passed": False,
        "timing_scope": "Client compile/serialization, native HTTP, queue and model; not GPU-only.",
        "cache_control": "Distinct native cache salts versus a shared salt; reset before each arm.",
        "warm_condition": "Prime the same state with a distinct question; exclude priming from "
        "the measured question time but retain its time and native counters.",
        "kernel_warmup": "No separate kernel warmup; randomized order cannot remove all first-use "
        "compilation effects. This diagnostic is not an optimized latency claim.",
    }
    atomic_json(out / "run.json", record)
    observations, comparisons = [], []
    try:
        for case_index, case in enumerate(suite["cases"]):
            state, questions = parse_request(case["request"])
            _, primer_questions = parse_request({
                "model": case["request"]["model"], "state": state,
                "questions": {"primer": case["priming_question"]},
            })
            for repeat in range(repeats):
                order = list(CONDITIONS)
                random.Random(20260922 + 1009 * case_index + repeat).shuffle(order)
                block = {}
                for condition in order:
                    if deadline - time.monotonic() < 1:
                        raise TimeoutError("Finite serving study deadline reached.")
                    # This administrative action is restricted to our exclusive server.
                    reset = client.post("/flush_cache", params={"timeout": 5},
                                        timeout=min(10, deadline - time.monotonic()))
                    reset.raise_for_status()
                    transport.namespace = f"{namespace}:{case_index}:{repeat}:{condition}"
                    transport.shared = "shared" in condition
                    batch_mode = condition.endswith("batch")
                    groups = [questions] if batch_mode else [[q] for q in questions]
                    row = {"case_id": case["id"], "repeat": repeat, "condition": condition,
                           "questions": len(questions), "status": "running", "native_calls": [],
                           "cache_state_at_measurement": (
                               "primed_with_distinct_question" if condition.startswith("warm")
                               else "cold_reset"
                           )}
                    condition_began, began, scores = time.monotonic(), None, []
                    call_error = None
                    try:
                        if condition.startswith("warm"):
                            priming_began = time.monotonic()
                            scorer.score(state, primer_questions)
                            row["priming_wall_seconds"] = time.monotonic() - priming_began
                            row["priming_native_call"] = copy.deepcopy(scorer.last_measurement)
                        began = time.monotonic()
                        for group in groups:
                            values, _ = scorer.score(state, group)
                            scores.extend(values)
                            row["native_calls"].append(copy.deepcopy(scorer.last_measurement))
                        row.update(status="completed", logits=scores)
                    except Exception as error:
                        call_error = error
                        row.update(status="failed", error_type=type(error).__name__,
                                   error=str(error)[:1000])
                    row["wall_seconds"] = time.monotonic() - began if began is not None else None
                    row["including_priming_wall_seconds"] = time.monotonic() - condition_began
                    row["completed_questions"] = len(scores)
                    observations.append(row)
                    with (out / "observations.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    block[condition] = row
                    if call_error is not None:
                        raise RuntimeError(
                            "Native cache-study failure; end the dedicated server before "
                            "another condition. Remote quiescence is unverified."
                        ) from call_error
                reference = block["isolated_sequential"]
                for condition in CONDITIONS:
                    candidate = block[condition]
                    if reference["status"] != "completed" or candidate["status"] != "completed":
                        comparisons.append({"case_id": case["id"], "repeat": repeat,
                                            "condition": condition, "status": "missing_arm"})
                        continue
                    comparisons.append({
                        "case_id": case["id"], "repeat": repeat, "condition": condition,
                        "status": "compared", "reference": "isolated_sequential",
                        "probability_changes": distribution_changes(
                            reference["logits"], candidate["logits"],
                        ),
                        "reference_total_http_wall_seconds": reference["wall_seconds"],
                        "condition_total_http_wall_seconds": candidate["wall_seconds"],
                        "batch_total_divided_by_questions_is_sequential_latency": False,
                    })
        record["status"] = (
            "completed" if all(r["status"] == "completed" for r in observations)
            else "completed_with_failures"
        )
    except TimeoutError as error:
        record.update(status="deadline", error=str(error))
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        record.update(
            finished_at=datetime.now(UTC).isoformat(), wall_seconds=time.monotonic() - start,
            planned_arms=len(suite["cases"]) * repeats * len(CONDITIONS),
            attempted_arms=len(observations),
            failed_arms=sum(r["status"] != "completed" for r in observations),
        )
        atomic_json(out / "comparisons.json", comparisons)
        atomic_json(out / "run.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "preflight", "run"))
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--upstream", default="http://127.0.0.1:30000")
    parser.add_argument("--upstream-model-path", default="/models/GLM-5.3-Flash")
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dedicated-server", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve an existing artifact; choose a new output.")
    if args.action == "build":
        atomic_json(args.out, build_cases())
        return
    if not all((args.model_dir, args.source, args.suite)):
        parser.error("Supply pinned source, original tokenizer and serving suite.")
    source = json.loads(args.source.read_text())
    compiler = GLMCompiler(args.model_dir, source)
    suite = json.loads(args.suite.read_text())
    if args.action == "preflight":
        atomic_json(args.out, {
            "schema": "bobcat-glm-cache-preflight-v1", "suite_sha256": file_hash(args.suite),
            "cases": validate_cases(suite, compiler), "model_inference_performed": False,
        })
        return
    address = urlparse(args.upstream)
    if (address.scheme != "http" or address.hostname not in ("127.0.0.1", "localhost")
            or address.username or address.password):
        parser.error("Use the exclusive local research server.")
    import httpx
    with httpx.Client(base_url=args.upstream, timeout=60) as client:
        result = run(
            suite, compiler, client, args.upstream_model_path, args.out,
            max_seconds=args.max_seconds, repeats=args.repeats,
            dedicated_server=args.dedicated_server,
        )
    print(json.dumps({k: result[k] for k in ("status", "attempted_arms", "failed_arms")}))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
