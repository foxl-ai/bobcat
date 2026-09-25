"""Finite native GLM diagnostics after the first residual-head failure.

These studies diagnose behavior and serving cost. They do not train weights,
fit calibration, establish Jev parity or pass a release gate.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_cache_eval import SaltedTransport, validate_cases
from bobcat.glm_cache_eval import run as run_cache
from bobcat.glm_identifier_controls import preflight as identifier_preflight
from bobcat.glm_identifier_controls import run as run_identifiers
from bobcat.glm_profile import profile_generation
from bobcat.glm_readout import SGLangScorer, extract_scores, native_payload
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash
from bobcat.workflow_probes import run_suite as run_workflows
from bobcat.workflow_probes import validate_suite as validate_workflows

SCHEMA = "bobcat-glm-diagnostic-plan-v1"
PROFILE_CONDITIONS = ("isolated_batch", "shared_batch", "warm_shared_batch")


def load_plan(path: Path, root: Path):
    plan = json.loads(path.read_text())
    expected = {
        "schema", "suites", "allowances_seconds", "cache_repeats", "profile_case_id",
        "profile_blocks", "expected_tp_ranks", "content_sha256",
    }
    if (set(plan) != expected or plan["schema"] != SCHEMA
            or plan["content_sha256"] != json_hash({
                k: v for k, v in plan.items() if k != "content_sha256"
            }) or set(plan["suites"]) != {"identifier", "workflow", "cache"}
            or set(plan["allowances_seconds"]) != {"identifier", "workflow", "cache", "profile"}
            or any(type(x) is not int or not 60 <= x <= 1800
                   for x in plan["allowances_seconds"].values())
            or sum(plan["allowances_seconds"].values()) > 4800
            or plan["cache_repeats"] not in (1, 2)
            or plan["profile_blocks"] not in (1, 2)
            or plan["expected_tp_ranks"] != 8):
        raise ValueError("Use the finite, checksummed Hopper diagnostic plan.")
    suites = {}
    for name, item in plan["suites"].items():
        relative = Path(item["path"])
        target = root / relative
        if (set(item) != {"path", "sha256"} or relative.is_absolute()
                or ".." in relative.parts or target.is_symlink()
                or file_hash(target) != item["sha256"]):
            raise ValueError("A diagnostic suite changed after freezing.")
        suites[name] = json.loads(target.read_text())
    return plan, suites


def workflow_inputs(suite):
    """Compile all possible upstream assignments, including incorrect predictions."""
    validate_workflows(suite)
    payloads = []
    for case in suite["cases"]:
        def add(state, **questions):
            payloads.append({
                "model": "bobcat-latest", "state": state, "questions": questions,
            })

        add(case["state"], assign=case["assign"])
        add(case["state"], plan=case["joint"])
        add(case["state"], assign=case["assign"], destination=case["downstream"])
        for route in case["state"]["destinations"]:
            state = copy.deepcopy(case["state"])
            state["recorded_assignment"] = {"route": route}
            add(state, destination=case["downstream"])
        state = copy.deepcopy(case["state"])
        state["recorded_assignment"] = {"route": case["intervention_route"]}
        add(state, destination=case["downstream"])
    return payloads


def preflight(plan, suites, compiler):
    identifiers = identifier_preflight(suites["identifier"], compiler)
    cache = validate_cases(suites["cache"], compiler)
    if sum(row["case_id"] == plan["profile_case_id"] for row in cache) != 1:
        raise ValueError("The profile must name one frozen serving case.")
    workflow = []
    for payload in workflow_inputs(suites["workflow"]):
        state, questions = parse_request(payload)
        compiled = compiler.compile(state, questions)
        workflow.append({
            "payload_sha256": json_hash(payload),
            "compiled_input_sha256": json_hash(compiled.input_ids),
            "questions": len(questions),
            "max_branch_tokens": max(map(len, compiled.input_ids)),
        })
    return {
        "identifier": identifiers, "workflow_all_possible_inputs": workflow,
        "cache": cache, "model_executed": False,
    }


def run_profiles(plan, suites, compiler, client, model_path, out, trace_root, deadline):
    """Matched native input, randomized conditions and all eight TP rank traces."""
    case = next(c for c in suites["cache"]["cases"] if c["id"] == plan["profile_case_id"])
    state, questions = parse_request(case["request"])
    compiled = compiler.compile(state, questions)
    _, primer_questions = parse_request({
        "model": "bobcat-latest", "state": state,
        "questions": {"primer": case["priming_question"]},
    })
    records = []
    for block in range(plan["profile_blocks"]):
        order = list(PROFILE_CONDITIONS)
        random.Random(20260922 + block).shuffle(order)
        for condition in order:
            left = deadline - time.monotonic()
            if left < 30:
                raise TimeoutError("No time remains for another bounded profile.")
            reset = client.post("/flush_cache", params={"timeout": 5}, timeout=min(10, left))
            reset.raise_for_status()
            namespace = f"bobcat-profile-{out.name}-{block}-{condition}"
            payload = native_payload(compiled)
            payload["cache_salt"] = (
                [f"{namespace}:{i}" for i in range(len(questions))]
                if condition == "isolated_batch" else [namespace] * len(questions)
            )
            row = {
                "case_id": case["id"], "block": block, "condition": condition,
                "compiled_input_sha256": json_hash(compiled.input_ids),
                "logical_input_tokens": compiled.logical_input_tokens,
                "native_prompt_tokens": sum(map(len, compiled.input_ids)),
                "training_performed": False,
            }
            if condition == "warm_shared_batch":
                transport = SaltedTransport(client, deadline, namespace)
                transport.shared = True
                scorer = SGLangScorer(compiler, "", model_path, client=transport)
                scorer.score(state, primer_questions)
                row["primer"] = copy.deepcopy(scorer.last_measurement)
            body, trace = profile_generation(
                client, payload, trace_root, "/bobcat-profiles",
                expected_ranks=plan["expected_tp_ranks"],
                deadline=min(deadline, time.monotonic() + 150), dedicated_server=True,
            )
            if not isinstance(body, list) or len(body) != len(questions):
                raise ValueError("The profiled response does not match the frozen batch.")
            row["raw_scores"] = [
                extract_scores(value["meta_info"], ids, len(prompt))
                for value, ids, prompt in zip(
                    body, compiled.option_token_ids, compiled.input_ids, strict=True,
                )
            ]
            row["cached_tokens_reported"] = [
                value["meta_info"].get("cached_tokens") for value in body
            ]
            row["profile"] = trace
            records.append(row)
            atomic_json(out / "profiles.json", {
                "observations": records, "physical_prefix_reuse_verified": False,
                "timing_scope": "Instrumented, per-rank kernel clocks; not optimized latency.",
            })
            if not trace["gpu_kernel_profile_collected"] or not trace["profiler_stop_confirmed"]:
                raise RuntimeError("Incomplete GPU evidence; end this dedicated server.")
    return {
        "status": "completed", "instrumented_calls": len(records),
        "gpu_kernel_profile_collected": True, "physical_prefix_reuse_verified": False,
        "profile_trace_count": sum(len(row["profile"]["traces"]) for row in records),
    }


def run(plan, suites, compiler, client, model_path, out, trace_root, *,
        max_seconds, dedicated_server=False):
    if (not dedicated_server or client.base_url.host not in ("127.0.0.1", "::1", "localhost")
            or out.exists() or not math.isfinite(max_seconds) or not 120 <= max_seconds <= 4800
            or not trace_root.is_dir()):
        raise ValueError("Use a fresh, bounded diagnostic run on an exclusively owned server.")
    checks = preflight(plan, suites, compiler)
    out.mkdir(parents=True)
    atomic_json(out / "preflight.json", checks)
    atomic_json(out / "plan.json", plan)
    started = time.monotonic()
    deadline = started + max_seconds
    record = {
        "schema": "bobcat-glm-diagnostic-study-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "max_seconds": max_seconds,
        "plan_sha256": plan["content_sha256"], "evaluator_sha256": file_hash(Path(__file__)),
        "training_performed": False, "calibration_fitted": False,
        "physical_prefix_reuse_verified": False, "release_gate_passed": False,
        "studies": {},
    }
    atomic_json(out / "diagnostics.json", record)
    try:
        for name in ("identifier", "workflow", "cache", "profile"):
            left = min(plan["allowances_seconds"][name], deadline - time.monotonic())
            if left < 60:
                record.update(status="deadline", unfinished_study=name)
                break
            record["current_study"] = name
            atomic_json(out / "diagnostics.json", record)
            if name == "identifier":
                result = run_identifiers(
                    suites[name], compiler, client, model_path, out / name, max_seconds=left,
                )
            elif name == "workflow":
                transport = SaltedTransport(
                    client, time.monotonic() + left, "bobcat-workflow-" + out.name,
                )
                scorer = SGLangScorer(compiler, "", model_path, client=transport)
                result = run_workflows(suites[name], scorer, out / name, max_seconds=left)
            elif name == "cache":
                result = run_cache(
                    suites[name], compiler, client, model_path, out / name,
                    max_seconds=left, repeats=plan["cache_repeats"], dedicated_server=True,
                )
            else:
                result = run_profiles(
                    plan, suites, compiler, client, model_path, out, trace_root,
                    time.monotonic() + left,
                )
            record["studies"][name] = result
            atomic_json(out / "diagnostics.json", record)
            if result["status"] != "completed":
                record.update(status="incomplete", unfinished_study=name)
                break
        else:
            record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      wall_seconds=time.monotonic() - started)
        atomic_json(out / "diagnostics.json", record)
    return record
