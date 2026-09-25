"""Compare identical GLM decisions with zero/one completion and explicit state priming.

The parent is exactly the compiler's shared token prefix. Native hybrid cache
matching may reuse only an aligned subset; this module makes no state-handle,
zero-copy, numerical-equivalence or GPU-compute claim from API counters.
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
from bobcat.glm_cache_eval import distribution_changes
from bobcat.glm_profile import profile_generation
from bobcat.glm_readout import CompiledRequest, extract_scores, native_payload
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash

SCHEMA = "bobcat-glm-prefill-plan-v1"
MODES = ("one_token", "prefill_only")
CACHE_CONDITIONS = ("isolated", "shared_cold", "state_prepared")
ARMS = tuple((mode, cache) for mode in MODES for cache in CACHE_CONDITIONS)


def load_plan(path: Path):
    plan = json.loads(path.read_text())
    expected = {
        "schema", "cases", "repeats", "profile_case_id", "profile_blocks",
        "expected_tp_ranks", "max_study_seconds", "content_sha256",
    }
    if (set(plan) != expected or plan["schema"] != SCHEMA
            or plan["content_sha256"] != json_hash({
                k: v for k, v in plan.items() if k != "content_sha256"
            }) or type(plan["repeats"]) is not int or plan["repeats"] not in (2, 3)
            or type(plan["profile_blocks"]) is not int or plan["profile_blocks"] not in (1, 2)
            or plan["expected_tp_ranks"] != 8
            or type(plan["max_study_seconds"]) is not int
            or not 300 <= plan["max_study_seconds"] <= 3600
            or not isinstance(plan["cases"], list) or not 1 <= len(plan["cases"]) <= 24):
        raise ValueError("Use the finite, checksummed prefill study plan.")
    return plan


def state_prefix(compiled: CompiledRequest):
    length = compiled.shared_prefix_tokens
    if (type(length) is not int or length <= 0 or not compiled.input_ids
            or any(len(ids) <= length for ids in compiled.input_ids)):
        raise ValueError("A state primer must be a strict shared prefix of every branch.")
    prefix = compiled.input_ids[0][:length]
    if any(ids[:length] != prefix for ids in compiled.input_ids):
        raise ValueError("Question branches do not share the same state prefix.")
    # Read one arbitrary score to verify completion of prefill; never a task answer.
    return CompiledRequest(
        input_ids=[prefix], option_token_ids=[compiled.option_token_ids[0][:1]],
        shared_prefix_tokens=length, logical_input_tokens=length,
    )


def preflight(plan, compiler):
    identifiers, compiled, records = set(), {}, []
    for case in plan["cases"]:
        if (set(case) != {"id", "language", "request", "source"}
                or not isinstance(case["id"], str) or not case["id"]
                or case["id"] in identifiers or case["language"] not in ("ko", "en")):
            raise ValueError("Use unique, explicitly sourced Korean/English cases.")
        identifiers.add(case["id"])
        state, questions = parse_request(case["request"])
        if not 1 <= len(questions) <= 32:
            raise ValueError("This finite study accepts at most 32 question branches.")
        current = compiler.compile(state, questions)
        separate = [compiler.compile(state, [q]).input_ids[0] for q in questions]
        if current.input_ids != separate:
            raise ValueError("Standalone and batch token inputs differ.")
        primer = state_prefix(current)
        a, b = (native_payload(current, readout_mode=mode) for mode in MODES)
        a["sampling_params"]["max_new_tokens"] = 0
        if a != b:
            raise ValueError("The zero/one-completion arms must have identical model inputs.")
        records.append({
            "case_id": case["id"], "language": case["language"],
            "source": case["source"], "questions": len(questions),
            "candidate_counts": [len(ids) for ids in current.option_token_ids],
            "compiled_input_sha256": json_hash(current.input_ids),
            "option_ids_sha256": json_hash(current.option_token_ids),
            "state_prefix_sha256": json_hash(primer.input_ids[0]),
            "shared_prefix_tokens": current.shared_prefix_tokens,
            "logical_input_tokens": current.logical_input_tokens,
            "native_prompt_tokens": sum(map(len, current.input_ids)),
        })
        compiled[case["id"]] = current
    if plan["profile_case_id"] not in identifiers:
        raise ValueError("Profile one of the frozen cases.")
    return compiled, records


def wait_scheduler_idle(client, deadline, *, not_before=0.0, expected_accelerators=8):
    """Require a fresh scheduler snapshot, not just a completed HTTP response.

    This is reported scheduler quiescence. It does not synchronize CUDA streams
    or substitute for per-rank profiling.
    """
    started, observations = time.monotonic(), []
    while True:
        left = min(deadline - time.monotonic(), 15 - (time.monotonic() - started))
        if left <= 0:
            raise TimeoutError("No fresh idle scheduler snapshot within the finite allowance.")
        result = client.get("/v1/loads", params={"include": "core"}, timeout=min(5, left))
        result.raise_for_status()
        body = result.json()
        loads = body.get("loads") if isinstance(body, dict) else None
        if (not isinstance(loads, list) or len(loads) != 1
                or body.get("num_accelerators") != expected_accelerators
                or loads[0].get("dp_rank") != 0):
            raise ValueError("The dedicated TP server must expose its one DP scheduler.")
        row = loads[0]
        for field in ("num_running_reqs", "num_waiting_reqs"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ValueError("Malformed scheduler request counts.")
        stamp = row.get("timestamp")
        if (isinstance(stamp, bool) or not isinstance(stamp, (float, int))
                or not math.isfinite(stamp)):
            raise ValueError("Scheduler freshness is missing.")
        observations.append(copy.deepcopy(row))
        if stamp >= not_before and row["num_running_reqs"] == row["num_waiting_reqs"] == 0:
            return {
                "scheduler_reported_idle": True, "gpu_streams_synchronized": False,
                "wall_seconds": time.monotonic() - started,
                "snapshots": observations, "not_before_unix": not_before,
            }
        time.sleep(min(0.05, left))


def read_response(body, compiled, mode):
    if not isinstance(body, list) or len(body) != len(compiled.input_ids):
        raise ValueError("Readout returned a different branch count.")
    if mode == "prefill_only" and any(
        row.get("text") not in (None, "") or row.get("output_ids") not in (None, [])
        for row in body
    ):
        raise ValueError("Prefill-only readout returned generated output.")
    return [
        extract_scores(row["meta_info"], options, len(ids), readout_mode=mode)
        for row, options, ids in zip(
            body, compiled.option_token_ids, compiled.input_ids, strict=True,
        )
    ]


def request_scores(client, compiled, mode, salts, deadline, *, trace_root=None):
    if len(salts) != len(compiled.input_ids):
        raise ValueError("Every branch needs one explicit cache namespace.")
    payload = native_payload(compiled, readout_mode=mode)
    payload["cache_salt"] = salts
    remaining = deadline - time.monotonic()
    if remaining < 10:
        raise TimeoutError("Insufficient time for a bounded native readout.")
    started, began_at = time.monotonic(), time.time()
    trace = None
    if trace_root is None:
        result = client.post("/generate", json=payload, timeout=min(90, remaining))
        result.raise_for_status()
        body = result.json()
        native_seconds = time.monotonic() - started
    else:
        body, trace = profile_generation(
            client, payload, trace_root, "/bobcat-profiles", expected_ranks=8,
            deadline=min(deadline, time.monotonic() + 180), dedicated_server=True,
        )
        native_seconds = trace["instrumented_generate_wall_seconds"]
        if not trace["gpu_kernel_profile_collected"] or not trace["profiler_stop_confirmed"]:
            raise RuntimeError("Incomplete per-rank profile; end this dedicated server.")
    scores = read_response(body, compiled, mode)
    idle = wait_scheduler_idle(client, deadline, not_before=began_at)
    return {
        "payload_sha256": json_hash(payload),
        "compiled_input_sha256": json_hash(compiled.input_ids),
        "candidate_ids": compiled.option_token_ids,
        "readout_mode": mode, "logits": scores, "native_responses": body,
        "native_http_seconds": native_seconds,
        "including_profile_and_idle_seconds": time.monotonic() - started,
        "native_prompt_tokens": sum(map(len, compiled.input_ids)),
        "native_completion_tokens": sum(row["meta_info"]["completion_tokens"] for row in body),
        "native_scored_positions": len(scores),
        "reported_cached_tokens": [row["meta_info"].get("cached_tokens") for row in body],
        "scheduler_after": idle, "profile": trace,
        "zero_decode_steps_verified": False,
    }


def run_arm(client, compiled, mode, cache, namespace, deadline, *, trace_root=None):
    if (mode, cache) not in ARMS:
        raise ValueError("Unrecognized frozen study arm.")
    wait_scheduler_idle(client, deadline)
    reset = client.post("/flush_cache", params={"timeout": 5},
                        timeout=min(10, max(0.1, deadline - time.monotonic())))
    reset.raise_for_status()
    row = {"readout_mode": mode, "cache_condition": cache, "status": "running"}
    began = time.monotonic()
    if cache == "state_prepared":
        row["state_preparation"] = request_scores(
            client, state_prefix(compiled), "prefill_only", [namespace], deadline,
            trace_root=trace_root,
        )
        row["state_preparation"]["meaning"] = "Cache primer, no question or task answer."
    salts = (
        [f"{namespace}:{i}" for i in range(len(compiled.input_ids))]
        if cache == "isolated" else [namespace] * len(compiled.input_ids)
    )
    row["decision"] = request_scores(
        client, compiled, mode, salts, deadline, trace_root=trace_root,
    )
    row.update(status="completed", including_state_preparation_seconds=time.monotonic() - began)
    return row


def compare_observations(observations):
    """Keep paired modes, cache effects and exact-input repeats separate."""
    index = {
        (r["case_id"], r["repeat"], r["readout_mode"], r["cache_condition"]): r
        for r in observations
    }
    if len(index) != len(observations):
        raise ValueError("Duplicate observation identity.")
    comparisons = []
    for key, row in index.items():
        case_id, repeat, mode, cache = key
        refs = [
            ("readout_mode", (case_id, repeat, "one_token", cache)),
            ("cache_condition", (case_id, repeat, mode, "isolated")),
            ("exact_input_repeat", (case_id, 0, mode, cache)),
        ]
        for kind, reference_key in refs:
            if key == reference_key:
                continue
            before = index.get(reference_key)
            if before is None:
                raise ValueError("A paired readout arm is missing.")
            if before["decision"]["compiled_input_sha256"] != (
                row["decision"]["compiled_input_sha256"]
            ) or before["decision"]["candidate_ids"] != row["decision"]["candidate_ids"]:
                raise ValueError("Comparison inputs or candidate identifiers changed.")
            comparisons.append({
                "kind": kind, "case_id": case_id, "repeat": repeat,
                "readout_mode": mode, "cache_condition": cache,
                "reference": list(reference_key),
                "question_changes": distribution_changes(
                    before["decision"]["logits"], row["decision"]["logits"],
                ),
            })
    return comparisons


def run(plan, compiler, client, model_path, out, trace_root, *, max_seconds,
        dedicated_server=False):
    if (not dedicated_server or client.base_url.host not in ("127.0.0.1", "::1", "localhost")
            or out.exists() or not trace_root.is_dir()
            or not math.isfinite(max_seconds) or not 300 <= max_seconds <= 3600):
        raise ValueError("Use a fresh, bounded study on an exclusively owned local server.")
    compiled, checks = preflight(plan, compiler)
    info = client.get("/get_model_info", timeout=5)
    info.raise_for_status()
    if info.json().get("model_path") != model_path:
        raise ValueError("The dedicated server loaded another model.")
    out.mkdir()
    atomic_json(out / "plan.json", plan)
    atomic_json(out / "preflight.json", checks)
    start = time.monotonic()
    deadline = start + min(max_seconds, plan["max_study_seconds"])
    record = {
        "schema": "bobcat-glm-prefill-study-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "plan_sha256": plan["content_sha256"],
        "evaluator_sha256": file_hash(Path(__file__)),
        "max_seconds": deadline - start, "training_performed": False,
        "calibration_fitted": False, "release_gate_passed": False,
        "zero_decode_steps_verified": False, "physical_prefix_reuse_verified": False,
        "explicit_state_handle_implemented": False, "observations_completed": 0,
        "warmup_completed": 0, "profiles_completed": 0,
        "scope": "Matched runtime diagnostics, not an independent quality benchmark.",
        "state_preparation": "Exact compiler prefix, waited for reported idle; "
        "actual hybrid cache matches may be aligned partial prefixes.",
        "timing_scope": "HTTP/queue/model plus separately recorded preparation and idle; "
        "profiles are instrumented. No optimized latency or independent-sample claim.",
    }
    observations, profiles = [], []
    atomic_json(out / "study.json", record)

    def append(name, row):
        with (out / name).open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    try:
        for case_number, case in enumerate(plan["cases"]):
            current = compiled[case["id"]]
            # Warm each case/mode separately. Preserve first-use costs and responses.
            for mode in MODES:
                namespace = f"bobcat-prefill:{out.name}:warmup:{case_number}:{mode}"
                row = run_arm(client, current, mode, "isolated", namespace, deadline)
                row.update(case_id=case["id"], quality_observation=False)
                append("warmup.jsonl", row)
                record["warmup_completed"] += 1
                atomic_json(out / "study.json", record)
            for repeat in range(plan["repeats"]):
                order = list(ARMS)
                random.Random(20260922 + 1009 * case_number + repeat).shuffle(order)
                for mode, cache in order:
                    identity = f"{case_number}:{repeat}:{mode}:{cache}"
                    record["current_arm"] = identity
                    atomic_json(out / "study.json", record)
                    row = run_arm(
                        client, current, mode, cache, f"bobcat-prefill:{out.name}:{identity}",
                        deadline,
                    )
                    row.update(case_id=case["id"], language=case["language"], repeat=repeat)
                    append("observations.jsonl", row)
                    observations.append(row)
                    record["observations_completed"] += 1
                    atomic_json(out / "study.json", record)
        atomic_json(out / "comparisons.json", {
            "comparisons": compare_observations(observations),
            "correlated_questions_not_independent_samples": True,
        })
        current = compiled[plan["profile_case_id"]]
        for block in range(plan["profile_blocks"]):
            order = list(ARMS)
            random.Random(20260922 + block).shuffle(order)
            for mode, cache in order:
                namespace = f"bobcat-prefill-profile:{out.name}:{block}:{mode}:{cache}"
                record["current_arm"] = namespace
                atomic_json(out / "study.json", record)
                row = run_arm(
                    client, current, mode, cache, namespace, deadline, trace_root=trace_root,
                )
                row.update(case_id=plan["profile_case_id"], block=block)
                append("profiles.jsonl", row)
                profiles.append(row)
                record["profiles_completed"] += 1
                atomic_json(out / "study.json", record)
        record.update(
            status="completed",
            profile_trace_count=sum(
                len(r[k]["profile"]["traces"])
                for r in profiles for k in ("state_preparation", "decision") if k in r
            ),
        )
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - start)
        atomic_json(out / "study.json", record)
    return record
