"""AH-04 diagnostics: explicit host transitions versus isolated question branches.

These synthetic workflows have no external side effects. A downstream question
reads the assignment actually committed by the host, never the evaluator's gold.
Joint planning and a flat pair of isolated questions are separate conditions.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from bobcat.corpus import atomic_json
from bobcat.protocol import parse_request, response
from bobcat.schema import file_hash, json_hash

SCHEMA = "bobcat-workflow-probes-v1"


def build_suite(seed=20260922):
    rng, cases = random.Random(seed), []
    for language in ("ko", "en"):
        ko = language == "ko"
        families = [
            ("payment", ("duplicate_charge", "cancel_requested"),
             ("결제", "주문", "일반") if ko else ("billing", "orders", "general")),
            ("access", ("account_locked", "app_crashed"),
             ("계정", "기술", "일반") if ko else ("accounts", "technical", "general")),
        ]
        for family, fields, routes in families:
            for bits in itertools.product((False, True), repeat=2):
                desks = [f"창구-{c}" if ko else f"desk-{c}" for c in ("K", "M", "R")]
                # A host table is part of the supplied policy, not fixed label knowledge.
                rng.shuffle(desks)
                mapping = dict(zip(routes, desks, strict=True))
                rules = [
                    {"field": fields[0], "equals": True, "route": routes[0]},
                    {"field": fields[1], "equals": True, "route": routes[1]},
                    {"otherwise": routes[2]},
                ]
                state = {
                    "facts": dict(zip(fields, bits, strict=True)),
                    "routing_rules": rules, "destinations": mapping,
                }
                route = routes[0] if bits[0] else routes[1] if bits[1] else routes[2]
                unassigned = "아직 배정되지 않음" if ko else "not assigned yet"
                route_order, desk_order = list(routes), [*desks, unassigned]
                rng.shuffle(route_order)
                rng.shuffle(desk_order)
                assign = {
                    "type": "choice",
                    "instructions": (
                        "facts에 맞는 routing_rules를 위에서부터 적용해 처음 일치하는 "
                        "route를 고르세요. otherwise는 마지막 기본값입니다. "
                        "이 답을 쓰는 것만으로 state의 배정 기록이 바뀌지는 않습니다."
                        if ko else
                        "Apply routing_rules from top to bottom to facts and choose the first "
                        "matching route. otherwise is the final default. Answering this "
                        "question does not itself change the assignment record in state."
                    ),
                    "criteria": dict.fromkeys(route_order),
                }
                downstream = {
                    "type": "choice",
                    "instructions": (
                        "state.recorded_assignment에 호스트가 실제로 기록한 route의 창구를 "
                        "destinations에서 고르세요. 기록이 없으면 '아직 배정되지 않음'입니다. "
                        "facts로 앞으로 선택할 route를 추정하지 마세요."
                        if ko else
                        "Choose the destination of the route actually recorded by the host "
                        "in state.recorded_assignment. If the record is absent, choose "
                        "'not assigned yet'. Do not infer a future route from facts."
                    ),
                    "criteria": dict.fromkeys(desk_order),
                }
                pairs = list(itertools.product(routes, desks))
                rng.shuffle(pairs)
                joint_labels = {f"{r} → {d}": {"route": r, "destination": d} for r, d in pairs}
                joint = {
                    "type": "choice",
                    "instructions": (
                        "하나의 실행 계획을 고르세요. facts에 routing_rules를 위에서부터 "
                        "적용해 첫 일치 route를 정하고 그 route의 destinations 창구와 "
                        "짝지으세요. otherwise는 기본값입니다. 실제 배정 기록은 아직 없으며 "
                        "이 질문은 배정할 계획을 묻습니다."
                        if ko else
                        "Choose a single execution plan: apply routing_rules in order to facts, "
                        "take the first matching route, and pair it with its destination. "
                        "otherwise is the default. There is no committed assignment yet; "
                        "this question asks for a plan to commit."
                    ),
                    "criteria": joint_labels,
                }
                world = f"{family}-{int(bits[0])}{int(bits[1])}"
                alternative = next(r for r in routes if r != route)
                cases.append({
                    "id": f"{language}-{world}", "world": world, "language": language,
                    "state": state, "assign": assign, "downstream": downstream, "joint": joint,
                    "gold": {"route": route, "destination": mapping[route],
                             "joint": f"{route} → {mapping[route]}", "unassigned": unassigned},
                    "intervention_route": alternative,
                })
    result = {
        "schema": SCHEMA, "seed": seed, "purpose": "dependency_diagnostic_not_final_quality",
        "cases": cases,
    }
    result["content_sha256"] = json_hash(result)
    return result


def _payload(state, **questions):
    return {"model": "bobcat-latest", "state": copy.deepcopy(state),
            "questions": copy.deepcopy(questions)}


def validate_suite(suite):
    if (suite.get("schema") != SCHEMA or not 1 <= len(suite.get("cases", [])) <= 128
            or suite.get("content_sha256") != json_hash({
                k: v for k, v in suite.items() if k != "content_sha256"
            })):
        raise ValueError("Use a bounded, checksummed workflow diagnostic suite.")
    seen = set()
    for case in suite["cases"]:
        if case["id"] in seen or case["language"] not in ("ko", "en"):
            raise ValueError("Workflow case IDs and languages must be explicit.")
        seen.add(case["id"])
        state, gold = case["state"], case["gold"]
        if "recorded_assignment" in state:
            raise ValueError("Initial state cannot contain a committed assignment.")
        for kind in ("assign", "downstream", "joint"):
            _, questions = parse_request(_payload(state, decision=case[kind]))
            if questions[0].kind != "choice":
                raise ValueError("This diagnostic uses explicit categorical host transitions.")
        routes, destinations = case["assign"]["criteria"], state["destinations"]
        intervention = case["intervention_route"]
        expected = None
        for rule in state["routing_rules"]:
            if "otherwise" in rule:
                expected = rule["otherwise"]
                break
            if state["facts"].get(rule["field"]) == rule["equals"]:
                expected = rule["route"]
                break
        if (set(routes) != set(destinations) or gold["route"] not in routes
                or gold["route"] != expected
                or intervention not in routes or intervention == gold["route"]
                or gold["destination"] != destinations[gold["route"]]
                or gold["joint"] not in case["joint"]["criteria"]
                or case["joint"]["criteria"][gold["joint"]] != {
                    "route": gold["route"], "destination": gold["destination"],
                }
                or not {*destinations.values(), gold["unassigned"]}
                <= set(case["downstream"]["criteria"])):
            raise ValueError("Host mapping, interventions and diagnostic gold must align.")
    return {"cases": len(seen), "worlds": len({c["world"] for c in suite["cases"]})}


def _call(scorer, payload, condition, deadline):
    row = {"condition": condition, "request": copy.deepcopy(payload),
           "request_sha256": json_hash(payload), "status": "running"}
    start = time.monotonic()
    try:
        if start >= deadline:
            raise TimeoutError("Workflow diagnostic deadline reached.")
        state, questions = parse_request(copy.deepcopy(payload))
        before = json_hash(state)
        scores, tokens = scorer.score(state, questions)
        if json_hash(state) != before:
            raise ValueError("The scorer mutated shared input state.")
        if time.monotonic() >= deadline:
            raise TimeoutError("Workflow request completed after the diagnostic deadline.")
        row.update(
            status="completed", raw_scores=scores, logical_input_tokens=tokens,
            response=response(scorer.model_name, questions, scores,
                              getattr(scorer, "temperatures", {}), tokens),
            native_measurement=copy.deepcopy(getattr(scorer, "last_measurement", None)),
        )
    except Exception as error:
        row.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        row["transport_or_deadline_failure"] = isinstance(
            error, (httpx.TransportError, TimeoutError),
        )
    row["wall_seconds"] = time.monotonic() - start
    return row


def run_suite(suite, scorer, out, *, max_seconds=900):
    """Run only host-simulated transitions, with no network effects besides scoring.

    The supplied scorer must have its own finite request timeout. After a
    transport/deadline error no further scoring is attempted. Request timeouts
    do not establish that remote GPU execution has quiesced.
    """
    counts = validate_suite(suite)
    if out.exists() or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 3600:
        raise ValueError("Use a fresh output and finite workflow study allowance.")
    out.mkdir(parents=True)
    started, observations = time.monotonic(), []
    deadline = started + max_seconds
    record = {
        "schema": "bobcat-workflow-run-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "max_seconds": max_seconds,
        "suite_sha256": suite["content_sha256"], "evaluator_sha256": file_hash(Path(__file__)),
        "scorer_provenance": copy.deepcopy(getattr(scorer, "provenance", {})),
        "planned_cases": counts["cases"], "worlds": counts["worlds"],
        "external_actions_executed": False, "marginal_probabilities_multiplied": False,
        "release_gate_passed": False,
        "timing_scope": "Sequential host/compiler/native/API time, not GPU-only.",
        "joint_scope": "A complete route-and-destination plan, not a sibling answer lookup.",
        "timeout_scope": "Caller supplies finite HTTP timeout; remote quiescence is not inferred.",
    }
    atomic_json(out / "run.json", record)
    stop = False
    try:
        for case in suite["cases"]:
            if stop or time.monotonic() >= deadline:
                record["status"] = "incomplete"
                break
            row = {"id": case["id"], "world": case["world"], "language": case["language"],
                   "calls": [], "status": "running"}

            def call(payload, condition, *, case_record=row):
                nonlocal stop
                if stop:
                    return {"condition": condition, "status": "blocked",
                            "reason": "prior_transport_or_deadline_failure"}
                value = _call(scorer, payload, condition, deadline)
                case_record["calls"].append(value)
                stop = value.get("transport_or_deadline_failure", False)
                return value

            staged_began = time.monotonic()
            route_call = call(_payload(case["state"], assign=case["assign"]), "staged_assign")
            destination_call = {"status": "blocked", "reason": "upstream_failed"}
            selected = None
            if route_call["status"] == "completed":
                selected = route_call["response"]["answers"]["assign"]["choice"]
                committed = copy.deepcopy(case["state"])
                committed["recorded_assignment"] = {"route": selected}
                destination_call = call(
                    _payload(committed, destination=case["downstream"]), "staged_downstream",
                )
            destination = (
                destination_call["response"]["answers"]["destination"]["choice"]
                if destination_call["status"] == "completed" else None
            )
            row["staged"] = {
                "selected_route": selected, "selected_destination": destination,
                "controller_wall_seconds": time.monotonic() - staged_began,
                "downstream_status": destination_call["status"],
                "downstream_reads_actual_assignment": (
                    selected is not None and destination == case["state"]["destinations"][selected]
                ),
                "task_success": selected == case["gold"]["route"]
                and destination == case["gold"]["destination"],
            }

            joint_began = time.monotonic()
            joint_call = call(_payload(case["state"], plan=case["joint"]), "joint_plan")
            row["joint"] = {
                "status": joint_call["status"], "task_success":
                joint_call["status"] == "completed"
                and joint_call["response"]["answers"]["plan"]["choice"] == case["gold"]["joint"],
                "controller_wall_seconds": time.monotonic() - joint_began,
            }
            flat_call = call(_payload(
                case["state"], assign=case["assign"], destination=case["downstream"],
            ), "flat_isolated_questions")
            row["flat"] = {
                "status": flat_call["status"], "assignment_was_committed": False,
                "correctly_reports_unassigned": flat_call["status"] == "completed"
                and flat_call["response"]["answers"]["destination"]["choice"]
                == case["gold"]["unassigned"],
            }
            overridden = copy.deepcopy(case["state"])
            overridden["recorded_assignment"] = {"route": case["intervention_route"]}
            override_call = call(
                _payload(overridden, destination=case["downstream"]), "host_intervention",
            )
            row["intervention"] = {
                "status": override_call["status"],
                "host_route": case["intervention_route"],
                "followed_record_instead_of_original_policy": override_call["status"] == "completed"
                and override_call["response"]["answers"]["destination"]["choice"]
                == case["state"]["destinations"][case["intervention_route"]],
            }
            row["status"] = "completed" if (
                len(row["calls"]) == 5 and all(c["status"] == "completed" for c in row["calls"])
            ) else "incomplete"
            observations.append(row)
            with (out / "observations.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            record["status"] = "completed" if all(
                row["status"] == "completed" for row in observations
            ) else "completed_with_failures"
        if stop:
            record["status"] = "incomplete_transport_or_deadline"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        record.update(
            finished_at=datetime.now(UTC).isoformat(), wall_seconds=time.monotonic() - started,
            attempted_cases=len(observations),
            completed_cases=sum(row["status"] == "completed" for row in observations),
            native_score_calls=sum(len(row["calls"]) for row in observations),
            stopped_after_transport_or_deadline_failure=stop,
        )
        record["by_language"] = {}
        for language in ("ko", "en"):
            rows = [row for row in observations if row["language"] == language]
            record["by_language"][language] = {
                "attempted_cases": len(rows),
                "staged_task_successes": sum(row["staged"]["task_success"] for row in rows),
                "joint_task_successes": sum(row["joint"]["task_success"] for row in rows),
                "flat_unassigned_successes": sum(
                    row["flat"]["correctly_reports_unassigned"] for row in rows
                ),
                "intervention_successes": sum(
                    row["intervention"]["followed_record_instead_of_original_policy"]
                    for row in rows
                ),
            }
        atomic_json(out / "run.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve existing diagnostic suites.")
    suite = build_suite()
    validate_suite(suite)
    atomic_json(args.out, suite)


if __name__ == "__main__":
    main()
