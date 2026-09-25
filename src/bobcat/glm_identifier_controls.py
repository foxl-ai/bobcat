"""Separate candidate order from identifier binding in a GLM development study.

The same candidate meanings receive a crossed order/binding manipulation plus
an identical-input repeat. This diagnoses a baseline failure, not final quality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_cache_eval import SaltedTransport
from bobcat.glm_readout import GLMCompiler, SGLangScorer
from bobcat.protocol import parse_request, probabilities
from bobcat.schema import file_hash, json_hash

SCHEMA = "bobcat-glm-identifier-controls-v1"
ARMS = ("base", "order_only", "identifier_only", "both", "base_repeat")


def ordered_hash(value):
    """Candidate object order changes model input, so preserve it in this hash."""
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
    ).encode()).hexdigest()


def build(source_suite: dict, *, seed=20260923, blocks=2):
    if (source_suite.get("schema") != "bobcat-architecture-probes-v2"
            or source_suite.get("content_sha256") != json_hash({
                k: v for k, v in source_suite.items() if k != "content_sha256"
            }) or not 1 <= blocks <= 4):
        raise ValueError("Use the unchanged architecture development suite and 1–4 blocks.")
    grouped = defaultdict(list)
    for case in source_suite["cases"]:
        if case["family"] == "many_choices":
            grouped[case["group"]].append(case)
    if not 1 <= len(grouped) <= 24:
        raise ValueError("Use a bounded set of complete semantic groups.")
    cases = []
    for group, original in sorted(grouped.items()):
        anchor = min(original, key=lambda row: row["id"])
        request = copy.deepcopy(anchor["request"])
        _, questions = parse_request(request)
        if len(questions) != 1 or questions[0].kind != "choice":
            raise ValueError("This diagnostic requires a single Choice per request.")
        qid = next(iter(request["questions"]))
        meanings = request["questions"][qid]["criteria"]
        labels = sorted(meanings)  # Does not use gold or prior predictions.
        for row in original:
            if (json_hash(row["request"]) != json_hash(request)
                    or row["gold"] != anchor["gold"]):
                raise ValueError("Group variants changed the decision's meaning or gold.")
        for block in range(blocks):
            rng = random.Random(json_hash([seed, group, block]))
            shuffled_labels, shuffled_slots = labels[:], list(range(len(labels)))
            rng.shuffle(shuffled_labels)
            rng.shuffle(shuffled_slots)
            if shuffled_labels == labels:
                shuffled_labels = labels[1:] + labels[:1]
            if shuffled_slots == list(range(len(labels))):
                shuffled_slots = shuffled_slots[1:] + shuffled_slots[:1]
            base_binding = dict(zip(labels, range(len(labels)), strict=True))
            changed_binding = dict(zip(labels, shuffled_slots, strict=True))
            block_cases = []
            for arm in ARMS:
                order = shuffled_labels if arm in {"order_only", "both"} else labels
                binding = changed_binding if arm in {"identifier_only", "both"} else base_binding
                payload = copy.deepcopy(request)
                payload["questions"][qid]["criteria"] = {
                    label: copy.deepcopy(meanings[label]) for label in order
                }
                block_cases.append({
                    "id": f"{group}:block{block}:{arm}",
                    "semantic_group": group, "block": block, "arm": arm,
                    "language": anchor["language"], "request": payload,
                    "labels_in_order": list(order),
                    "identifier_slots": [binding[label] for label in order],
                    "gold": anchor["gold"],
                })
            rng.shuffle(block_cases)
            cases.extend(block_cases)
    suite = {
        "schema": SCHEMA, "seed": seed, "blocks_per_group": blocks,
        "source_suite_content_sha256": source_suite["content_sha256"],
        "source_suite_ordered_sha256": ordered_hash(source_suite),
        "scope": "Development diagnosis of observed failures; not fresh final or training data.",
        "training_use": False, "cases": cases,
    }
    suite["content_sha256"] = ordered_hash(suite)
    validate(suite)
    return suite


def validate(suite):
    if (suite.get("schema") != SCHEMA or suite.get("training_use") is not False
            or not 5 <= len(suite.get("cases", [])) <= 480
            or suite.get("content_sha256") != ordered_hash({
                k: v for k, v in suite.items() if k != "content_sha256"
            })):
        raise ValueError("Use a bounded suite with its order-preserving checksum.")
    groups, seen = defaultdict(dict), set()
    for case in suite["cases"]:
        _, questions = parse_request(case["request"])
        labels = list(questions[0].labels)
        slots = case["identifier_slots"]
        if (case["id"] in seen or len(questions) != 1 or questions[0].kind != "choice"
                or not 2 <= len(labels) <= 255 or labels != case["labels_in_order"]
                or any(type(i) is not int for i in slots)
                or sorted(slots) != list(range(len(labels))) or case["gold"] not in labels
                or case["arm"] not in ARMS):
            raise ValueError("Candidate order, binding, label or case identity is invalid.")
        seen.add(case["id"])
        block = groups[(case["semantic_group"], case["block"])]
        if case["arm"] in block:
            raise ValueError("Duplicate crossed arm.")
        block[case["arm"]] = case
    for values in groups.values():
        if set(values) != set(ARMS):
            raise ValueError("All four crossed arms and an exact repeat are required.")
        base = values["base"]
        bindings = {
            arm: dict(zip(row["labels_in_order"], row["identifier_slots"], strict=True))
            for arm, row in values.items()
        }
        if (any(json_hash(row["request"]) != json_hash(base["request"])
                or row["gold"] != base["gold"] for row in values.values())
                or bindings["base"] != bindings["order_only"]
                or bindings["identifier_only"] != bindings["both"]
                or bindings["base"] == bindings["identifier_only"]
                or values["order_only"]["labels_in_order"] == base["labels_in_order"]
                or values["both"]["labels_in_order"] != values["order_only"]["labels_in_order"]
                or values["identifier_only"]["labels_in_order"] != base["labels_in_order"]
                or ordered_hash(values["base_repeat"]["request"])
                != ordered_hash(base["request"])
                or bindings["base_repeat"] != bindings["base"]):
            raise ValueError("A crossed arm changed more than order or identifier binding.")


def compile_case(compiler: GLMCompiler, case):
    state, questions = parse_request(case["request"])
    slots = case["identifier_slots"]
    if (len(questions) != 1 or list(questions[0].labels) != case["labels_in_order"]
            or any(type(i) is not int for i in slots)
            or sorted(slots) != list(range(len(questions[0].labels)))
            or len(slots) > len(compiler.identifier_ids)):
        raise ValueError("Use one complete, bijective candidate-to-identifier assignment.")
    controlled = copy.copy(compiler)
    # Preserve the original object and all tokenizer/template behavior.
    controlled.identifiers = [compiler.identifiers[i] for i in slots]
    controlled.identifier_ids = [compiler.identifier_ids[i] for i in slots]
    return controlled.compile(state, questions)


def preflight(suite, compiler):
    validate(suite)
    records, seen = [], {}
    for case in suite["cases"]:
        compiled = compile_case(compiler, case)
        record = {
            "case_id": case["id"], "input_ids_sha256": json_hash(compiled.input_ids),
            "option_token_ids": compiled.option_token_ids[0],
            "prompt_tokens": len(compiled.input_ids[0]),
        }
        records.append(record)
        seen[(case["semantic_group"], case["block"], case["arm"])] = record
    for (group, block, arm), record in seen.items():
        if arm == "base_repeat":
            baseline = seen[(group, block, "base")]
            if (record["input_ids_sha256"] != baseline["input_ids_sha256"]
                    or record["option_token_ids"] != baseline["option_token_ids"]):
                raise ValueError("The noise control changed compiled model input.")
    return records


def run(suite, compiler, client, model_path, out, *, max_seconds=900):
    if out.exists() or not 0 < max_seconds <= 1800:
        raise ValueError("Use a new result directory and at most 1,800 seconds.")
    compiled_records = preflight(suite, compiler)
    out.mkdir(parents=True)
    atomic_json(out / "suite.json", suite)
    atomic_json(out / "preflight.json", {"compiled": compiled_records})
    started = time.monotonic()
    transport = SaltedTransport(client, started + max_seconds, "bobcat-identifiers-" + out.name)
    manifest = {
        "schema": "bobcat-glm-identifier-run-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "suite_sha256": suite["content_sha256"], "evaluator_sha256": file_hash(Path(__file__)),
        "temperature": 1.0,
        "training_performed": False, "calibration_fitted": False,
        "release_gate_passed": False, "planned_calls": len(suite["cases"]),
        "max_seconds": max_seconds, "remote_quiescence_proven": False,
    }
    atomic_json(out / "run.json", manifest)
    try:
        scorer = SGLangScorer(compiler, str(client.base_url), model_path, client=transport)
        manifest["base_provenance"] = scorer.provenance
    except Exception as error:
        manifest.update(
            status="failed", attempted_calls=0, failed_calls=0,
            error_type=type(error).__name__, error=str(error)[:1000],
        )
        atomic_json(out / "run.json", manifest)
        return manifest
    rows = []
    with (out / "predictions.jsonl").open("x") as stream:
        for case in suite["cases"]:
            if time.monotonic() - started >= max_seconds:
                break
            row = {key: case[key] for key in (
                "id", "semantic_group", "block", "arm", "language", "gold",
                "labels_in_order", "identifier_slots",
            )}
            try:
                values, _, _ = scorer._native_compiled(
                    compile_case(compiler, case), capture_last_hidden=False,
                )
                p = probabilities(values[0], 1.0)
                selected = max(range(len(p)), key=p.__getitem__)
                row.update(
                    status="scored", raw_scores=values[0],
                    probabilities=dict(zip(case["labels_in_order"], p, strict=True)),
                    prediction=case["labels_in_order"][selected],
                    correct=case["labels_in_order"][selected] == case["gold"],
                    native_measurement=scorer.last_measurement,
                )
            except Exception as error:
                row.update(
                    status="failed", error_type=type(error).__name__, error=str(error)[:1000],
                )
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            if row["status"] == "failed":
                break  # A timed-out native request may still be running; never retry it.
    manifest.update(
        status=("failed" if any(r["status"] == "failed" for r in rows) else
                "completed" if len(rows) == len(suite["cases"]) else "deadline"),
        attempted_calls=len(rows), failed_calls=sum(r["status"] == "failed" for r in rows),
        wall_seconds=time.monotonic() - started,
        predictions_sha256=file_hash(out / "predictions.jsonl"),
    )
    atomic_json(out / "run.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve the previous frozen suite.")
    suite = build(json.loads(args.source_suite.read_text()))
    atomic_json(args.out, suite)
    print(json.dumps({"cases": len(suite["cases"]), "sha256": suite["content_sha256"]}))


if __name__ == "__main__":
    main()
