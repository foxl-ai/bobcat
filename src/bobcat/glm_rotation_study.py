"""A bounded, prior-free AnyJev-style rotation study with a matched repeat control."""
from __future__ import annotations

import copy
import json
import time
from collections import Counter, defaultdict, deque
from datetime import UTC, datetime

import numpy as np

from bobcat.checkpoint_metrics import aggregate, compare, enrich
from bobcat.corpus import atomic_json
from bobcat.glm_serving_checkpoint import CompiledReadout, distribution_comparison
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash


def aligned_logmean(scores, orders, original_order):
    """Normalized geometric mean, not arithmetic averaging or an online prior."""
    if not scores or len(scores) != len(orders) or len(set(original_order)) != len(original_order):
        raise ValueError("Require nonempty, unambiguous aligned readouts.")
    aligned = []
    for values, labels in zip(scores, orders, strict=True):
        logits = np.asarray(values, dtype=np.float64)
        if (len(labels) != len(original_order) or set(labels) != set(original_order)
                or logits.shape != (len(labels),) or not np.isfinite(logits).all()):
            raise ValueError("A rotation dropped, duplicated or invented an option.")
        logp = logits - np.logaddexp.reduce(logits)
        aligned.append([float(logp[labels.index(label)]) for label in original_order])
    result = np.mean(aligned, axis=0)
    return (result - np.logaddexp.reduce(result)).tolist()


def validate_plan(plan):
    if (plan.get("schema") != "bobcat-glm-rotation-study-plan-v1"
            or plan.get("content_sha256") != json_hash({
                k: v for k, v in plan.items() if k != "content_sha256"})
            or not plan.get("cases") or plan.get("training_or_calibration") is not False):
        raise ValueError("Require the frozen development-only rotation study.")
    components = set()
    for case in plan["cases"]:
        original = case["original"]
        labels = original["candidate_ids"]
        if (original["group_id"] in components or original["supervision"] != "hard_label"
                or original["kind"] != "choice" or not 2 <= len(labels) <= 7
                or len(set(labels)) != len(labels)
                or not 0 <= original["target_index"] < len(labels)
                or original["split"] != "dev_train" or original["input_tokens"] > 2048
                or len(case["rotations"]) != min(4, len(labels))):
            raise ValueError("The bounded Choice study changed its data scope.")
        components.add(original["group_id"])
        for rotation in case["rotations"]:
            order = rotation["candidate_ids"]
            if (len(order) != len(labels) or set(order) != set(labels)
                    or rotation["id"] != original["id"]
                    or rotation["group_id"] != original["group_id"]
                    or rotation["split"] != "dev_train"
                    or rotation["input_tokens"] != len(rotation["input_ids"])
                    or rotation["option_token_ids"] != original["option_token_ids"]
                    or rotation["target_index"] != order.index(labels[original["target_index"]])
                    or rotation["input_sha256"] != json_hash({
                        "input_ids": rotation["input_ids"],
                        "option_token_ids": rotation["option_token_ids"]})):
                raise ValueError("Rotation input or label mapping changed.")
        if case["rotations"][0]["input_sha256"] != original["input_sha256"]:
            raise ValueError("The unrotated prompt must equal the prior compiled baseline.")
        if len({tuple(row["candidate_ids"]) for row in case["rotations"]}) != min(4, len(labels)):
            raise ValueError("The planned rotations must be distinct.")


def prepare_plan(suite_rows, raw_path, compiler, *, source_provenance):
    """Select by language/task and stable suite order, without reading predictions."""
    buckets = defaultdict(lambda: defaultdict(deque))
    for row in suite_rows:
        if (row["kind"] == "choice" and row["supervision"] == "hard_label"
                and 2 <= len(row["candidate_ids"]) <= 7 and row["input_tokens"] <= 2048):
            buckets[row["language"]][row["task"]].append(row)
    selected = []
    for language in ("ko", "en"):
        chosen = []
        while len(chosen) < 32:
            added = False
            for task in sorted(buckets[language]):
                if buckets[language][task] and len(chosen) < 32:
                    chosen.append(buckets[language][task].popleft())
                    added = True
            if not added:
                raise ValueError("The fixed suite cannot supply 32 eligible cases per language.")
        selected.append(chosen)
    ordered = [row for pair in zip(*selected, strict=True) for row in pair]
    wanted = {row["id"] for row in ordered}
    raw = {}
    with raw_path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item["id"] in wanted:
                if item["id"] in raw:
                    raise ValueError("Duplicate raw development ID.")
                raw[item["id"]] = item
    cases = []
    for original in ordered:
        source = raw[original["id"]]
        if (source["split"] != "dev_train"
                or source["input_sha256"] != json_hash(source["request"])
                or source["input_sha256"] != original["source_request_sha256"]
                or source["candidate_ids"] != original["candidate_ids"]):
            raise ValueError("Raw and compiled development identities differ.")
        labels = original["candidate_ids"]
        count = min(4, len(labels))
        offsets = [index * len(labels) // count for index in range(count)]
        rotations = []
        for offset in offsets:
            request = copy.deepcopy(source["request"])
            question = next(iter(request["questions"].values()))
            items = list(question["criteria"].items())
            question["criteria"] = dict(items[offset:] + items[:offset])
            state, questions = parse_request(request)
            compiled = compiler.compile(state, questions)
            order = list(questions[0].labels)
            inputs = {"input_ids": compiled.input_ids[0],
                      "option_token_ids": compiled.option_token_ids[0]}
            rotated = {
                **original, **inputs, "candidate_ids": order,
                "target_index": order.index(labels[original["target_index"]]),
                "input_sha256": json_hash(inputs), "input_tokens": len(inputs["input_ids"]),
                "last_input_position": len(inputs["input_ids"]) - 1,
                "rotation_offset": offset, "source_request_sha256": json_hash(request),
            }
            rotations.append(rotated)
        if (rotations[0]["input_ids"] != original["input_ids"]
                or rotations[0]["option_token_ids"] != original["option_token_ids"]):
            raise ValueError("Recompilation changed the existing unrotated GLM input.")
        cases.append({"original": original, "rotations": rotations})
    plan = {
        "schema": "bobcat-glm-rotation-study-plan-v1",
        "created_at": datetime.now(UTC).isoformat(), "training_or_calibration": False,
        "selection": (
            "32 cases per language, task round-robin, original suite order; no predictions"),
        "rotation": "min(4,K) evenly spaced cyclic offsets; full cyclic set only when K<=4",
        "comparison": "same number of serial unchanged-input repeats, interleaved treatment order",
        "source_provenance": source_provenance, "raw_source_sha256": file_hash(raw_path),
        "language_counts": dict(Counter(row["language"] for row in ordered)),
        "task_counts": dict(Counter(row["task"] for row in ordered)),
        "planned_native_requests": sum(2 * len(case["rotations"]) for case in cases),
        "cases": cases,
    }
    plan["content_sha256"] = json_hash(plan)
    validate_plan(plan)
    return plan


def run(plan_path, out, *, client, model_path, seconds=900):
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if out.exists() or not 120 <= seconds <= 900:
        raise ValueError("Use a fresh bounded diagnostic directory.")
    out.mkdir(parents=True)
    start, deadline = time.monotonic(), time.monotonic() + seconds
    record = {
        "schema": "bobcat-glm-rotation-study-v1", "status": "running",
        "started_at": datetime.now(UTC).isoformat(), "plan_sha256": file_hash(plan_path),
        "planned_components": len(plan["cases"]), "completed_components": 0,
        "native_requests": 0, "processed_prompt_tokens": 0,
        "online_prior_correction": False, "calibration_fitted": False,
        "weights_updated": False, "release_gate_passed": False,
        "runtime_may_have_failed_numerical_controls": True,
        "scope": "Bounded Choice rotations; matched repeated-input compute; development only.",
    }
    atomic_json(out / "run.json", record)
    scorer = CompiledReadout(client, model_path=model_path, run_id="rotation-study")
    complete = []
    try:
        with (out / "observations.jsonl").open("x") as stream:
            for index, case in enumerate(plan["cases"]):
                observations = {"repeat": [], "rotation": []}
                failed = False
                for repeat, rotated in enumerate(case["rotations"]):
                    # Alternate treatment order; neither arm receives a systematic warm-cache lead.
                    arms = [("repeat", case["original"]), ("rotation", rotated)]
                    if (index + repeat) % 2:
                        arms.reverse()
                    for arm, row in arms:
                        if deadline - time.monotonic() < 65:
                            failed = True
                            break
                        values, measured = scorer.read(
                            [row], label=f"{index}-{repeat}-{arm}")
                        record["native_requests"] += 1
                        record["processed_prompt_tokens"] += row["input_tokens"]
                        observations[arm].append({
                            "rotation_index": repeat, "candidate_ids": row["candidate_ids"],
                            "input_sha256": row["input_sha256"],
                            "prediction": values[0], "measurement": measured,
                        })
                    if failed:
                        break
                item = {
                    "id": case["original"]["id"], "group_id": case["original"]["group_id"],
                    "status": "partial_time_budget" if failed else "completed",
                    "observations": observations,
                }
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if failed:
                    record["status"] = "partial_time_budget"
                    break
                original_order = case["original"]["candidate_ids"]
                aligned = {}
                for arm in ("repeat", "rotation"):
                    obs = sorted(observations[arm], key=lambda row: row["rotation_index"])
                    aligned[arm] = {
                        **case["original"], "logits": aligned_logmean(
                            [r["prediction"]["logits"] for r in obs],
                            [r["candidate_ids"] for r in obs], original_order),
                        "model_passes": len(obs),
                    }
                raw = {
                    **case["original"],
                    "logits": observations["repeat"][0]["prediction"]["logits"],
                    "model_passes": 1,
                }
                repeat_rows = [row["prediction"] for row in observations["repeat"]]
                item["repeat_variation"] = [
                    distribution_comparison([repeat_rows[0]], [row])
                    for row in repeat_rows[1:]]
                complete.append({
                    "raw": raw, **aligned, "repeat_variation": item["repeat_variation"]})
                record["completed_components"] = len(complete)
                record["updated_at"] = datetime.now(UTC).isoformat()
                atomic_json(out / "run.json", record)
            else:
                record["status"] = "completed"
        if complete:
            for arm in ("raw", "repeat", "rotation"):
                rows = [item[arm] for item in complete]
                atomic_json(out / f"{arm}.json", rows)
                record[f"{arm}_metrics_completed_subset"] = aggregate(list(map(enrich, rows)))
            atomic_json(out / "raw-to-rotation.json", compare(
                [item["raw"] for item in complete], [item["rotation"] for item in complete]))
            atomic_json(out / "matched-repeat-to-rotation.json", compare(
                [item["repeat"] for item in complete], [item["rotation"] for item in complete]))
            atomic_json(out / "repeat-variation.json",
                        [item["repeat_variation"] for item in complete])
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1200])
        raise
    finally:
        record.update(finished_at=datetime.now(UTC).isoformat(),
                      elapsed_seconds=time.monotonic() - start)
        atomic_json(out / "run.json", record)
    return record
