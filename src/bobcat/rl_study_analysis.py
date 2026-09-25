"""Compare immutable native RL evaluations on the same development components.

Never accesses a live optimizer, fits deployment thresholds, or claims a final
benchmark. Korean/English versions of a workflow world are clustered together.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bobcat.checkpoint_metrics import aggregate, compare, enrich
from bobcat.corpus import atomic_json
from bobcat.development_temperature_control import crossfit_categorical
from bobcat.glm_native_evaluate import read_suite
from bobcat.schema import file_hash

LABELS = ("parent556", "proper_score_reinforce-final", "direct_brier-final",
          "correctness_reinforce-final")
POLICY_THRESHOLDS = (.5, .75, .9, .95, .99)


def verify_workflow_traces(episodes, *, first_seed, worlds):
    """Replay recorded actions in the pinned, side-effect-free environment."""
    from bobcat.rl_workflow import EvidenceWorkflow, make_world
    from bobcat.schema import json_hash

    expected = {(lang, seed) for lang in ("ko", "en")
                for seed in range(first_seed, first_seed + worlds)}
    actual = {(r["language"], r["seed"]) for r in episodes}
    if len(episodes) != len(expected) or actual != expected:
        raise ValueError("Missing, duplicated or changed workflow world/language membership.")
    for row in episodes:
        env = EvidenceWorkflow(make_world(row["seed"], row["language"]))
        if (row["component_id"] != env.world.component_id
                or not 1 <= len(row["trace"]) <= env.world.maximum_steps):
            raise ValueError("The workflow component or trace length changed.")
        total, last = 0., None
        for step in row["trace"]:
            if (env.terminated or env.truncated
                    or json_hash(env.observation()) != step["observation_sha256"]):
                raise ValueError("Recorded workflow observation does not follow its actions.")
            last = env.step(step["action"])
            if last["reward"] != step["reward"]:
                raise ValueError("Recorded workflow reward differs from the host outcome.")
            total += last["reward"]
        if (not (env.terminated or env.truncated)
                or row["reward"] != total or row["steps"] != env.steps
                or row["success"] != last["info"]["success"]
                or row["failure"] != last["info"]["failure"]
                or row["extra_collective_padding_forwards"] !=
                env.world.maximum_steps - env.steps):
            raise ValueError("The workflow terminal outcome or reported totals changed.")
    return {"episodes": len(episodes), "independent_worlds": worlds,
            "recorded_actions_replayed": True, "external_effects_executed": False}


def read_evaluation(folder: Path, expected: list[dict], *,
                    workflow_schedule=None) -> tuple[list, list, dict]:
    marker = json.loads((folder / "complete.json").read_text())
    if (marker.get("schema") != "bobcat-native-rl-evaluation-v1"
            or folder.name != "evaluation-" + marker["label"]
            or marker.get("same_process_restored_rng_and_actor") is not True
            or set(marker["files"]) != {f"rank-{rank}.json" for rank in range(8)}):
        raise ValueError("Require a completed eight-rank native RL evaluation.")
    observed, episodes = {}, []
    for name, digest in marker["files"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("An immutable evaluation rank changed.")
        result = json.loads((folder / name).read_text())
        if (result.get("label") != marker["label"] or not result.get("development_only")
                or not result.get("same_generator_workflow_holdout")
                or result.get("production_latency_measured") is not False):
            raise ValueError("Do not promote development predictions to release evidence.")
        for row in result["predictions"]:
            if row["id"] in observed:
                raise ValueError("Duplicate development component.")
            observed[row["id"]] = row
        episodes.extend(result["workflow_episodes"])
    if set(observed) != {row["id"] for row in expected}:
        raise ValueError("Missing or extra predictions must not change the denominator.")
    fields = ("id", "group_id", "language", "task", "kind", "supervision",
              "target_index", "score_mean", "candidate_ids", "input_sha256", "input_tokens")
    rows = []
    for gold in expected:
        row = observed[gold["id"]]
        logits, probs = np.asarray(row["logits"]), np.asarray(row["probabilities"])
        if (any(row.get(field) != gold.get(field) for field in fields)
                or logits.ndim != 1 or len(logits) != len(gold["option_token_ids"])
                or probs.shape != logits.shape or not np.isfinite(logits).all()
                or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
                or not np.isclose(probs.sum(), 1., atol=1e-5, rtol=0)):
            raise ValueError("Native predictions differ from the frozen inputs or labels.")
        expected_probs = np.exp(logits.astype(np.float64) - float(logits.max()))
        expected_probs /= expected_probs.sum()
        if not np.allclose(probs, expected_probs, atol=2e-6, rtol=0):
            raise ValueError("Logged probabilities do not correspond to the native logits.")
        rows.append({**row, "language_origin": gold["language_origin"],
                     "checkpoint": marker["label"]})
    keys = [(e["language"], e["seed"]) for e in episodes]
    if len(set(keys)) != len(keys) or not episodes:
        raise ValueError("Duplicate or missing workflow results.")
    if workflow_schedule is not None:
        verify_workflow_traces(episodes, **workflow_schedule)
    return rows, episodes, marker


def workflow_change(before: list[dict], after: list[dict], *, samples=2000) -> dict:
    def index(rows):
        return {(r["language"], r["seed"]): r for r in rows}

    a, b = index(before), index(after)
    if (len(a) != len(before) or len(b) != len(after) or set(a) != set(b)
            or any(a[key]["component_id"] != b[key]["component_id"] for key in a)):
        raise ValueError("Workflow comparison requires identical worlds and languages.")
    worlds = sorted({key[1] for key in a})
    differences = np.asarray([
        np.mean([int(b[key]["success"]) - int(a[key]["success"])
                 for key in a if key[1] == world]) for world in worlds
    ])
    draws = np.random.default_rng(20260923).integers(
        0, len(worlds), size=(samples, len(worlds)),
    )
    result = {
        "episodes": len(a), "independent_world_seeds": len(worlds),
        "delta_success_rate": float(differences.mean()),
        "paired_world_bootstrap_95pct": np.quantile(
            differences[draws].mean(1), [.025, .975],
        ).tolist(),
        "same_generator_development_holdout": True, "production_safety_claimed": False,
        "by_language": {},
    }
    for language in sorted({key[0] for key in a}):
        result["by_language"][language] = {}
        for name, rows in (("baseline", a), ("candidate", b)):
            values = [v for key, v in rows.items() if key[0] == language]
            result["by_language"][language][name] = {
                "episodes": len(values), "successes": sum(int(r["success"]) for r in values),
                "success_rate": float(np.mean([r["success"] for r in values])),
                "mean_reward": float(np.mean([r["reward"] for r in values])),
                "mean_steps": float(np.mean([r["steps"] for r in values])),
                "failures": dict(Counter(str(r["failure"]) for r in values if not r["success"])),
            }
    return result


def update_summary(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    arms = {}
    for arm in sorted({r["arm"] for r in rows}):
        group = [r for r in rows if r["arm"] == arm]
        outcomes = [c for r in group for c in r.get("sampled_outcomes", [])]
        arms[arm] = {
            "logged_updates": len(group),
            "unique_component_ids": len({r["component"]["group_id"] for r in group}),
            "sampled_outcomes": len(outcomes),
            "mean_sampled_correctness": float(np.mean(outcomes)) if outcomes else None,
            "zero_group_advantage_fraction": float(np.mean([
                r["zero_group_advantage"] for r in group
            ])) if all("zero_group_advantage" in r for r in group) else None,
            "last_actor_update": max(r["actor_update"] for r in group),
            "local_rank_only": True,
            "sampled_outcomes_are_not_new_independent_annotations": True,
        }
    return arms


def policy_diagnostics(before: list[dict], after: list[dict]) -> dict:
    """Fixed raw-probability development diagnostics, not a calibrated policy.

    Count both correct automation and confidently wrong automation. No threshold
    is selected here, and these labels do not describe the safety of a real
    pending tool call. Mean-only scores and trivial singletons are excluded.
    """
    a, b = ({row["id"]: row for row in rows} for rows in (before, after))
    if (len(a) != len(before) or len(b) != len(after) or set(a) != set(b)
            or any(any(a[k].get(field) != b[k].get(field) for field in (
                "group_id", "language", "supervision", "target_index", "candidate_ids",
            )) for k in a)):
        raise ValueError("Policy diagnostics require the same components and outcomes.")
    keys = [k for k in a if a[k]["supervision"] == "hard_label"
            and len(a[k]["candidate_ids"]) > 1]
    slices = {"overall": keys} | {
        language: [k for k in keys if a[k]["language"] == language]
        for language in sorted({a[k]["language"] for k in keys})
    }
    result = {
        "thresholds": list(POLICY_THRESHOLDS),
        "signal": "native_top_candidate_probability_not_adapter_confidence",
        "denominator": "hard_label_non_singleton_judgments",
        "excluded_rows": len(a) - len(keys),
        "threshold_fitted": False, "deployment_policy": False,
        "equal_risk_comparison": False, "production_tool_safety_measured": False,
        "slices": {},
    }
    for language, selected in slices.items():
        pairs = []
        for threshold in POLICY_THRESHOLDS:
            pair = {"threshold": threshold, "total": len(selected)}
            for name, rows in (("baseline", a), ("candidate", b)):
                accepted = [k for k in selected
                            if max(rows[k]["probabilities"]) >= threshold]
                wrong = sum(int(np.argmax(rows[k]["probabilities"]))
                            != rows[k]["target_index"] for k in accepted)
                pair[name] = {
                    "accepted": len(accepted), "correct_accepted": len(accepted) - wrong,
                    "wrong_accepted": wrong, "referred": len(selected) - len(accepted),
                    "coverage": len(accepted) / len(selected) if selected else None,
                    "error_among_accepted": wrong / len(accepted) if accepted else None,
                    "wrong_automation_per_request": wrong / len(selected) if selected else None,
                }
            pair["delta_correct_accepted"] = (
                pair["candidate"]["correct_accepted"] - pair["baseline"]["correct_accepted"]
            )
            pair["delta_wrong_accepted"] = (
                pair["candidate"]["wrong_accepted"] - pair["baseline"]["wrong_accepted"]
            )
            pairs.append(pair)
        result["slices"][language] = pairs
    return result


def analyze(run: Path, suite: Path, curriculum: Path, *, partial=False,
            temperature=False, job: Path | None = None) -> dict:
    from bobcat.glm_native_train import read_curriculum

    _, gold = read_suite(suite)
    _, training = read_curriculum(curriculum)
    if {r["group_id"] for r in gold} & {r["group_id"] for r in training["train"]}:
        raise ValueError("Training/evaluation component overlap.")
    workflow_schedule = None
    workflow_proof = {"performed": False}
    if job is not None:
        contract = json.loads(job.read_text())
        environment = Path(__file__).with_name("rl_workflow.py")
        if (contract.get("schema") != "bobcat-native-glm-rl-job-v1"
                or contract["world_size"] != 8
                or type(contract["eval_world_start"]) is not int
                or type(contract["eval_worlds"]) is not int
                or contract["eval_worlds"] < 4 or contract["eval_worlds"] % 4
                or contract["files"]["src/bobcat/rl_workflow.py"] != file_hash(environment)):
            raise ValueError("The workflow schedule or recorded environment source changed.")
        workflow_schedule = {
            "first_seed": contract["eval_world_start"], "worlds": contract["eval_worlds"],
        }
        workflow_proof = {
            "performed": True, "job_sha256": file_hash(job),
            "environment_sha256": file_hash(environment), **workflow_schedule,
            "expected_episodes_per_model": 2 * contract["eval_worlds"],
            "external_effects_executed": False,
            "native_checkpoint_resume_claimed": False,
        }
    readings, markers = {}, {}
    for label in LABELS:
        folder = run / f"evaluation-{label}"
        if (folder / "complete.json").exists():
            rows, episodes, marker = read_evaluation(
                folder, gold, workflow_schedule=workflow_schedule,
            )
            readings[label] = (rows, episodes)
            markers[label] = marker
        elif not partial:
            raise ValueError(f"Missing completed arm: {label}")
    if "parent556" not in readings:
        raise ValueError("The fixed starting checkpoint has not completed evaluation.")
    baseline, base_episodes = readings["parent556"]
    result = {
        "schema": "bobcat-native-rl-study-analysis-v1",
        "evaluation_role": "repeated_development_not_final",
        "complete_four_arm_evaluation": len(readings) == len(LABELS),
        "suite_sha256": file_hash(suite / "manifest.json"),
        "curriculum_sha256": file_hash(curriculum / "manifest.json"),
        "independent_public_components": len(gold), "training_overlap": False,
        "baseline": aggregate(list(map(enrich, baseline))), "comparisons": {},
        "markers": markers, "release_gate_passed": False,
        "production_latency_measured": False, "jev_comparison_executed": False,
        "workflow_trace_verification": workflow_proof,
    }
    for label, (rows, episodes) in readings.items():
        if label == "parent556":
            continue
        result["comparisons"][label] = {
            "public_judgments": compare(baseline, rows),
            "workflow": workflow_change(base_episodes, episodes),
            "fixed_threshold_diagnostics": policy_diagnostics(baseline, rows),
        }
    for control in ("direct_brier-final", "correctness_reinforce-final"):
        if control in readings and "proper_score_reinforce-final" in readings:
            result["comparisons"]["RL-vs-" + control] = {
                "public_judgments": compare(readings[control][0],
                                            readings["proper_score_reinforce-final"][0]),
                "workflow": workflow_change(readings[control][1],
                                             readings["proper_score_reinforce-final"][1]),
                "fixed_threshold_diagnostics": policy_diagnostics(
                    readings[control][0], readings["proper_score_reinforce-final"][0],
                ),
            }
    if temperature:
        result["development_temperature_controls"] = {
            label: crossfit_categorical([r for r in rows if r["supervision"] == "hard_label"])
            for label, (rows, _) in readings.items()
        }
    result["rank_training_summaries"] = {}
    for rank in range(8):
        path = run / f"updates-rank-{rank}.jsonl"
        if path.exists():
            result["rank_training_summaries"][str(rank)] = update_summary(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--development-temperature-control", action="store_true")
    parser.add_argument("--job", type=Path,
                        help="Frozen job for exact workflow membership and host trace replay.")
    args = parser.parse_args()
    result = analyze(args.run, args.suite, args.curriculum, partial=args.partial,
                     temperature=args.development_temperature_control, job=args.job)
    atomic_json(args.out, result)
    print(json.dumps({k: result[k] for k in (
        "evaluation_role", "complete_four_arm_evaluation", "independent_public_components",
        "baseline", "release_gate_passed",
    )}, indent=2))


if __name__ == "__main__":
    main()
