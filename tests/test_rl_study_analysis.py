import copy
import json

import numpy as np
import pytest

from bobcat.rl_study_analysis import (
    policy_diagnostics,
    read_evaluation,
    verify_workflow_traces,
    workflow_change,
)
from bobcat.schema import file_hash


def make_evaluation(root):
    folder = root / "evaluation-parent556"
    folder.mkdir()
    rows = []
    for rank in range(8):
        row = {
            "id": f"row-{rank}", "group_id": f"component-{rank}", "language": "ko",
            "language_origin": "native", "task": "intent", "kind": "choice",
            "supervision": "hard_label", "target_index": 0, "score_mean": 0.,
            "candidate_ids": ["billing", "shipping"], "input_sha256": str(rank),
            "input_tokens": 32, "option_token_ids": [11, 12],
        }
        rows.append(row)
        prediction = row | {
            "logits": [2., 0.], "probabilities": [float(1 / (1 + np.exp(-2))),
                                               float(1 / (1 + np.exp(2)))],
        }
        result = {
            "label": "parent556", "development_only": True,
            "same_generator_workflow_holdout": True, "production_latency_measured": False,
            "predictions": [prediction],
            "workflow_episodes": [{"language": "ko", "seed": rank}],
        }
        (folder / f"rank-{rank}.json").write_text(json.dumps(result))
    marker = {
        "schema": "bobcat-native-rl-evaluation-v1", "label": "parent556",
        "same_process_restored_rng_and_actor": True,
        "files": {p.name: file_hash(p) for p in folder.iterdir()},
    }
    (folder / "complete.json").write_text(json.dumps(marker))
    return folder, rows


def test_native_reader_preserves_denominator_and_candidate_mapping(tmp_path):
    folder, gold = make_evaluation(tmp_path)
    rows, _, _ = read_evaluation(folder, gold)
    assert len(rows) == 8
    with pytest.raises(ValueError, match="denominator"):
        read_evaluation(folder, gold[:-1])
    changed = copy.deepcopy(gold)
    changed[0]["candidate_ids"].reverse()
    with pytest.raises(ValueError, match="frozen inputs"):
        read_evaluation(folder, changed)
    (folder / "rank-0.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        read_evaluation(folder, gold)


def test_workflow_languages_are_one_world_cluster_not_two_independent_samples():
    before = [
        {"language": lang, "seed": seed, "component_id": f"world-{seed}",
         "success": False, "reward": 0., "steps": 2, "failure": "wrong"}
        for seed in range(2) for lang in ("ko", "en")
    ]
    after = [r | {"success": True, "reward": 1., "failure": None} for r in before]
    result = workflow_change(before, after)
    assert result["independent_world_seeds"] == 2
    assert result["episodes"] == 4
    assert result["delta_success_rate"] == 1
    assert result["paired_world_bootstrap_95pct"] == [1., 1.]
    with pytest.raises(ValueError):
        workflow_change(before, after[:-1])


def test_more_confidence_reports_wrong_automation_and_excludes_mean_scores():
    before = [
        {"id": str(i), "group_id": str(i), "language": "ko", "supervision": "hard_label",
         "target_index": 0, "candidate_ids": ["yes", "no"], "probabilities": p}
        for i, p in enumerate(([.95, .05], [.6, .4], [.4, .6]))
    ] + [{
        "id": "score", "group_id": "score", "language": "ko", "supervision": "score_mean",
        "target_index": None, "candidate_ids": ["low", "high"], "probabilities": [.01, .99],
    }]
    after = copy.deepcopy(before)
    after[1]["probabilities"] = [.95, .05]   # One useful new automatic answer.
    after[2]["probabilities"] = [.05, .95]   # One confidently wrong automatic answer.
    result = policy_diagnostics(before, after)
    point = next(p for p in result["slices"]["ko"] if p["threshold"] == .9)
    assert result["excluded_rows"] == 1
    assert point["total"] == 3
    assert point["delta_correct_accepted"] == point["delta_wrong_accepted"] == 1
    assert point["candidate"]["coverage"] == 1.
    assert point["candidate"]["error_among_accepted"] == pytest.approx(1 / 3)
    empty = result["slices"]["ko"][-1]
    assert empty["candidate"]["error_among_accepted"] is None
    assert result["deployment_policy"] is result["equal_risk_comparison"] is False
    with pytest.raises(ValueError, match="same components"):
        policy_diagnostics(before, after[:-1])


def test_workflow_validation_replays_real_actions_and_rejects_fabricated_success():
    from bobcat.rl_workflow import EvidenceWorkflow, make_world, visible_rule_policy
    from bobcat.schema import json_hash

    rows = []
    for language in ("ko", "en"):
        for seed in range(20, 24):
            env = EvidenceWorkflow(make_world(seed, language))
            trace, total = [], 0.
            while not (env.terminated or env.truncated):
                observation = env.observation()
                action = visible_rule_policy(observation)
                result = env.step(action)
                total += result["reward"]
                trace.append({"action": action, "observation_sha256": json_hash(observation),
                              "reward": result["reward"]})
            rows.append({
                "language": language, "seed": seed, "component_id": env.world.component_id,
                "trace": trace, "reward": total, "steps": env.steps,
                "success": result["info"]["success"], "failure": result["info"]["failure"],
                "extra_collective_padding_forwards": 4 - env.steps,
            })
    proof = verify_workflow_traces(rows, first_seed=20, worlds=4)
    assert proof["independent_worlds"] == 4
    assert proof["episodes"] == 8
    assert proof["recorded_actions_replayed"]
    with pytest.raises(ValueError, match="membership"):
        verify_workflow_traces(rows[:-1], first_seed=20, worlds=4)
    changed = copy.deepcopy(rows)
    changed[0]["success"] = not changed[0]["success"]
    with pytest.raises(ValueError, match="outcome"):
        verify_workflow_traces(changed, first_seed=20, worlds=4)
    changed = copy.deepcopy(rows)
    changed[0]["trace"][0]["observation_sha256"] = "forged"
    with pytest.raises(ValueError, match="observation"):
        verify_workflow_traces(changed, first_seed=20, worlds=4)
    changed = copy.deepcopy(rows)
    changed[0]["trace"][0]["reward"] += .5
    with pytest.raises(ValueError, match="reward"):
        verify_workflow_traces(changed, first_seed=20, worlds=4)
