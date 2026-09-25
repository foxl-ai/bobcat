import json
import math

import httpx
import pytest

from bobcat.architecture_probes import (
    build_suite,
    compare,
    digest,
    observe,
    run_suite,
    summarize,
)


class Scores:
    model_name = "test-only"
    temperatures = {}

    def __init__(self, values=None):
        self.values = values
        self.seen = []

    def score(self, state, questions):
        self.seen.append((state, questions))
        return [
            self.values or [0.0] * len(q.labels) for q in questions
        ], 1


def small_suite():
    return build_suite(worlds=1, blocks=2)


def test_suite_is_frozen_bilingual_and_many_candidate_gold_is_computed():
    suite = small_suite()
    assert suite == small_suite()
    assert {case["language"] for case in suite["cases"]} == {"ko", "en"}
    for case in suite["cases"]:
        assert "gold" not in case["request"]
        if case["family"] != "many_choices":
            continue
        candidates = case["request"]["questions"]["decision"]["criteria"]
        eligible = {
            k: v for k, v in candidates.items() if v["fee"] <= 70 and v["days"] <= 4
        }
        assert max(eligible, key=lambda name: eligible[name]["quality"]) == case["gold"]
        assert max(candidates, key=lambda name: candidates[name]["quality"]) != case["gold"]
        assert len(candidates) == case["candidate_count"]
        assert list(candidates).index(case["gold"]) == case["gold_position"]


def test_transport_timeout_does_not_queue_more_gpu_requests(tmp_path):
    scorer = Scores()

    def timeout(state, questions):
        scorer.seen.append((state, questions))
        raise httpx.ReadTimeout("The GPU request may still be running.")

    scorer.score = timeout
    result = run_suite(small_suite(), scorer, tmp_path / "probe", max_seconds=30)
    assert len(scorer.seen) == 1
    assert result["attempted_cases"] == 1
    assert result["status"] == "aborted_transport"


def test_blind_model_does_not_pass_isolation_positive_controls():
    cases = [c for c in small_suite()["cases"] if c["family"] == "isolation"]
    # This always selects "not stated"; numerical isolation alone is insufficient.
    scorer = Scores([0.0, 0.0, 10.0])
    scorer.score = lambda state, qs: ([
        [0.0, 0.0, 10.0] if len(q.labels) == 3 else [0.0, 0.0] for q in qs
    ], 1)
    report = summarize([observe(c, scorer) for c in cases])
    assert all(c["absence_and_siblings_correct"]
               for c in report["isolation_positive_controls"])
    assert not any(c["state_both_correct"] for c in report["isolation_positive_controls"])
    assert all(pair["tv"] == 0 for pair in report["pairs"])


def test_reference_controls_retain_the_rule_and_flip_gold():
    cases = [c for c in small_suite()["cases"] if c["family"] == "reference"]
    assert {c["reference_position"] for c in cases} == {0, 1, 2}
    assert {c["template_family"] for c in cases} == {"review_status", "assigned_route"}
    grouped = {}
    for case in cases:
        key = case["language"], case["evidence_location"], case["template_family"]
        grouped.setdefault(key, set()).add(case["gold"])
    assert all(len(gold) == 2 for gold in grouped.values())


def test_policy_can_change_without_an_argmax_flip():
    case = next(c for c in small_suite()["cases"] if c["family"] == "permutation")
    first = observe(case, Scores([math.log(0.89), math.log(0.1), math.log(0.01)]))
    second = observe(case, Scores([math.log(0.91), math.log(0.08), math.log(0.01)]))
    result = compare(first, second)
    assert result["execution_threshold_crossed"]
    assert not result["argmax_changed"]
    assert result["tv"] == pytest.approx(0.02)
    assert first["adapter_confidence"] != first["p_max"]
    assert result["diagnostic_policy_changes"]["0.9"]["execution_threshold_crossed"]
    assert not result["diagnostic_policy_changes"]["0.8"]["execution_threshold_crossed"]
    assert not result["diagnostic_policy_changes"]["0.95"]["execution_threshold_crossed"]


def test_iia_uses_logit_differences_and_keeps_repeat_controls():
    cases = [c for c in small_suite()["cases"] if c["family"] == "iia"]
    rows = []
    for case in cases:
        count = len(case["request"]["questions"]["decision"]["criteria"])
        # Adding a large irrelevant score changes probabilities, not the fixed log odds.
        scorer = Scores([1.0, 2.0, -1000.0, -1001.0] + ([1000.0] if count == 5 else []))
        rows.append(observe(case, scorer))
    report = summarize(rows)
    assert len(report["iia_blocks"]) == 4
    for block in report["iia_blocks"]:
        assert block["append_effect_fixed_t_log_odds"] == 0
        assert block["base_repeat_noise_fixed_t_log_odds"] == 0
    assert all(row["fixed_t_log_odds"] == 1 for row in rows)
    assert all(row["correct"] is None for row in rows)


def test_iia_effect_averages_identical_conditions_before_differencing():
    cases = [c for c in small_suite()["cases"] if c["family"] == "iia"]
    values = {"base": 1.0, "base_repeat": 3.0, "append": 5.0,
              "append_repeat": 9.0, "replace": 8.0}
    rows = []
    for case in cases:
        count = len(case["request"]["questions"]["decision"]["criteria"])
        scorer = Scores([0.0, 0.0, values[case["condition"]], 0.0]
                        + ([0.0] if count == 5 else []))
        scorer.temperatures = {"choice": 0.5}
        rows.append(observe(case, scorer))
    for block in summarize(rows)["iia_blocks"]:
        assert block["base_mean_fixed_t_log_odds"] == 2
        assert block["append_mean_fixed_t_log_odds"] == 7
        assert block["append_effect_fixed_t_log_odds"] == 5
        assert block["same_k_replacement_fixed_t_log_odds"] == 1
        assert block["append_effect_deployment_log_odds"] == 10
        assert block["base_repeat_noise_fixed_t_log_odds"] == 2
        assert block["append_repeat_noise_fixed_t_log_odds"] == 4


def test_reference_position_and_paired_value_controls_are_reported():
    cases = [c for c in small_suite()["cases"] if c["family"] == "reference"]
    rows = []
    for case in cases:
        labels = list(case["request"]["questions"]["decision"]["criteria"])
        rows.append(observe(case, Scores([10.0 if k == case["gold"] else 0.0 for k in labels])))
    report = summarize(rows)
    assert len(report["reference_counterfactuals"]) == 48
    assert all(r["both_correct"] for r in report["reference_counterfactuals"])
    assert all(r["correct"] == r["scored_questions"] for r in report["reference_positions"])
    assert not any(r["reference_selected"] for r in report["reference_positions"])


def test_finite_runner_preserves_failed_attempts_and_rejects_mutation(tmp_path):
    suite = small_suite()
    suite["cases"] = suite["cases"][:2]
    suite["content_sha256"] = digest({k: v for k, v in suite.items() if k != "content_sha256"})

    class Broken(Scores):
        def score(self, state, questions):
            raise ValueError("missing all-option scores")

    result = run_suite(suite, Broken(), tmp_path / "run")
    assert result["status"] == "completed"
    assert result["failed_cases"] == 2
    assert not result["release_gate_passed"]
    lines = (tmp_path / "run/predictions.jsonl").read_text().splitlines()
    assert all(json.loads(line)["status"] == "failed" for line in lines)
    with pytest.raises(ValueError, match="new run directory"):
        run_suite(suite, Scores(), tmp_path / "run")
    suite["cases"][0]["gold"] = "tampered"
    with pytest.raises(ValueError, match="frozen content hash"):
        run_suite(suite, Scores(), tmp_path / "another")
