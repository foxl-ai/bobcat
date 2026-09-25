import copy

from bobcat.output_contract_probe import paired_cases, semantic_comparison
from bobcat.protocol import parse_request


def test_native_probe_keeps_facts_and_separates_caller_instruction_changes():
    cases = paired_cases()
    assert len(cases) == 96
    assert len({c["id"] for c in cases}) == 96
    assert {c["language"] for c in cases} == {"ko", "en"}
    for offset in range(0, len(cases), 4):
        clean, repeat, state_attack, instruction_attack = cases[offset:offset + 4]
        assert clean["payload"] == repeat["payload"]
        assert clean["payload"]["questions"] == state_attack["payload"]["questions"]
        assert (clean["payload"]["state"]["customer_message"]
                == state_attack["payload"]["state"]["customer_message"])
        assert clean["payload"]["state"] == instruction_attack["payload"]["state"]
        assert state_attack["semantic_oracle_valid"]
        assert not instruction_attack["semantic_oracle_valid"]
        for case in (clean, repeat, state_attack, instruction_attack):
            _, questions = parse_request(case["payload"])
            assert {q.kind for q in questions} == {"choice", "score", "noul"}
            assert all(0 <= case["expected_indices"][q.id] < len(q.labels) for q in questions)


def test_semantic_attack_rate_uses_clean_correct_cases_and_reports_repeat_noise():
    cases = paired_cases()
    scores = {}
    for case in cases:
        _, questions = parse_request(case["payload"])
        scores[case["id"]] = {
            q.id: [8. if i == case["expected_indices"][q.id] else -8.
                   for i in range(len(q.labels))] for q in questions
        }
    clean = cases[0]
    attacked_id = f"{clean['group']}-state_attack"
    repeat_id = f"{clean['group']}-repeat"
    scores[attacked_id]["route"] = [-8., 8., -8.]
    scores[repeat_id]["route"] = [-8., 8., -8.]
    result = semantic_comparison(cases, scores)
    assert result["languages"]["ko"]["eligible_clean_correct"] == 36
    assert result["languages"]["ko"]["correct_to_wrong_after_state_attack"] == 1
    assert result["languages"]["ko"]["repeat_argmax_flips"] == 1
    assert result["languages"]["ko"]["conditional_semantic_failure_rate"] == 1 / 36
    assert result["languages"]["en"]["conditional_semantic_failure_rate"] == 0
    assert result["language_specific_case_groups"] == 24
    assert result["paired_bilingual_scenarios"] == 12
    assert not result["independent_statistical_sample"]
    assert len(result["pairs"]) == 72
    ko = result["languages"]["ko"]["by_primitive"]
    assert ko["choice"]["conditional_semantic_failure_rate"] == 1 / 12
    assert ko["choice"]["repeat_argmax_flips"] == 1
    assert ko["score"]["correct_to_wrong_after_state_attack"] == 0
    assert ko["noul"]["correct_to_wrong_after_state_attack"] == 0
    assert result["languages"]["ko"]["score_clean_mean_absolute_error"] < 1e-5
    assert all(pair["deployment_policy"] is False for pair in result["pairs"])
    assert all("clean_score" in pair for pair in result["pairs"]
               if pair["primitive"] == "score")
    assert all(pair["event_threshold"] == .5 for pair in result["pairs"]
               if pair["primitive"] == "noul")
    assert not result["release_quality_passed"]

    wrong_before = copy.deepcopy(scores)
    wrong_before[clean["id"]]["route"] = [-8., 8., -8.]
    changed = semantic_comparison(cases, wrong_before)["languages"]["ko"]
    assert changed["eligible_clean_correct"] == 35
    assert changed["correct_to_wrong_after_state_attack"] == 0
