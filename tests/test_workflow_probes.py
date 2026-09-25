import copy
import json

import httpx
import pytest

from bobcat.schema import json_hash
from bobcat.workflow_probes import build_suite, run_suite, validate_suite


def small_suite():
    suite = build_suite()
    suite["cases"] = suite["cases"][:1]
    suite["content_sha256"] = json_hash({
        key: value for key, value in suite.items() if key != "content_sha256"
    })
    return suite


class HostTableFixture:
    """A deterministic test interpreter, not learned model evidence."""

    model_name = "host-table-test-fixture"
    provenance = {"is_model": False}
    temperatures = {}
    last_measurement = None

    def __init__(self, first_error=None, wrong_first=False, mutate=False):
        self.calls = []
        self.first_error = first_error
        self.wrong_first = wrong_first
        self.mutate = mutate

    def score(self, state, questions):
        self.calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
        if len(self.calls) == 1 and self.first_error is not None:
            if isinstance(self.first_error, Exception):
                raise self.first_error
            return self.first_error, 1
        correct_route = None
        for rule in state["routing_rules"]:
            if "otherwise" in rule:
                correct_route = rule["otherwise"]
                break
            if state["facts"].get(rule["field"]) == rule["equals"]:
                correct_route = rule["route"]
                break
        values = []
        for question in questions:
            labels = question.labels
            if set(labels) == set(state["destinations"]):
                selected = correct_route
                if self.wrong_first and len(self.calls) == 1:
                    selected = next(label for label in labels if label != selected)
            elif all(isinstance(value, dict) for value in question.criteria.values()):
                selected = next(
                    key for key, value in question.criteria.items()
                    if value == {"route": correct_route,
                                 "destination": state["destinations"][correct_route]}
                )
            else:
                assigned = state.get("recorded_assignment")
                selected = (
                    state["destinations"][assigned["route"]] if assigned
                    else next(
                        label for label in labels if label not in state["destinations"].values()
                    )
                )
            values.append([10.0 if label == selected else -10.0 for label in labels])
        if self.mutate:
            state["unexpected_mutation"] = True
        return values, 100


def observations(out):
    return [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()]


def test_bilingual_gold_and_worlds_are_checked_before_scoring():
    suite = build_suite()
    assert validate_suite(suite) == {"cases": 16, "worlds": 8}
    assert {case["language"] for case in suite["cases"]} == {"ko", "en"}
    assert all("recorded_assignment" not in case["state"] for case in suite["cases"])
    changed = copy.deepcopy(suite)
    case = changed["cases"][0]
    case["gold"]["route"] = case["intervention_route"]
    case["gold"]["destination"] = case["state"]["destinations"][case["gold"]["route"]]
    case["gold"]["joint"] = f'{case["gold"]["route"]} → {case["gold"]["destination"]}'
    changed["content_sha256"] = json_hash({
        k: v for k, v in changed.items() if k != "content_sha256"
    })
    with pytest.raises(ValueError, match="gold"):
        validate_suite(changed)


def test_host_transition_joint_plan_and_flat_isolation_have_distinct_semantics(tmp_path):
    scorer, suite, out = HostTableFixture(), small_suite(), tmp_path / "workflow"
    frozen = copy.deepcopy(suite)
    result = run_suite(suite, scorer, out, max_seconds=30)
    row = observations(out)[0]
    assert result["status"] == "completed"
    assert result["native_score_calls"] == 5
    assert sum(len(qs) for _, qs in scorer.calls) == 6
    assert row["staged"]["task_success"] and row["joint"]["task_success"]
    assert row["flat"]["correctly_reports_unassigned"]
    assert row["flat"]["assignment_was_committed"] is False
    assert row["intervention"]["followed_record_instead_of_original_policy"]
    assert result["external_actions_executed"] is False
    assert result["marginal_probabilities_multiplied"] is False
    assert result["release_gate_passed"] is False
    assert suite == frozen
    for state, _ in scorer.calls:
        assert not {"gold", "world", "id", "language", "intervention_route"} & set(state)
    assert "recorded_assignment" not in scorer.calls[2][0]  # Joint has original state.
    assert "recorded_assignment" not in scorer.calls[3][0]  # Flat cannot see route output.


def test_wrong_upstream_answer_is_forwarded_instead_of_gold(tmp_path):
    scorer, suite, out = HostTableFixture(wrong_first=True), small_suite(), tmp_path / "wrong"
    result = run_suite(suite, scorer, out, max_seconds=30)
    row = observations(out)[0]
    assert result["status"] == "completed"  # Execution, not model correctness.
    assert row["staged"]["task_success"] is False
    assert row["staged"]["downstream_reads_actual_assignment"] is True
    actual = row["staged"]["selected_route"]
    assert actual != suite["cases"][0]["gold"]["route"]
    assert scorer.calls[1][0]["recorded_assignment"] == {"route": actual}


def test_failed_upstream_does_not_invent_or_run_a_downstream_assignment(tmp_path):
    scorer = HostTableFixture(first_error=[[float("nan"), 0.0, 0.0]])
    out = tmp_path / "bad-output"
    result = run_suite(small_suite(), scorer, out, max_seconds=30)
    row = observations(out)[0]
    assert result["status"] == "completed_with_failures"
    assert row["staged"]["downstream_status"] == "blocked"
    assert row["staged"]["selected_route"] is None
    assert "staged_downstream" not in [call["condition"] for call in row["calls"]]
    assert row["calls"][0]["error_type"] == "ValueError"
    assert row["joint"]["task_success"]  # Independent control remains available.


def test_transport_failure_stops_further_remote_work(tmp_path):
    scorer = HostTableFixture(first_error=httpx.ReadTimeout("Initialization not complete"))
    out = tmp_path / "timeout"
    result = run_suite(build_suite(), scorer, out, max_seconds=30)
    assert result["status"] == "incomplete_transport_or_deadline"
    assert result["attempted_cases"] == 1 and result["native_score_calls"] == 1
    assert result["stopped_after_transport_or_deadline_failure"] is True
    assert len(scorer.calls) == 1
    assert observations(out)[0]["staged"]["downstream_status"] == "blocked"


def test_mutated_shared_state_is_not_silently_accepted(tmp_path):
    scorer, suite, out = HostTableFixture(mutate=True), small_suite(), tmp_path / "mutation"
    frozen = copy.deepcopy(suite)
    result = run_suite(suite, scorer, out, max_seconds=30)
    assert result["completed_cases"] == 0 and suite == frozen
    assert observations(out)[0]["calls"][0]["error"] == "The scorer mutated shared input state."


def test_invalid_suite_or_existing_output_does_not_invoke_model(tmp_path):
    scorer, suite = HostTableFixture(), small_suite()
    suite["cases"][0]["state"]["facts"]["duplicate_charge"] = True
    with pytest.raises(ValueError, match="checksummed"):
        run_suite(suite, scorer, tmp_path / "bad")
    with pytest.raises(ValueError, match="fresh"):
        run_suite(small_suite(), scorer, tmp_path)
    assert not scorer.calls
