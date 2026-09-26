import json
import math

import pytest

from scripts import semif_bench as sb


def choice_row(row_id, state, question="Assess the claim.", label=0):
    return {"id": row_id, "group_id": "g", "family": "evidence_interpretation", "state": state,
            "question": question, "label": label,
            "options": [{"id": "supported", "description": "The evidence establishes it"},
                        {"id": "insufficient", "description": "Neither"},
                        {"id": "contradicted", "description": "The opposite"}]}


def yes_no_row(row_id, state, question="Is it so?"):
    return {"id": row_id, "group_id": "g", "family": "every_judge-grid", "state": state,
            "question": question,
            "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]}


def test_rows_sharing_a_state_become_one_request_with_isolated_questions():
    rows = [choice_row("a", "S1"), choice_row("b", "S1", "Other claim."), choice_row("c", "S2")]
    built = sb.build_requests(rows, "choice")
    assert len(built) == 2
    first = built[0]
    assert first["request"]["state"] == "S1"
    assert list(first["request"]["questions"]) == ["q0", "q1"]
    q0 = first["request"]["questions"]["q0"]
    assert q0 == {"type": "choice", "instructions": "Assess the claim.",
                  "criteria": {"supported": "The evidence establishes it",
                               "insufficient": "Neither", "contradicted": "The opposite"}}
    # Candidate order is SemIf's display order (option-reversal perturbations rely on it).
    assert list(q0["criteria"]) == ["supported", "insufficient", "contradicted"]
    assert first["rows"]["q1"] == {"id": "b",
                                   "option_ids": ["supported", "insufficient", "contradicted"]}


def test_json_states_group_by_content_and_requests_split_at_the_question_limit(monkeypatch):
    monkeypatch.setattr(sb, "MAX_QUESTIONS", 2)
    rows = [yes_no_row(f"r{i}", {"b": 1, "a": [1, 2]}) for i in range(3)]
    rows.append(yes_no_row("other", {"a": [1, 2], "b": 1}))  # same content, other key order
    built = sb.build_requests(rows, "every")
    assert [len(b["rows"]) for b in built] == [2, 2]
    assert built[0]["request"]["questions"]["q0"] == {"type": "noul",
                                                     "instructions": "Is it so?"}


def test_shape_rows_are_nouls_with_the_yes_no_descriptions_as_criteria():
    row = yes_no_row("s", "state")
    row["options"] = [{"id": "yes", "description": "Yes, satisfied."},
                      {"id": "no", "description": "No, not satisfied."}]
    question = sb.build_requests([row], "shape")[0]["request"]["questions"]["q0"]
    assert question["criteria"] == {"true": "Yes, satisfied.", "false": "No, not satisfied."}


def test_typesafe_rows_use_the_original_question_and_document(tmp_path):
    payload = {"eval": {"questions": [{"type": "noul", "instructions": "Unauthorized?",
                                       "criteria": {"true": "Yes", "false": "No"}},
                                      {"type": "choice", "instructions": "Kind?",
                                       "criteria": {"meal": "food", "travel": "trip"}}],
                        "documents": [{"alert": "x"}]}}
    (tmp_path / "typesafe-w-cases.js").write_text(f"__VIEWER_DATA__({json.dumps(payload)});")
    payloads = {"w": sb.parse_payload(tmp_path / "typesafe-w-cases.js")}
    rows = [{"id": "t1", "options": [{"id": "true", "description": "true: Yes"},
                                     {"id": "false", "description": "false: No"}],
             "provenance": {"workflow": "w", "question_index": 0, "document_index": 0,
                            "primitive": "noul"}},
            {"id": "t2", "options": [{"id": "meal", "description": "meal: food"},
                                     {"id": "travel", "description": "travel: trip"}],
             "provenance": {"workflow": "w", "question_index": 1, "document_index": 0,
                            "primitive": "choice"}}]
    built = sb.build_requests(rows, "typesafe", payloads)
    assert len(built) == 1
    assert built[0]["request"]["state"] == {"alert": "x"}
    assert built[0]["request"]["questions"]["q0"] == payload["eval"]["questions"][0]
    assert built[0]["request"]["questions"]["q1"] == payload["eval"]["questions"][1]
    rows[0]["provenance"]["primitive"] = "choice"
    with pytest.raises(ValueError):
        sb.build_requests(rows, "typesafe", payloads)


def test_answers_map_to_semif_option_order_and_failures_stay_missing():
    requests = sb.build_requests([choice_row("a", "S"), choice_row("b", "S")], "choice")
    requests += sb.build_requests([yes_no_row("y", "T")], "every")
    responses = [
        {"request_id": "r0", "status": 200, "client_ms": 12.0, "answers": {
            "q0": {"type": "choice", "probabilities": {"contradicted": 0.2, "supported": 0.5,
                                                       "insufficient": 0.3}}}},
        {"request_id": "r1", "status": 500, "client_ms": 3.0, "answers": {}},
    ]
    # build_requests numbers each call from r0; make the Noul request r1.
    requests[1]["request_id"] = "r1"
    rows = {row["id"]: row for row in sb.predictions(requests, responses)}
    assert rows["a"]["option_ids"] == ["supported", "insufficient", "contradicted"]
    assert rows["a"]["probabilities"] == pytest.approx([0.5, 0.3, 0.2])
    assert rows["a"]["option_logits"] == pytest.approx([math.log(0.5), math.log(0.3),
                                                        math.log(0.2)])
    assert rows["b"]["status"] == "missing"  # no answer for q1: never invented
    assert rows["y"]["status"] == "missing" and "probabilities" not in rows["y"]


def test_noul_probability_is_the_yes_or_true_option():
    assert sb.distribution({"type": "noul", "noul": 0.8}, "every", ["yes", "no"]) == \
        pytest.approx([0.8, 0.2])
    assert sb.distribution({"type": "noul", "noul": 0.8}, "typesafe", ["true", "false"]) == \
        pytest.approx([0.8, 0.2])
    assert sb.distribution({"type": "noul", "noul": 0.8}, "typesafe", ["false", "true"]) == \
        pytest.approx([0.2, 0.8])
