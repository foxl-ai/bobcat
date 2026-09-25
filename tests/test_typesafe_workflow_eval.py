import json

import pytest

from scripts import typesafe_workflow_eval as ev


def node(name, doc, questions, answers, ran=True):
    return {"node": name, "ran": ran, "doc": doc, "questions": questions, "answers": answers}


def write_cases(folder, workflow, cases, questions, documents):
    examples = [{"case_id": case_id, "label": "All three agree"} for case_id in cases]
    data = {"eval": {"cases": cases, "questions": questions, "documents": documents,
                     "examples": examples}}
    (folder / f"{workflow}-cases.js").write_text(f"__VIEWER_DATA__({json.dumps(data)});")


@pytest.fixture
def cases(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "WORKFLOWS", ["w"])
    questions = [{"type": "noul", "instructions": "Unauthorized?", "criteria": None},
                 {"type": "choice", "instructions": "Kind?",
                  "criteria": {"meal": "food", "travel": "trip"}},
                 {"type": "noul", "instructions": "Pair A?", "criteria": None},
                 {"type": "noul", "instructions": "Pair B?", "criteria": None}]
    noul = lambda p: {"type": "noul", "noul": p}  # noqa: E731
    choice = lambda probs: {"type": "choice", "probabilities": probs}  # noqa: E731
    models = {
        "opus": {"nodes": [node("s", 0, {"u": 0, "k": 1, "pair": 2},
                                {"u": noul(0.9), "k": choice({"meal": 0.2, "travel": 0.8}),
                                 "pair": noul(0.1)})], "cost": {"usd": 0.1}, "seconds": 9},
        "sol": {"nodes": [node("s", 0, {"u": 0, "k": 1, "pair": 3},
                               {"u": noul(0.4), "k": choice({"meal": 0.9, "travel": 0.1}),
                                "pair": noul(0.2)})], "cost": {"usd": 0.1}, "seconds": 9},
        "typesafe": {"nodes": [node("s", 0, {"u": 0, "k": 1},
                                    {"u": noul(0.7), "k": choice({"meal": 0.6, "travel": 0.4})}),
                               node("later", None, {}, {}, ran=False)],
                     "cost": {"usd": 0.0001}, "seconds": 0.2},
    }
    reference = {"s": {
        # Astra 0.8 / Fable 0.4 on true: the mean is 0.6, so the consensus is true.
        "u": {"type": "noul", "sets": [
            {"value": True, "probabilities": {"true": 0.8, "false": 0.2}},
            {"value": False, "probabilities": {"true": 0.4, "false": 0.6}}]},
        # No probabilities published: the shared value is the consensus.
        "k": {"type": "choice", "sets": [{"value": "meal", "probabilities": None},
                                         {"value": "meal", "probabilities": None}]},
        "pair": {"type": "noul", "sets": [{"value": False, "probabilities": None},
                                          {"value": False, "probabilities": None}]},
    }, "later": {
        # References disagree without probabilities: no consensus.
        "x": {"type": "choice", "sets": [{"value": "a", "probabilities": None},
                                         {"value": "b", "probabilities": None}]},
    }}
    case = {"models": models, "reference_answers": reference}
    write_cases(tmp_path, "w", {"c1": case}, questions, [{"alert": "x"}])
    return tmp_path


def test_consensus_uses_mean_probabilities_and_shared_values():
    target, _ = ev.consensus([{"value": True, "probabilities": {"true": 0.8, "false": 0.2}},
                              {"value": False, "probabilities": {"true": 0.4, "false": 0.6}}])
    assert target == "true"
    assert ev.consensus([{"value": False, "probabilities": None}] * 2) == ("false", None)
    assert ev.consensus([{"value": "a", "probabilities": None},
                         {"value": "b", "probabilities": None}])[0] is None
    # A tie has no modal answer.
    assert ev.modal({"true": 0.5, "false": 0.5}) is None


def test_build_skips_questions_whose_wording_differs_between_models(cases, tmp_path):
    out = tmp_path / "requests.jsonl"
    ev.build(cases, out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    assert set(rows[0]["request"]["questions"]) == {"u", "k"}  # "pair" is dynamic
    assert rows[0]["request"]["state"] == {"alert": "x"}


def test_score_counts_failures_as_wrong_and_compares_on_common_items(cases, tmp_path):
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps({
        "workflow": "w", "case": "c1", "step": "s", "status": 200, "attempts": 1,
        "client_ms": 50.0, "engine_ms": 40.0, "processed_tokens": 10,
        "usage": {"input_tokens": 1000, "output_tokens": 0},
        "answers": {"u": {"type": "noul", "noul": 0.9},
                    "k": {"type": "choice", "probabilities": {"meal": 0.7, "travel": 0.3}}},
    }) + "\n")
    out = tmp_path / "report.json"
    ev.score(cases, responses, out)
    report = json.loads(out.read_text())
    assert report["items"] == {"reference": 4, "disputed_or_tied": 1, "dynamic_question": 1,
                               "common_to_all_models": 2}
    common = report["common"]["all"]
    assert common["bobcat"]["agreement"] == 1.0
    assert common["typesafe"]["agreement"] == 1.0
    assert common["opus"]["agreement"] == 0.5  # "travel" misses "meal"
    assert common["sol"]["agreement"] == 0.5   # 0.4 misses "true"
    assert report["sensitivity"]["neutral_cases_only"]["all"]["bobcat"]["n"] == 2
    cost = report["cost_time_per_case"]["w"]["c1"]["bobcat"]
    assert cost["usd"] == pytest.approx(1000 * 0.042 / 1e6)

    failed = tmp_path / "failed.jsonl"
    failed.write_text(json.dumps({
        "workflow": "w", "case": "c1", "step": "s", "status": 500, "attempts": 1,
        "client_ms": 5.0, "engine_ms": 0.0, "processed_tokens": 0, "usage": None,
        "answers": {}}) + "\n")
    ev.score(cases, failed, out)
    assert json.loads(out.read_text())["common"]["all"]["bobcat"]["agreement"] == 0.0
