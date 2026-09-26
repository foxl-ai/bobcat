"""Flash-first routing (bobcat.route_server): rules, closed replies and question isolation."""

import asyncio
import math

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from test_api_server import MODELS, char_student  # noqa: E402

from bobcat import api_server  # noqa: E402
from bobcat.output_contract_audit import inspect_reply  # noqa: E402
from bobcat.protocol import parse_request  # noqa: E402
from bobcat.route_server import (  # noqa: E402
    RoutedEngine,
    RoutePolicy,
    Tier,
    known_script_share,
    top_probability,
)

T_FLASH, T_BIG = 0.9, 1.2
IDS = [chr(c) for c in range(40, 120)]  # 80 one-character identifiers: room for 70 candidates


class Recording:
    """Scores by the question's instructions, read back from its own compiled sequence."""

    def __init__(self, name, compiler, table, fail=False):
        self.name, self.compiler, self.table, self.fail = name, compiler, table, fail
        self.seen = []

    async def logits(self, sequences, option_ids, prefix):
        if self.fail:
            raise RuntimeError(f"{self.name} raw failure text")
        out = []
        for sequence, options in zip(sequences, option_ids, strict=True):
            text = "".join(self.compiler.host_tokenizer.id_to_token(i) or "" for i in sequence)
            key = next(k for k in self.table if f'"instructions":"{k}"' in text)
            self.seen.append(key)
            values = self.table[key]
            out.append(list(values[:len(options)]) + [-9.0] * (len(options) - len(values)))
        return out


FLASH_TABLE = {"confident": [4.0, 0.0], "unsure": [0.1, 0.0], "many": [2.0, 0.0],
               "other": [3.0, 0.0]}
BIG_TABLE = {"confident": [0.0, 4.0], "unsure": [0.0, 2.0], "many": [0.0, 3.0],
             "other": [0.0, 1.0]}


def routed(tmp_path, *, big_fail=False, allow_override=False, big_limit=None):
    flash_compiler = api_server.cache_state_encoding(char_student(tmp_path / "flash", IDS))
    big_compiler = api_server.cache_state_encoding(char_student(tmp_path / "big", IDS))
    if big_limit is not None:
        big_compiler.max_branch_tokens = big_limit
    flash = Recording("flash", flash_compiler, FLASH_TABLE)
    big = Recording("big", big_compiler, BIG_TABLE, fail=big_fail)
    engine = RoutedEngine(Tier("flash", flash, flash_compiler, T_FLASH),
                          Tier("bobcat-1.1", big, big_compiler, T_BIG), RoutePolicy(),
                          allow_override=allow_override)
    app = api_server.create_api(engine, flash_compiler, model_name="bobcat-flash-1.1",
                                aliases={"bobcat-latest"}, temperature=T_FLASH, models=MODELS,
                                edge_secret=None)
    return TestClient(app), flash, big


def noul(text):
    return {"type": "noul", "instructions": text}


def softmax(values, t):
    peak = max(values)
    weights = [math.exp((v - peak) / t) for v in values]
    return [w / sum(weights) for w in weights]


def test_rules_are_decided_from_the_input_before_any_pass():
    policy = RoutePolicy()
    state, (small, big, japanese, korean) = parse_request({
        "model": "m", "state": {"doc": "plain English text"}, "questions": {
            "small": {"type": "choice", "instructions": "pick",
                      "criteria": {f"c{i}": None for i in range(64)}},
            "big": {"type": "choice", "instructions": "pick",
                    "criteria": {f"c{i}": None for i in range(65)}},
            "japanese": {"type": "noul", "instructions": "これは日本語の質問ですか？"
                                                         "とても長い日本語の文章です"},
            "korean": {"type": "noul", "instructions": "한국어 질문인가?"}}})
    assert policy.out_of_range(state, small) is None
    assert policy.out_of_range(state, big) == "candidates"
    assert policy.out_of_range(state, korean) is None
    assert known_script_share(state, japanese) < 0.5
    assert policy.out_of_range(state, japanese) == "script"
    assert known_script_share("", parse_request({"model": "m", "state": "", "questions": {
        "q": {"type": "noul", "instructions": "123"}}})[1][0]) == 1.0
    assert top_probability([0.0, 0.0], 1.0) == 0.5
    assert abs(top_probability([2.0, 0.0], 2.0) - 1 / (1 + math.exp(-1))) < 1e-12


def test_flash_answers_first_and_bobcat_takes_low_confidence_and_out_of_range(tmp_path):
    api, flash, big = routed(tmp_path)
    many = {"type": "choice", "instructions": "many",
            "criteria": {f"c{i}": f"option {i}" for i in range(70)}}
    body = {"model": "bobcat-latest", "state": {"doc": "shared state"},
            "questions": {"a": noul("confident"), "b": noul("unsure"), "c": many}}
    reply = api.post("/v1/systemone", json=body)
    assert reply.status_code == 200
    inspect_reply(body, reply, expected_model="bobcat-flash-1.1")
    assert reply.headers["x-bobcat-route"] == "flash,bobcat,bobcat"
    assert reply.headers["x-bobcat-route-reasons"] == "in_range,low_confidence,candidates"
    # Flash never scored the 70-candidate question; Bobcat scored only what was routed.
    assert sorted(flash.seen) == ["confident", "unsure"]
    assert sorted(big.seen) == ["many", "unsure"]
    answers = reply.json()["answers"]
    # Each answer carries its own model's probabilities at that model's temperature.
    assert abs(answers["a"]["noul"] - softmax(FLASH_TABLE["confident"], T_FLASH)[1]) < 1e-12
    assert abs(answers["b"]["noul"] - softmax(BIG_TABLE["unsure"], T_BIG)[1]) < 1e-12
    assert answers["c"]["choice"] == "c1"
    processed = int(reply.headers["x-bobcat-processed-tokens"])
    assert processed > int(reply.json()["usage"]["input_tokens"])


def test_a_question_is_routed_and_answered_the_same_whatever_is_asked_beside_it(tmp_path):
    api, _, _ = routed(tmp_path)
    state = {"doc": "shared state"}
    many = {"type": "choice", "instructions": "many",
            "criteria": {f"c{i}": None for i in range(70)}}
    groups = [{"b": noul("unsure")}, {"a": noul("confident"), "b": noul("unsure")},
              {"c": many, "b": noul("unsure"), "d": noul("other")},
              {"b": noul("unsure"), "a": noul("confident"), "c": many}]
    seen = {}
    for questions in groups:
        reply = api.post("/v1/systemone", json={"model": "bobcat-latest", "state": state,
                                                "questions": questions})
        assert reply.status_code == 200
        routes = dict(zip(questions, reply.headers["x-bobcat-route"].split(","), strict=True))
        for qid, item in reply.json()["answers"].items():
            seen.setdefault(qid, set()).add((routes[qid], repr(item)))
    assert all(len(v) == 1 for v in seen.values()), seen


def test_override_needs_the_flag_and_failures_do_not_leak(tmp_path):
    body = {"model": "bobcat-latest", "state": "s", "questions": {"q": noul("confident")}}
    forced = {"x-bobcat-route-mode": "bobcat"}
    api, _, _ = routed(tmp_path / "off")
    assert api.post("/v1/systemone", json=body, headers=forced).headers[
        "x-bobcat-route"] == "flash"                   # ignored without --allow-route-override
    api, _, big = routed(tmp_path / "on", allow_override=True)
    assert api.post("/v1/systemone", json=body, headers=forced).headers[
        "x-bobcat-route"] == "bobcat"
    unsure = {"model": "bobcat-latest", "state": "s", "questions": {"q": noul("unsure")}}
    only = api.post("/v1/systemone", json=unsure, headers={"x-bobcat-route-mode": "flash"})
    assert only.headers["x-bobcat-route"] == "flash"  # flash-only mode never falls back
    bad = api.post("/v1/systemone", json=body, headers={"x-bobcat-route-mode": "gpt"})
    assert bad.status_code == 500
    api, _, _ = routed(tmp_path / "fail", big_fail=True)
    reply = api.post("/v1/systemone", json=unsure)
    assert reply.status_code == 500 and "raw failure" not in reply.text
    assert "raw failure" not in "".join(reply.headers.values())


def test_a_question_bobcat_cannot_compile_keeps_the_flash_answer(tmp_path):
    api, flash, big = routed(tmp_path, big_limit=20)
    body = {"model": "bobcat-latest", "state": "s", "questions": {"q": noul("unsure")}}
    reply = api.post("/v1/systemone", json=body)
    assert reply.status_code == 200
    inspect_reply(body, reply, expected_model="bobcat-flash-1.1")
    assert reply.headers["x-bobcat-route"] == "flash"
    assert reply.headers["x-bobcat-route-reasons"] == "bobcat_limit"
    assert big.seen == []


def test_out_of_range_questions_start_on_bobcat_beside_flash(tmp_path):
    flash_compiler = char_student(tmp_path / "f", IDS)
    big_compiler = char_student(tmp_path / "b", IDS)
    events = []

    class Slow:
        def __init__(self, name):
            self.name = name

        async def logits(self, sequences, option_ids, prefix):
            events.append(f"start {self.name}")
            await asyncio.sleep(0.01)
            events.append(f"end {self.name}")
            return [[3.0] + [0.0] * (len(o) - 1) for o in option_ids]

    engine = RoutedEngine(Tier("flash", Slow("flash"), flash_compiler, T_FLASH),
                          Tier("bobcat-1.1", Slow("big"), big_compiler, T_BIG))
    _, questions = parse_request({"model": "m", "state": "s", "questions": {
        "a": noul("confident"),
        "c": {"type": "choice", "instructions": "many",
              "criteria": {f"c{i}": None for i in range(70)}}}})
    compiled = [flash_compiler.compile("s", q) for q in questions]
    scores, temperatures, processed, headers = asyncio.run(engine.decide(
        "s", questions, [c[0] for c in compiled], [c[1] for c in compiled]))
    assert sorted(events[:2]) == ["start big", "start flash"]  # both run before either ends
    assert temperatures == [T_FLASH, T_BIG] and headers["x-bobcat-route"] == "flash,bobcat"
    assert processed == len(compiled[0][0]) + len(big_compiler.compile("s", questions[1])[0])
