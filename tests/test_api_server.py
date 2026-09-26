import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from test_student_readout import student  # noqa: E402

from bobcat import api_server  # noqa: E402
from bobcat.output_contract_audit import inspect_reply  # noqa: E402
from bobcat.student_readout import StudentCompiler  # noqa: E402

MODELS = [{"name": "bobcat-1", "description": "d", "release_date": "2026-09-25"}]


class FakeEngine:
    name = "fake"

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    async def logits(self, sequences, option_ids, prefix):
        self.calls.append((sequences, prefix))
        if self.fail:
            raise RuntimeError("raw engine text that must not leak")
        return [[float(i) for i in range(len(o))] for o in option_ids]


def client(tmp_path, engine=None, **options):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pinned = student(tmp_path)
    compiler = StudentCompiler(tmp_path, pinned, ["A", "B", "C"], max_branch_tokens=4096,
                               piecewise=True)
    app = api_server.create_api(engine or FakeEngine(), compiler, model_name="bobcat-1",
                                aliases={"bobcat-latest"}, temperature=1.0, models=MODELS,
                                **{"edge_secret": "s3cret", **options})
    return TestClient(app), compiler


def payload(**questions):
    return {"model": "bobcat-latest", "state": "state question",
            "questions": questions or {"q": {"type": "choice", "instructions": "pick",
                                             "criteria": {"x": "first", "y": None}},
                                       "n": {"type": "noul", "instructions": "yes"}}}


def test_decisions_follow_the_published_wire_contract(tmp_path):
    api, compiler = client(tmp_path)
    body = payload()
    reply = api.post("/v1/systemone", json=body, headers={"x-bobcat-edge-secret": "s3cret"})
    assert reply.status_code == 200
    inspect_reply(body, reply, expected_model="bobcat-1")
    result = reply.json()
    assert result["answers"]["q"]["choice"] == "y"
    assert result["usage"]["output_tokens"] == 0
    processed = int(reply.headers["x-bobcat-processed-tokens"])
    assert 0 < result["usage"]["input_tokens"] < processed
    assert api.get("/v1/models", headers={"x-bobcat-edge-secret": "s3cret"}).json() == {
        "models": MODELS}


def test_errors_use_status_codes_and_never_leak_engine_text(tmp_path):
    api, _ = client(tmp_path)
    assert api.post("/v1/systemone", json=payload()).status_code == 401
    secret = {"x-bobcat-edge-secret": "s3cret"}
    bad = api.post("/v1/systemone", json={"model": "bobcat-latest", "state": "x",
                                          "questions": {}}, headers=secret)
    assert bad.status_code == 422 and bad.json()["detail"][0]["loc"] == ["body"]
    unknown = api.post("/v1/systemone", json={**payload(), "model": "gpt"}, headers=secret)
    assert unknown.status_code == 422 and unknown.json()["detail"][0]["loc"] == ["body", "model"]
    duplicate = api.post("/v1/systemone", content=b'{"model":"a","model":"b"}', headers=secret)
    assert duplicate.status_code == 422
    failing, _ = client(tmp_path / "f", FakeEngine(fail=True))
    reply = failing.post("/v1/systemone", json=payload(), headers=secret)
    assert reply.status_code == 500 and "raw engine" not in reply.text
    busy, _ = client(tmp_path / "b", max_inflight=0)
    reply = busy.post("/v1/systemone", json=payload(), headers=secret)
    assert reply.status_code == 529 and reply.headers["retry-after"] == "1"


def test_billing_counts_the_state_once_and_excludes_the_template(tmp_path):
    api, compiler = client(tmp_path, edge_secret=None)
    one = api.post("/v1/systemone", json=payload(q={"type": "noul", "instructions": "yes"}))
    two = api.post("/v1/systemone", json=payload(
        q={"type": "noul", "instructions": "yes"}, r={"type": "noul", "instructions": "yes"}))
    state_tokens = len(compiler._data("state question"))
    single = one.json()["usage"]["input_tokens"]
    assert two.json()["usage"]["input_tokens"] == 2 * single - state_tokens
    assert json.loads(one.content)["answers"]["q"]["type"] == "noul"


def test_engine_arguments_parse_as_json_values_and_default_to_nothing():
    assert api_server.parse_engine_args(None) == {}
    assert api_server.parse_engine_args([]) == {}
    assert api_server.parse_engine_args(
        ["max_num_batched_tokens=8192", "enforce_eager=true", "kv_cache_dtype=fp8",
         "long_prefill_token_threshold=null", 'mamba_cache_mode="all"']) == {
        "max_num_batched_tokens": 8192, "enforce_eager": True, "kv_cache_dtype": "fp8",
        "long_prefill_token_threshold": None, "mamba_cache_mode": "all"}
    # The default context leaves exactly the old 16,384-token compile limit.
    assert api_server.DEFAULT_MAX_MODEL_LEN - 64 == 16384


@pytest.mark.parametrize("pairs", [["max_num_batched_tokens"], ["=3"], ["9x=1"],
                                   ["max_model_len=65600"], ["quantization=fp8"],
                                   ["block_size=16", "block_size=32"]])
def test_engine_arguments_refuse_malformed_owned_or_repeated_keys(pairs):
    with pytest.raises(ValueError):
        api_server.parse_engine_args(pairs)


def test_warmup_prefix_is_the_block_aligned_shared_prefix():
    shared = list(range(2000))
    sequences = [shared + [9001, 9002], shared + [9003], shared + [9004, 9005, 9006]]
    assert api_server.warmup_prefix(sequences, 784) == shared[:1568]
    assert api_server.warmup_prefix(sequences, 1000) == shared[:2000]
    # Nothing to share: one question, no block size, or a prefix shorter than one block.
    assert api_server.warmup_prefix(sequences[:1], 784) is None
    assert api_server.warmup_prefix(sequences, None) is None
    assert api_server.warmup_prefix([[1, 2, 3, 4], [1, 2, 5]], 16) is None


@pytest.mark.parametrize("schedule,count,prefix,plan", [
    ("default", 1, 9000, "all"), ("default", 8, 1023, "all"), ("default", 2, 1024, "parent"),
    ("rule", 2, 3000, "all"), ("rule", 8, 3000, "warm"), ("rule", 80, 200, "all"),
    ("rule", 5, 4096, "warm"), ("all", 64, 9000, "all"), ("parent", 2, 10, "parent"),
    ("warm", 3, 10, "warm"), ("warm", 1, 9000, "all")])
def test_schedule_plans(schedule, count, prefix, plan):
    assert api_server.plan_schedule(schedule, count, prefix, recompute_tokens=16384) == plan


def test_unknown_schedule_is_refused():
    with pytest.raises(ValueError):
        api_server.plan_schedule("fastest", 2, 2048)


@pytest.mark.parametrize("schedule,expected", [
    ("all", ["start 0", "start 1", "start 2", "end 0", "end 1", "end 2"]),
    ("parent", ["start 0", "end 0", "start 1", "start 2", "end 1", "end 2"]),
    ("default", ["start 0", "end 0", "start 1", "start 2", "end 1", "end 2"]),
    ("warm", ["warm 1568", "start 0", "start 1", "start 2", "end 0", "end 1", "end 2"]),
    ("rule", ["start 0", "start 1", "start 2", "end 0", "end 1", "end 2"])])
def test_vllm_engine_sends_questions_in_the_planned_order(schedule, expected):
    import asyncio

    engine = object.__new__(api_server.VLLMEngine)  # no model: only the dispatch is tested
    engine.schedule, engine.block_size, events = schedule, 784, []
    engine.recompute_tokens = api_server.RULE_RECOMPUTE_TOKENS
    shared = list(range(2000))
    sequences = [shared + [9000 + i] for i in range(3)]

    async def one(sequence, options):
        events.append(f"start {sequence[-1] - 9000}")
        await asyncio.sleep(0)
        events.append(f"end {sequence[-1] - 9000}")
        return [float(len(options))]

    async def warm(prefix_ids):
        events.append(f"warm {len(prefix_ids)}")

    engine._one, engine._warm = one, warm
    values = asyncio.run(engine.logits(sequences, [[1], [1, 2], [1, 2, 3]],
                                       api_server.common_prefix(sequences)))
    assert values == [[1.0], [2.0], [3.0]]
    assert events == expected


# --- state-once tokenization (ported from bobcat.flash_server, 2026-09-26) ---------------

def char_student(tmp_path, identifiers=("A", "B", "C")):
    """A pinned toy student whose tokenizer maps every ASCII character to its own id, so
    different texts give different token IDs (the word-level toy maps most JSON to [UNK])."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    from bobcat.schema import file_hash

    tmp_path.mkdir(parents=True, exist_ok=True)
    chars = [chr(c) for c in range(32, 127)]
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tokenizer.add_special_tokens(["<|system|>", "<|user|>", "<|assistant|>"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    template = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    pinned = {name: {"status": "ok", "bytes": (tmp_path / name).stat().st_size,
                     "sha256": file_hash(tmp_path / name)}
              for name in ("tokenizer.json", "tokenizer_config.json")}
    return StudentCompiler(tmp_path, pinned, list(identifiers), max_branch_tokens=100000,
                           piecewise=True)


STATES = ["plain text", {"doc": "a b c", "n": 3, "nested": {"k": [1, 2.5, None, True]}},
          {"doc": "a b c", "n": 4}, ["list", {"x": "y"}], "", {"quote": "\"esc\\aped\""}]


def many_questions(count):
    return {f"q{j}": {"type": "choice", "instructions": f"question {j}",
                      "criteria": {"x": f"first {j}", "y": None}} for j in range(count)}


def test_state_cache_gives_the_stock_token_ids(tmp_path):
    from bobcat.protocol import parse_request

    stock = char_student(tmp_path / "s")
    cached = api_server.cache_state_encoding(char_student(tmp_path / "c"), max_tokens=64)
    assert api_server.cache_state_encoding(cached) is cached      # idempotent
    for state in STATES * 2:                                        # the second pass hits
        _, questions = parse_request({"model": "m", "state": state,
                                      "questions": many_questions(4)})
        for question in questions:
            assert cached.compile(state, question) == stock.compile(state, question)
        sequences = [stock.compile(state, q)[0] for q in questions]
        assert (api_server.billed_tokens(cached, state, sequences)
                == api_server.billed_tokens(stock, state, sequences))
    memo = cached._data
    assert memo.hits > 0 and memo.tokens <= 64                      # bounded, LRU-evicted
    first = cached._data({"doc": "a b c", "n": 4})
    first.append(999)                                               # callers may mutate
    assert cached._data({"doc": "a b c", "n": 4}) == stock._data({"doc": "a b c", "n": 4})
    long_state = {"doc": "z" * 200}                                 # longer than the cache
    assert cached._data(long_state) == stock._data(long_state)
    assert _dumps_key(long_state) not in memo.entries


def _dumps_key(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def test_state_is_encoded_once_per_request(tmp_path):
    compiler = char_student(tmp_path)
    calls, encode = [], compiler._data

    def counting(value):
        calls.append(_dumps_key(value))
        return encode(value)

    compiler._data = counting
    api_server.cache_state_encoding(compiler)
    engine = FakeEngine()
    app = api_server.create_api(engine, compiler, model_name="bobcat-1", aliases=set(),
                                temperature=1.0, models=MODELS, edge_secret=None)
    api = TestClient(app)
    state = {"doc": "one long document", "n": 7}
    body = {"model": "bobcat-1", "state": state, "questions": many_questions(5)}
    assert api.post("/v1/systemone", json=body).status_code == 200
    assert calls == [_dumps_key(state)]            # five questions, one state encoding
    assert api.post("/v1/systemone", json=body).status_code == 200
    assert calls == [_dumps_key(state)]            # the identical state again: no encoding
    other = {**body, "state": {"doc": "another document"}}
    assert api.post("/v1/systemone", json=other).status_code == 200
    assert calls == [_dumps_key(state), _dumps_key(other["state"])]


class SequenceEngine:
    """Logits that depend only on each question's own sequence (a stand-in for a model)."""
    name = "sequence"

    def __init__(self):
        self.calls = []

    async def logits(self, sequences, option_ids, prefix):
        self.calls.append([list(s) for s in sequences])
        return [[float((sum(s) * (i + 3)) % 11) for i in range(len(o))]
                for s, o in zip(sequences, option_ids, strict=True)]


def test_server_sends_the_same_token_ids_with_and_without_the_cache(tmp_path):
    replies, calls = [], []
    for name, cache in (("stock", False), ("cached", True)):
        compiler = char_student(tmp_path / name)
        if cache:
            api_server.cache_state_encoding(compiler)
        engine = SequenceEngine()
        api = TestClient(api_server.create_api(
            engine, compiler, model_name="bobcat-1", aliases=set(), temperature=1.3,
            models=MODELS, edge_secret=None))
        got = []
        for state in STATES:
            body = {"model": "bobcat-1", "state": state, "questions": many_questions(3)}
            reply = api.post("/v1/systemone", json=body)
            assert reply.status_code == 200
            inspect_reply(body, reply, expected_model="bobcat-1")
            got.append((reply.json(), reply.headers["x-bobcat-processed-tokens"]))
        replies.append(got)
        calls.append(engine.calls)
    assert calls[0] == calls[1]
    assert replies[0] == replies[1]


def test_questions_stay_isolated_with_the_state_cache(tmp_path):
    compiler = api_server.cache_state_encoding(char_student(tmp_path))
    reference = char_student(tmp_path / "ref")
    engine = SequenceEngine()
    api = TestClient(api_server.create_api(
        engine, compiler, model_name="bobcat-1", aliases=set(), temperature=1.0,
        models=MODELS, edge_secret=None))
    state = {"doc": "shared state"}
    a = {"type": "noul", "instructions": "is it shared?"}
    b = {"type": "choice", "instructions": "which", "criteria": {"x": "one", "y": "two"}}
    c = {"type": "score", "instructions": "how much", "criteria": ["low", "mid", "high"]}
    answers = []
    for questions in ({"a": a}, {"a": a, "b": b}, {"b": b, "a": a, "c": c}):
        reply = api.post("/v1/systemone", json={"model": "bobcat-1", "state": state,
                                                "questions": questions})
        assert reply.status_code == 200
        answers.append(reply.json()["answers"]["a"])
    # The same question gets the same sequence and answer whatever else is asked beside it.
    assert answers[0] == answers[1] == answers[2]
    from bobcat.protocol import parse_request

    _, (alone,) = parse_request({"model": "m", "state": state, "questions": {"a": a}})
    expected = reference.compile(state, alone)[0]
    assert all(expected in call for call in engine.calls)
    # Each sequence holds its own question only: no other question's text is in it.
    for call in engine.calls:
        for sequence in call:
            texts = [t for t in ("is it shared?", "which", "how much")
                     if reference._text(t) and _contains(sequence, reference._text(t))]
            assert len(texts) == 1
    # A new state after a cached one: the sequences carry the new state's tokens only.
    reply = api.post("/v1/systemone", json={"model": "bobcat-1", "state": {"doc": "new one"},
                                            "questions": {"a": a}})
    assert reply.status_code == 200
    _, (fresh,) = parse_request({"model": "m", "state": {"doc": "new one"},
                                 "questions": {"a": a}})
    assert engine.calls[-1] == [reference.compile({"doc": "new one"}, fresh)[0]]


def _contains(sequence, part):
    return any(sequence[i:i + len(part)] == part for i in range(len(sequence) - len(part) + 1))


def test_per_question_temperatures_build_the_same_reply_as_one_temperature(tmp_path):
    from bobcat.protocol import parse_request

    _, questions = parse_request({"model": "m", "state": "s", "questions": {
        "n": {"type": "noul", "instructions": "yes?"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"x": None, "y": None}},
        "s": {"type": "score", "instructions": "rate", "criteria": ["a", "b", "c"]}}})
    scores = [[0.2, 1.1], [3.0, -1.0], [0.5, 0.1, 2.0]]
    one = api_server.build_response("m", questions, scores,
                                    {"choice": 0.9, "noul": 0.9, "score": 0.9}, 12)
    assert api_server.build_response("m", questions, scores, [0.9, 0.9, 0.9], 12) == one
    mixed = api_server.build_response("m", questions, scores, [0.9, 1.2, 0.9], 12)
    assert mixed["answers"]["n"] == one["answers"]["n"]
    assert mixed["answers"]["c"] != one["answers"]["c"]
    with pytest.raises(ValueError):
        api_server.build_response("m", questions, scores, [0.9, 0.9], 12)
