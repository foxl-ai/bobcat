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
