import json

import pytest

from bobcat.kobest_eval import components, freeze, record
from bobcat.public_eval import run
from bobcat.schema import file_hash


def examples():
    return {
        "boolq": {"paragraph": "매장은 월요일에 쉰다.", "question": "월요일은 휴무인가?",
                  "label": 1},
        "copa": {"premise": "도로가 젖어 있다.", "question": "원인",
                 "alternative_1": "비가 왔다.", "alternative_2": "햇볕이 내리쬤다.", "label": 0},
        "wic": {"word": "배", "context_1": "[배]를 타고 바다로 갔다.",
                "context_2": "과수원에서 [배]를 땄다.", "label": 0},
        "hellaswag": {"context": "가방을 닫고 현관으로 걸어갔다.",
                      "ending_1": "신발을 신었다.", "ending_2": "책상에서 가방을 열었다.",
                      "ending_3": "이불을 덮었다.", "ending_4": "씻으러 들어갔다.", "label": 0},
        "sentineg": {"sentence": "서비스가 나쁘지 않았다.", "label": 1},
    }


def test_explicit_inputs_exclude_gold_and_metadata_and_preserve_label_semantics():
    expected = {"boolq": "yes", "copa": "선택1", "wic": "no",
                "hellaswag": "선택1", "sentineg": "긍정"}
    for task, raw in examples().items():
        row = record(task, raw | {"outcome": "PRIVATE-GOLD", "id": "PRIVATE-ID"}, 0)
        assert row["target"] == expected[task]
        assert "PRIVATE" not in json.dumps(row["request"])
        changed = record(task, raw | {"label": 1 - raw["label"]}, 0)
        assert row["request"] == changed["request"] and row["id"] == changed["id"]
        assert row["target"] != changed["target"]
    cause = record("copa", examples()["copa"], 0)
    effect = record("copa", examples()["copa"] | {"question": "결과"}, 0)
    assert "원인" in cause["request"]["questions"]["q"]["instructions"]
    assert "결과" in effect["request"]["questions"]["q"]["instructions"]
    with pytest.raises(ValueError, match="cause/effect"):
        record("copa", examples()["copa"] | {"question": "모름"}, 0)


def test_shared_evidence_and_transitive_sentence_pairs_are_one_component():
    rows = [
        record("boolq", examples()["boolq"], 0),
        record("boolq", examples()["boolq"] | {"question": "화요일이 휴무인가?"}, 1),
        record("wic", examples()["wic"], 0),
        record("wic", examples()["wic"] | {"context_1": "새 문장"}, 1),
        record("wic", examples()["wic"] | {"context_1": "새 문장", "context_2": "다른 문장"}, 2),
    ]
    groups = components(rows)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3] == groups[4]
    assert groups[0] != groups[2]


def fixture(tmp_path):
    raw = tmp_path / "raw"
    files = []
    for task, row in examples().items():
        path = raw / task / "test.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
        files.append({"task": task, "path": f"{task}/test.jsonl", "rows": 1,
                      "sha256": file_hash(path), "bytes": path.stat().st_size})
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({
        "schema": "bobcat-kobest-evaluation-sources-v1", "repo": "skt/kobest_v1",
        "revision": "a" * 40, "license": "CC-BY-SA-4.0", "training_use": False,
        "attribution": "Software test fixture, not benchmark data", "files": files,
    }))
    return config, raw


class ValidResponseFixture:
    model_name = "software-fixture"
    temperatures = {}

    def score(self, state, questions):
        assert "label" not in json.dumps(state) and "target" not in json.dumps(state)
        return [[0.] * len(q.labels) for q in questions], 1


def test_frozen_korean_diagnostics_use_existing_typed_evaluator_without_training(tmp_path):
    config, raw = fixture(tmp_path)
    suite = freeze(config, raw, tmp_path / "suite.json", per_task=1)
    repeated = freeze(config, raw, tmp_path / "suite-again.json", per_task=1)
    assert suite == repeated and suite["question_count"] == 5
    assert suite["training_use"] is False
    assert {r["language"] for g in suite["groups"] for r in g["rows"]} == {"ko"}
    result = run(suite, ValidResponseFixture(), tmp_path / "evaluated")
    assert result["status"] == "completed" and result["attempted_questions"] == 5
    assert result["final_evaluation"] is False
    summary = json.loads((tmp_path / "evaluated/summary.json").read_text())
    assert summary["calibration_refitted"] is False
    assert result["release_gate_passed"] is False
    path = raw / "boolq/test.jsonl"
    path.write_text(path.read_text().replace("월요일", "화요일"))
    with pytest.raises(ValueError, match="checksum"):
        freeze(config, raw, tmp_path / "corrupt.json", per_task=1)


def test_upstream_training_and_originated_reviews_cannot_enter_the_test_suite(tmp_path):
    config, raw = fixture(tmp_path)
    source = json.loads(config.read_text())
    source["files"][-1]["path"] = "sentineg/test_originated.jsonl"
    config.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="five test files"):
        freeze(config, raw, tmp_path / "forbidden.json", per_task=1)
    source["training_use"] = True
    config.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="evaluation-only"):
        freeze(config, raw, tmp_path / "forbidden-training.json", per_task=1)
