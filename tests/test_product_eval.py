import copy
import json
from pathlib import Path

import pytest

from bobcat import product_eval as pe
from bobcat.protocol import parse_request

CATALOG = json.loads(Path("configs/tool-situations-v1.json").read_text())


def word(i: int) -> str:
    return "".join(chr(0xAC00 + (i * 131 + j * 1777) % 11172) for j in range(3))


def mrc(i: int, *, impossible=False, answer=None, category="사회", guid=None, context=None):
    answer = answer or f"{1700 + i}년"
    context = context or (f"{word(i)} 회사는 {answer}에 설립되었다. 본사는 서울에 있다. "
                          f"직원은 약 300명이다. {word(i + 5000)} 제품을 만든다.")
    return {"title": f"{word(i)} {word(i + 7000)}", "context": context, "news_category": category,
            "source": "hankyung", "guid": guid or f"klue-mrc-v1_dev_{i:05d}",
            "is_impossible": impossible, "question_type": 3 if impossible else 1,
            "question": f"{word(i)} 회사는 언제 설립되었나?",
            "answers": {"answer_start": [] if impossible else [10],
                        "text": [] if impossible else [answer]}}


def tables(n=240):
    rows = [mrc(i) for i in range(n)]
    rows += [mrc(i, impossible=True, guid=f"klue-mrc-v1_dev_imp_{i:05d}",
                 context=rows[i]["context"]) for i in range(0, n, 3)]
    rows += [mrc(i, impossible=True, guid=f"klue-mrc-v1_dev_imp_{i:05d}")
             for i in range(n, n + n // 2)]
    return {"mrc:validation": rows, "mrc:train": [], "wos:validation": [], "wos:train": []}


def test_partition_is_deterministic_and_uses_three_splits():
    names = [pe.partition(f"c{i}", 7) for i in range(400)]
    assert names == [pe.partition(f"c{i}", 7) for i in range(400)]
    assert set(names) == set(pe.SPLITS)
    assert 150 < names.count("dev") < 250


def test_perturbed_number_differs_and_is_absent_from_context():
    context = "회사는 1990년에 설립되었고 1995년에 상장했다."
    wrong = pe.perturb("1990년", context, "seed")
    assert wrong and wrong != "1990년" and wrong.endswith("년")
    assert wrong[:-1] not in context
    assert pe.perturb("서울", context, "seed") is None


def test_insert_places_a_whole_sentence():
    text = "첫 문장이다. 둘째 문장이다. 셋째 문장이다."
    assert pe.insert(text, "끼운 문장.", "start").startswith("끼운 문장.")
    assert pe.insert(text, "끼운 문장.", "end").endswith("끼운 문장.")
    assert pe.sentences(pe.insert(text, "끼운 문장.", "middle"))[1] == "끼운 문장."


def test_mrc_tasks_keep_every_component_in_one_split(monkeypatch):
    monkeypatch.setattr(pe, "TITLE_K", (3, 5))
    pool = pe.articles(tables(), 11, set())
    limits = {**pe.DEFAULT_LIMITS, "title_queries": 12, "citation_numeric": 20,
              "citation_plain": 0, "injection": 16, "topic": 12, "search_other": 12}
    taken: set = set()
    rows = pe.search_rows(pool, 11, limits)
    taken |= {r["group_id"].removeprefix("product:") for r in rows}
    rows += pe.title_rows(pool, 11, limits, taken)
    taken |= {r["group_id"].removeprefix("product:") for r in rows}
    rows += pe.citation_rows(pool, 11, limits, taken)
    rows += pe.injection_rows(pool, 11, limits, taken)
    rows += pe.topic_rows(pool, 11, limits, taken)
    checks = pe.audit(rows)
    assert not checks["components_crossing_splits"]
    by_family = {r["family"] for r in rows}
    assert {"answers_query", "document_title_k3", "document_title_k5", "citation_relation",
            "instruction_to_system", "evidence_under_injection", "taxonomy_fine",
            "taxonomy_coarse"} <= by_family
    for row in rows:
        _, questions = parse_request(copy.deepcopy(row["request"]))
        assert list(questions[0].labels) == row["candidate_ids"]
        assert row["target"] in row["candidate_ids"]
        assert "klue-mrc" not in json.dumps(row["request"], ensure_ascii=False)
    contradicted = [r for r in rows if r["target"] == "contradicts"]
    assert contradicted and all(r["review"] == "required" for r in contradicted)
    for row in rows:
        if row["family"] == "instruction_to_system":
            assert (row["target"] == "yes") == (row["counterfactual"]["variant"] == "injected")
    titles = [r for r in rows if r["family"].startswith("document_title")]
    assert all(len(r["candidate_ids"]) == int(r["family"].rsplit("k", 1)[1]) for r in titles)


def test_evidence_label_follows_the_upstream_answerable_flag(monkeypatch):
    pool = pe.articles(tables(), 5, set())
    rows = pe.injection_rows(pool, 5, {**pe.DEFAULT_LIMITS, "injection": 40}, set())
    evidence = [r for r in rows if r["family"] == "evidence_under_injection"]
    assert {r["target"] for r in evidence} == {"yes", "no"}
    for row in evidence:
        guid = row["source"]["guid"]
        assert (row["target"] == "no") == ("imp" in guid)


def test_tool_scope_uses_the_explicit_table_and_splits_whole_situations():
    rows = pe.tool_rows(CATALOG, 3)
    assert pe.audit(rows)["components_crossing_splits"] == []
    assert len({r["split"] for r in rows}) == 3
    reset = [r for r in rows if r["family"] == "scope"
             and r["request"]["state"]["대기 중인 동작"]["command"] == "npm run db:reset"]
    assert {r["source"]["task"]: r["target"] for r in reset} == {
        "reset": "expected", "column": "overreach"}
    irreversible = {r["source"]["command"]: r["target"] for r in rows
                    if r["family"] == "irreversible" and r["source"]["situation"] == "files"}
    assert irreversible == {"rm_cache": "no", "grep_logs": "no", "rm_raw": "yes", "ls": "no"}
    pairs = {}
    for row in rows:
        if row["family"] == "plan_mismatch":
            pairs.setdefault(row["counterfactual"]["pair_id"], set()).add(row["target"])
    assert pairs and all(targets == {"yes", "no"} for targets in pairs.values())
    skipped = [r for r in rows if r["family"] == "scope" and r["source"]["situation"] == "database"
               and r["source"]["task"] == "reset" and r["source"]["command"] == "migrate"]
    assert skipped == []


def test_incomplete_scope_table_is_rejected():
    broken = copy.deepcopy(CATALOG)
    del broken["situations"][0]["scope"]["reset"]["status"]
    with pytest.raises(ValueError, match="incomplete"):
        pe.tool_rows(broken, 3)


def test_routing_labels_come_from_new_annotated_slots():
    first = ["식당-지역-강남", "식당-종류-한식당"]
    booked = first + ["식당-예약 요일-토요일", "식당-예약 명수-2"]
    dialogue = {"guid": "wos-v1_dev_00001", "dialogue": [
        {"role": "user", "text": "강남에 있는 한식당 알려줘.", "state": first},
        {"role": "sys", "text": "두 곳이 있습니다.", "state": []},
        {"role": "user", "text": "토요일 2명으로 예약해줘.", "state": booked},
        {"role": "sys", "text": "예약했습니다.", "state": []},
        {"role": "user", "text": "근처 호텔도 찾아줘.", "state": booked + ["숙소-지역-강남"]},
    ]}
    data = {"wos:validation": [dialogue], "wos:train": []}
    rows = pe.routing_rows(data, 1, {"routing_per_service": 40})
    service = [(r["source"]["turn"], r["target"]) for r in rows if r["family"] == "service"]
    reservation = [(r["source"]["turn"], r["target"]) for r in rows if r["family"] == "reservation"]
    assert service == [(0, "식당"), (2, "식당")]
    assert reservation == [(0, "no"), (2, "yes")]
    assert rows[0]["request"]["state"]["이전 대화"] == []


def test_audit_rejects_a_component_in_two_splits():
    rows = pe.tool_rows(CATALOG, 3)[:2]
    moved = copy.deepcopy(rows[1])
    moved["split"] = next(s for s in pe.SPLITS if s != rows[0]["split"])
    moved["id"] += ":moved"
    with pytest.raises(ValueError, match="Split or identity"):
        pe.audit([rows[0], moved])


def test_written_rows_keep_the_presented_candidate_order(tmp_path):
    rows = [r for r in pe.tool_rows(CATALOG, 3) if r["family"] == "scope"][:3]
    path = tmp_path / "dev.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    pe.verify_written(path)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                            for r in rows))
    with pytest.raises(ValueError, match="changed its candidate order"):
        pe.verify_written(path)
