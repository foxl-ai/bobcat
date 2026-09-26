import json
import re
from pathlib import Path

import pytest

from bobcat import product_eval as pe
from bobcat import stress
from bobcat import student_data_v11 as v11
from bobcat.product_eval import text_key
from bobcat.protocol import parse_request

EVAL_PAD_KEYS = ("기타 자료 1", "기타 자료 2")  # scripts/long_context_eval.py


def row(i, *, task="product_search", family="answers_query", kind="noul", target="yes",
        state=None, language="ko", source="product", supervision="hard_label", group=None):
    if kind == "noul":
        question = pe.noul("질문인가?", "예", "아니요")
        labels = ["no", "yes"]
    else:
        question = pe.choice("고르라.", {"숙소": "숙소 예약", "식당": "식당 예약",
                                        "택시": "택시 호출"})
        labels = ["숙소", "식당", "택시"]
    state = state if state is not None else {
        "질의": f"질문 {i}은 무엇인가?",
        "문단": f"첫 문장 {i}이다. 둘째 문장이다. 셋째 문장은 조금 더 길게 쓴다. 넷째 문장이다."}
    request = {"model": "bobcat-latest", "state": state, "questions": {"q": question}}
    return {"id": f"{task}:{i}", "group_id": group or f"g{i}", "task": task, "family": family,
            "kind": "boolean" if kind == "noul" else "choice", "language": language,
            "request": request, "candidate_ids": labels, "target": target,
            "score_target": None, "supervision": supervision, "text_keys": [f"text:{i}"],
            "input_sha256": "x", "mixture_source": source}


def runs(text: str, n: int = 6) -> set[str]:
    text = text.replace("'{w}'", "\0").replace("{w}", "\0")
    return {text[i:i + n] for i in range(len(text) - n + 1) if "\0" not in text[i:i + n]}


def test_training_wording_is_disjoint_from_every_evaluation_attack():
    evaluation = (list(stress.INJECTIONS.values()) + pe.INJECTIONS + pe.BENIGN
                  + ["시스템 메모", "검토 이력", *EVAL_PAD_KEYS])
    training = [t for group in (v11.NOTES, v11.INLINE, v11.VERDICTS) for texts in group.values()
                for t in texts]
    training += [k for group in (v11.NOTE_KEYS, v11.VERDICT_KEYS) for keys in group.values()
                 for k in keys]
    training += [k for pairs in v11.PAD_KEYS.values() for pair in pairs for k in pair]
    training += list(v11.PAD_LIST_KEYS.values())
    held = set().union(*(runs(t) for t in evaluation))
    for text in training:
        assert not runs(text) & held, text
    assert not set(training) & set(evaluation)


def test_injected_copy_keeps_gold_candidates_and_component():
    rows = [row(i, target="yes" if i % 2 else "no") for i in range(200)]
    copies = [v11.inject_one(r, 7) for r in rows]
    for original, copy in zip(rows, copies, strict=True):
        assert copy["target"] == original["target"]
        assert copy["candidate_ids"] == original["candidate_ids"]
        assert copy["group_id"] == original["group_id"]
        assert copy["request"]["questions"] == original["request"]["questions"]
        assert copy["request"]["state"] != original["request"]["state"]
        assert copy["id"].startswith("inject:") and copy["derived"]["from"] == original["id"]
        dumped = json.dumps(copy["request"]["state"], ensure_ascii=False)
        assert f"'{copy['derived']['word']}'" in dumped
        meta = copy["derived"]
        if meta["named"] == "wrong":
            assert meta["named_label"] != original["target"]
        else:
            assert meta["named_label"] == original["target"]
        if meta["style"] == "inline":  # every original field is still present
            assert set(copy["request"]["state"]) == set(original["request"]["state"])
        else:
            assert list(original["request"]["state"].items()) == [
                kv for kv in copy["request"]["state"].items() if kv[0] != meta["field"]]
    styles = {c["derived"]["style"] for c in copies}
    assert styles == {"note", "inline", "verdict"}
    gold = sum(c["derived"]["named"] == "gold" for c in copies) / len(copies)
    assert 0.1 < gold < 0.3
    words = {c["derived"]["word"] for c in copies}
    assert {"true", "false"} & words


def test_screening_instruction_rows_and_score_means_are_never_injected():
    assert not v11.injectable(row(1, task="product_injection", family="instruction_to_system"))
    assert v11.injectable(row(2, task="product_injection", family="evidence_under_injection"))
    assert not v11.injectable(row(3, supervision="score_mean"))


def test_string_states_take_the_inline_style():
    copy = v11.inject_one(row(1, state="원문 첫 문장이다. 둘째 문장이다. 셋째 문장이다."), 3)
    assert copy["derived"]["style"] == "inline"
    assert isinstance(copy["request"]["state"], str)
    assert "원문 첫 문장이다." in copy["request"]["state"]


def test_choice_copies_name_a_candidate_or_its_meaning():
    rows = [row(i, kind="choice", target="식당", task="product_routing", family="service")
            for i in range(60)]
    meanings = {"숙소": "숙소 예약", "식당": "식당 예약", "택시": "택시 호출"}
    for copy in (v11.inject_one(r, 11) for r in rows):
        word = copy["derived"]["word"]
        assert word in meanings or word in meanings.values()


def test_round_robin_cycles_tasks_then_starts_new_passes():
    rows = [row(i, task="a") for i in range(3)] + [row(i + 10, task="b") for i in range(10)]
    picks = v11.round_robin(rows, 10, "salt")
    tasks = [r["task"] for r, _ in picks]
    assert tasks[:6] == ["a", "b"] * 3 and len(picks) == 10
    many = v11.round_robin(rows[:3], 7, "salt")
    assert [n for _, n in many] == [0, 0, 0, 1, 1, 1, 2]
    copies = [v11.inject_one(r, 5, n) for r, n in many]
    assert len({c["id"] for c in copies}) == 7


def fake_length(state):
    return len(json.dumps(state, ensure_ascii=False)) // 2


def test_long_copy_pads_with_other_components_and_keeps_the_state():
    passages = [row(i, task="product_citation", group=f"p{i}") for i in range(400)]
    for i, p in enumerate(passages):
        p["request"]["state"]["문단"] = (f"다른 문서 {i}의 내용이다. " * 30)
    target_row = row(1000, group="p3")
    pools = v11.passage_pool(passages)
    assert pools["ko"] and not pools["en"]
    pools["en"] = pools["ko"]
    copy = v11.long_copy(target_row, pools, 4000, fake_length, seed=1)
    assert copy is not None and copy["id"].startswith("long:")
    assert 0.9 * 4000 <= fake_length(copy["request"]["state"]) <= 4000
    state = copy["request"]["state"]
    for key, value in target_row["request"]["state"].items():
        assert state[key] == value
    keys = [k for k in state if k not in target_row["request"]["state"]]
    assert keys and not set(keys) & set(EVAL_PAD_KEYS)
    own = json.dumps(passages[3]["request"]["state"]["문단"], ensure_ascii=False)[1:40]
    assert own not in json.dumps({k: state[k] for k in keys}, ensure_ascii=False)
    assert copy["target"] == target_row["target"]
    assert set(target_row["text_keys"]) < set(copy["text_keys"])


def test_padding_never_returns_the_rows_own_passages():
    passages = [(f"g{i % 3}", f"문단 {i} " * 50) for i in range(30)]
    target = row(1, group="g0")
    parts = v11.padding(target, passages, 5000, "s")
    assert sum(len(p) for p in parts) <= 5000 + 2 * len(parts)
    own = {text for group, text in passages if group == "g0"}
    assert not own & set(parts[:-1])


def test_arrange_aligns_long_rows_to_rank_blocks(tmp_path):
    short = tmp_path / "short.jsonl"
    long = tmp_path / "long.jsonl"
    short.write_text("".join(json.dumps({"id": f"s{i}", "input_ids": [1] * (i % 7 + 1)}) + "\n"
                             for i in range(83)))
    long.write_text("".join(json.dumps({"id": f"long:{i}", "input_ids": [1] * (100 + i)}) + "\n"
                            for i in range(19)))
    meta = v11.arrange([short, long], tmp_path / "out.jsonl", world=8, seed=3)
    ids = [json.loads(line)["id"] for line in (tmp_path / "out.jsonl").open()]
    assert sorted(ids) == sorted([f"s{i}" for i in range(83)] + [f"long:{i}" for i in range(19)])
    blocks = {}
    for index, name in enumerate(ids):
        if name.startswith("long:"):
            blocks.setdefault(index // 8, []).append(index)
    assert meta["partial_long_blocks"] <= 1 and meta["blocks_with_long_rows"] == 3
    full = [b for b in blocks.values() if len(b) == 8]
    assert len(full) == 2 and all(b == list(range(b[0], b[0] + 8)) for b in full)


def test_insert_text_keeps_the_original_characters():
    text = "첫 문장이다.\n둘째 문장이다. 셋째 문장이다."
    for where in ("start", "middle", "end"):
        out = v11.insert_text(text, "끼움.", where)
        assert out.replace("끼움. ", "", 1).replace(" 끼움.", "", 1) == text


# ---------------------------------------------------------------- fresh final

def mrc(i: int, *, impossible=False, guid=None, context=None):
    """KLUE MRC-shaped rows (as in tests/test_product_eval.py)."""
    answer = f"{1700 + i}년"
    context = context or (f"{v11_word(i)} 회사는 {answer}에 설립되었다. 본사는 서울에 있다. "
                          f"직원은 약 300명이다. {v11_word(i + 5000)} 제품을 만든다.")
    return {"title": f"{v11_word(i)} {v11_word(i + 7000)}", "context": context,
            "news_category": "사회", "source": "hankyung",
            "guid": guid or f"klue-mrc-v1_dev_{i:05d}", "is_impossible": impossible,
            "question_type": 3 if impossible else 1,
            "question": f"{v11_word(i)} 회사는 언제 설립되었나?",
            "answers": {"answer_start": [] if impossible else [10],
                        "text": [] if impossible else [answer]}}


def tables(n):
    rows = [mrc(i) for i in range(n)]
    rows += [mrc(i, impossible=True, guid=f"klue-mrc-v1_dev_imp_{i:05d}",
                 context=rows[i]["context"]) for i in range(0, n, 3)]
    rows += [mrc(i, impossible=True, guid=f"klue-mrc-v1_dev_imp_{i:05d}")
             for i in range(n, n + n // 2)]
    return {"mrc:validation": rows, "mrc:train": [], "wos:validation": [], "wos:train": []}


def test_fresh_final_uses_only_components_no_split_or_training_used(tmp_path, monkeypatch):
    data = tables(900)
    monkeypatch.setattr(pe, "load_sources", lambda root, source: data)
    monkeypatch.setattr(pe, "training_sentences", lambda root: set())
    monkeypatch.setattr(pe, "TITLE_K", (3, 5))
    pool = pe.articles(data, 2026092402, set())
    held_articles = pool["dev"][:40]
    used_articles = pool["final"][:40]
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    for split in pe.SPLITS:
        (eval_dir / f"{split}.jsonl").write_text("".join(
            json.dumps({"group_id": f"product:{a.component}"}) + "\n"
            for a in (held_articles if split == "dev" else [])))
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps({"group_id": f"product:{a.component}"}) + "\n"
                             for a in used_articles))
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"revision": "r" * 40, "files": []}))
    limits = {**pe.DEFAULT_LIMITS, "title_queries": 20, "search_pairs": 40,
              "search_other": 20, "citation_numeric": 40, "citation_plain": 0,
              "injection": 40, "topic": 20}
    manifest = pe.fresh_final(sources, tmp_path, tmp_path, eval_dir, train,
                              Path("configs/tool-situations-v3-final.json"), tmp_path / "out",
                              seed=5, limits=limits)
    rows = [json.loads(line) for line in (tmp_path / "out" / "final.jsonl").open()]
    forbidden = {a.component for a in held_articles + used_articles}
    assert rows and all(r["split"] == "final" for r in rows)
    for r in rows:
        assert r["group_id"].removeprefix("product:") not in forbidden
        assert r.get("other_component") not in forbidden
        _, questions = parse_request(r["request"])
        assert list(questions[0].labels) == r["candidate_ids"]
    tasks = {r["task"] for r in rows}
    assert {"product_search", "product_citation", "product_injection",
            "product_classification", "product_tool_call"} <= tasks
    assert manifest["schema"] == "bobcat-product-eval-v2"
    assert set(manifest["files"]) == {"final"}
    assert manifest["excluded_components"]["training_build"] == 40
    # Evaluation wording, not the training templates, in the screening task.
    injected = [r for r in rows if r["family"] == "instruction_to_system"
                and r["counterfactual"]["variant"] == "injected"]
    assert injected and all(any(t in r["request"]["state"]["문서"] for t in pe.INJECTIONS)
                            for r in injected)


def test_new_tool_situations_are_new_and_complete():
    catalog = json.loads(Path("configs/tool-situations-v3-final.json").read_text())
    earlier = set()
    for name in ("tool-situations-v1.json", "tool-situations-v2-ood.json"):
        earlier |= {s["id"] for s in json.loads(Path("configs", name).read_text())["situations"]}
    ids = {s["id"] for s in catalog["situations"]}
    assert len(ids) == 8 and not ids & earlier
    rows = pe.tool_rows(catalog, 1)
    assert {r["family"] for r in rows} == {"scope", "irreversible", "plan_mismatch"}
    assert {r["target"] for r in rows if r["family"] == "irreversible"} == {"yes", "no"}
    commands = [c["command"] for s in catalog["situations"] for c in s["commands"].values()]
    old = [c["command"] for name in ("tool-situations-v1.json", "tool-situations-v2-ood.json")
           for s in json.loads(Path("configs", name).read_text())["situations"]
           for c in s["commands"].values()]
    assert not set(commands) & set(old)


def test_text_keys_cover_inserted_passages():
    passage = "다른 문서의 긴 내용이다. " * 20
    copy = v11.derived(row(1), {**row(1)["request"]["state"], "부록": passage}, kind="long",
                       meta={"level": 1}, texts=[passage])
    assert text_key(passage) in copy["text_keys"]


@pytest.mark.parametrize("language", ["ko", "en"])
def test_templates_format_cleanly(language):
    for group in (v11.NOTES, v11.INLINE, v11.VERDICTS):
        for template in group[language]:
            text = template.format(w="X")
            assert "'X'" in text and not re.search(r"[{}]", text)


# ---------------------------------------------------------------- abstain copies

def citation(i, target="supports"):
    state = {"주장": {"질문": f"회사 {i}는 언제 세워졌나?", "주장한 답": f"{1900 + i}년"},
             "인용 문단": (f"회사 {i}는 {1900 + i}년에 세워졌다. 본사는 서울에 있다. "
                       f"직원은 300명이다. 주력 제품은 가전이다.")}
    r = row(i, task="product_citation", family="citation_relation", kind="choice",
            target="숙소", state=state)
    r["request"]["questions"] = {"relation": pe.RELATION}
    r["candidate_ids"], r["target"] = list(pe.RELATION["criteria"]), target
    r["source"] = {"guid": f"g{i}"}
    return r


def test_answer_removal_turns_support_into_says_nothing():
    copy = v11.removal_copy(citation(1), {"g1": ["1901년"]})
    assert copy["target"] == "says_nothing" and copy["label_basis"] == "evidence_removed"
    passage = copy["request"]["state"]["인용 문단"]
    assert "1901" not in passage and "본사는 서울에 있다." in passage
    assert copy["request"]["state"]["주장"] == citation(1)["request"]["state"]["주장"]
    assert v11.removal_copy(citation(2, target="contradicts"), {}) is None


def test_answer_removal_refuses_when_the_answer_survives_or_too_little_is_left():
    assert v11.without_answer("가는 1이다. 나는 2이다.", ["1"]) is None  # one sentence left
    assert v11.without_answer("가는 1이다. 나는 2이다. 다는 3이다.", ["없음"]) is None
    assert v11.without_answer("가 2다. 나 2다. 다 2다. 라 3다. 마 4다.", ["2"]) is None  # >half
    assert v11.without_answer("A는 X다. B는 Y다. C는 Z다.", ["X"]) == "B는 Y다. C는 Z다."


def nli(i, family="nli_grounding", target="함의", encoded=False, task="kornli_multinli"):
    body = {"전제": f"{v11_word(i)}{v11_word(i + 50)} 공원에서 달린다.",
            "가설": f"{v11_word(i)} 움직인다."}
    r = row(i, task=task, family=family, kind="choice", target="숙소",
            state=json.dumps(body, ensure_ascii=False) if encoded else body)
    r["request"]["questions"] = {"decision": pe.choice("관계를 고르라.", {
        "함의": "참", "중립": "모름", "모순": "거짓"})}
    r["candidate_ids"], r["target"] = ["함의", "중립", "모순"], target
    return r


def v11_word(i):
    return "".join(chr(0xAC00 + (i * 97 + j * 1231) % 11172) for j in range(3))


def test_unrelated_premise_makes_nli_neutral_and_keeps_the_encoding():
    rows = [nli(i) for i in range(20)]
    copy = v11.premise_copy(rows[0], rows)
    assert copy["target"] == "중립"
    assert copy["request"]["state"]["가설"] == rows[0]["request"]["state"]["가설"]
    assert copy["request"]["state"]["전제"] != rows[0]["request"]["state"]["전제"]
    encoded = [nli(i, encoded=True, task="klue_nli", family="klue_nli_relation")
               for i in range(20)]
    copy = v11.premise_copy(encoded[3], encoded)
    assert isinstance(copy["request"]["state"], str) and copy["target"] == "중립"
    assert json.loads(copy["request"]["state"])["가설"] == json.loads(
        encoded[3]["request"]["state"])["가설"]


def test_boolq_claims_are_three_way_with_an_abstain_on_a_swapped_passage():
    rows = []
    for i in range(30):
        state = {"passage": f"Passage about topic{chr(97 + i % 26)}{i} with facts.",
                 "question": f"is thing{i} special"}
        r = row(i, task="boolq", family="reading_grounding", state=state, language="en",
                target="yes" if i % 2 else "no", source="public")
        rows.append(r)
    copies = v11.claim_rows(rows[1], rows, 3)
    assert len(copies) == 2
    original, swapped = copies
    assert original["request"]["state"]["passage"] == rows[1]["request"]["state"]["passage"]
    assert swapped["request"]["state"]["passage"] != original["request"]["state"]["passage"]
    labels = set(original["candidate_ids"])
    assert labels in ({"supported", "contradicted", "not_stated"}, {"true", "false", "unknown"})
    assert swapped["target"] in ("not_stated", "unknown")
    polarity = original["derived"]["polarity"]
    assert (original["target"] in ("supported", "true")) == (polarity == "yes")
    for copy in copies:
        _, questions = parse_request(copy["request"])
        assert list(questions[0].labels) == copy["candidate_ids"]


def test_evidence_schemes_map_neutral_to_the_abstain_label():
    for instructions, labels, meanings in v11.EVIDENCE_SCHEMES:
        assert len(labels) == len(meanings) == 3 and instructions
        assert labels[1] in ("insufficient", "unknown", "not_stated")
        assert labels[0] in ("supported", "true") and labels[2] in ("contradicted", "false")


def test_permuted_copy_keeps_gold_and_changes_order():
    r = row(1, kind="choice", target="식당", task="snli", family="nli_grounding", language="en")
    copy = v11.permuted_copy(r, 3)
    assert copy["target"] == "식당" and copy["candidate_ids"] != r["candidate_ids"]
    assert sorted(copy["candidate_ids"]) == sorted(r["candidate_ids"])
    _, questions = parse_request(copy["request"])
    assert list(questions[0].labels) == copy["candidate_ids"]
    assert v11.permuted_copy(row(2), 3) is None  # Noul rows are left alone


def test_stated_schemes_put_the_abstain_label_second():
    for instructions, labels, meanings in v11.STATED_SCHEMES:
        assert labels[1] in ("not_stated", "insufficient", "cannot_tell")
        assert len(meanings) == 2 and instructions
