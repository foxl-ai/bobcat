"""Bobcat Flash corpus: generated families are valid requests with correct gold, the builder
drops evaluation text and never builds tool-call rows, and compiled rows keep their IDs."""

import json
from collections import Counter

from bobcat import flash_data
from bobcat import flash_families as fam
from bobcat.protocol import parse_request


def test_families_parse_and_gold_is_offered():
    titles = [f"Page {i}" for i in range(40)]
    counts = Counter()
    for lang in ("en", "ko"):
        for i in range(60):
            rows = (fam.tictactoe(i, lang) + fam.gridworld(i, lang) + fam.shooter(i, lang)
                    + fam.invoice(i, lang) + fam.security(i, lang)
                    + fam.linkrace(i, lang, titles))
            for row in rows:
                _, questions = parse_request(row["request"])
                assert list(questions[0].labels) == row["candidate_ids"]
                if row["supervision"] == "hard_label":
                    assert row["target"] in row["candidate_ids"]
                else:
                    assert row["target"] is None
                assert "tool" not in row["task"]
                counts[row["family"]] += 1
    assert counts["game_shooter_strategy"] > 50 and counts["workflow_invoice_route"] == 120


def test_tictactoe_policy():
    board = list("XX.OO....")
    assert fam.ttt_rule_move(board, "X") == 2         # win now
    assert fam.ttt_rule_move(list("OO.X.X..."), "X") == 4      # own win before block
    assert fam.ttt_rule_move(list("OO......."), "X") == 2       # block


def test_shooter_rules_follow_text():
    assert fam.shooter_rule("pacifist", 90, 30, [{"distance_m": 5}], []) == "retreat"
    assert fam.shooter_rule("survive", 20, 5, [], [{"type": "medkit", "distance_m": 3}]) \
        == "pick_up_medkit"
    assert fam.shooter_rule("aggressive", 80, 0, [{"distance_m": 5}], []) == "retreat"
    assert fam.shooter_rule("collector", 80, 2, [{"distance_m": 20}],
                            [{"type": "ammo", "distance_m": 4}]) == "pick_up_ammo"


def test_requestion_nli_maps_labels():
    source = {"id": "k1", "task": "kornli_multinli", "language": "ko", "group_id": "g1",
              "text_keys": ["text:a"], "target": "중립",
              "request": {"state": {"전제": "비가 온다.", "가설": "우산을 샀다."}}}
    rows = fam.requestion(source)
    assert len(rows) == 1
    row = rows[0]
    assert row["target"] in ("insufficient", "근거 부족")
    assert row["group_id"] == "g1" and row["text_keys"] == ["text:a"]


def mixture_row(i, text_key):
    request = {"model": "bobcat-latest", "state": f"state {i}",
               "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    return {"id": f"m{i}", "group_id": f"g{i}", "task": "klue_nli", "family": "x",
            "kind": "boolean", "language": "ko", "request": request,
            "candidate_ids": ["no", "yes"], "target": "yes", "score_target": None,
            "supervision": "hard_label", "text_keys": [text_key], "mixture_source": "public"}


def test_build_drops_eval_text_and_external_strings(tmp_path):
    mixture = tmp_path / "mixture"
    mixture.mkdir()
    rows = [mixture_row(i, f"text:{i}") for i in range(20)]
    rows.append({**mixture_row(99, "text:x"), "task": "product_tool_call"})
    external_text = "this exact sentence appears in an evaluation request file"
    leaked = mixture_row(50, "text:50")
    leaked["request"]["state"] = {"doc": external_text}
    rows.append(leaked)
    (mixture / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "expanded.jsonl").write_text("")
    (tmp_path / "policy.jsonl").write_text("")
    evaluation = tmp_path / "dev.jsonl"
    evaluation.write_text(json.dumps({"text_keys": ["text:3"]}) + "\n")
    external = tmp_path / "semif.jsonl"
    external.write_text(json.dumps({"request": {"state": external_text}}) + "\n")
    saved = dict(flash_data.GENERATED)
    try:
        for key in flash_data.GENERATED:
            flash_data.GENERATED[key] = 2
        manifest = flash_data.build(mixture, tmp_path / "expanded.jsonl",
                                    tmp_path / "policy.jsonl", [evaluation], tmp_path / "out",
                                    external=[external])
    finally:
        flash_data.GENERATED.update(saved)
    written = [json.loads(line) for name in ("train", "monitor")
               for line in (tmp_path / "out" / f"{name}.jsonl").open()]
    ids = {r["id"] for r in written}
    assert "m3" not in ids and "m99" not in ids and "m50" not in ids
    assert manifest["dropped"]["bobcat1_mixture:eval_text_overlap"] == 1
    assert manifest["dropped"]["bobcat1_mixture:tool_call"] == 1
    assert manifest["dropped"]["bobcat1_mixture:external_eval_text"] == 1
    assert not any("tool_call" in r["task"] for r in written)
