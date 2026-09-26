"""Flash long-context continuation data (bobcat.flash_longctx) and the length-threshold
selection (scripts/length_route_select.py): teacher mapping, evaluation-text exclusion,
replay sampling, routing simulation and the pre-registered choice."""

import json

from bobcat import flash_longctx as fl
from bobcat import student_data_v11
from scripts import length_route_select as sel


def corpus_row(i, source, task="product_search", keys=(), state=None):
    return {"id": f"row{i}", "group_id": f"g{i}", "task": task, "language": "ko",
            "kind": "choice", "supervision": "hard_label", "target": "a",
            "candidate_ids": ["a", "b"], "flash_source": source,
            "mixture_source": "product" if source == fl.MIXTURE else None,
            "text_keys": list(keys),
            "request": {"model": "bobcat-latest", "state": state or {"doc": f"text {i}"},
                        "questions": {"q": {"type": "choice", "instructions": "pick",
                                            "criteria": {"a": "A", "b": "B"}}}}}


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def test_replay_takes_half_from_the_mixture_and_is_seeded():
    rows = [corpus_row(i, fl.MIXTURE if i % 3 == 0 else "generated") for i in range(60)]
    first = fl.replay_rows(rows, 10, 7, set())
    assert first == fl.replay_rows(rows, 10, 7, set())
    assert sum(r["flash_source"] == fl.MIXTURE for r in first) == 5
    assert len({r["id"] for r in first}) == 10
    assert fl.replay_rows(rows, 10, 8, set()) != first


def test_build_leaves_out_evaluation_text_and_maps_teachers(tmp_path, monkeypatch):
    rows = [corpus_row(i, fl.MIXTURE, keys=[f"k{i}"]) for i in range(8)]
    rows += [corpus_row(100 + i, "generated", task="tictactoe") for i in range(8)]
    rows[0]["text_keys"] = ["EVAL"]            # meets an evaluation row: never used
    corpus = write(tmp_path / "train.jsonl", rows)
    evaluation = write(tmp_path / "dev.jsonl", [{"id": "d", "text_keys": ["EVAL", "PAD"]}])

    def fake_long_copies(mixture, seed, model_dir, identifiers, workers, pool, plan, levels,
                         passes):
        assert all(r["id"] != "row0" for r in mixture + pool)
        copies = []
        for row in mixture[:4]:
            state = {"배경 자료": "padding " + row["id"], **row["request"]["state"]}
            texts = ["PADTEXT"] if row["id"] == "row2" else []
            copy = student_data_v11.derived(row, state, kind="long", texts=texts, meta={
                "level": 4096, "tokens": 4000, "language": "ko", "layout": "before",
                "keys": ["배경 자료", "첨부 문서"], "passages": 1})
            if row["id"] == "row2":
                copy["text_keys"].append("PAD")  # its padding meets evaluation text
            copies.append(copy)
        return copies

    monkeypatch.setattr(student_data_v11, "long_copies", fake_long_copies)
    out = tmp_path / "data"
    manifest = fl.build(corpus, tmp_path, tmp_path / "ids.json", out, seed=1, workers=1,
                        replay=6, forbidden=(evaluation,))
    long_rows = [json.loads(line) for line in (out / "long.jsonl").open()]
    assert {r["derived"]["from"] for r in long_rows} == {"row1", "row3", "row4"}
    assert all(r["flash_source"] == "long_context" for r in long_rows)
    assert manifest["forbidden"]["corpus_rows_left_out"] == 1
    assert manifest["forbidden"]["padded_copies_left_out"] == 1
    replay = [json.loads(line) for line in (out / "replay.jsonl").open()]
    assert len(replay) == 6 and all(r["id"] != "row0" for r in replay)
    sources = [json.loads(line) for line in (out / "sources.jsonl").open()]
    assert sorted(r["id"] for r in sources) == ["row1", "row3", "row4"]
    audit = fl.audit([out / "long.jsonl", out / "replay.jsonl"], [evaluation])
    assert audit["overlapping_rows"] == 0
    # The teacher on each unpadded source question answers its padded copies.
    teacher = write(tmp_path / "teacher.jsonl",
                    [{"id": r["id"], "logits": [float(i), 0.0]} for i, r in
                     enumerate(sources + replay)])
    mapped = fl.teacher_map(out / "long.jsonl", [teacher], tmp_path / "mapped.jsonl")
    table = {r["id"]: r for r in map(json.loads, (tmp_path / "mapped.jsonl").open())}
    source_logits = {r["id"]: r["logits"] for r in map(json.loads, teacher.open())}
    for row in long_rows:
        assert table[row["id"]]["logits"] == source_logits[row["derived"]["from"]]
    assert mapped["long_rows_without_teacher"] == 0
    assert all(r["id"] in table for r in replay)


def request(state, candidates=2):
    return {"model": "m", "state": state, "questions": {"q": {
        "type": "choice", "instructions": "pick",
        "criteria": {f"c{i}": f"option {i}" for i in range(candidates)}}}}


def dev_row(i, candidates=2, task="product_search"):
    return {"id": f"d{i}", "task": task, "group_id": f"g{i}", "split": "dev",
            "candidate_ids": [f"c{j}" for j in range(candidates)], "target": "c0",
            "request": request({"doc": f"text {i}"}, candidates)}


def test_routing_simulation_follows_the_server_rules_in_order():
    rows = [dev_row(0), dev_row(1), dev_row(2), dev_row(3, candidates=70)]
    lengths = {"d0": 500, "d1": 5000, "d2": 500, "d3": 500}
    flash = {"d0": [5.0, 0.0], "d1": [5.0, 0.0], "d2": [0.1, 0.0], "d3": [5.0] + [0.0] * 69}
    bobcat = {"d0": [0.0, 5.0], "d1": [0.0, 5.0], "d2": [5.0, 0.0], "d3": [0.0] + [5.0] * 69}
    level = sel.Level("x:0", rows, lengths, flash, bobcat, "accuracy")
    assert [level.route(r, 4096) for r in rows] == [
        ("flash", "in_range"), ("bobcat", "length"), ("bobcat", "low_confidence"),
        ("bobcat", "candidates")]
    assert level.route(rows[1], None) == ("flash", "in_range")   # no length rule
    table, share, reasons = level.outcome(4096)
    assert [table[r["id"]][0] for r in rows] == [True, False, True, False]
    assert share == 0.25 and reasons["length"] == 1
    del bobcat["d1"]                                              # Bobcat cannot compile it
    assert level.route(rows[1], 4096) == ("flash", "bobcat_limit")


def test_the_largest_qualifying_threshold_is_chosen():
    rows = [dev_row(i) for i in range(4)]
    dev = {r["id"]: r for r in rows}
    # Long rows (d2, d3) are right only on Bobcat; short rows right on both.
    lengths = {"d0": 100, "d1": 100, "d2": 3000, "d3": 7000}
    flash = {r["id"]: ([5.0, 0.0] if r["id"] in ("d0", "d1") else [0.0, 5.0]) for r in rows}
    bobcat = {r["id"]: [5.0, 0.0] for r in rows}
    level = sel.Level("x:0", rows, lengths, flash, bobcat, "accuracy")
    table = sel.evaluate([level], [1024, 2048, 4096, 8192, None], 1.0, dev)
    assert [table[k]["qualifies"] for k in ("1024", "2048", "4096", "8192", "none")] == [
        True, True, False, False, False]
    assert sel.choose(table, [1024, 2048, 4096, 8192, None])["T"] == 2048
    none = sel.evaluate([level], [8192, None], 1.0, dev)
    assert sel.choose(none, [8192, None]) == {
        "T": 8192, "qualified": False,
        "note": "no candidate met the constraint; the smallest candidate is used"}


def test_a_serving_entry_leaves_the_frozen_fields_alone(tmp_path):
    import pytest

    from scripts import add_serving_artifact as asa

    manifest = {"schema": "x", "status": "frozen", "weights": {"sha256": "a"},
                "serving_artifacts": {"nvfp4": {"x": 1}}}
    path = tmp_path / "m.json"
    path.write_text(asa.dumps(manifest))
    frozen = asa.frozen_sha256(manifest)
    result = asa.add(path, "routing_length_rule", {"T": 4096}, frozen)
    after = json.loads(path.read_text())
    assert after["serving_artifacts"] == {"nvfp4": {"x": 1}, "routing_length_rule": {"T": 4096}}
    assert result["frozen_sha256"] == frozen and after["weights"] == manifest["weights"]
    with pytest.raises(ValueError):
        asa.add(path, "routing_length_rule", {"T": 1}, frozen)   # no silent overwrite
    with pytest.raises(ValueError):
        asa.add(path, "other", {}, "0" * 64)                    # frozen fields must match


def test_the_released_flash_manifest_keeps_its_frozen_sha256():
    from pathlib import Path

    import pytest

    from scripts import add_serving_artifact as asa

    if not Path("release/bobcat-flash-1.1-final-opening.json").exists():
        pytest.skip("the internal release manifests are not in the public tree")
    manifest = json.loads(Path("release/bobcat-flash-1.1-manifest.json").read_text())
    opening = json.loads(Path("release/bobcat-flash-1.1-final-opening.json").read_text())
    assert asa.frozen_sha256(manifest) == opening["flash_manifest"]["sha256"]
