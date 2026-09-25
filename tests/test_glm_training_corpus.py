import copy
import json

import pytest
import test_glm_readout

from bobcat.glm_training_corpus import compile_record, source_inventory
from bobcat.schema import file_hash, json_hash

compiler = test_glm_readout.compiler


def record(split="train", kind="choice"):
    q = {"type": kind, "instructions": "주어진 기준으로 판단하라.",
         "criteria": {"승인": "규정 충족", "거절": "규정 미충족"}}
    labels, target, mean = ["승인", "거절"], "거절", None
    if kind == "score":
        q["criteria"] = ["없음", "일부", "많음"]
        labels, target, mean = ["0", "1", "2"], None, 1.25
    request = {"model": "bobcat", "state": "고객이 반품을 신청했습니다.",
               "questions": {"PRIVATE-QUESTION-ID": q}}
    return {
        "id": "PRIVATE-EXAMPLE-ID", "group_id": "PRIVATE-GROUP-ID",
        "observation_id": "PRIVATE-OBSERVATION-ID", "split": split, "source_split": "train",
        "language": "ko", "task": "klue_fixture", "family": "fixture",
        "kind": "choice" if kind == "choice" else "ordinal",
        "supervision": "hard_label" if target is not None else "score_mean",
        "request": request, "input_sha256": json_hash(request), "candidate_ids": labels,
        "target": target, "score_target": mean, "context_weight": 1.0,
    }


def test_all_gold_and_identifiers_stay_out_of_compiled_inputs(compiler):
    compiler = compiler[0]
    row = record()
    result = compile_record(row, compiler, seed=42, group_rows=3)
    assert "PRIVATE" not in compiler.host_tokenizer.decode(result["input_ids"])
    assert result["candidate_ids"][result["target_index"]] == "거절"
    assert result["context_weight"] == 1 / 3
    changed = copy.deepcopy(row)
    changed["id"], changed["target"] = "another-id", "승인"
    changed["split"] = "dev_train"
    original_dev = compile_record({**row, "split": "dev_train"}, compiler, seed=42, group_rows=3)
    changed_dev = compile_record(changed, compiler, seed=42, group_rows=3)
    assert changed_dev["input_ids"] == original_dev["input_ids"]
    assert changed_dev["target_index"] != original_dev["target_index"]


def test_full_data_means_are_not_rounded_or_candidate_shuffled(compiler):
    compiler = compiler[0]
    result = compile_record(record(kind="score"), compiler, seed=42, group_rows=1)
    assert result["target_index"] == -100
    assert result["score_mean"] == 1.25
    assert result["candidate_ids"] == ["0", "1", "2"]
    assert result["last_input_position"] == len(result["input_ids"]) - 1


def test_source_validation_precedes_any_training_compilation(tmp_path):
    files = {}
    for split in ("train", "dev_train", "cal_temperature", "cal_policy", "dev_public"):
        row = record(split)
        row["id"], row["group_id"] = split, split
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    manifest = {"schema": "bobcat-public-decisions-v1", "files": files}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, groups, _ = source_inventory(tmp_path)
    assert groups == {"train": 1, "dev_train": 1}
    path = tmp_path / "dev_public.jsonl"
    changed = json.loads(path.read_text())
    changed["group_id"] = "train"
    path.write_text(json.dumps(changed) + "\n")
    files[path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="crosses"):
        source_inventory(tmp_path)


def test_changed_candidate_alignment_and_upstream_test_are_rejected(compiler):
    compiler = compiler[0]
    row = record()
    row["candidate_ids"].reverse()
    with pytest.raises(ValueError, match="misaligned"):
        compile_record(row, compiler, seed=42, group_rows=1)
    row = record()
    row["source_split"] = "test"
    with pytest.raises(ValueError, match="train-derived"):
        compile_record(row, compiler, seed=42, group_rows=1)


def test_curriculum_excludes_entire_component_and_preserves_other_complete_inputs(
    compiler, tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    import pyarrow.parquet as pq

    import bobcat.glm_training_corpus as corpus

    model, model_dir, source = compiler
    rows = [record() for _ in range(3)]
    for index, row in enumerate(rows):
        row["id"] = str(index)
        row["group_id"] = "shared" if index < 2 else "independent"
    rows[1]["request"] = copy.deepcopy(rows[1]["request"])
    rows[1]["request"]["state"] = "Long complete source " * 1000
    rows[1]["input_sha256"] = json_hash(rows[1]["request"])
    limit = max(256, compile_record(rows[0], model, seed=42, group_rows=2)["input_tokens"] + 10)
    data = tmp_path / "data"
    data.mkdir()
    files = {}
    for split in ("train", "dev_train", "cal_temperature", "cal_policy", "dev_public"):
        path = data / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows)
                        if split == "train" else "")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    (data / "manifest.json").write_text(json.dumps({
        "schema": "bobcat-public-decisions-v1", "files": files,
    }))

    class InlinePool:
        def __init__(self, _workers, *, initializer, initargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def imap(self, function, values, chunksize):
            return map(function, values)

    monkeypatch.setattr(corpus.mp, "get_context", lambda _: SimpleNamespace(Pool=InlinePool))
    out = tmp_path / "compiled"
    result = corpus.build(data, model_dir, source, out, workers=1, chunk_rows=16,
                          seed=42, max_input_tokens=limit)
    assert result["overlength_components"] == 1
    assert result["directly_overlength_questions"] == 1
    assert result["excluded_component_questions"] == 2
    assert result["questions_completed"] == 1 and result["truncation"] is False
    kept = pq.read_table(out / result["files"][0]["path"]).to_pylist()[0]
    expected = compile_record(rows[2], model, seed=42, group_rows=1)
    assert kept == expected and kept["context_weight"] == 1
