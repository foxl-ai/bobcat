import copy
import io
import json
from pathlib import Path

import pytest

from bobcat.klue import convert, download, examples_for, request_for, validate_sources
from bobcat.protocol import parse_request


def source():
    path = Path(__file__).parents[1] / "configs/korean-decisions-klue-v1.json"
    return json.loads(path.read_text())


def nli(identity, premise, hypothesis, label=0, split="train"):
    return {
        "identity": identity, "task": "nli", "upstream_split": split,
        "raw": {"guid": identity, "premise": premise, "hypothesis": hypothesis, "label": label,
                "source": "ANNOTATOR-ONLY"},
    }


@pytest.mark.parametrize("label,expected", [
    (0, ["함의", "yes", "no"]), (1, ["중립", "no", "no"]), (2, ["모순", "no", "yes"]),
])
def test_gold_mapping_and_oracle_fields_do_not_enter_requests(label, expected):
    row = nli("PRIVATE-ID", "지수가 책을 읽습니다.", "지수가 독서를 합니다.", label)
    examples = examples_for(row, "PRIVATE-GROUP", "train", source())
    assert [e.target for e in examples] == expected
    changed = copy.deepcopy(row)
    changed["raw"]["label"] = (label + 1) % 3
    changed["identity"] = "OTHER-PRIVATE-ID"
    changed["raw"]["source"] = "OTHER-ANNOTATOR"
    other = examples_for(changed, "OTHER-GROUP", "dev_public", source())
    assert [request_for(e) for e in examples] == [request_for(e) for e in other]
    for example in examples:
        request = request_for(example)
        assert "PRIVATE" not in json.dumps(request)
        assert "ANNOTATOR" not in json.dumps(request)
        _, questions = parse_request(request)
        assert set(questions[0].labels) == {c.id for c in example.choices}
        assert example.target in questions[0].labels


def test_transitive_shared_sentences_keep_public_validation_out_of_training():
    rows = [
        nli("train-a", "공유 문장 A", "공유 문장 B"),
        nli("train-b", "공유 문장 B", "공유 문장 C"),
        nli("validation-c", "공유 문장 C", "공유 문장 D", split="validation"),
        nli("separate", "독립 전제", "독립 가설"),
    ]
    examples, audit, _ = convert(rows, source())
    assert audit["removal_counts"] == {"shares_component_with_public_validation": 2}
    public = [e for e in examples if e.split == "dev_public"]
    assert len(public) == 3
    assert len({e.group_id for e in public}) == 1
    assert len({e.metadata["independent_observation"] for e in public}) == 1
    assert not any(e.metadata["source_id"] in {"train-a", "train-b"} for e in examples)
    reverse, _, _ = convert(list(reversed(copy.deepcopy(rows))), source())
    assert sorted((e.id, e.group_id, e.split) for e in examples) == sorted(
        (e.id, e.group_id, e.split) for e in reverse
    )


def test_url_grouping_conflicts_and_duplicate_training_input():
    rows = [
        {"task": "ynat", "upstream_split": split, "raw": {
            "guid": f"title-{i}", "title": title, "url": "https://example.com/article?id=7",
            "label": 0,
        }}
        for i, (split, title) in enumerate([
            ("train", "원래 기사 제목"), ("validation", "수정한 기사 제목"),
        ])
    ]
    rows.extend([
        nli("conflict-a", "충돌하는 전제", "충돌하는 가설", 0),
        nli("conflict-b", "충돌하는 전제", "충돌하는 가설", 2),
        nli("duplicate-a", "독립 전제", "독립 가설"),
        nli("duplicate-b", "독립 전제", "독립 가설"),
    ])
    examples, audit, _ = convert(rows, source())
    assert audit["removal_counts"] == {
        "shares_component_with_public_validation": 1,
        "conflicting_annotations": 2, "duplicate_training_input": 1,
    }
    assert len(examples) == 4
    assert len({e.group_id for e in examples}) == 2
    assert audit["group_overlap_count"] == 0


def test_download_failure_preserves_evidence_and_cannot_publish_complete(tmp_path, monkeypatch):
    config = tmp_path / "source.json"
    config.write_text(json.dumps(source()))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(RuntimeError, match="checksum"):
        download(config, tmp_path / "download")
    assert (tmp_path / "download/failed.json").exists()
    assert not (tmp_path / "download/downloads.json").exists()
    assert len(list((tmp_path / "download").rglob("*.part"))) == 4


def test_reject_test_partition_and_incompatible_mapping():
    config = source()
    config["files"][0]["split"] = "test"
    with pytest.raises(ValueError, match="test"):
        validate_sources(config)
    config = source()
    config["label_names"]["nli"][1] = "contradiction"
    with pytest.raises(ValueError, match="mapping"):
        validate_sources(config)
