import csv
import json

from bobcat.expanded_decisions import (
    ATTRIBUTES,
    Components,
    nli_rows,
    rating_index,
    record,
    sentiment_rows,
)
from bobcat.public_decisions import SPLITS
from bobcat.schema import json_hash


def test_transitive_links_quarantine_conflicting_frozen_partitions():
    c = Components()
    a = c.add(["a"], 1 << SPLITS.index("train"))
    b = c.add(["b"], 1 << SPLITS.index("cal_temperature"))
    bridge = c.add(["a", "b"])
    for index in (a, b, bridge):
        assert c.assignment(index, "train")[2] == "conflicting_frozen_partitions"


def test_parallel_languages_follow_holdout_and_do_not_become_training():
    c = Components()
    ko = c.add(["ko-text", "massive-id:1"])
    en = c.add(["en-text", "massive-id:1"], 1 << SPLITS.index("dev_public"))
    assert c.assignment(ko, "train")[0] is None
    assert c.assignment(en, "validation")[0] == "dev_public"
    assert c.assignment(ko, "train")[1] == c.assignment(en, "validation")[1]


def test_record_keeps_annotator_scores_and_reasons_out_of_request():
    row = record(
        "ratings", "train", 1, state={"prompt": "Explain a receipt.", "response": "Example."},
        question={"type": "score", "instructions": "Judge clarity.",
                  "criteria": ["unclear", "partially clear", "clear"]},
        language="en", language_origin="original", family="judgment",
        source={"license": "fixture", "reasoning": "PRIVATE-ANNOTATOR-GOLD"},
        texts=["Explain a receipt."], mean=4 / 3, votes=[1, 1, 2],
    )
    assert row["score_target"] == 4 / 3
    assert row["source"]["individual_ratings"] == [1, 1, 2]
    assert "PRIVATE" not in json.dumps(row["request"])
    assert row["target"] is None


def test_duplicate_rating_rows_do_not_invent_votes_or_choose_ambiguous_annotations():
    from collections import Counter

    original = {"prompt": "prompt", "response": "response",
                **{attribute: [1, 2, 3] for attribute in ATTRIBUTES}}
    key = json_hash(["prompt", "response"])
    counts = Counter()
    ratings, ambiguous = rating_index([original, original], counts)
    assert ratings[key]["helpfulness"] == [1, 2, 3]
    assert not ambiguous
    assert counts["helpsteer2_exact_duplicate_rating_rows"] == 1
    ratings, ambiguous = rating_index(
        [original, {**original, "helpfulness": [1, 2, 4]}, original], counts,
    )
    assert key not in ratings and ambiguous == {key}


def test_korean_source_adapters_do_not_turn_translations_into_native(tmp_path):
    from collections import Counter

    sources = {}
    for name in ("multinli.train.ko.tsv", "snli_1.0_train.ko.tsv",
                 "xnli.dev.ko.tsv", "xnli.test.ko.tsv"):
        path = "kornlu--KorNLI--" + name
        sources[path] = {"license": "fixture"}
        with (tmp_path / path).open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["sentence1", "sentence2", "gold_label"])
            writer.writerow(["사람이 걷는다.", "누군가 움직인다.", "entailment"])
    rows = list(nli_rows(tmp_path, sources, Counter()))
    assert [r["language_origin"] for r in rows] == [
        "machine_translation", "machine_translation", "human_translation", "human_translation",
    ]
    assert all(r["target"] == "함의" for r in rows)
    for name in ("ratings_train.txt", "ratings_test.txt"):
        path = "nsmc--" + name
        sources[path] = {"license": "fixture"}
        (tmp_path / path).write_text("id\tdocument\tlabel\n123\t재밌어요!\t1\n124\t\t0\n")
    counters = Counter()
    rows = list(sentiment_rows(tmp_path, sources, counters))
    assert counters["nsmc_invalid"] == 2
    assert all(r["language_origin"] == "native" for r in rows)
    assert all("123" not in json.dumps(r["request"]) for r in rows)
