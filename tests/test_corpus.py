import json
import sqlite3
import unicodedata

import numpy as np
import pytest

from bobcat.corpus import (
    ExclusionIndex,
    MMapCorpus,
    document_key,
    document_sequences,
    document_split,
    fit_tokenizer,
    pack,
    records,
)
from bobcat.schema import file_hash
from bobcat.tokenization import BOS, EOS, SPECIAL, ScratchTokenizer

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture(scope="module")
def real_format_corpus(tmp_path_factory):
    """Synthetic fixtures in the actual Parquet format, not quality training data."""
    root = tmp_path_factory.mktemp("parquet")
    files = []
    for language in ("ko", "en"):
        texts = [
            (f"문서 {i}입니다. 배송과 반품 조건을 확인합니다. " * 8
             if language == "ko" else
             f"Document {i} discusses weather, plants and deliveries. " * 8)
            + "Literal source text includes [PAD] and [MASK]."
            for i in range(1400)
        ]
        path = root / f"{language}.parquet"
        pq.write_table(pa.table({"text": texts}), path, row_group_size=128)
        files.append({
            "language": language, "source_id": f"unit-fixture-{language}",
            "path": str(path), "sha256": file_hash(path),
        })
    downloads = root / "downloads.json"
    downloads.write_text(json.dumps({
        "source_manifest_sha256": "0" * 64, "files": files,
    }))
    tokenizer_path = root / "tokenizer.json"
    manifest = fit_tokenizer(downloads, tokenizer_path, 80_000, vocab_size=512)
    return root, downloads, tokenizer_path, manifest


def test_short_korean_exclusions_normalization_and_overlapping_patterns(tmp_path):
    path = tmp_path / "inputs.jsonl"
    patterns = ["배송이 아직 도착하지 않았습니다", "abcdeabcde", "deabcdefgh", "반품"]
    path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n"
                            for t in patterns))
    index = ExclusionIndex(path)
    assert index.matches("고객 문의: 배송이 아직 도착하지 않았습니다. 확인해주세요.")
    assert index.matches(unicodedata.normalize("NFD", patterns[0]))
    assert index.matches("prefix abcdeabcdefgh suffix")
    assert index.matches("반품")
    assert not index.matches("반품 절차를 일반적으로 설명하는 다른 문서입니다.")
    assert not index.matches("다른 요청입니다.")
    assert index.policy["very_short_whole_document_patterns"] == 1


def test_document_partition_is_stable_for_unicode_equivalent_text():
    text = "한글 문서\n검증 조건"
    decomposed = unicodedata.normalize("NFD", text).replace("\n", "\r\n")
    assert document_key(text) == document_key(decomposed)
    assert document_split(document_key(text)) == document_split(document_key(decomposed))


def test_source_special_token_spellings_are_data(real_format_corpus):
    tokenizer = ScratchTokenizer(real_format_corpus[2])
    text = " ".join(SPECIAL)
    encoded = tokenizer.encode(text)
    assert min(encoded) >= len(SPECIAL)
    assert tokenizer.backend.decode(encoded) == text
    # The batch encoder used by packing must apply the same boundary policy.
    assert tokenizer.backend.encode_batch([text])[0].ids == encoded


def test_sequence_chunks_preserve_every_source_token_and_boundaries():
    source = list(range(6, 49))
    sequences = list(document_sequences(source, 16))
    restored = []
    for sequence in sequences:
        actual = sequence[sequence != 0].tolist()
        assert actual[0] == BOS and actual[-1] == EOS
        restored.extend(actual[1:-1])
    assert restored == source
    assert all(sequence.dtype == np.uint16 for sequence in sequences)


def test_pack_finishes_document_and_accounts_for_its_full_text(real_format_corpus, tmp_path):
    _, downloads_path, tokenizer_path, tokenizer_manifest = real_format_corpus
    tokenizer = ScratchTokenizer(tokenizer_path)
    assert tokenizer_manifest["training_partition"] == "train"
    assert tokenizer_manifest["ko_documents"] >= 100
    assert tokenizer_manifest["en_documents"] >= 100
    downloads = json.loads(downloads_path.read_text())
    first = {}
    for record in records(downloads, "ko"):
        first.setdefault(record["split"], record)
        if len(first) == 2:
            break
    result = pack(downloads_path, tokenizer_path, tmp_path, "ko", 1, 1, length=16)
    with sqlite3.connect(tmp_path / "ko-documents.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 2
    for split in ("train", "validation"):
        dataset = MMapCorpus(tmp_path, "ko", split, tokenizer.digest)
        restored = []
        for sequence in dataset.array:
            actual = sequence[sequence != 0].tolist()
            assert actual[0] == BOS and actual[-1] == EOS
            restored.extend(actual[1:-1])
        assert restored == tokenizer.encode(first[split]["text"])
        assert result[f"{split}_documents"] == 1
        assert result[f"{split}_characters"] == len(first[split]["text"])
        assert result["token_cap_overrun"][split] == result[f"{split}_tokens"] - 1
    assert first["train"]["key"] != first["validation"]["key"]


def test_pack_outputs_are_immutable(real_format_corpus, tmp_path):
    _, downloads, tokenizer, _ = real_format_corpus
    pack(downloads, tokenizer, tmp_path, "en", 1, 1, length=16)
    with pytest.raises(ValueError, match="immutable"):
        pack(downloads, tokenizer, tmp_path, "en", 1, 1, length=16)


def test_mmap_rejects_wrong_tokenizer_and_corrupt_size(real_format_corpus, tmp_path):
    _, downloads, tokenizer, _ = real_format_corpus
    pack(downloads, tokenizer, tmp_path, "en", 1, 1, length=16)
    with pytest.raises(ValueError, match="tokenizers differ"):
        MMapCorpus(tmp_path, "en", "train", "f" * 64)
    data = tmp_path / "en-train.bin"
    with data.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="incomplete"):
        MMapCorpus(tmp_path, "en", "train", file_hash(tokenizer))
