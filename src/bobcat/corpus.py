"""Pinned real-language documents -> scratch tokenizer -> fixed-length mmap records.

No pretrained tokenizer, remote dataset code, or model-generated labels are used.
Each packed sequence stays inside one document. Padding is not counted as language
tokens. Corpus validation is selected before tokenizer fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from bobcat.schema import file_hash
from bobcat.tokenization import BOS, EOS, SPECIAL, ScratchTokenizer


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def normalized_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n")).strip()


def document_key(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def document_split(key: str) -> str:
    # Content, never row order, determines the partition.
    return "validation" if int(key[:12], 16) % 1000 < 5 else "train"


def validate_sources(manifest: dict) -> None:
    if manifest.get("schema") != "bobcat-corpus-sources-v1":
        raise ValueError("Unknown corpus source format.")
    for source in manifest["sources"]:
        revision = source["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Every corpus revision must be an immutable commit.")
        if source["upstream_split"] != "train" or not source.get("license"):
            raise ValueError("Only attributed upstream training data may enter the corpus.")
        if not source["files"] or source["language"] not in {"ko", "en"}:
            raise ValueError("A source needs files and an explicit supported language.")
        for item in source["files"]:
            if not item["path"].endswith(".parquet") or not item.get("sha256"):
                raise ValueError("Pin parquet contents, not executable dataset scripts.")


def fetch(manifest_path: Path, cache: Path, out: Path, workers: int = 4) -> dict:
    from huggingface_hub import hf_hub_download

    manifest = json.loads(manifest_path.read_text())
    validate_sources(manifest)
    cache.mkdir(parents=True, exist_ok=True)
    items = [(source, item) for source in manifest["sources"] for item in source["files"]]
    results = []

    def download(pair):
        source, item = pair
        start = time.monotonic()
        path = Path(hf_hub_download(
            repo_id=source["repo"], filename=item["path"], revision=source["revision"],
            repo_type="dataset", cache_dir=cache,
        ))
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"Downloaded content failed verification: {item['path']}")
        result = {
            "source_id": source["id"], "repo": source["repo"],
            "revision": source["revision"], "language": source["language"],
            "filename": item["path"], "path": str(path.resolve()),
            "bytes": item["bytes"], "sha256": item["sha256"],
            "seconds": round(time.monotonic() - start, 3),
        }
        print(json.dumps({"event": "source_verified", **result}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(download, items):
            results.append(result)
    result = {
        "schema": "bobcat-corpus-download-v1",
        "source_manifest_sha256": file_hash(manifest_path),
        "files": results,
    }
    atomic_json(out, result)
    return result


def records(downloads: dict, language: str, batch_size: int = 1024) -> Iterator[dict]:
    import pyarrow.parquet as pq

    files = [entry for entry in downloads["files"] if entry["language"] == language]
    # Interleave files, so a bounded prefix includes each selected crawl shard.
    iterators = []
    for entry in files:
        parquet = pq.ParquetFile(entry["path"])
        columns = [name for name in ("text", "url", "id") if name in parquet.schema.names]
        batches = parquet.iter_batches(batch_size=batch_size, columns=columns)

        def rows(stream, provenance):
            for batch in stream:
                for row in batch.to_pylist():
                    text = normalized_text(row.get("text") or "")
                    if not 100 <= len(text) <= 250_000:
                        continue
                    # Restrict controls while preserving ordinary Unicode/line breaks.
                    if any(ord(char) < 32 and char not in "\n\t" for char in text):
                        continue
                    if language == "ko":
                        hangul = sum("\uac00" <= char <= "\ud7a3" for char in text)
                        if hangul / max(1, len(text)) < 0.15:
                            continue
                    key = document_key(text)
                    yield {
                        "text": text, "key": key, "split": document_split(key),
                        "language": language, "source_id": provenance["source_id"],
                    }

        iterators.append(iter(rows(batches, entry)))
    while iterators:
        alive = []
        for iterator in iterators:
            try:
                yield next(iterator)
                alive.append(iterator)
            except StopIteration:
                pass
        iterators = alive


class ExclusionIndex:
    """Exact normalized substring matching, including short Korean inputs.

    Only evaluation INPUTS belong here, never labels. This is not a guarantee
    against paraphrases. Very short inputs (<8 characters) match whole documents
    only; otherwise common words would exclude unrelated documents.
    """

    def __init__(self, path: Path | None = None, width: int = 80):
        if width < 8:
            raise ValueError("Long-input exclusion windows must be at least eight characters.")
        self.width = width
        self.fragments: set[str] = set()
        self.whole_documents: set[str] = set()
        self.digest = None
        if path is not None:
            self.digest = file_hash(path)
            for line in path.read_text().splitlines():
                text = normalized_text(json.loads(line)["text"])
                if len(text) >= width:
                    for i in range(0, len(text) - width + 1, width):
                        self.fragments.add(text[i:i + width])
                    self.fragments.add(text[-width:])
                elif len(text) >= 8:
                    self.fragments.add(text)
                elif text:
                    self.whole_documents.add(document_key(text))
        # Aho-Corasick: one scan over a document, not one scan per excluded input.
        self.edges: list[dict[str, int]] = [{}]
        self.failure = [0]
        self.terminal = [False]
        for fragment in sorted(self.fragments):
            state = 0
            for char in fragment:
                if char not in self.edges[state]:
                    self.edges[state][char] = len(self.edges)
                    self.edges.append({})
                    self.failure.append(0)
                    self.terminal.append(False)
                state = self.edges[state][char]
            self.terminal[state] = True
        queue = deque(self.edges[0].values())
        while queue:
            state = queue.popleft()
            for char, target in self.edges[state].items():
                fallback = self.failure[state]
                while fallback and char not in self.edges[fallback]:
                    fallback = self.failure[fallback]
                self.failure[target] = self.edges[fallback].get(char, 0)
                self.terminal[target] |= self.terminal[self.failure[target]]
                queue.append(target)
        self.policy = {
            "method": "NFC exact substrings; no paraphrase or semantic guarantee",
            "long_input_window_characters": width,
            "minimum_substring_characters": 8,
            "substring_patterns": len(self.fragments),
            "very_short_whole_document_patterns": len(self.whole_documents),
            "inputs_supplied": path is not None,
        }

    def matches(self, text: str) -> bool:
        if not self.fragments and not self.whole_documents:
            return False
        text = normalized_text(text)
        if self.whole_documents and document_key(text) in self.whole_documents:
            return True
        state = 0
        for char in text:
            while state and char not in self.edges[state]:
                state = self.failure[state]
            state = self.edges[state].get(char, 0)
            if self.terminal[state]:
                return True
        return False


def fit_tokenizer(
    downloads_path: Path, output: Path, characters_per_language: int,
    vocab_size: int = 64000, exclusions: Path | None = None,
) -> dict:
    if output.exists():
        raise ValueError("A tokenizer is immutable; choose a new output path.")
    if not 262 <= vocab_size <= 65536:
        raise ValueError("Vocabulary must fit uint16, including special tokens.")
    if characters_per_language < 1:
        raise ValueError("Tokenizer text budget must be positive.")
    downloads = json.loads(downloads_path.read_text())
    excluded = ExclusionIndex(exclusions)
    counts: Counter = Counter()
    seen = set()

    def texts():
        streams = {language: iter(records(downloads, language)) for language in ("ko", "en")}
        while streams:
            for language in list(streams):
                if counts[f"{language}_characters"] >= characters_per_language:
                    del streams[language]
                    continue
                try:
                    row = next(streams[language])
                except StopIteration:
                    del streams[language]
                    continue
                if row["split"] != "train" or row["key"] in seen:
                    continue
                if excluded.matches(row["text"]):
                    counts["excluded_documents"] += 1
                    continue
                seen.add(row["key"])
                counts[f"{language}_characters"] += len(row["text"])
                counts[f"{language}_documents"] += 1
                yield row["text"]

    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True,
    )
    backend.train_from_iterator(texts(), trainer=trainer)
    if any(counts[f"{language}_documents"] < 100 for language in ("ko", "en")):
        raise ValueError("Both languages need substantive training text.")
    output.parent.mkdir(parents=True, exist_ok=True)
    backend.save(str(output))
    result = {
        "origin": "scratch_byte_bpe", "normalization": "NFC", "training_partition": "train",
        "vocab_size": backend.get_vocab_size(), "sha256": file_hash(output),
        "downloads_manifest_sha256": file_hash(downloads_path),
        "source_manifest_sha256": downloads["source_manifest_sha256"],
        "exclusion_inputs_sha256": excluded.digest,
        "exclusion_policy": excluded.policy,
        "encoding_profile": ScratchTokenizer(output).encoding_profile,
        "validation_scope": "exact normalized document hashes; not a near-duplicate audit",
        "characters_per_language_target": characters_per_language, **dict(counts),
    }
    atomic_json(output.with_suffix(".manifest.json"), result)
    return result


def document_sequences(ids: list[int], length: int) -> Iterator[np.ndarray]:
    if length < 4:
        raise ValueError("A sequence must have room for text and document boundaries.")
    # Every chunk is a within-document sample. Remainders are padded, not dropped.
    for begin in range(0, len(ids), length - 2):
        sequence = [BOS, *ids[begin:begin + length - 2], EOS]
        result = np.zeros(length, dtype=np.uint16)
        result[:len(sequence)] = sequence
        yield result


def pack(
    downloads_path: Path, tokenizer_path: Path, out: Path, language: str,
    max_train_tokens: int, max_validation_tokens: int, length: int = 512,
    exclusions: Path | None = None,
) -> dict:
    if language not in {"ko", "en"} or min(max_train_tokens, max_validation_tokens) < 1:
        raise ValueError("Choose a supported language and positive token caps.")
    if length < 4:
        raise ValueError("Sequence length must preserve document boundaries and text.")
    tokenizer = ScratchTokenizer(tokenizer_path)
    if tokenizer.vocab_size > 65536:
        raise ValueError("uint16 token storage cannot hold this tokenizer.")
    downloads = json.loads(downloads_path.read_text())
    excluded = ExclusionIndex(exclusions)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{language}.manifest.json"
    if manifest_path.exists():
        raise ValueError("Corpus output is immutable; choose a new directory.")
    temporary = {split: out / f"{language}-{split}.bin.partial"
                 for split in ("train", "validation")}
    if any(path.exists() for path in temporary.values()):
        raise ValueError("Incomplete packing exists; inspect it before restarting.")
    journal_path = out / f"{language}-documents.sqlite"
    if journal_path.exists() or any(
        (out / f"{language}-{split}.bin").exists() for split in temporary
    ):
        raise ValueError("Corpus files or a document journal already exist.")
    writers = {split: path.open("wb") for split, path in temporary.items()}
    db = sqlite3.connect(journal_path)
    db.execute("CREATE TABLE docs (digest TEXT PRIMARY KEY, split TEXT, source TEXT)")
    counts: Counter = Counter()
    caps = {"train": max_train_tokens, "validation": max_validation_tokens}
    start = time.monotonic()
    try:
        pending = []

        def flush():
            if not pending:
                return
            encoded = tokenizer.backend.encode_batch([row["text"] for row in pending])
            for row, encoding in zip(pending, encoded, strict=True):
                split = row["split"]
                if counts[f"{split}_tokens"] >= caps[split]:
                    db.execute("DELETE FROM docs WHERE digest = ?", (row["key"],))
                    continue
                for sequence in document_sequences(encoding.ids, length):
                    # Finish the selected document. Report the bounded overrun
                    # instead of counting its full text after dropping its tail.
                    writers[split].write(sequence.tobytes())
                    counts[f"{split}_sequences"] += 1
                    counts[f"{split}_tokens"] += int(np.count_nonzero(sequence))
                    counts[f"{split}_text_tokens"] += int(np.count_nonzero(sequence)) - 2
                    counts[f"{split}_slots"] += length
                counts[f"{split}_documents"] += 1
                counts[f"{split}_characters"] += len(row["text"])
                counts[f"{split}_whitespace_words"] += len(row["text"].split())
                counts[f"{split}_unknown_tokens"] += encoding.ids.count(1)
            pending.clear()
            db.commit()

        for row in records(downloads, language):
            split = row["split"]
            if all(counts[f"{name}_tokens"] >= cap for name, cap in caps.items()):
                break
            if counts[f"{split}_tokens"] >= caps[split]:
                continue
            if excluded.matches(row["text"]):
                counts["excluded_documents"] += 1
                continue
            try:
                db.execute("INSERT INTO docs VALUES (?, ?, ?)",
                           (row["key"], split, row["source_id"]))
            except sqlite3.IntegrityError:
                counts["duplicate_documents"] += 1
                continue
            pending.append(row)
            if len(pending) >= 256:
                flush()
                if (counts["train_documents"] + counts["validation_documents"]) % 10240 < 256:
                    print(json.dumps({
                        "event": "pack_progress", "language": language,
                        "seconds": round(time.monotonic() - start, 1), **dict(counts),
                    }), flush=True)
        flush()
    finally:
        for writer in writers.values():
            writer.flush()
            os.fsync(writer.fileno())
            writer.close()
        db.close()
    files = {}
    if any(counts[f"{split}_sequences"] < 2 for split in temporary):
        raise ValueError(f"No useful {language} train/validation partitions were produced.")
    for split, path in temporary.items():
        destination = out / f"{language}-{split}.bin"
        path.replace(destination)
        files[split] = {"path": destination.name, "sha256": file_hash(destination),
                        "bytes": destination.stat().st_size}
    result = {
        "schema": "bobcat-mmap-corpus-v1", "language": language, "dtype": "uint16",
        "sequence_length": length, "tokenizer_sha256": tokenizer.digest,
        "tokenizer_encoding_profile": tokenizer.encoding_profile,
        "downloads_manifest_sha256": file_hash(downloads_path),
        "source_manifest_sha256": downloads["source_manifest_sha256"],
        "exclusion_inputs_sha256": excluded.digest, "files": files,
        "exclusion_policy": excluded.policy,
        "split_rule": "sha256(NFC(text)) first 12 hex digits mod 1000 < 5 => validation",
        "validation_scope": (
            "Exact document duplicates share a split; same-URL revisions, near duplicates, "
            "translations and historical benchmark exposure are not excluded by this rule."
        ),
        "cross_document_attention": False, "boundary_tokens_counted": True,
        "token_cap_policy": "finish the last selected document, maximum 250000 source characters",
        "token_caps_requested": caps,
        "token_cap_overrun": {
            split: max(0, counts[f"{split}_tokens"] - cap) for split, cap in caps.items()
        },
        "seconds": round(time.monotonic() - start, 3), **dict(counts),
    }
    for split in ("train", "validation"):
        words = counts[f"{split}_whitespace_words"]
        result[f"{split}_tokens_per_whitespace_word"] = (
            counts[f"{split}_text_tokens"] / words if words else None
        )
    atomic_json(manifest_path, result)
    return result


class MMapCorpus:
    def __init__(self, root: Path, language: str, split: str, tokenizer_sha256: str):
        self.manifest_path = root / f"{language}.manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if (self.manifest.get("schema") != "bobcat-mmap-corpus-v1"
                or self.manifest.get("dtype") != "uint16"
                or self.manifest.get("language") != language
                or split not in {"train", "validation"}):
            raise ValueError("Unsupported corpus schema, language, dtype or split.")
        if self.manifest["tokenizer_sha256"] != tokenizer_sha256:
            raise ValueError("Corpus and model tokenizers differ.")
        length = self.manifest["sequence_length"]
        if type(length) is not int or length < 4:
            raise ValueError("Invalid corpus sequence length.")
        info = self.manifest["files"][split]
        path = root / info["path"]
        if path.stat().st_size != info["bytes"]:
            raise ValueError("Corpus is incomplete.")
        if not info["bytes"] or info["bytes"] % (length * 2):
            raise ValueError("Corpus bytes do not contain complete uint16 sequences.")
        if info["bytes"] // (length * 2) != self.manifest[f"{split}_sequences"]:
            raise ValueError("Corpus sequence count differs from its manifest.")
        self.array = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, length)
        self.language = language

    def __len__(self):
        return len(self.array)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        indexes = rng.integers(0, len(self), size=size)
        return np.asarray(self.array[indexes], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire = sub.add_parser("fetch")
    acquire.add_argument("--sources", type=Path, required=True)
    acquire.add_argument("--cache", type=Path, required=True)
    acquire.add_argument("--out", type=Path, required=True)
    acquire.add_argument("--workers", type=int, default=4)
    for name in ("tokenizer", "pack"):
        child = sub.add_parser(name)
        child.add_argument("--downloads", type=Path, required=True)
        child.add_argument("--tokenizer", type=Path, required=True)
        child.add_argument("--exclusions", type=Path)
        if name == "tokenizer":
            child.add_argument("--characters-per-language", type=int, default=100_000_000)
            child.add_argument("--vocab-size", type=int, default=64000)
        else:
            child.add_argument("--out", type=Path, required=True)
            child.add_argument("--language", choices=["ko", "en"], required=True)
            child.add_argument("--max-train-tokens", type=int, default=800_000_000)
            child.add_argument("--max-validation-tokens", type=int, default=1_000_000)
            child.add_argument("--length", type=int, default=512)
    args = parser.parse_args()
    if args.command == "fetch":
        result = fetch(args.sources, args.cache, args.out, args.workers)
    elif args.command == "tokenizer":
        result = fit_tokenizer(args.downloads, args.tokenizer, args.characters_per_language,
                               args.vocab_size, args.exclusions)
    else:
        result = pack(args.downloads, args.tokenizer, args.out, args.language,
                      args.max_train_tokens, args.max_validation_tokens,
                      args.length, args.exclusions)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
