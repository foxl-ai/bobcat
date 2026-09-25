"""Compile complete split-frozen judgment partitions into bounded Arrow shards.

This is the training-data path, not the small feature-extraction sampler. Every
accepted source row is represented once. Tokenization workers run independently;
ordered shard publication and the final manifest remain sequential.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_readout import PROFILE, GLMCompiler
from bobcat.protocol import RequestLimitError, parse_request
from bobcat.schema import file_hash, json_hash
from bobcat.supervision import target_values

SCHEMA = "bobcat-glm-training-corpus-v1"
SOURCE_SCHEMAS = {"bobcat-public-decisions-v1", "bobcat-expanded-decisions-v1"}
FIT_SPLITS = ("train", "dev_train")
_compiler = None
_length_limit = None


def initialize_worker(model_dir: str, source: dict, length_limit: int | None = None) -> None:
    global _compiler, _length_limit
    _compiler = GLMCompiler(Path(model_dir), source)
    _length_limit = length_limit


def compile_record(row: dict, compiler: GLMCompiler, *, seed: int, group_rows: int) -> dict:
    if (
        row["split"] not in FIT_SPLITS or row["source_split"] != "train"
        or row["language"] not in ("ko", "en")
        or type(group_rows) is not int or group_rows < 1
    ):
        raise ValueError("Training compilation requires intact train-derived components.")
    original_target, original_mean = target_values(row)
    request = copy.deepcopy(row["request"])
    _, original_questions = parse_request(request)
    if (
        len(original_questions) != 1
        or list(original_questions[0].labels) != row["candidate_ids"]
        or row["input_sha256"] != json_hash(row["request"])
    ):
        raise ValueError("Source request, candidate order, and supervision are misaligned.")
    question = next(iter(request["questions"].values()))
    if row["split"] == "train" and question["type"] == "choice":
        items = list(question["criteria"].items())
        random.Random(json_hash([seed, row["id"]])).shuffle(items)
        question["criteria"] = dict(items)
    state, questions = parse_request(request)
    labels = list(questions[0].labels)
    target = labels.index(row["target"]) if original_target >= 0 else original_target
    compiled = compiler.compile(state, questions)
    ids, options = compiled.input_ids[0], compiled.option_token_ids[0]
    if len(options) != len(labels) or len(set(options)) != len(options) or not ids:
        raise ValueError("Compiled data must retain every complete input and candidate.")
    inputs = {"input_ids": ids, "option_token_ids": options}
    origin = row.get("language_origin")
    if origin is None:
        native = row["language"] == "ko" and row["task"].startswith("klue_")
        origin = "native" if native else "original"
    return {
        "id": row["id"], "group_id": row["group_id"],
        "observation_id": row["observation_id"], "split": row["split"],
        "task": row["task"], "family": row["family"], "language": row["language"],
        "language_origin": origin, "kind": row["kind"],
        "supervision": row["supervision"], "candidate_ids": labels,
        **inputs, "input_sha256": json_hash(inputs),
        "source_request_sha256": row["input_sha256"],
        "target_index": target, "score_mean": original_mean,
        "context_weight": 1.0 / group_rows,
        "source_context_weight": row["context_weight"],
        "input_tokens": len(ids), "last_input_position": len(ids) - 1,
    }


def compile_chunk(payload: tuple[list[dict], int, dict[str, int]]) -> dict:
    rows, seed, group_counts = payload
    if _compiler is None:
        raise RuntimeError("Initialize each tokenizer worker independently.")
    accepted, rejected = [], []
    for row in rows:
        reason, tokens = None, None
        try:
            result = compile_record(
                row, _compiler, seed=seed, group_rows=group_counts[row["group_id"]],
            )
            tokens = result["input_tokens"]
            if _length_limit is not None and tokens > _length_limit:
                reason = "complete_prompt_exceeds_curriculum_limit"
        except RequestLimitError as error:
            if _length_limit is None:
                raise
            reason = f"original_compiler_limit: {error}"
        if reason is None:
            accepted.append(result)
        else:
            rejected.append({
                "id": row["id"], "group_id": row["group_id"], "split": row["split"],
                "reason": reason, "full_input_tokens": tokens,
                "source_request_sha256": row["input_sha256"],
            })
    return {"split": rows[0]["split"], "rows": accepted, "rejected": rejected}


def source_inventory(root: Path) -> tuple[dict, dict, dict]:
    """Check all five partitions before any worker can see training text."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") not in SOURCE_SCHEMAS:
        raise ValueError("Use an existing split-frozen judgment dataset.")
    counts, component_split, seen, source_files = Counter(), {}, set(), {}
    for split in ("train", "dev_train", "cal_temperature", "cal_policy", "dev_public"):
        path = root / f"{split}.jsonl"
        item = manifest["files"][path.name]
        if (path.is_symlink() or path.stat().st_size != item["bytes"]
                or file_hash(path) != item["sha256"]):
            raise ValueError("Source partition checksum mismatch.")
        source_files[path.name] = item
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["split"] != split or row["id"] in seen:
                    raise ValueError("Duplicate source identity or wrong partition.")
                seen.add(row["id"])
                group = row["group_id"]
                if group in component_split and component_split[group] != split:
                    raise ValueError("A source component crosses fitting/evaluation partitions.")
                component_split[group] = split
                if split in FIT_SPLITS:
                    if row["source_split"] != "train":
                        raise ValueError("Held-out upstream examples cannot enter training.")
                    counts[group] += 1
    return manifest, dict(counts), source_files


def source_chunks(root: Path, group_counts: dict, *, seed: int, chunk_rows: int):
    for split in FIT_SPLITS:
        chunk = []
        with (root / f"{split}.jsonl").open() as stream:
            for line in stream:
                chunk.append(json.loads(line))
                if len(chunk) == chunk_rows:
                    yield chunk, seed, {r["group_id"]: group_counts[r["group_id"]] for r in chunk}
                    chunk = []
        if chunk:
            yield chunk, seed, {r["group_id"]: group_counts[r["group_id"]] for r in chunk}


def arrow_schema():
    import pyarrow as pa

    fields = [
        (key, pa.string()) for key in (
            "id", "group_id", "observation_id", "split", "task", "family", "language",
            "language_origin", "kind", "supervision", "input_sha256", "source_request_sha256",
        )
    ]
    fields += [
        ("candidate_ids", pa.list_(pa.string())),
        ("input_ids", pa.list_(pa.int32())), ("option_token_ids", pa.list_(pa.int32())),
        ("target_index", pa.int32()), ("score_mean", pa.float64()),
        ("context_weight", pa.float64()), ("source_context_weight", pa.float64()),
        ("input_tokens", pa.int32()), ("last_input_position", pa.int32()),
    ]
    return pa.schema(fields)


def build(root: Path, model_dir: Path, source: dict, out: Path, *,
          workers: int = 4, chunk_rows: int = 512, seed: int = 20260922,
          max_input_tokens: int | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if out.exists() or not 1 <= workers <= 16 or not 16 <= chunk_rows <= 8192:
        raise ValueError("Use a new directory and bounded workers/shards.")
    if max_input_tokens is not None and (
        type(max_input_tokens) is not int or not 256 <= max_input_tokens <= 32768
    ):
        raise ValueError("An explicit curriculum limit must be within 256–32768 tokens.")
    out.mkdir(parents=True)
    staging = out / "staging"
    staging.mkdir()
    started = time.monotonic()
    run = {
        "schema": SCHEMA, "status": "validating_sources",
        "started_at": datetime.now(UTC).isoformat(), "workers": workers,
        "chunk_rows": chunk_rows, "seed": seed, "questions_completed": 0,
        "training_performed": False, "curriculum_max_input_tokens": max_input_tokens,
    }
    atomic_json(out / "progress.json", run)
    try:
        _, group_counts, source_files = source_inventory(root)
        run["status"] = "compiling"
        atomic_json(out / "progress.json", run)
        files, by_task, tokens, candidates = [], Counter(), Counter(), Counter()
        languages, maximum, total = Counter(), 0, 0
        groups, observations, group_weight_sums = {}, {}, Counter()
        staged, rejected = [], []
        compiled_count = 0
        payloads = source_chunks(root, group_counts, seed=seed, chunk_rows=chunk_rows)
        # Pool.imap bounds the number of outstanding chunks; unlike Executor.map
        # on Python 3.12 it does not eagerly serialize the complete corpus.
        with mp.get_context("spawn").Pool(
            workers, initializer=initialize_worker,
            initargs=(str(model_dir), source, max_input_tokens),
        ) as pool:
            for number, chunk in enumerate(pool.imap(compile_chunk, payloads, chunksize=1)):
                rows, split = chunk["rows"], chunk["split"]
                rejected.extend(chunk["rejected"])
                if any(row["split"] != split for row in rows):
                    raise ValueError("A shard crosses fitting partitions.")
                name = f"{split}-{number:06d}.parquet"
                if rows:
                    path = staging / name
                    pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema()), path,
                                   compression="zstd", row_group_size=chunk_rows)
                    if pq.read_table(path).to_pylist() != rows:
                        raise ValueError("Arrow roundtrip changed tokens or supervision.")
                    staged.append((name, split))
                    compiled_count += len(rows)
                run.update(
                    questions_compiled=compiled_count, staged_shards=len(staged),
                    directly_overlength_questions=len(rejected),
                    wall_seconds=time.monotonic() - started,
                )
                atomic_json(out / "progress.json", run)
        bad_groups = {row["group_id"] for row in rejected}
        run["status"] = "publishing_complete_components"
        atomic_json(out / "progress.json", run)
        with (out / "quarantine.jsonl").open("w") as quarantine:
            for row in rejected:
                quarantine.write(json.dumps(row, ensure_ascii=False) + "\n")
            for name, split in staged:
                original_rows = pq.read_table(staging / name).to_pylist()
                rows = [row for row in original_rows if row["group_id"] not in bad_groups]
                for row in original_rows:
                    if row["group_id"] in bad_groups:
                        quarantine.write(json.dumps({
                            "id": row["id"], "group_id": row["group_id"], "split": split,
                            "reason": "component_contains_overlength_question",
                            "full_input_tokens": row["input_tokens"],
                            "source_request_sha256": row["source_request_sha256"],
                        }) + "\n")
                if not rows:
                    continue
                path = out / name
                if len(rows) == len(original_rows):
                    (staging / name).replace(path)
                else:
                    pq.write_table(pa.Table.from_pylist(rows, schema=arrow_schema()), path,
                                   compression="zstd", row_group_size=chunk_rows)
                    if pq.read_table(path).to_pylist() != rows:
                        raise ValueError("Component filtering changed an accepted record.")
                files.append({"path": name, "bytes": path.stat().st_size,
                              "sha256": file_hash(path), "rows": len(rows), "split": split})
                for row in rows:
                    key = f"{split}/{row['language']}"
                    by_task[f"{split}/{row['task']}"] += 1
                    languages[f"{key}/{row['language_origin']}"] += 1
                    tokens[key] += row["input_tokens"]
                    candidates[f"{split}/{len(row['option_token_ids'])}"] += 1
                    groups.setdefault(split, set()).add(row["group_id"])
                    observations.setdefault(split, set()).add(row["observation_id"])
                    group_weight_sums[row["group_id"]] += row["context_weight"]
                    maximum = max(maximum, row["input_tokens"])
                    total += 1
                run.update(questions_completed=total, shards_completed=len(files),
                           wall_seconds=time.monotonic() - started, prompt_tokens=dict(tokens))
                atomic_json(out / "progress.json", run)
        if set(group_weight_sums) != set(group_counts) - bad_groups:
            raise ValueError("Accepted source components disappeared during publication.")
        if any(not math.isclose(v, 1.0, abs_tol=1e-9) for v in group_weight_sums.values()):
            raise ValueError("Complete components must contribute one unit of source weight.")
        result = {
            **run, "status": "completed", "finished_at": datetime.now(UTC).isoformat(),
            "source_manifest_sha256": file_hash(root / "manifest.json"),
            "source_files": source_files, "model_source": source,
            "compiler_profile": PROFILE, "compiler_sha256": file_hash(
                Path(__file__).with_name("glm_readout.py")),
            "generator_sha256": file_hash(Path(__file__)),
            "files": files, "questions": dict(by_task), "prompt_tokens": dict(tokens),
            "candidate_counts": dict(candidates), "language_origins": dict(languages),
            "components": {key: len(value) for key, value in groups.items()},
            "observations": {key: len(value) for key, value in observations.items()},
            "maximum_input_tokens": maximum,
            "weight_rule": "Each complete source component has total row weight one.",
            "model_input_fields": ["input_ids", "option_token_ids"],
            "prompt_language_buckets_are_not_lexical_token_languages": True,
            "processed_training_tokens": 0, "calibration_and_public_validation_included": False,
            "truncation": False, "shortlisting": False,
            "directly_overlength_questions": len(rejected),
            "overlength_components": len(bad_groups),
            "excluded_component_questions": sum(group_counts[group] for group in bad_groups),
            "quarantine_sha256": file_hash(out / "quarantine.jsonl"),
            "staging_is_not_training_data": True,
        }
        result["content_sha256"] = json_hash(result)
        atomic_json(out / "manifest.json", result)
        atomic_json(out / "progress.json", {**run, "status": "completed"})
        return result
    except BaseException as error:
        atomic_json(out / "progress.json", {
            **run, "status": "failed", "error": f"{type(error).__name__}: {error}",
            "wall_seconds": time.monotonic() - started,
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument(
        "--max-input-tokens", type=int,
        help="Explicitly exclude complete overlength components; never truncate a prompt.",
    )
    args = parser.parse_args()
    result = build(args.data, args.model_dir, json.loads(args.source.read_text()),
                   args.out, workers=args.workers, chunk_rows=args.chunk_rows,
                   max_input_tokens=args.max_input_tokens)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("source_files", "model_source", "files")}, indent=2))


if __name__ == "__main__":
    main()
