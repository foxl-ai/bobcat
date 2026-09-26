"""Teacher logits for the Bobcat Flash corpus: Bobcat 1 (merged, vLLM) at the first answer
position of every compiled row (2026-09-26).

Runs in the vLLM venv, one process per GPU (`--shard i --shards n` over the rows by a stable
hash of the row ID). Each row's raw identifier logits are read with `logprob_token_ids`
(`logprobs_mode="raw_logits"`) in fixed 64-wide chunks, exactly as `bobcat.api_server` reads
them; the one sampled token is discarded. Output rows are {id, logits, n}; an existing output
file is resumed (rows already written are skipped). Nothing here is trained or tuned, and the
teacher never sees the gold label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

WIDTH = 64


def shard_of(row_id: str, shards: int) -> int:
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) % shards


def chunks(options: list[int], pool: list[int]):
    """Requests of exactly WIDTH identifier ids covering `options` in order."""
    parts = []
    for start in range(0, len(options), WIDTH):
        real = list(options[start:start + WIDTH])
        taken = set(real)
        pad = [t for t in pool if t not in taken][:WIDTH - len(real)]
        parts.append((real, real + pad))
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=16448)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    done, kept = set(), []
    if args.out.exists():
        for line in args.out.open():
            try:
                done.add(json.loads(line)["id"])
                kept.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError):
                break  # a torn last line from an interrupted run: dropped below
        partial = args.out.with_suffix(".resume")
        partial.write_text("".join(kept))
        partial.replace(args.out)
    rows = []
    for line in args.compiled.open():
        record = json.loads(line)
        if shard_of(record["id"], args.shards) != args.shard or record["id"] in done:
            continue
        rows.append({"id": record["id"], "input_ids": record["input_ids"],
                     "option_ids": record["option_ids"]})
    rows.sort(key=lambda r: len(r["input_ids"]))
    pool = sorted({t for r in rows for t in r["option_ids"]})
    print(json.dumps({"shard": args.shard, "rows": len(rows), "already": len(done)}), flush=True)
    if not rows:
        return
    import os

    backend = os.environ.get("FLASH_ATTENTION_BACKEND")  # B200: FlashInfer JIT fails, use FA
    extra = {"attention_backend": backend} if backend else {}
    llm = LLM(model=str(args.model), dtype="bfloat16", quantization=args.quantization, **extra,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
              max_logprobs=-1, logprobs_mode="raw_logits", seed=0,
              max_num_seqs=args.max_num_seqs, limit_mm_per_prompt={"image": 0, "video": 0})
    started, tokens = time.time(), 0
    with args.out.open("a") as sink:
        for begin in range(0, len(rows), args.batch):
            batch = rows[begin:begin + args.batch]
            prompts, params, owners = [], [], []
            for index, row in enumerate(batch):
                for real, ids in chunks(row["option_ids"], pool):
                    prompts.append({"prompt_token_ids": row["input_ids"]})
                    params.append(SamplingParams(max_tokens=1, temperature=0.0,
                                                 detokenize=False, logprobs=WIDTH,
                                                 logprob_token_ids=ids))
                    owners.append((index, real))
            outs = llm.generate(prompts, params, use_tqdm=False)
            values = [[] for _ in batch]
            for out, (index, real) in zip(outs, owners, strict=True):
                table = out.outputs[0].logprobs[0]
                values[index] += [float(table[t].logprob) for t in real]
            for row, value in zip(batch, values, strict=True):
                sink.write(json.dumps({"id": row["id"], "logits": value,
                                       "n": len(value)}) + "\n")
                tokens += len(row["input_ids"])
            sink.flush()
            elapsed = time.time() - started
            print(json.dumps({"shard": args.shard, "done": begin + len(batch),
                              "rows": len(rows), "tokens_per_second": tokens / elapsed}),
                  flush=True)


if __name__ == "__main__":
    main()
