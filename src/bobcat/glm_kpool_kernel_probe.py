"""Bounded synthetic controls for the installed GLM pooled top-k kernel.

This runs inside the pinned serving container after every model client finishes.
It never loads, changes, or evaluates model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path


def check_selection(scores, lengths, seq_lens, selected, *, pool_size=4, topk=2048):
    """Check top-k score optimality, complete groups, padding, and the unpooled tail."""
    if not (len(scores) == len(lengths) == len(seq_lens) == len(selected)):
        raise ValueError("Row counts changed.")
    for values, length, seq_len, row in zip(scores, lengths, seq_lens, selected, strict=True):
        if (not 0 <= length <= len(values) or topk % pool_size
                or not 0 <= seq_len - length * pool_size < pool_size
                or len(row) != topk + pool_size - 1):
            raise ValueError("Invalid pooled shape.")
        valid = [index for index in row if index >= 0]
        group_count = min(length, topk // pool_size)
        if (any(index < -1 or index >= seq_len for index in row)
                or len(valid) != len(set(valid))
                or len(valid) != group_count * pool_size + seq_len % pool_size):
            raise ValueError("Invalid or duplicated token selection.")
        history = {index for index in valid if index < length * pool_size}
        groups = sorted({index // pool_size for index in history})
        if (len(groups) != group_count
                or history != {group * pool_size + i
                               for group in groups for i in range(pool_size)}
                or set(valid) - history != set(range(length * pool_size, seq_len))):
            raise ValueError("Incomplete history group or missing tail.")
        chosen = sorted((values[group] for group in groups), reverse=True)
        if chosen != sorted(values[:length], reverse=True)[:group_count]:
            raise ValueError("The kernel did not select optimal valid group scores.")
    return True


def run(*, rank, seconds=70, expected_sources=None):
    import torch
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        topk_from_pooled_history_logits,
    )

    if not 0 <= rank < 8 or not 20 <= seconds <= 90:
        raise ValueError("Use one bounded probe per assigned GPU.")
    root = Path("/sgl-workspace/sglang/python/sglang")
    paths = [
        root / "srt/layers/attention/dsa/kpool_fp8_index.py",
        root / "kernels/ops/moe/kpool_topk_transform.py",
    ]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    if expected_sources is not None and hashes != expected_sources:
        raise ValueError("The installed pooled kernel wrapper differs from the frozen source.")
    config = json.loads(Path("/models/bobcat/config.json").read_text())["text_config"]
    if config["index_kpool"] != 4 or config["index_topk"] != 2048:
        raise ValueError("This probe must match the actual GLM pooled configuration.")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)  # The caller binds a different CUDA_VISIBLE_DEVICES per process.
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(20260923 + rank)
    record = {
        "schema": "bobcat-glm-kpool-kernel-probe-v1", "rank": rank,
        "started_at": datetime.now(UTC).isoformat(), "source_sha256": hashes,
        "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "pool_size": 4, "token_topk": 2048, "group_topk": 512,
        "cases": [], "status": "running", "maximum_seconds": seconds,
        "model_weights_loaded": False, "model_weights_changed": False,
        "real_model_causality_demonstrated": False, "release_gate_passed": False,
        "scope": "synthetic unique/tied pooled scores; kernel correctness and repeatability only",
    }
    started = time.monotonic()
    for cols in (512, 519, 600, 880):
        for family in ("unique", "ties"):
            if time.monotonic() - started > seconds - 10:
                record["status"] = "partial_time_budget"
                break
            cpu = torch.stack([torch.randperm(cols, generator=generator) for _ in range(8)])
            scores_cpu = (cpu.float() / cols if family == "unique"
                          else torch.div(cpu, 16, rounding_mode="floor").float())
            lengths = [min(cols, value) for value in (127, 255, 511, 512, cols, cols, cols, cols)]
            seq_lens = [length * 4 + i % 4 for i, length in enumerate(lengths)]
            scores = scores_cpu.cuda()
            length_tensor = torch.tensor(lengths, dtype=torch.int32, device="cuda")
            seq_tensor = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
            case_start = time.monotonic()

            def invoke(data=scores, lens=length_tensor, seq=seq_tensor):
                return topk_from_pooled_history_logits(
                    data, lens, pool_size=4, topk=2048, seq_lens=seq)

            first = invoke()
            torch.cuda.synchronize()
            first_rows = first.cpu().tolist()
            check_selection(scores_cpu.tolist(), lengths, seq_lens, first_rows)
            first_sorted = first.sort(dim=1).values
            part = {
                "columns": cols, "family": family,
                "first_call_seconds": time.monotonic() - case_start,
                "initial_selection_correct": True, "repeats": 0,
                "order_changes": 0, "set_changes": 0, "changed_selections_checked": 0,
                "input_sha256": hashlib.sha256(scores_cpu.numpy().tobytes()).hexdigest(),
                "first_output_sha256": hashlib.sha256(first.cpu().numpy().tobytes()).hexdigest(),
            }
            repeat_start = time.monotonic()
            for _ in range(64):
                if time.monotonic() - started > seconds - 5:
                    break
                value = invoke()
                same = torch.equal(first, value)
                part["repeats"] += 1
                part["order_changes"] += int(not same)
                part["set_changes"] += int(not torch.equal(
                    first_sorted, value.sort(dim=1).values))
                if not same:
                    check_selection(scores_cpu.tolist(), lengths, seq_lens, value.cpu().tolist())
                    part["changed_selections_checked"] += 1
            torch.cuda.synchronize()
            part["repeat_wall_seconds"] = time.monotonic() - repeat_start
            singles = torch.cat([
                invoke(scores[i:i + 1], length_tensor[i:i + 1], seq_tensor[i:i + 1])
                for i in range(8)])
            check_selection(scores_cpu.tolist(), lengths, seq_lens, singles.cpu().tolist())
            part["single_row_vs_batch_order_equal"] = torch.equal(first, singles)
            part["single_row_vs_batch_sets_equal"] = torch.equal(
                first_sorted, singles.sort(dim=1).values)
            record["cases"].append(part)
        if record["status"] == "partial_time_budget":
            break
    else:
        record["status"] = "completed"
    record.update(
        finished_at=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started,
        peak_probe_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_probe_reserved_bytes=torch.cuda.max_memory_reserved())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=70)
    parser.add_argument("--expected-sources", required=True)
    args = parser.parse_args()
    result = run(rank=args.rank, seconds=args.seconds,
                 expected_sources=json.loads(args.expected_sources))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
