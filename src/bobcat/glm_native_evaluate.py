"""Bounded checkpoint evaluation on a separate eight-GPU worker.

Loads the original base once, measures its unchanged readout, then reads only
completed immutable adapter checkpoints. It sends no command to the trainer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bobcat.checkpoint_watch import (
    candidate_markers,
    download_checkpoint,
    producer_terminal,
    read_version,
)
from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import verify_source
from bobcat.glm_adapter_scope import validate_checkpoint_adapter_scope
from bobcat.glm_native_data import single_rank_batch
from bobcat.glm_native_train import REVISION, decision_loss, tensor_digest, validate_record
from bobcat.schema import file_hash, json_hash


def read_suite(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("schema") != "bobcat-checkpoint-monitor-suite-v1"
            or manifest.get("evaluation_role") != "development_monitoring"
            or manifest.get("source_revision") != REVISION
            or manifest.get("training_or_calibration") is not False
            or manifest["content_sha256"] != json_hash({
                k: v for k, v in manifest.items() if k != "content_sha256"
            }) or file_hash(folder / "records.jsonl") != manifest["records_sha256"]):
        raise ValueError("Use the frozen monitoring suite, separate from training/calibration.")
    rows = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    if len(rows) != manifest["rows"] or len(rows) % 8:
        raise ValueError("Monitoring must use complete distributed batches.")
    groups = set()
    for row in rows:
        validate_record(row, manifest["max_input_tokens"])
        if row["split"] != "dev_train" or row["group_id"] in groups:
            raise ValueError("Monitoring rows must be distinct development components.")
        groups.add(row["group_id"])
    return manifest, rows


def execute(args, record):
    import boto3
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from botocore.config import Config
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    from bobcat.glm_native_loader import load_native_model

    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    begun = time.monotonic()
    suite, rows = read_suite(args.suite)
    record["suite_sha256"] = file_hash(args.suite / "manifest.json")
    source = json.loads(args.source.read_text())
    if source["revision"] != REVISION:
        raise ValueError("The evaluation base differs from the training base.")
    identifiers = None
    if args.decision_identifiers is not None:
        mapping = json.loads(args.decision_identifiers.read_text())
        identifiers = mapping["token_ids"]
        if (mapping["source_revision"] != REVISION or len(identifiers) != 255
                or len(set(identifiers)) != 255
                or any(type(i) is not int or i < 0 for i in identifiers)
                or any(not set(row["option_token_ids"]).issubset(identifiers) for row in rows)):
            raise ValueError("Use the original 255 identifiers covering every offered option.")
        record["decision_identifiers_sha256"] = file_hash(args.decision_identifiers)

    def status(phase, **fields):
        record.update(phase=phase, **fields, updated_at=datetime.now(UTC).isoformat())
        record["elapsed_seconds"] = time.monotonic() - begun
        record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        atomic_json(args.out / f"rank-{rank}.json", record)
        if rank == 0:
            print(json.dumps({"phase": phase, **fields}, default=str), flush=True)

    model, adapters = load_native_model(
        args.model_dir, source, args.out, status, cpu_offload=args.cpu_offload,
        expert_backend=args.expert_backend, decision_token_ids=identifiers,
        adapter_last_layers=args.adapter_last_layers,
        compact_bank_dir=args.compact_decision_bank,
        source_manifest_sha256=file_hash(args.source),
    )
    record["expert_backend"] = args.expert_backend
    model.eval()
    # Hash the frozen shards once before and once after this worker's evaluations.
    frozen = {n: tensor_digest(p) for n, p in model.state_dict().items() if "lora_" not in n}
    atomic_json(args.out / f"frozen-state-hashes-rank-{rank}.json", frozen)
    if args.frozen_state_reference is not None:
        from bobcat.glm_compact_head import compare_frozen_body_hashes

        reference_folder = args.frozen_state_reference
        reference_manifest = json.loads((reference_folder / "manifest.json").read_text())
        name = f"frozen-state-hashes-rank-{rank}.json"
        if (reference_manifest.get("schema") != "bobcat-original-frozen-state-reference-v1"
                or reference_manifest.get("source_revision") != REVISION
                or reference_manifest.get("world_size") != 8
                or reference_manifest.get("original_vocabulary_head") is not True
                or set(reference_manifest.get("files", {})) != {
                    f"frozen-state-hashes-rank-{r}.json" for r in range(8)
                }
                or file_hash(reference_folder / name) != reference_manifest["files"][name]):
            raise ValueError("Use the versioned original-head worker's frozen-shard evidence.")
        comparison = compare_frozen_body_hashes(
            frozen, json.loads((reference_folder / name).read_text()),
        )
        atomic_json(args.out / f"body-value-comparison-rank-{rank}.json", {
            **comparison,
            "reference_manifest_sha256": file_hash(reference_folder / "manifest.json"),
        })
        record["body_matches_original_reference"] = comparison["body_values_exact"]
    options = StateDictOptions(ignore_frozen_params=True, cpu_offload=True, strict=False)
    client = boto3.client("s3", region_name=args.region, config=Config(
        connect_timeout=5, read_timeout=30, retries={"total_max_attempts": 1},
    ))
    head = model.get_output_embeddings()
    audit_folder = args.compact_decision_bank if args.compact_projection_audit \
        else args.external_bank_audit
    if audit_folder is not None:
        from bobcat.glm_compact_head import load_verified_native_bank

        audit_bank, audit_binding = load_verified_native_bank(
            audit_folder, source=source,
            source_manifest_sha256=file_hash(args.source), token_ids=identifiers,
        )
        # A diagnostic tensor, not a Parameter/buffer or a replacement FSDP head.
        head.external_audit_bank = audit_bank.to(device)
        record["external_bank_audit_binding"] = audit_binding

    def row_forward(row, padded):
        batch = single_rank_batch({"inputs": {"input_ids": row["input_ids"]}},
                                  padded, device=device)
        logits = model(**batch).logits[0, -1]
        if identifiers is not None:
            return head.select_batch(logits[None], [row["option_token_ids"]])[0]
        indices = torch.tensor(row["option_token_ids"], device=device)
        return logits.index_select(-1, indices).float()

    def projection_control():
        status("checking_decision_projection")
        controls = []
        with torch.no_grad():
            for offset in range(0, 32, 8):
                items = rows[offset:offset + 8]
                row = items[rank]
                padded = (max(r["input_tokens"] for r in items) + 127) // 128 * 128
                head.decision_only = False
                head.audit_option_ids = row["option_token_ids"]
                original = row_forward(row, padded)
                same_hidden = dict(head.last_audit)
                head.audit_option_ids = None
                repeated = row_forward(row, padded)
                head.decision_only = True
                selected = row_forward(row, padded)
                head.decision_only = False

                def tv(a, b):
                    return float((a.softmax(-1) - b.softmax(-1)).abs().sum() / 2)

                controls.append({
                    "id": row["id"], "input_sha256": row["input_sha256"],
                    "same_hidden": same_hidden,
                    "repeat_tv": tv(original, repeated),
                    "selected_full_forward_tv": tv(repeated, selected),
                    "repeat_argmax_equal": bool(original.argmax() == repeated.argmax()),
                    "selected_argmax_equal": bool(repeated.argmax() == selected.argmax()),
                    "finite": bool(torch.isfinite(selected).all()),
                    "original_logits": original.cpu().tolist(),
                    "repeated_logits": repeated.cpu().tolist(),
                    "selected_logits": selected.cpu().tolist(),
                })
        local_passed = all(
            item["same_hidden"]["finite"]
            and item["same_hidden"]["maximum_probability_tv"] <= .001
            and item["same_hidden"]["argmax_equal"]
            and item["repeat_tv"] <= .001 and item["repeat_argmax_equal"]
            and item["selected_full_forward_tv"] <= .001
            and item["selected_argmax_equal"] and item["finite"]
            and (args.external_bank_audit is None or (
                item["same_hidden"]["external_original_rows_equal"]
                and item["same_hidden"]["external_finite"]
                and item["same_hidden"]["external_maximum_probability_tv"] <= .001
                and item["same_hidden"]["external_argmax_equal"]
            ))
            for item in controls
        )
        vote = torch.tensor(int(local_passed), device=device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        passed = bool(vote)
        atomic_json(args.out / f"decision-projection-control-rank-{rank}.json", {
            "rank": rank, "questions": controls, "local_passed": local_passed,
            "global_passed": passed, "allowed_probability_tv": .001,
            "weight_layout_preserved": True, "full_weight_all_gather_removed": False,
            "training_updates": 0, "production_latency_measured": False,
        })
        record["decision_projection_gate_passed"] = passed
        return passed

    def compact_projection_control():
        status("checking_materialized_compact_projection")
        controls = []
        with torch.no_grad():
            for offset in range(0, 32, 8):
                items = rows[offset:offset + 8]
                row = items[rank]
                padded = (max(r["input_tokens"] for r in items) + 127) // 128 * 128
                head.audit_option_ids = row["option_token_ids"]
                original = row_forward(row, padded)
                same_hidden = dict(head.last_audit)
                head.audit_option_ids = None
                repeated = row_forward(row, padded)
                controls.append({
                    "id": row["id"], "input_sha256": row["input_sha256"],
                    "same_hidden": same_hidden,
                    "repeat_tv": float(
                        (original.softmax(-1) - repeated.softmax(-1)).abs().sum() / 2,
                    ),
                    "repeat_argmax_equal": bool(original.argmax() == repeated.argmax()),
                    "original_logits": original.cpu().tolist(),
                    "repeated_logits": repeated.cpu().tolist(),
                })
        passed = all(
            row["same_hidden"]["loaded_rows_bitwise_equal"]
            and row["same_hidden"]["finite"]
            and row["same_hidden"]["maximum_probability_tv"] <= .001
            and row["same_hidden"]["argmax_equal"]
            and row["repeat_tv"] <= .001 and row["repeat_argmax_equal"]
            for row in controls
        )
        vote = torch.tensor(int(passed), device=device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        atomic_json(args.out / f"compact-projection-control-rank-{rank}.json", {
            "rank": rank, "questions": controls, "local_passed": passed,
            "global_passed": bool(vote), "allowed_probability_tv": .001,
            "weight_layout_preserved": False, "actual_compact_weight_audited": True,
            "production_latency_measured": False, "training_updates": 0,
        })
        record["compact_projection_gate_passed"] = bool(vote)

    def evaluate(label, checkpoint=None):
        status("evaluating", checkpoint=label, completed_questions=0)
        results = []
        with torch.no_grad():
            for offset in range(0, len(rows), 8):
                items = rows[offset:offset + 8]
                row = items[rank]
                padded = (max(r["input_tokens"] for r in items) + 127) // 128 * 128
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                decision = row_forward(row, padded)
                if not torch.isfinite(decision).all():
                    raise ValueError("A checkpoint returned non-finite monitoring logits.")
                torch.cuda.synchronize(device)
                results.append({
                    **{k: row[k] for k in ("id", "group_id", "task", "language",
                                           "language_origin", "kind", "target_index",
                                           "score_mean", "supervision", "input_sha256")},
                    "logits": decision.cpu().tolist(),
                    "loss": float(decision_loss(decision, row)),
                    "forward_seconds": time.perf_counter() - start,
                    "input_tokens": row["input_tokens"], "padded_tokens": padded,
                    "checkpoint": label, "suite_sha256": record["suite_sha256"],
                    "decision_only": bool(identifiers is not None and head.decision_only),
                    "production_latency_measured": False,
                })
                if offset % 128 == 0:
                    status("evaluating", checkpoint=label, completed_questions=offset + 8)
        atomic_json(args.out / f"{label}-rank-{rank}.json", results)
        dist.barrier()
        if rank == 0:
            atomic_json(args.out / f"{label}-complete.json", {
                "checkpoint": label, "checkpoint_receipt": checkpoint,
                "suite_sha256": record["suite_sha256"], "rows": len(rows),
                "source_revision": REVISION, "evaluation_role": "development_monitoring",
                "files": {f"{label}-rank-{r}.json": file_hash(
                    args.out / f"{label}-rank-{r}.json",
                ) for r in range(8)},
                "final_evaluation": False, "quality_gate_passed": False,
                "training_modified": False,
                "expert_backend": args.expert_backend,
                "adapter_scope": record["adapter_scope"],
                "adapter_scope_sha256": json_hash(record["adapter_scope"]),
                "compact_output_binding": record.get("compact_output_binding"),
            })
        dist.barrier()

    baseline_started = time.monotonic()
    evaluate("baseline")
    baseline_seconds = time.monotonic() - baseline_started
    if args.compact_projection_audit:
        compact_projection_control()
    if identifiers is not None and args.compact_decision_bank is None:
        projection_passed = projection_control()
        # Baseline and checkpoint must still use the same original readout if
        # numerical equivalence fails. A failed optimization does not erase a
        # valid unchanged-base development comparison.
        if (projection_passed
                and time.monotonic() - begun + 2.5 * baseline_seconds + 300 < args.max_seconds):
            head.decision_only = True
            evaluate("baseline-selected")
            head.decision_only = False
        else:
            record["selected_suite_skipped"] = (
                "projection_control_failed" if not projection_passed else "finite_time_budget"
            )
    evaluated, skipped = [], []
    last_work = time.monotonic()
    while len(evaluated) < args.max_checkpoints:
        reserve = max(180., baseline_seconds * 1.3 + 180.)
        stopping = torch.tensor(
            int(time.monotonic() - begun > args.max_seconds - reserve), device=device,
        )
        dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        if bool(stopping):
            break
        command = [None]
        if rank == 0:
            keys = [args.fixed_checkpoint_marker] if args.fixed_checkpoint_marker else []
            if not args.fixed_checkpoint_marker:
                for page in client.get_paginator("list_objects_v2").paginate(
                    Bucket=args.bucket, Prefix=args.producer_prefix,
                ):
                    keys.extend(item["Key"] for item in page.get("Contents", []))
            choice, backlog = candidate_markers(
                keys, args.producer_prefix, evaluated_steps=evaluated + skipped,
            )
            if choice:
                step, key = choice
                directory = args.out / f"download-{step:06d}"
                receipt = download_checkpoint(
                    client, args.bucket, key, directory, revision=REVISION,
                    curriculum_sha256=args.curriculum_sha256,
                )
                if (args.fixed_checkpoint_sha256 is not None
                        and receipt["complete_sha256"] != args.fixed_checkpoint_sha256):
                    raise ValueError("The fixed checkpoint marker changed.")
                command[0] = {"step": step, "directory": str(directory),
                              "receipt": receipt, "skipped": backlog}
            else:
                from botocore.exceptions import ClientError
                try:
                    raw, producer_receipt = read_version(
                        client, args.bucket,
                        args.producer_prefix.removesuffix("train/") + "execution.json",
                        max_bytes=4 * 1024**2,
                    )
                except ClientError as error:
                    if error.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                        raise
                else:
                    if producer_terminal(
                        json.loads(raw), prefix=args.producer_prefix, revision=REVISION,
                        curriculum_sha256=args.curriculum_sha256,
                    ):
                        command[0] = {"stop": "producer_finished",
                                      "producer_receipt": producer_receipt}
                if command[0] is None and time.monotonic() - last_work > args.max_idle_seconds:
                    command[0] = {"stop": "no_ready_checkpoint"}
        dist.broadcast_object_list(command, src=0)
        if command[0] and "stop" in command[0]:
            record["stop_reason"] = command[0]
            break
        if command[0] is None:
            # A finite small interval; no GPU kernel or trainer request is used to wait.
            time.sleep(10)
            continue
        item = command[0]
        marker = json.loads((Path(item["directory"]) / "complete.json").read_text())
        validate_checkpoint_adapter_scope(marker, record["adapter_scope"])
        state = get_model_state_dict(model, options=options)
        parameter_names = set(dict(model.named_parameters()))
        if not state or any("lora_" not in name and name in parameter_names for name in state):
            raise ValueError("Adapter-only evaluation state includes frozen weights.")
        dcp.load({"model": state}, checkpoint_id=item["directory"])
        set_model_state_dict(model, state, options=options)
        model.eval()
        evaluate(f"checkpoint-{item['step']:06d}", item["receipt"])
        evaluated.append(item["step"])
        skipped.extend(item["skipped"])
        last_work = time.monotonic()
        status("checkpoint_evaluated", evaluated_steps=evaluated, coalesced_steps=skipped)
    if {n: tensor_digest(p) for n, p in model.state_dict().items()
            if "lora_" not in n} != frozen:
        raise ValueError("The evaluator changed the pretrained base or persistent buffers.")
    status("completed", status="completed", evaluated_steps=evaluated,
           coalesced_steps=skipped, frozen_base_preserved=True, training_modified=False,
           quality_gate_passed=False)


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "git-tree", "model-dir", "source", "suite", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("bucket", "producer-prefix", "curriculum-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--max-seconds", type=int, default=14400)
    parser.add_argument("--max-checkpoints", type=int, default=4)
    parser.add_argument("--max-idle-seconds", type=int, default=180)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--expert-backend", choices=("torch", "torch_mm"), default="torch")
    parser.add_argument("--decision-identifiers", type=Path)
    parser.add_argument("--adapter-last-layers", type=int)
    parser.add_argument("--compact-decision-bank", type=Path)
    parser.add_argument("--external-bank-audit", type=Path)
    parser.add_argument("--compact-projection-audit", action="store_true")
    parser.add_argument("--frozen-state-reference", type=Path)
    parser.add_argument("--short-reference-control", action="store_true")
    parser.add_argument("--fixed-checkpoint-marker")
    parser.add_argument("--fixed-checkpoint-sha256")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    if (int(os.environ.get("WORLD_SIZE", "0")) != 8
            or not 1 <= args.max_checkpoints <= 8 or not 0 <= args.max_idle_seconds <= 180
            or not (1200 if args.short_reference_control else 1800) <= args.max_seconds <= 21600
            or args.adapter_last_layers is not None and not 1 <= args.adapter_last_layers <= 45
            or not args.producer_prefix.endswith("/train/")):
        parser.error("Use a finite eight-GPU evaluator with a bounded idle period.")
    if args.compact_decision_bank is not None and args.decision_identifiers is None:
        parser.error("A compact bank requires its original identifier mapping.")
    if args.compact_projection_audit:
        _, audit_rows = read_suite(args.suite)
        if (args.compact_decision_bank is None or args.frozen_state_reference is None
                or len(audit_rows) != 64 or args.max_checkpoints != 1
                or args.max_seconds > 1800 or not args.fixed_checkpoint_marker):
            parser.error("Use the finite 64-row physical compact and body-value audit.")
    elif args.frozen_state_reference is not None:
        parser.error("Frozen-state comparison belongs to the explicit compact audit.")
    if args.short_reference_control:
        _, short_rows = read_suite(args.suite)
        if (len(short_rows) != 64 or args.max_checkpoints != 1
                or args.max_seconds > 1500 or args.compact_decision_bank is not None
                or args.external_bank_audit is None or not args.fixed_checkpoint_marker):
            parser.error("The short profile is a frozen 64-row original-head control.")
    if args.external_bank_audit is not None and (
            args.decision_identifiers is None or args.compact_decision_bank is not None):
        parser.error("Compare an external bank inside the unchanged original output head.")
    if (bool(args.fixed_checkpoint_marker) != bool(args.fixed_checkpoint_sha256)
            or args.fixed_checkpoint_marker and (
                not args.fixed_checkpoint_marker.startswith(args.producer_prefix)
                or args.max_checkpoints != 1
                or len(args.fixed_checkpoint_sha256) != 64
                or any(c not in "0123456789abcdef" for c in args.fixed_checkpoint_sha256))):
        parser.error("Use one explicitly frozen producer checkpoint and digest.")
    if args.deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            parser.error("Set the original deterministic cuBLAS environment before startup.")
        torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    rank = dist.get_rank()
    record = {"schema": "bobcat-native-checkpoint-evaluation-v1", "rank": rank,
              "status": "running", "started_at": datetime.now(UTC).isoformat(),
              "trainer_source_modified": False, "final_evaluation": False}
    try:
        if rank == 0:
            args.out.mkdir(parents=True, exist_ok=False)
            record["vendor_verification"] = verify_source(args.source_root, args.git_tree)
        dist.barrier()
        sys.path.insert(0, str(args.source_root.resolve()))
        execute(args, record)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:3000])
        raise
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        if args.out.exists():
            atomic_json(args.out / f"rank-{rank}.json", record)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
