"""Finite real-GLM candidate readout over the frozen adversarial output probe.

No optimizer, generation loop or external API is used. This records native
scores for later HTTP-contract replay; it is not a serving latency benchmark.
Run only on eight otherwise unoccupied GPUs after the training job completes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_adapter_probe import verify_source
from bobcat.glm_native_train import REVISION
from bobcat.output_contract_probe import read_probe
from bobcat.schema import file_hash


def verify_probe_inputs(args):
    from safetensors.torch import load_file

    manifest, cases, branches = read_probe(args.probe)
    source = json.loads(args.source.read_text())
    decode = json.loads(args.parent_decode.read_text())
    if (source["revision"] != REVISION or manifest["source_revision"] != REVISION
            or manifest["source_sha256"] != file_hash(args.source)
            or manifest["tokenizer_sha256"] != file_hash(args.model_dir / "tokenizer.json")
            or manifest["chat_template_sha256"] != file_hash(args.model_dir / "chat_template.jinja")
            or manifest["compiler_sha256"] != file_hash(Path(__file__).with_name("glm_readout.py"))
            or file_hash(args.parent_adapter) != args.parent_adapter_sha256
            or file_hash(args.parent_decode) != args.parent_decode_sha256
            or decode.get("schema") != "bobcat-native-rl-cpu-decode-v1"
            or decode["adapter"]["sha256"] != args.parent_adapter_sha256
            or decode.get("all_eight_local_actor_shards_match_dcp") is not True
            or decode["adapter"].get("decoded_values_exact") is not True
            or decode["loop"]["arm"] != "proper_score_reinforce"
            or decode["actor_optimizer_updates"] != 35):
        raise ValueError("Require the exact independently decoded primary RL actor and probe.")
    tensors = load_file(str(args.parent_adapter), device="cpu")
    if len(tensors) != 360 or sum(t.numel() for t in tensors.values()) != 17649664:
        raise ValueError("Unexpected primary actor layout.")
    # Recompile only caller-visible fields, never the evaluator's expected labels.
    from bobcat.glm_readout import GLMCompiler
    from bobcat.protocol import parse_request

    compiler = GLMCompiler(args.model_dir, source, max_branch_tokens=2048,
                           max_request_tokens=8192)
    for case in cases:
        state, questions = parse_request(case["payload"])
        compiled = compiler.compile(state, questions)
        if compiled.logical_input_tokens != case["logical_input_tokens"]:
            raise ValueError("Logical input accounting changed after probe preparation.")
        for index, ids, options in zip(
            case["branch_indices"], compiled.input_ids, compiled.option_token_ids, strict=True,
        ):
            row = branches[index]
            if row["input_ids"] != ids or row["option_token_ids"] != options:
                raise ValueError("The real tokenizer no longer reproduces the frozen inputs.")
    return manifest, cases, branches, source, tensors


def execute(args, record):
    import torch
    import torch.distributed as dist

    from bobcat.glm_fsdp_adapter_probe import expected_local, local_copy
    from bobcat.glm_native_data import single_rank_batch
    from bobcat.glm_native_loader import load_native_model
    from bobcat.glm_native_rl import restore_local_parameters
    from bobcat.glm_parent_adapter import parent_adapter_bindings
    from bobcat.native_resume import state_signature

    started, rank = time.monotonic(), dist.get_rank()
    if dist.get_world_size() != 8:
        raise ValueError("The full GLM probe requires eight exclusive ranks.")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    stopped = False

    def request_stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    def status(phase, **fields):
        record.update(phase=phase, **fields, elapsed_seconds=time.monotonic() - started,
                      updated_at=datetime.now(UTC).isoformat())
        atomic_json(args.out / f"rank-{rank}.json", record)
        if rank == 0:
            print(json.dumps({"phase": phase, **fields}), flush=True)

    manifest, cases, branches, source, parent = verify_probe_inputs(args)
    record.update(
        probe_manifest_sha256=file_hash(args.probe / "manifest.json"),
        source_revision=source["revision"], parent_adapter_sha256=args.parent_adapter_sha256,
        parent_decode_sha256=args.parent_decode_sha256, parent_actor_updates=35,
        model_weights_used=True, generated_text_tokens=0, training_updates=0,
        original_vocabulary_projection_retained=True, autoregressive_generation_called=False,
        http_inference_executed=False, serving_latency_measured=False,
        current_probe_is_not_a_release_quality_evaluation=True,
        numerical_path="native_resident_eval_no_grad_original_torch_experts",
    )
    model, _ = load_native_model(
        args.model_dir, source, args.out, status, cpu_offload=False,
        expert_backend="torch", activation_checkpointing=True,
        source_model_set=args.source_model_set, source_verification=args.source_verification,
        parent_adapter_path=args.parent_adapter,
    )
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    binding = parent_adapter_bindings(model, parent)
    parts = {name: expected_local(binding[name], p).clone() for name, p in parameters.items()}
    restore_local_parameters(parameters, parts)
    before = state_signature(parts)
    del parent, binding, parts
    model.eval()
    generation_calls, head_calls = 0, 0

    def forbidden_generation(*_args, **_kwargs):
        nonlocal generation_calls
        generation_calls += 1
        raise RuntimeError("Autoregressive generation is outside this decision probe.")

    def count_head(_module, _inputs):
        nonlocal head_calls
        head_calls += 1

    model.generate = forbidden_generation
    head_hook = model.get_output_embeddings().register_forward_pre_hook(count_head)
    record["parent_actor_values_exact"] = True
    padded = manifest["max_padded_tokens"]
    if padded > 2048:
        raise ValueError("The fixed complete-input probe exceeds its prepared bound.")
    record["fixed_padded_tokens_per_branch"] = padded
    count = 0
    absolute = datetime.fromisoformat(args.absolute_deadline_utc)
    status("scoring_native_adversarial_branches", completed_local_branches=0)
    with (args.out / f"predictions-rank-{rank}.jsonl").open("x") as stream:
        for offset in range(0, len(branches), 8):
            stop = torch.tensor(int(
                stopped or time.monotonic() - started >= args.max_seconds - 90
                or datetime.now(UTC) >= absolute - timedelta(seconds=90)
            ), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop):
                record["status"] = "interrupted"
                break
            row = branches[offset + rank]
            batch = single_rank_batch({"inputs": {"input_ids": row["input_ids"]}},
                                      padded, device=device)
            begin = time.monotonic()
            heads_before = head_calls
            with torch.no_grad():
                output = model(**batch).logits
                if tuple(output.shape) != (1, 1, 154880) or head_calls != heads_before + 1:
                    raise ValueError("Expected one original vocabulary score position.")
                vocab = output[0, 0].float()
                indices = torch.tensor(row["option_token_ids"], device=device)
                selected = vocab.index_select(0, indices)
                if not bool(torch.isfinite(vocab).all()):
                    raise ValueError("The native probe produced nonfinite logits.")
                scores = selected.cpu().tolist()
                candidate_log_mass = float(selected.logsumexp(0) - vocab.logsumexp(0))
                raw_argmax = int(vocab.argmax())
                del output, vocab, indices, selected
            torch.cuda.synchronize(device)
            result = {
                "branch_index": row["index"], "case_id": row["case_id"],
                "question_id": row["question_id"], "input_sha256": row["input_sha256"],
                "option_token_ids": row["option_token_ids"], "logits": scores,
                "candidate_log_probability_mass": candidate_log_mass,
                "unconstrained_vocabulary_argmax_id": raw_argmax,
                "unconstrained_argmax_is_requested": raw_argmax in row["option_token_ids"],
                "public_output_uses_only_requested_identifiers": True,
                "no_tokens_sampled_or_decoded": True,
                "native_round_seconds_not_http_latency": time.monotonic() - begin,
            }
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            count += 1
            status("scoring_native_adversarial_branches", completed_local_branches=count)
    head_hook.remove()
    after = state_signature({n: local_copy(p) for n, p in parameters.items()})
    if before != after:
        raise ValueError("The read-only probe changed the primary actor.")
    record.update(
        status="completed" if count * 8 == len(branches) else "interrupted",
        actor_values_unchanged=True, completed_local_branches=count,
        observed_output_projection_calls=head_calls,
        attempted_generation_calls=generation_calls,
        finished_at=datetime.now(UTC).isoformat(),
        predictions_sha256=file_hash(args.out / f"predictions-rank-{rank}.jsonl"),
    )
    status(record["status"])
    dist.barrier()
    if rank == 0:
        ranks = [json.loads((args.out / f"rank-{r}.json").read_text()) for r in range(8)]
        complete = all(r["status"] == "completed" for r in ranks)
        atomic_json(args.out / ("complete.json" if complete else "interrupted.json"), {
            "schema": "bobcat-native-glm-output-probe-result-v1",
            "status": "completed" if complete else "interrupted",
            "job_sha256": args.job_sha256, "probe_manifest_sha256": record["probe_manifest_sha256"],
            "source_revision": REVISION, "parent_adapter_sha256": args.parent_adapter_sha256,
            "parent_decode_sha256": args.parent_decode_sha256, "world_size": 8,
            "model_weights_used": True, "generated_text_tokens": 0, "training_updates": 0,
            "files": {path.name: file_hash(path) for path in sorted(args.out.iterdir())
                      if path.is_file() and path.name not in ("complete.json", "interrupted.json")},
            "release_gate_passed": False,
        })


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "source", "probe", "out", "source-root", "git-tree",
                 "source-model-set", "source-verification", "parent-adapter", "parent-decode"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("parent-adapter-sha256", "parent-decode-sha256", "job-sha256",
                 "absolute-deadline-utc"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 300 <= args.max_seconds <= 1200:
        parser.error("Use a bounded 5–20 minute probe.")
    if sys.version_info[:2] != (3, 13) or torch.__version__ != "2.12.1+cu130":
        parser.error("Use the pinned producer runtime without replacing Torch.")
    verify_source(args.source_root, args.git_tree)
    if args.preflight_only:
        manifest, cases, branches, _, _ = verify_probe_inputs(args)
        print(json.dumps({"preflight": "passed", "cases": len(cases), "branches": len(branches),
                          "probe_manifest_sha256": file_hash(args.probe / "manifest.json"),
                          "max_padded_tokens": manifest["max_padded_tokens"],
                          "gpu_used": False, "training_updates": 0}))
        return
    deadline = datetime.fromisoformat(args.absolute_deadline_utc)
    if (deadline.utcoffset() is None
            or (deadline - datetime.now(UTC)).total_seconds() < args.max_seconds + 60):
        parser.error("The prepared probe no longer fits its original absolute time window.")
    if torch.cuda.device_count() != 8 or any(
        "B300" not in torch.cuda.get_device_name(i) for i in range(8)
    ):
        parser.error("Require the original eight otherwise unoccupied B300 GPUs.")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / f"rank-{os.environ['RANK']}.json").exists():
        parser.error("Use a fresh output directory; preserve earlier probe evidence.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    record = {
        "schema": "bobcat-native-glm-output-probe-rank-v1", "rank": dist.get_rank(),
        "status": "initializing", "started_at": datetime.now(UTC).isoformat(),
        "job_sha256": args.job_sha256,
    }
    try:
        execute(args, record)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:2000],
                      finished_at=datetime.now(UTC).isoformat())
        atomic_json(args.out / f"rank-{dist.get_rank()}.json", record)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
