"""Verify captured real-GLM probe scores, then replay them through the typed API."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_readout import GLMCompiler
from bobcat.output_contract_audit import audit
from bobcat.output_contract_probe import read_probe, semantic_comparison
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash


def read_native(folder: Path, probe: Path):
    manifest, cases, branches = read_probe(probe)
    complete = json.loads((folder / "complete.json").read_text())
    if (complete.get("schema") != "bobcat-native-glm-output-probe-result-v1"
            or complete.get("status") != "completed" or complete.get("world_size") != 8
            or complete["probe_manifest_sha256"] != file_hash(probe / "manifest.json")
            or complete["source_revision"] != manifest["source_revision"]
            or complete.get("model_weights_used") is not True
            or complete.get("generated_text_tokens") != 0
            or complete.get("training_updates") != 0):
        raise ValueError("Require the completed real-model readout for the exact probe.")
    for name, digest in complete["files"].items():
        if (Path(name).name != name or (folder / name).is_symlink()
                or file_hash(folder / name) != digest):
            raise ValueError("A completed native artifact changed.")
    predictions = {}
    for rank in range(8):
        status_file, prediction_file = f"rank-{rank}.json", f"predictions-rank-{rank}.jsonl"
        if not {status_file, prediction_file}.issubset(complete["files"]):
            raise ValueError("The native completion marker omits a rank.")
        status = json.loads((folder / status_file).read_text())
        if (status.get("status") != "completed" or status["rank"] != rank
                or status.get("parent_actor_values_exact") is not True
                or status.get("actor_values_unchanged") is not True
                or status.get("model_weights_used") is not True
                or status.get("generated_text_tokens") != 0
                or status.get("training_updates") != 0
                or status.get("attempted_generation_calls") != 0
                or status.get("observed_output_projection_calls") != len(branches) // 8
                or status.get("completed_local_branches") != len(branches) // 8
                or status["parent_adapter_sha256"] != complete["parent_adapter_sha256"]
                or status["parent_decode_sha256"] != complete["parent_decode_sha256"]
                or status["source_revision"] != complete["source_revision"]
                or status["probe_manifest_sha256"] != complete["probe_manifest_sha256"]
                or status["job_sha256"] != complete["job_sha256"]
                or status["predictions_sha256"] != complete["files"][prediction_file]):
            raise ValueError("The native actor/rank proof is incomplete.")
        rows = [json.loads(line) for line in (folder / prediction_file).read_text().splitlines()]
        if len(rows) * 8 != len(branches):
            raise ValueError("A native rank omitted readout branches.")
        for offset, row in enumerate(rows):
            index = offset * 8 + rank
            expected = branches[index]
            if (row["branch_index"] != index or index in predictions
                    or any(row[key] != expected[key] for key in (
                        "case_id", "question_id", "input_sha256", "option_token_ids"
                    )) or row.get("no_tokens_sampled_or_decoded") is not True
                    or len(row["logits"]) != len(expected["option_token_ids"])
                    or any(type(x) not in (int, float) or not math.isfinite(x)
                           for x in row["logits"])):
                raise ValueError("Native prediction membership, ordering or scores changed.")
            predictions[index] = row
    if set(predictions) != set(range(len(branches))):
        raise ValueError("The native eight-rank readout is incomplete.")
    return complete, cases, branches, predictions


def run(folder: Path, probe: Path, model_dir: Path, source_path: Path, out: Path):
    from fastapi.testclient import TestClient

    from bobcat.serve import create_app

    complete, cases, branches, predictions = read_native(folder, probe)
    compiler = GLMCompiler(model_dir, json.loads(source_path.read_text()))
    scores = {case["id"]: {
        branches[i]["question_id"]: predictions[i]["logits"] for i in case["branch_indices"]
    } for case in cases}

    class RecordedScorer:
        model_name = f"bobcat-glm53-rl35-probe-{complete['parent_adapter_sha256'][:12]}"
        release_gate_passed = False
        readout_mode = "prefill_only"
        temperatures = {}
        cursor = 0

        def score(self, state, questions):
            expected = cases[self.cursor]
            original_state, original_questions = parse_request(expected["payload"])
            if (json_hash(state) != json_hash(original_state) or questions != original_questions):
                raise ValueError("Replay must use the exact request, not a new unscored prompt.")
            compiled = compiler.compile(state, questions)
            for index, ids, options in zip(
                expected["branch_indices"], compiled.input_ids,
                compiled.option_token_ids, strict=True,
            ):
                if (json_hash(ids) != predictions[index]["input_sha256"]
                        or options != predictions[index]["option_token_ids"]):
                    raise ValueError("HTTP replay differs from real-model compiled inputs.")
            self.cursor += 1
            return ([scores[expected["id"]][q.id] for q in questions],
                    compiled.logical_input_tokens)

    scorer = RecordedScorer()
    with TestClient(create_app(scorer), raise_server_exceptions=False) as client:
        report = audit(client, cases, model=scorer.model_name, out=out, scope={
            "mode": "real_native_glm_scores_replayed_through_http_contract",
            "model_weights_used_for_native_scores": True,
            "http_client_replays_captured_scores": True, "live_http_gpu_inference": False,
            "serving_latency_measured": False,
            "parent_adapter_sha256": complete["parent_adapter_sha256"],
            "parent_decode_sha256": complete["parent_decode_sha256"],
            "native_complete_sha256": file_hash(folder / "complete.json"),
            "probe_manifest_sha256": file_hash(probe / "manifest.json"),
            "source_revision": complete["source_revision"],
        })
    semantic = semantic_comparison(cases, scores)
    atomic_json(out / "semantic-pairs.json", semantic)
    report.update(
        semantic_jailbreak_success_measured=True,
        semantic_evaluation_scope=semantic["scope"],
        semantic_pair_artifact_sha256=file_hash(out / "semantic-pairs.json"),
        semantic_analysis_source_sha256=file_hash(
            Path(__file__).with_name("output_contract_probe.py"),
        ),
        replay_source_sha256=file_hash(Path(__file__)),
        native_branches=len(predictions),
        unconstrained_vocabulary_argmax_outside_requested_options=sum(
            not p["unconstrained_argmax_is_requested"] for p in predictions.values()
        ),
        unconstrained_tokens_were_never_sampled_decoded_or_returned=True,
    )
    atomic_json(out / "report.json", report)
    if (report["output_contract_violations"] or scorer.cursor != len(cases)
            or report["statuses"] != {"200": len(cases)}):
        raise ValueError("The real-model output-contract replay failed.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-run", "probe", "model-dir", "source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.native_run, args.probe, args.model_dir, args.source, args.out)
    print(json.dumps({k: result[k] for k in (
        "scope", "requests", "unique_payloads", "statuses", "typed_answers",
        "output_contract_violations", "unconstrained_vocabulary_argmax_outside_requested_options",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
