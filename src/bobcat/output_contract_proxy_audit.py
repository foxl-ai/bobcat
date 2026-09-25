"""Actual GLM tokenizer/proxy audit with a simulated numeric engine, no weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.glm_readout import GLMCompiler, SGLangScorer
from bobcat.output_contract_audit import attack_cases, audit
from bobcat.schema import file_hash


def run(model_dir: Path, source_path: Path, out: Path):
    import httpx
    from fastapi.testclient import TestClient

    from bobcat.serve import create_app

    source = json.loads(source_path.read_text())
    compiler = GLMCompiler(model_dir, source)
    expected_controls = [
        token for token in compiler.before + compiler.between + compiler.after
        if token in compiler.reserved_ids
    ]
    counters = {"native_batches": 0, "native_branches": 0,
                "data_did_not_add_host_control_tokens": 0,
                "requested_new_tokens": 0}

    def engine(request):
        if request.url.path == "/get_model_info":
            return httpx.Response(200, json={"model_path": "/numeric-fixture/glm53"})
        if request.url.path != "/generate":
            raise ValueError("Unexpected native route.")
        body = json.loads(request.content)
        if (body["sampling_params"]["max_new_tokens"] != 0
                or body["return_text_in_logprobs"] is not False
                or body["top_logprobs_num"] != 0):
            raise ValueError("The decision proxy requested generation or incomplete options.")
        counters["native_batches"] += 1
        rows = []
        for ids, options in zip(body["input_ids"], body["token_ids_logprob"], strict=True):
            controls = [token for token in ids if token in compiler.reserved_ids]
            if controls != expected_controls:
                raise ValueError("Untrusted input altered the host control-token sequence.")
            if len(options) != len(set(options)) or not 1 <= len(options) <= 255:
                raise ValueError("The exact candidate readout changed.")
            counters["native_branches"] += 1
            counters["data_did_not_add_host_control_tokens"] += 1
            rows.append({
                "text": "", "output_ids": [], "meta_info": {
                    "prompt_tokens": len(ids), "completion_tokens": 0,
                    "output_token_ids_logprobs": [[
                        [-2. - .1 * index, token, None] for index, token in enumerate(options)
                    ]],
                },
            })
        return httpx.Response(200, json=rows)

    with httpx.Client(transport=httpx.MockTransport(engine),
                      base_url="http://numeric-fixture") as native:
        scorer = SGLangScorer(compiler, "http://numeric-fixture", "/numeric-fixture/glm53",
                             client=native, readout_mode="prefill_only")
        with TestClient(create_app(scorer), raise_server_exceptions=False) as client:
            report = audit(client, attack_cases(), model=scorer.model_name, out=out, scope={
                "mode": "actual_glm_tokenizer_and_proxy_with_numeric_engine_fixture",
                "original_glm_tokenizer_used": True, "original_chat_template_used": True,
                "model_weights_used": False, "gpu_used": False,
                "native_engine_is_simulated": True, "physical_decode_profiled": False,
                "source_revision": source["revision"],
                "source_sha256": file_hash(source_path),
                "tokenizer_sha256": file_hash(model_dir / "tokenizer.json"),
                "chat_template_sha256": file_hash(model_dir / "chat_template.jinja"),
            })
    report["native_request_checks"] = counters
    atomic_json(out / "report.json", report)
    if (report["output_contract_violations"] or report["statuses"] != {"200": len(attack_cases())}
            or counters["native_branches"] != report["typed_answers"]):
        raise ValueError("The actual-tokenizer/proxy structural audit did not pass.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.model_dir, args.source, args.out)
    print(json.dumps({key: result[key] for key in (
        "scope", "requests", "unique_payloads", "statuses", "typed_answers",
        "output_contract_violations", "native_request_checks",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
