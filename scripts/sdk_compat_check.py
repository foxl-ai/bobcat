"""Drive the Bobcat API with TypeSafe's official Python SDK (`typesafe-sdk`, MIT).

Starts `api_server.create_api` on localhost with the real student compiler and a
deterministic stand-in engine (no weights), then calls it through the unmodified SDK with
`base_url` pointed here: `system_one` with Noul/Choice/Score questions, `models.list()`,
and the SDK's 422 error class. This checks wire compatibility only, not model quality.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path


class StandIn:
    name = "stand-in"

    async def logits(self, sequences, option_ids, prefix):
        return [[float((len(s) + i) % 5) for i in range(len(o))]
                for s, o in zip(sequences, option_ids, strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-model", type=Path, required=True)
    parser.add_argument("--identifiers", type=Path,
                        default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    args = parser.parse_args()

    import uvicorn
    from tokenizers import Tokenizer
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    from typesafe_sdk import __version__ as sdk_version

    from bobcat.api_server import create_api
    from bobcat.student_readout import StudentCompiler, identifier_scheme

    receipt = json.loads((args.compiler_model / "bobcat-download.json").read_text())
    tokenizer = args.compiler_model / "tokenizer.json"
    reserved = {t["id"] for t in json.loads(tokenizer.read_text()).get("added_tokens", [])}
    identifiers = identifier_scheme(Tokenizer.from_file(str(tokenizer)), reserved,
                                    json.loads(args.identifiers.read_text())["identifiers"])
    compiler = StudentCompiler(args.compiler_model, receipt["files"], identifiers,
                               max_branch_tokens=16384, piecewise=True)
    models = [{"name": "bobcat-1.1", "description": "check", "release_date": "2026-09-25"}]
    app = create_api(StandIn(), compiler, model_name="bobcat-1.1", aliases={"bobcat-latest"},
                     temperature=1.0, models=models, edge_secret=None)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    client = TypeSafeClient(api_key="local-check", base_url=f"http://127.0.0.1:{port}",
                            model="bobcat-latest")
    result = client.system_one("결제가 두 번 됐어요. 빨리 확인해 주세요.", {
        "billing": Noul(instructions="결제 문의인가?"),
        "tone": Choice(instructions="어조는?", criteria={"calm": None, "angry": "화가 남"}),
        "urgency": Score(instructions="얼마나 급한가?", criteria=["낮음", "보통", "높음"]),
    })
    report = {
        "sdk": f"typesafe-sdk {sdk_version}",
        "noul": result.nouls["billing"].noul, "choice": result.choices["tone"].choice,
        "score": result.scores["urgency"].score, "model": result.model,
        "usage": {"input_tokens": result.usage.input_tokens,
                  "output_tokens": result.usage.output_tokens},
        "models": [m.name for m in client.models.list().models],
    }
    try:
        client.system_one("x", {"q": Choice(instructions="?", criteria={})})
        report["validation_error"] = "not raised"
    except Exception as error:  # the SDK maps 422 to its own exception class
        report["validation_error"] = type(error).__name__
    server.should_exit = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
