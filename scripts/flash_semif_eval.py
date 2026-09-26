"""SemIf's unmodified evaluators on one system's predictions, plus a compact summary.

Usage: flash_semif_eval.py --semif DIR --eval DIR --pred DIR --out DIR
  --semif  the SemIf-OpenJev checkout (benchmarks/*.py, benchmarks/data)
  --eval   built evaluation rows (wanli256.jsonl, typesafe102.jsonl, gold154.jsonl,
           inference204.jsonl, firewall-actions.json) from the 2026-09-26 SemIf bundle
  --pred   <set>.predictions.jsonl from `scripts/semif_bench.py predictions`
The system's predictions fill both of SemIf's `direct` and `reranker` slots; only `direct`
is read back. Jev is never called; its published figures are not recomputed here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr[-2000:]}")
    return result.stdout


def stability(report: dict) -> dict:
    """Balanced accuracy of the system on the originals, each variant and missing evidence."""
    direct = report["systems"]["direct_logits"]

    def bal(block):
        block = block.get("evaluation", block)
        return block.get("mean_family_balanced_accuracy")

    out = {"base_original": bal(direct["base_original"]),
           "missing_evidence": bal(direct["missing_evidence"])}
    out.update({name: bal(block) for name, block in direct["variants"].items()})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("semif", "eval", "pred", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    bench = args.semif / "benchmarks"
    py = sys.executable
    args.out.mkdir(parents=True, exist_ok=True)
    pred = {p.name.split(".")[0]: p for p in args.pred.glob("*.predictions.jsonl")}
    summary, errors = {}, {}

    def step(name, command, parse, stdout=False):
        target = args.out / f"{name}.json"
        if target.exists():
            target.unlink()
        try:
            printed = run(command)
            if stdout:  # evaluate_external prints its report instead of writing a file
                target.write_text(printed)
            summary[name] = parse(json.loads(target.read_text()))
        except Exception as error:  # recorded, the other sets still run
            errors[name] = str(error)[-600:]

    for name, gold in (("authored144", bench / "data" / "authored144.jsonl"),
                       ("wanli256", args.eval / "wanli256.jsonl"),
                       ("perturbations108", bench / "data" / "perturbations108.jsonl")):
        if name in pred:
            step(name, [py, str(bench / "evaluate.py"), "--gold", str(gold),
                        "--predictions", str(pred[name]), "--output",
                        str(args.out / f"{name}.json")],
                 lambda r: {"mean_family_balanced_accuracy": r["mean_family_balanced_accuracy"],
                            "scored": r["scored"], "coverage": r["coverage"]})
    if "authored144" in pred and "perturbations108" in pred:
        step("stability", [py, str(bench / "evaluate_perturbations.py"),
                           "--gold", str(bench / "data" / "authored144.jsonl"),
                           "--perturbations", str(bench / "data" / "perturbations108.jsonl"),
                           "--direct-base", str(pred["authored144"]),
                           "--direct-perturbations", str(pred["perturbations108"]),
                           "--reranker-base", str(pred["authored144"]),
                           "--reranker-perturbations", str(pred["perturbations108"]),
                           "--output", str(args.out / "stability.json")],
             stability)
    if "typesafe102" in pred:
        step("typesafe102", [py, str(bench / "evaluate_external.py"), "--source", "typesafe",
                             "--gold", str(args.eval / "typesafe102.jsonl"),
                             "--direct", str(pred["typesafe102"]),
                             "--reranker", str(pred["typesafe102"])],
             lambda r: r.get("direct", r), stdout=True)
    if "every204" in pred:
        step("every204", [py, str(bench / "evaluate_external.py"), "--source", "every",
                          "--gold", str(args.eval / "gold154.jsonl"),
                          "--inference", str(args.eval / "inference204.jsonl"),
                          "--firewall-actions", str(args.eval / "firewall-actions.json"),
                          "--direct", str(pred["every204"]),
                          "--reranker", str(pred["every204"])],
             lambda r: r.get("direct", r), stdout=True)
    (args.out / "summary.json").write_text(json.dumps({"summary": summary, "errors": errors},
                                                      indent=1) + "\n")
    print(json.dumps({"summary": summary, "errors": list(errors)}))


if __name__ == "__main__":
    main()
