"""Training mixture and compiled inputs for the student model (PLAN step 3).

Three gold sources, sampled by component so derived questions cannot dominate:
  product  – six-task rows built from components no evaluation split uses
             (`product_eval train-split`), injection wording disjoint from evaluation;
  policy   – policy-transfer questions on real text (synthetic rules, exact interpreter);
  public   – prepared public decisions (NLI, sentiment, topic, intent, QA, STS/HelpSteer
             means), capped per task.
Rows whose text keys appear anywhere in the evaluation are dropped. Choice, Noul and Score
all stay: Score means keep `score_target` for an expected-level loss. Tool-call review is
never trained, so it measures transfer to an unseen task.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from bobcat.corpus import atomic_json
from bobcat.protocol import parse_request
from bobcat.schema import file_hash, json_hash
from bobcat.student_readout import StudentCompiler, identifier_scheme

PUBLIC_CAPS = {
    "kornli_multinli": 1500, "kornli_snli_1": 800, "nsmc": 2000, "klue_nli": 2500,
    "klue_ynat": 2000, "klue_sts": 1500, "helpsteer2": 1200, "helpsteer3": 400,
    "massive_ko-KR": 1200, "massive_en-US": 1200, "banking77": 1500, "boolq": 1500,
    "arc_easy": 800, "arc_challenge": 500,
}
POLICY_ROWS = 8000
MONITOR_SHARE = 0.03


def order(value: str, salt: str) -> str:
    return json_hash([salt, value])


def eval_text_keys(eval_dir: Path) -> set[str]:
    keys = set()
    for split in ("dev", "calibration", "final"):
        for line in (eval_dir / f"{split}.jsonl").open():
            keys.update(json.loads(line)["text_keys"])
    return keys


def one_per_group(rows, cap: int, salt: str, per_group: int = 1) -> list[dict]:
    rows = sorted(rows, key=lambda r: order(r["id"], salt))
    seen, kept = Counter(), []
    for row in rows:
        if seen[row["group_id"]] < per_group:
            seen[row["group_id"]] += 1
            kept.append(row)
            if len(kept) == cap:
                break
    return kept


def build(product: Path, policy: Path, public: Path, eval_dir: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("Outputs are immutable; choose a new path.")
    forbidden = eval_text_keys(eval_dir)
    dropped = Counter()

    def clean(rows, source):
        kept = []
        for row in rows:
            if set(row.get("text_keys", [])) & forbidden:
                dropped[source] += 1
                continue
            kept.append({**row, "mixture_source": source})
        return kept

    product_rows = clean([json.loads(line) for line in product.open()], "product")
    policy_rows = clean(one_per_group(map(json.loads, policy.open()), POLICY_ROWS,
                                      "policy", per_group=2), "policy")
    by_task = defaultdict(list)
    for line in public.open():
        row = json.loads(line)
        if row["task"] in PUBLIC_CAPS:
            by_task[row["task"]].append(row)
    public_rows = []
    for task, cap in PUBLIC_CAPS.items():
        public_rows += clean(one_per_group(by_task[task], cap, f"public:{task}"), "public")
    rows = product_rows + policy_rows + public_rows
    groups = sorted({r["group_id"] for r in rows}, key=lambda g: order(g, "monitor"))
    monitor_groups = set(groups[: round(len(groups) * MONITOR_SHARE)])
    out.mkdir(parents=True)
    counts = {}
    for name, keep in (("train", lambda r: r["group_id"] not in monitor_groups),
                       ("monitor", lambda r: r["group_id"] in monitor_groups)):
        chosen = sorted((r for r in rows if keep(r)), key=lambda r: order(r["id"], name))
        with (out / f"{name}.jsonl").open("x") as stream:
            for row in chosen:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = {
            "rows": len(chosen), "components": len({r["group_id"] for r in chosen}),
            "by_source": dict(Counter(r["mixture_source"] for r in chosen)),
            "by_task": dict(sorted(Counter(r["task"] for r in chosen).items())),
            "by_kind": dict(Counter(r["kind"] for r in chosen)),
            "by_supervision": dict(Counter(r["supervision"] for r in chosen)),
            "by_language": dict(Counter(r["language"] for r in chosen)),
            "sha256": file_hash(out / f"{name}.jsonl"),
        }
    manifest = {
        "schema": "bobcat-student-mixture-v1", "public_caps": PUBLIC_CAPS,
        "policy_rows": POLICY_ROWS, "monitor_share": MONITOR_SHARE,
        "inputs": {"product": file_hash(product), "policy": file_hash(policy),
                   "public": file_hash(public)},
        "dropped_for_eval_text_overlap": dict(dropped), "splits": counts,
        "held_out_task": "product_tool_call", "teacher_outputs_used": False,
        "jev_outputs_used": False,
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def compile_rows(rows_path: Path, model_dir: Path, identifiers_path: Path, out: Path,
                 max_tokens: int) -> dict:
    """Piecewise-compiled training inputs: IDs, offered identifiers, candidate ends, targets."""
    receipt = json.loads((model_dir / "bobcat-download.json").read_text())
    glm = json.loads(identifiers_path.read_text())["identifiers"]
    from tokenizers import Tokenizer

    host = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    reserved = {t["id"] for t in json.loads((model_dir / "tokenizer.json").read_text())
                .get("added_tokens", [])}
    identifiers = identifier_scheme(host, reserved, glm)
    compiler = StudentCompiler(model_dir, receipt["files"], identifiers,
                               max_branch_tokens=max_tokens, piecewise=True)
    written, skipped, tokens = 0, Counter(), 0
    with out.open("x") as stream:
        for line in rows_path.open():
            row = json.loads(line)
            state, questions = parse_request(row["request"])
            if list(questions[0].labels) != row["candidate_ids"]:
                raise RuntimeError(f"Presented candidate order differs from the row: {row['id']}")
            try:
                sequence, options, ends = compiler.compile_detailed(state, questions[0])
            except Exception as error:  # over-long or unencodable rows are skipped, counted
                skipped[type(error).__name__] += 1
                continue
            target = (row["candidate_ids"].index(row["target"])
                      if row["supervision"] == "hard_label" else None)
            stream.write(json.dumps({
                "id": row["id"], "group_id": row["group_id"], "task": row["task"],
                "family": row.get("family"), "kind": row["kind"], "language": row["language"],
                "source": row.get("mixture_source"), "supervision": row["supervision"],
                "target": target, "score_target": row.get("score_target"),
                "input_ids": sequence, "option_ids": options, "candidate_ends": ends,
                "input_sha256": json_hash(sequence),
            }) + "\n")
            written += 1
            tokens += len(sequence)
    return {"rows": written, "skipped": dict(skipped), "tokens": tokens,
            "sha256": file_hash(out), "identifiers_sha256": json_hash(identifiers),
            "template_sha256": compiler.template_sha256, "repo": receipt["repo"],
            "revision": receipt["revision"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    mix = sub.add_parser("build")
    for name in ("product", "policy", "public", "eval-dir", "out"):
        mix.add_argument(f"--{name}", type=Path, required=True)
    comp = sub.add_parser("compile")
    for name in ("rows", "model-dir", "out"):
        comp.add_argument(f"--{name}", type=Path, required=True)
    comp.add_argument("--identifiers", type=Path,
                      default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    comp.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "build":
        manifest = build(args.product, args.policy, args.public, args.eval_dir, args.out)
        print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    else:
        meta = compile_rows(args.rows, args.model_dir, args.identifiers, args.out,
                            args.max_tokens)
        (args.out.parent / f"{args.out.stem}.compile.json").write_text(
            json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
