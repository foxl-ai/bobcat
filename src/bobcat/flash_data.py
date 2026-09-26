"""Bobcat Flash distillation corpus and compiled inputs (2026-09-26).

`build` writes one component-disjoint row set from
  mixture    the Bobcat 1 training mixture (gold; product, policy, public), unchanged;
  expanded   extra prepared public decisions (gold), sampled per task by a row hash, at most
             two rows per component, never a row the mixture already has;
  policy     extra policy-transfer questions (gold, exact interpreter);
  requestion new question wording / candidate sets over sampled public states
             (`flash_families.requestion`; gold where the source label decides it);
  generated  games and workflow checks (`flash_families`; gold from the generator rule, or
             teacher-only for semantic families).
Every row whose text keys meet any evaluation split is dropped, and the tool-call review
task never appears. Teacher (Bobcat 1) probabilities are added later by `flash_teacher`;
gold stays gold. `compile` turns rows into token IDs for one tokenizer (piecewise compile,
the serving contract) in parallel, keeping the row ID so any tokenizer's rows align with the
teacher's logits.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

from bobcat import flash_families as fam
from bobcat.corpus import atomic_json
from bobcat.schema import file_hash, json_hash

EXPANDED_CAPS = {
    "kornli_multinli": 30000, "kornli_snli_1": 12000, "klue_nli": 20000, "nsmc": 24000,
    "klue_ynat": 20000, "klue_sts": 9000, "helpsteer2": 9000, "helpsteer3": 12000,
    "massive_ko-KR": 10000, "massive_en-US": 10000, "banking77": 9500, "boolq": 7500,
    "arc_easy": 2100, "arc_challenge": 1060,
}
REQUESTION_TASKS = {"kornli_multinli": 14000, "kornli_snli_1": 6000, "klue_nli": 10000,
                    "nsmc": 8000, "klue_ynat": 10000, "massive_en-US": 5000,
                    "massive_ko-KR": 5000, "banking77": 4000, "boolq": 4000}
POLICY_CAP = 40000
GENERATED = {  # family -> rows-producing calls per language
    "tictactoe": 4000, "gridworld": 3000, "shooter": 5000, "linkrace": 2500,
    "invoice": 3500, "security": 3500,
}
TICKETS = 9000
MONITOR_SHARE = 0.03
FORBIDDEN_TASK_WORDS = ("tool_call", "tool-call", "toolcall")


def hash_unit(*parts) -> float:
    return int(json_hash(list(parts))[:12], 16) / float(16 ** 12)


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def eval_keys(paths: list[Path]) -> set[str]:
    keys = set()
    for path in paths:
        for row in read_jsonl(path):
            keys.update(row.get("text_keys", []))
    return keys


def title_pools(rows: list[dict]) -> dict[str, list[str]]:
    """Page-like titles for the link-choice game: Korean news headlines and the lead noun
    phrase of English BoolQ passages (Wikipedia pages)."""
    ko, en = set(), set()
    for row in rows:
        state = row["request"]["state"]
        if row["task"] == "klue_ynat":
            text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
            try:
                title = json.loads(text).get("뉴스 제목")
            except (json.JSONDecodeError, AttributeError):
                title = None
            if title and len(title) <= 40:
                ko.add(title)
        elif row["task"] == "boolq" and isinstance(state, dict):
            passage = state.get("passage", "")
            lead = passage.split(" (")[0].split(" is ")[0].split(" was ")[0].strip()
            if 2 <= len(lead) <= 40 and len(lead.split()) <= 5:
                en.add(lead)
    return {"ko": sorted(ko), "en": sorted(en)}


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def strings(value, found: set[str]) -> set[str]:
    if isinstance(value, str):
        if len(value) >= 16:
            found.add(norm(value))
        stripped = value.strip()
        if stripped[:1] in "{[":
            try:
                strings(json.loads(stripped), found)
            except json.JSONDecodeError:
                pass
    elif isinstance(value, dict):
        for item in value.values():
            strings(item, found)
    elif isinstance(value, list):
        for item in value:
            strings(item, found)
    return found


def external_eval_strings(paths: list[Path]) -> set[str]:
    """Every string (>= 16 chars, normalised) in the TypeSafe / SemIf / Every evaluation
    requests and rows; any corpus row containing one is dropped."""
    found: set[str] = set()
    for path in paths:
        if path.suffix == ".json":
            strings(json.loads(path.read_text()), found)
            continue
        for record in read_jsonl(path):
            strings(record, found)
    return found


ENGLISH_NLI = {"multi_nli": 24000, "snli": 10000}
NLI_LABELS = {0: "supported", 1: "insufficient", 2: "contradicted"}


def english_nli_rows(folder: Path, scale: float, forbid: set[str]) -> list[dict]:
    """English MultiNLI / SNLI training pairs as evidence / claim decisions (gold labels)."""
    import pyarrow.parquet as pq

    out = []
    for name, cap in ENGLISH_NLI.items():
        table = pq.read_table(folder / f"{name}-train.parquet",
                              columns=["premise", "hypothesis", "label"]).to_pylist()
        order = sorted(range(len(table)), key=lambda i: json_hash(["flash-en-nli", name, i]))
        seen_premise, taken = set(), 0
        for i in order:
            if taken >= round(cap * scale):
                break
            r = table[i]
            if r["label"] not in NLI_LABELS or not r["premise"] or not r["hypothesis"]:
                continue
            premise = norm(r["premise"])
            if premise in seen_premise or premise in forbid or norm(r["hypothesis"]) in forbid:
                continue
            seen_premise.add(premise)
            rng = fam.seeded("en-nli", name, i)
            group = f"flash:en-nli:{name}:{json_hash(premise)[:20]}"
            state = {"evidence": r["premise"], "claim": r["hypothesis"]}
            target = NLI_LABELS[r["label"]]
            if rng.random() < 0.7:
                question = {"type": "choice", "instructions": rng.choice([
                    "Judge the claim only from the evidence.",
                    "Does the evidence establish, rule out, or leave open the claim?",
                    "Using only the evidence, classify the claim. Choose insufficient when the "
                    "evidence does not settle it."]),
                    "criteria": {k: fam.NLI_EN[k] for k in fam.shuffled(rng, fam.NLI_EN)}}
                made = fam.row(family="english_nli_evidence", task="flash_english_nli",
                               lang="en", state=state, qid="verdict", question=question,
                               target=target, group=group, basis=f"{name}_label")
            else:
                probe = rng.choice(list(NLI_LABELS.values()))
                text = {"supported": "Does the evidence establish the claim?",
                        "insufficient": "Is the evidence insufficient to decide the claim?",
                        "contradicted": "Does the evidence rule the claim out?"}[probe]
                made = fam.row(family="english_nli_noul", task="flash_english_nli", lang="en",
                               state=state, qid="check", question={
                                   "type": "noul", "instructions": text, "criteria": None},
                               target="yes" if probe == target else "no", group=group,
                               basis=f"{name}_label")
            made["source"] = {"dataset": name, "index": i}
            out.append(made)
            taken += 1
    return out


def build(mixture: Path, expanded: Path, policy: Path, evals: list[Path], out: Path,
          scale: float = 1.0, english_nli: Path | None = None,
          external: list[Path] | None = None) -> dict:
    if out.exists():
        raise ValueError("Outputs are immutable; choose a new path.")
    forbidden = eval_keys(evals)
    forbid_strings = external_eval_strings(external or [])
    dropped = Counter()
    rows: list[dict] = []
    seen_ids: set[str] = set()

    def add(row: dict, source: str) -> bool:
        if any(word in row["task"] for word in FORBIDDEN_TASK_WORDS):
            dropped[f"{source}:tool_call"] += 1
            return False
        if set(row.get("text_keys", [])) & forbidden:
            dropped[f"{source}:eval_text_overlap"] += 1
            return False
        if forbid_strings and strings(row["request"]["state"], set()) & forbid_strings:
            dropped[f"{source}:external_eval_text"] += 1
            return False
        if row["id"] in seen_ids:
            dropped[f"{source}:duplicate_id"] += 1
            return False
        seen_ids.add(row["id"])
        rows.append({**row, "flash_source": source})
        return True

    for row in read_jsonl(mixture / "train.jsonl"):
        add(row, "bobcat1_mixture")
    mixture_ids = set(seen_ids)

    counts = Counter(r["task"] for r in read_jsonl(expanded))
    per_group, picked, pool = Counter(), Counter(), defaultdict(list)
    for row in read_jsonl(expanded):
        task = row["task"]
        cap = round(EXPANDED_CAPS.get(task, 0) * scale)
        want = round(REQUESTION_TASKS.get(task, 0) * scale)
        if not cap and not want:
            continue
        # A state may carry both its original question and a re-asked one (same component).
        if want and len(pool[task]) < want and hash_unit("flash-requestion", row["id"]) \
                < 1.3 * want / counts[task]:
            if not set(row.get("text_keys", [])) & forbidden:
                pool[task].append(row)
        if row["id"] in mixture_ids or per_group[row["group_id"]] >= 2:
            continue
        unit = hash_unit("flash-expanded", row["id"])
        if cap and picked[task] < cap and unit < 1.3 * cap / counts[task]:
            if add(row, "expanded"):
                picked[task] += 1
                per_group[row["group_id"]] += 1

    policy_group = Counter()
    policy_rows = sorted(read_jsonl(policy), key=lambda r: json_hash(["flash-policy", r["id"]]))
    taken = 0
    for row in policy_rows:
        if taken >= round(POLICY_CAP * scale):
            break
        if row["id"] in mixture_ids or policy_group[row["group_id"]] >= 3:
            continue
        if add(row, "policy"):
            policy_group[row["group_id"]] += 1
            taken += 1

    requestioned = Counter()
    for task, sources in sorted(pool.items()):
        for source in sources:
            if task == "klue_nli":
                label = (source.get("source") or {}).get("source_label")
                source = {**source, "target": {0: "entailment", 1: "neutral",
                                               2: "contradiction"}.get(label)}
            for new in fam.requestion(source):
                if add(new, "requestion"):
                    requestioned[new["family"]] += 1

    tickets = [r for r in pool.get("banking77", []) + pool.get("massive_ko-KR", [])
               + pool.get("massive_en-US", [])]
    ticket_rows = 0
    for index, source in enumerate(sorted(tickets, key=lambda r: json_hash(["ticket", r["id"]]))):
        if ticket_rows >= round(TICKETS * scale):
            break
        state = source["request"]["state"]
        text = (state.get("customer_message") or state.get("utterance")) \
            if isinstance(state, dict) else None
        if not text:
            continue
        ticket = {"id": source["id"], "language": source["language"], "task": source["task"],
                  "text": text, "label": source.get("target"), "group_id": source["group_id"],
                  "text_keys": source.get("text_keys", [])}
        for new in fam.ticket(index, ticket):
            add(new, "generated")
        ticket_rows += 1

    english = Counter()
    if english_nli is not None:
        for new in english_nli_rows(english_nli, scale, forbid_strings):
            if add(new, "english_nli"):
                english[new["family"]] += 1

    titles = title_pools(rows)
    generated = Counter()
    for family, count in GENERATED.items():
        make = getattr(fam, family)
        for lang in ("en", "ko"):
            for index in range(round(count * scale)):
                made = (make(index, lang, titles[lang]) if family == "linkrace"
                        else make(index, lang))
                for new in made:
                    if add(new, "generated"):
                        generated[new["family"]] += 1

    groups = sorted({r["group_id"] for r in rows if r["flash_source"] != "bobcat1_mixture"},
                    key=lambda g: json_hash(["flash-monitor", g]))
    monitor_groups = set(groups[: round(len(groups) * MONITOR_SHARE)])
    out.mkdir(parents=True)
    splits = {}
    for name, keep in (("train", lambda r: r["group_id"] not in monitor_groups),
                       ("monitor", lambda r: r["group_id"] in monitor_groups)):
        chosen = sorted((r for r in rows if keep(r)), key=lambda r: json_hash([name, r["id"]]))
        with (out / f"{name}.jsonl").open("x") as stream:
            for row in chosen:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        splits[name] = {
            "rows": len(chosen), "components": len({r["group_id"] for r in chosen}),
            "by_source": dict(Counter(r["flash_source"] for r in chosen)),
            "by_task": dict(sorted(Counter(r["task"] for r in chosen).items())),
            "by_family": dict(sorted(Counter(r["family"] for r in chosen).items())),
            "by_kind": dict(Counter(r["kind"] for r in chosen)),
            "by_supervision": dict(Counter(r["supervision"] for r in chosen)),
            "by_language": dict(Counter(r["language"] for r in chosen)),
            "sha256": file_hash(out / f"{name}.jsonl"),
        }
    manifest = {
        "schema": "bobcat-flash-corpus-v1", "scale": scale, "expanded_caps": EXPANDED_CAPS,
        "requestion_tasks": REQUESTION_TASKS, "policy_cap": POLICY_CAP, "generated": GENERATED,
        "tickets": TICKETS, "monitor_share": MONITOR_SHARE,
        "inputs": {"mixture_train": file_hash(mixture / "train.jsonl"),
                   "expanded": file_hash(expanded), "policy": file_hash(policy),
                   "evaluation_files": {p.name: file_hash(p) for p in evals}},
        "forbidden_text_keys": len(forbidden), "dropped": dict(dropped),
        "expanded_picked": dict(picked), "requestioned": dict(requestioned),
        "generated_rows": dict(generated), "english_nli_rows": dict(english),
        "english_nli_caps": ENGLISH_NLI, "external_eval_strings": len(forbid_strings),
        "external_eval_files": {p.name: file_hash(p) for p in external or []},
        "link_titles": {k: len(v) for k, v in titles.items()},
        "splits": splits, "held_out_task": "product_tool_call (never built)",
        "teacher_outputs_used": "added by flash_teacher; gold labels unchanged",
        "jev_outputs_used": False, "typesafe_or_semif_rows_used": False,
        "generator_sha256": {"flash_data": file_hash(Path(__file__)),
                             "flash_families": file_hash(Path(fam.__file__))},
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


# -------------------------------------------------------------------------- compile

_COMPILER = None


def _init(model_dir: str, identifiers_path: str, max_tokens: int) -> None:
    global _COMPILER
    from tokenizers import Tokenizer

    from bobcat.student_readout import StudentCompiler, identifier_scheme

    model = Path(model_dir)
    receipt = json.loads((model / "bobcat-download.json").read_text())
    glm = json.loads(Path(identifiers_path).read_text())["identifiers"]
    host = Tokenizer.from_file(str(model / "tokenizer.json"))
    reserved = {t["id"] for t in json.loads((model / "tokenizer.json").read_text())
                .get("added_tokens", [])}
    identifiers = identifier_scheme(host, reserved, glm)
    _COMPILER = StudentCompiler(model, receipt["files"], identifiers,
                                max_branch_tokens=max_tokens, piecewise=True)


def _compile(line: str):
    from bobcat.protocol import parse_request

    row = json.loads(line)
    state, questions = parse_request(row["request"])
    if list(questions[0].labels) != row["candidate_ids"]:
        return {"error": "candidate_order", "id": row["id"]}
    try:
        sequence, options, _ = _COMPILER.compile_detailed(state, questions[0])
    except Exception as error:  # over-long or unencodable: counted, never truncated
        return {"error": type(error).__name__, "id": row["id"]}
    target = (row["candidate_ids"].index(row["target"])
              if row["supervision"] == "hard_label" else None)
    return {"id": row["id"], "group_id": row["group_id"], "task": row["task"],
            "family": row.get("family"), "kind": row["kind"], "language": row["language"],
            "source": row.get("flash_source") or row.get("mixture_source"),
            "supervision": row["supervision"], "target": target,
            "score_target": row.get("score_target"), "input_ids": sequence,
            "option_ids": options}


def compile_rows(rows_path: Path, model_dir: Path, identifiers: Path, out: Path,
                 max_tokens: int, workers: int) -> dict:
    written, skipped, tokens = 0, Counter(), 0
    with rows_path.open() as source, out.open("x") as sink, mp.get_context("fork").Pool(
            workers, initializer=_init,
            initargs=(str(model_dir), str(identifiers), max_tokens)) as pool:
        for record in pool.imap(_compile, source, chunksize=64):
            if "error" in record:
                skipped[record["error"]] += 1
                continue
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            tokens += len(record["input_ids"])
    _init(str(model_dir), str(identifiers), max_tokens)
    receipt = json.loads((model_dir / "bobcat-download.json").read_text())
    meta = {"rows": written, "skipped": dict(skipped), "tokens": tokens,
            "sha256": file_hash(out), "source_rows_sha256": file_hash(rows_path),
            "identifiers_sha256": json_hash(_COMPILER.identifiers),
            "template_sha256": _COMPILER.template_sha256, "repo": receipt["repo"],
            "revision": receipt["revision"], "max_tokens": max_tokens}
    atomic_json(out.with_suffix(".compile.json"), meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--mixture", type=Path, required=True, help="Bobcat 1 mixture folder")
    b.add_argument("--expanded", type=Path, required=True)
    b.add_argument("--policy", type=Path, required=True)
    b.add_argument("--eval", type=Path, action="append", required=True,
                   help="every evaluation split whose text keys are forbidden")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--english-nli", type=Path, help="folder with multi_nli / snli train parquet")
    b.add_argument("--external-eval", type=Path, action="append", default=[],
                   help="TypeSafe / SemIf / Every request or row files; their text is excluded")
    c = sub.add_parser("compile")
    c.add_argument("--rows", type=Path, required=True)
    c.add_argument("--model-dir", type=Path, required=True)
    c.add_argument("--identifiers", type=Path,
                   default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--max-tokens", type=int, default=8192)
    c.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.command == "build":
        manifest = build(args.mixture, args.expanded, args.policy, args.eval, args.out,
                         args.scale, args.english_nli, args.external_eval)
        print(json.dumps(manifest["splits"], ensure_ascii=False, indent=1))
    else:
        meta = compile_rows(args.rows, args.model_dir, args.identifiers, args.out,
                            args.max_tokens, args.workers)
        print(json.dumps(meta))


if __name__ == "__main__":
    main()
