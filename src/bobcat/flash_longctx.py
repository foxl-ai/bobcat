"""Bobcat Flash long-context continuation data (2026-09-26).

Flash 1.1 loses accuracy as unrelated text fills the state (dev: -5.3 pt at 8K, -7.3 pt at
30K; the untrained Gemma base loses the same), and its corpus had no rows above 8,192 tokens.
This builds the continuation data of the pre-registered arm
(.aws-local/flashlc-20260926-preregistration.json), mirroring the Bobcat 1.1 long-context rows:

  build        padded copies of Flash corpus rows from the Bobcat 1 mixture
               (`bobcat.student_data_v11.long_copy`: the original state unchanged; padding
               from OTHER training components only, KLUE MRC training contexts for Korean,
               BoolQ / HelpSteer2 for English; training padding keys and layouts, never the
               evaluation's `기타 자료 1/2`; compiled length fitted to the Flash tokenizer at
               4K-30K), plus a replay sample of short corpus rows. Writes long.jsonl,
               replay.jsonl, sources.jsonl (the unpadded source rows, for the teacher) and
               manifest.json. Tool-call rows never exist (held-out task).
  teacher-map  teacher logits keyed by training row: a padded copy takes the teacher's
               distribution on its UNPADDED source question (same candidates, same order);
               a replay row its own.
  audit        text keys of the training rows (padding passages included) against
               evaluation splits and sealed finals: must be empty.

The teacher is Bobcat 1.1 r16semif only (bobcat.flash_teacher on its merged weights).
Nothing here reads the evaluation rows except to exclude their text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bobcat.schema import file_hash, json_hash

PLAN = {
    "product": {"product_search": 350, "product_citation": 250, "product_injection": 300,
                "product_routing": 250, "product_classification": 350},
    "public": {"boolq": 120, "banking77": 120, "massive_en-US": 80, "massive_ko-KR": 80,
               "helpsteer3": 60, "arc_easy": 30, "arc_challenge": 30, "nsmc": 60,
               "kornli_multinli": 70},
    "policy": 250,
}
LEVELS = (4096, 6144, 8192, 12288, 16384, 24576, 30720)
REPLAY = 4000
MIXTURE = "bobcat1_mixture"
FORBIDDEN_TASK_WORDS = ("tool_call", "tool-call", "toolcall")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def replay_rows(rows: list[dict], count: int, seed: int, exclude: set[str]) -> list[dict]:
    """Half from the Bobcat 1 mixture rows, half from the rest of the corpus, each a seeded
    uniform sample (so the corpus proportions are kept within each half)."""
    halves = ([r for r in rows if r.get("flash_source") == MIXTURE and r["id"] not in exclude],
              [r for r in rows if r.get("flash_source") != MIXTURE and r["id"] not in exclude])
    picked = []
    for index, half in enumerate(halves):
        want = count // 2 + (count % 2 if index else 0)
        ordered = sorted(half, key=lambda r: json_hash([seed, "replay", r["id"]]))
        picked += ordered[:want]
    return picked


def forbidden_keys(paths: list[Path]) -> set[str]:
    keys = set()
    for path in paths:
        for line in path.open():
            keys.update(json.loads(line).get("text_keys", []))
    return keys


def build(corpus: Path, model_dir: Path, identifiers: Path, out: Path, *, seed: int,
          workers: int, plan: dict = PLAN, levels: tuple = LEVELS, replay: int = REPLAY,
          passes: int = 2, forbidden: tuple[Path, ...] = ()) -> dict:
    """Rows whose text meets `forbidden` (evaluation splits, sealed finals) are left out, as
    source rows, replay rows, padding passages (their copies) alike."""
    from bobcat.student_data_v11 import long_copies, long_plan

    if out.exists():
        raise ValueError("Outputs are immutable; choose a new directory.")
    rows = read_jsonl(corpus)
    if any(any(w in r["task"] for w in FORBIDDEN_TASK_WORDS) for r in rows):
        raise ValueError("The corpus holds tool-call rows; the task must stay held out.")
    keys = forbidden_keys(list(forbidden))
    meets = {r["id"] for r in rows if set(r.get("text_keys", [])) & keys}
    rows = [r for r in rows if r["id"] not in meets]
    mixture = [r for r in rows if r.get("flash_source") == MIXTURE]
    copies = long_copies(mixture, seed, model_dir, identifiers, workers, mixture, plan,
                         tuple(levels), passes)
    padded_meeting = [c["id"] for c in copies if set(c.get("text_keys", [])) & keys]
    copies = [c for c in copies if c["id"] not in set(padded_meeting)]
    for copy in copies:
        copy["flash_source"] = "long_context"
    by_id = {r["id"]: r for r in mixture}
    sources = {c["derived"]["from"]: by_id[c["derived"]["from"]] for c in copies}
    short = replay_rows(rows, replay, seed, set())
    out.mkdir(parents=True)
    write_jsonl(out / "long.jsonl", copies)
    write_jsonl(out / "replay.jsonl", short)
    write_jsonl(out / "sources.jsonl", sorted(sources.values(), key=lambda r: r["id"]))
    ids = Counter(r["id"] for r in copies + short)
    if any(n > 1 for n in ids.values()):
        raise ValueError("Duplicate training row IDs.")
    manifest = {
        "schema": "bobcat-flash-longctx-data-v1", "seed": seed, "plan": plan,
        "levels": list(levels), "passes": passes, "replay": replay,
        "planned_long_rows": len(long_plan(mixture, plan, passes)),
        "forbidden": {"files": {p.name: file_hash(p) for p in forbidden},
                      "text_keys": len(keys), "corpus_rows_left_out": len(meets),
                      "padded_copies_left_out": len(padded_meeting)},
        "corpus_sha256": file_hash(corpus), "generator_sha256": file_hash(Path(__file__)),
        "long": describe(copies) | {
            "levels": dict(sorted(Counter(c["derived"]["level"] for c in copies).items())),
            "tokens": {"sum": sum(c["derived"]["tokens"] for c in copies),
                       "max": max((c["derived"]["tokens"] for c in copies), default=0)},
            "padding_language": dict(Counter(c["derived"]["language"] for c in copies)),
            "layout": dict(Counter(c["derived"]["layout"] for c in copies)),
            "sha256": file_hash(out / "long.jsonl")},
        "replay_rows": describe(short) | {"sha256": file_hash(out / "replay.jsonl")},
        "sources": {"rows": len(sources), "sha256": file_hash(out / "sources.jsonl")},
        "teacher": "Bobcat 1.1 r16semif only; padded copies take the teacher on the unpadded "
                   "source question",
        "held_out_task": "product_tool_call (never built)", "jev_outputs_used": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    return manifest


def more_replay(corpus: Path, out: Path, *, count: int, seed: int, exclude: list[Path],
                forbidden: tuple[Path, ...] = ()) -> dict:
    """Another replay sample (the pre-registered lc2b doubles the replay): new rows only,
    the same halves and the same evaluation-text exclusion as `build`."""
    if out.exists():
        raise ValueError("Outputs are immutable; choose a new path.")
    keys = forbidden_keys(list(forbidden))
    taken = {json.loads(line)["id"] for path in exclude for line in path.open()}
    rows = [r for r in read_jsonl(corpus) if not set(r.get("text_keys", [])) & keys]
    picked = replay_rows(rows, count, seed, taken)
    write_jsonl(out, picked)
    return describe(picked) | {"sha256": file_hash(out), "seed": seed,
                               "excluded_rows": len(taken)}


def describe(rows: list[dict]) -> dict:
    return {"rows": len(rows), "components": len({r["group_id"] for r in rows}),
            "by_flash_source": dict(Counter(r.get("flash_source") for r in rows)),
            "by_task": dict(sorted(Counter(r["task"] for r in rows).items())),
            "by_language": dict(Counter(r["language"] for r in rows)),
            "by_supervision": dict(Counter(r["supervision"] for r in rows))}


def teacher_map(long_rows: Path, teacher_paths: list[Path], out: Path) -> dict:
    """Teacher logits for every training row: padded copies inherit their source's."""
    teacher = {}
    for path in teacher_paths:
        for line in path.open():
            record = json.loads(line)
            teacher[record["id"]] = record["logits"]
    written, missing = 0, 0
    with out.open("x") as sink:
        for line in long_rows.open():
            row = json.loads(line)
            logits = teacher.get(row["derived"]["from"])
            if logits is None or len(logits) != len(row["candidate_ids"]):
                missing += 1
                continue
            sink.write(json.dumps({"id": row["id"], "logits": logits,
                                   "from": row["derived"]["from"]}) + "\n")
            written += 1
        for row_id, logits in teacher.items():
            sink.write(json.dumps({"id": row_id, "logits": logits}) + "\n")
            written += 1
    return {"rows": written, "long_rows_without_teacher": missing, "sha256": file_hash(out)}


def audit(paths: list[Path], forbidden_files: list[Path]) -> dict:
    """Training text keys (padding passages included) that meet evaluation or final text."""
    forbidden = set()
    for path in forbidden_files:
        for line in path.open():
            forbidden.update(json.loads(line).get("text_keys", []))
    hits, rows = [], 0
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            rows += 1
            if set(row.get("text_keys", [])) & forbidden:
                hits.append(row["id"])
    return {"rows": rows, "forbidden_text_keys": len(forbidden), "overlapping_rows": len(hits),
            "examples": hits[:20], "files": {p.name: file_hash(p) for p in forbidden_files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--corpus", type=Path, required=True, help="Flash corpus train.jsonl")
    b.add_argument("--model-dir", type=Path, required=True, help="Flash base (tokenizer)")
    b.add_argument("--identifiers", type=Path,
                   default=Path("reports/2026-09-22-glm-readout-preflight.json"))
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=2026092631)
    b.add_argument("--workers", type=int, default=64)
    b.add_argument("--plan", type=json.loads)
    b.add_argument("--levels", help="comma-separated Flash-token targets")
    b.add_argument("--replay", type=int, default=REPLAY)
    b.add_argument("--forbidden", type=Path, action="append", default=[],
                   help="evaluation / final JSONL: rows meeting their text_keys are left out")
    t = sub.add_parser("teacher-map")
    t.add_argument("--long", type=Path, required=True)
    t.add_argument("--teacher", type=Path, action="append", required=True)
    t.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("replay")
    r.add_argument("--corpus", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--count", type=int, default=REPLAY)
    r.add_argument("--seed", type=int, default=2026092632)
    r.add_argument("--exclude", type=Path, action="append", default=[],
                   help="rows already used (their IDs are not drawn again)")
    r.add_argument("--forbidden", type=Path, action="append", default=[])
    a = sub.add_parser("audit")
    a.add_argument("--rows", type=Path, action="append", required=True)
    a.add_argument("--forbidden", type=Path, action="append", required=True,
                   help="evaluation / final JSONL whose text_keys must not appear")
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        levels = tuple(int(x) for x in args.levels.split(",")) if args.levels else LEVELS
        manifest = build(args.corpus, args.model_dir, args.identifiers, args.out,
                         seed=args.seed, workers=args.workers, plan=args.plan or PLAN,
                         levels=levels, replay=args.replay,
                         forbidden=tuple(args.forbidden))
        print(json.dumps({k: manifest[k] for k in ("planned_long_rows", "sources")}
                         | {"long": manifest["long"]["rows"],
                            "levels": manifest["long"]["levels"],
                            "replay": manifest["replay_rows"]["rows"]}))
    elif args.command == "replay":
        print(json.dumps(more_replay(args.corpus, args.out, count=args.count, seed=args.seed,
                                     exclude=args.exclude, forbidden=tuple(args.forbidden))))
    elif args.command == "teacher-map":
        print(json.dumps(teacher_map(args.long, args.teacher, args.out)))
    else:
        result = audit(args.rows, args.forbidden)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("rows", "overlapping_rows")}))
        if result["overlapping_rows"]:
            raise SystemExit("training text meets evaluation text")


if __name__ == "__main__":
    main()

