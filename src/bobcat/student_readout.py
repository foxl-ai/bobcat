"""Zero-shot first-position readout for student candidates (PLAN step 2, SPEC B3).

The prompt contract mirrors `glm_readout.GLMCompiler`: the same system text, the same
compact JSON of state, question type, instructions and candidate meanings, and the
same 255 identifier strings the GLM B0 readout verified. Only the chat template and
tokenizer change. Data text is encoded with every added-token matcher disabled, so a
state cannot emit the model's control tokens. The next-token logits at the first
assistant position are read for the offered identifiers; nothing is generated.

This measures an untrained student. It is not a calibrated Bobcat model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tokenizers import Tokenizer

from bobcat.glm_readout import SYSTEM
from bobcat.metrics import basic_metrics, grouped_interval, scored_row
from bobcat.protocol import Question, RequestLimitError, parse_request
from bobcat.schema import file_hash, json_hash

PROFILE = "student_first_position_glm_contract_v1"
STATE_MARKER = "BOBCAT_HOST_STATE_INSERTION_41891"
QUESTION_MARKER = "BOBCAT_HOST_QUESTION_INSERTION_91732"
PINNED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
          "encoding/encoding.py")


def chat_template(model_dir: Path) -> str:
    path = model_dir / "chat_template.jinja"
    if path.exists():
        return path.read_text()
    template = json.loads((model_dir / "tokenizer_config.json").read_text()).get("chat_template")
    if isinstance(template, list):
        template = next((t["template"] for t in template if t.get("name") == "default"), None)
    if not isinstance(template, str):
        raise ValueError("The student has no usable chat template.")
    return template


def render_prompt(model_dir: Path, messages: list[dict]) -> tuple[str, str]:
    """Render with the model's own prompt format: a Jinja chat template, or the official
    `encoding/encoding.py` reference for models that ship no template (DeepSeek-V4.1)."""
    reference = model_dir / "encoding" / "encoding.py"
    has_template = ((model_dir / "chat_template.jinja").exists() or json.loads(
        (model_dir / "tokenizer_config.json").read_text()).get("chat_template"))
    if reference.exists() and not has_template:
        import importlib.util

        spec = importlib.util.spec_from_file_location("model_prompt_encoding", reference)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # "chat" closes the thinking block, so the next token is the answer.
        return module.encode_messages(messages, thinking_mode="chat"), "encoding.py:chat"
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"])
    environment.globals.update(
        raise_exception=raise_exception,
        strftime_now=lambda fmt: datetime(2026, 9, 24, tzinfo=UTC).strftime(fmt),
    )
    config = json.loads((model_dir / "tokenizer_config.json").read_text())
    tokens = {key: (value if isinstance(value, str) else (value or {}).get("content"))
              for key, value in config.items() if key.endswith("_token")}
    rendered = environment.from_string(chat_template(model_dir)).render(
        messages=messages, add_generation_prompt=True, tools=None, enable_thinking=False,
        **tokens,
    )
    return rendered, "jinja"


EXTENSION = [chr(c) for c in (*range(0x391, 0x3AA), *range(0x3B1, 0x3CA),
                               *range(0x410, 0x450), *range(0x5D0, 0x5EB),
                               *range(0xC0, 0x100), *range(0x531, 0x557),
                               *range(0x10D0, 0x10FB))]


def identifier_scheme(tokenizer: Tokenizer, reserved: set[int], preferred: list[str],
                      count: int = 255) -> list[str]:
    """GLM's identifier strings where they are one ordinary token, then fixed extras.

    Rows whose candidates fit in the shared prefix use exactly GLM's identifiers;
    longer candidate lists fall back to Greek/Cyrillic/Hebrew letters, recorded in
    the summary."""
    chosen, ids = [], set()
    for text in [*preferred, *EXTENSION]:
        encoded = tokenizer.encode(text, add_special_tokens=False).ids
        if (len(encoded) == 1 and encoded[0] not in reserved and encoded[0] not in ids
                and tokenizer.decode(encoded) == text):
            chosen.append(text)
            ids.add(encoded[0])
            if len(chosen) == count:
                return chosen
    raise ValueError(f"Only {len(chosen)} single-token identifiers are available.")


class StudentCompiler:
    def __init__(self, model_dir: Path, pinned: dict, identifiers: list[str], *,
                 max_branch_tokens: int, piecewise: bool = False):
        # piecewise=True encodes each candidate object separately so its last token
        # position is exact (pointer readout); the text stays byte-identical.
        self.piecewise = piecewise
        for name in PINNED:
            if (model_dir / name).exists():
                record = pinned.get(name)
                if not record or record.get("status") != "ok":
                    raise ValueError(f"{name} has no pinned survey record.")
                path = model_dir / name
                if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
                    raise ValueError(f"{name} does not match its pinned survey checksum.")
        if len(identifiers) != len(set(identifiers)) or not identifiers:
            raise ValueError("Identifiers must be distinct.")
        self.host_tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.host_tokenizer.no_padding()
        self.host_tokenizer.no_truncation()
        serialized = json.loads((model_dir / "tokenizer.json").read_text())
        self.reserved_ids = {item["id"] for item in serialized.get("added_tokens", [])}
        # Same boundary as the GLM compiler: data never matches added tokens,
        # including special=False control markers such as <think>.
        serialized["added_tokens"] = []
        serialized["padding"] = serialized["truncation"] = None
        self.data_tokenizer = Tokenizer.from_str(json.dumps(serialized))
        ids = []
        for text in identifiers:
            encoded = self.host_tokenizer.encode(text, add_special_tokens=False).ids
            if (len(encoded) != 1 or encoded[0] in self.reserved_ids or encoded[0] in ids
                    or self.host_tokenizer.decode(encoded) != text):
                raise ValueError(f"Identifier {text!r} is not one distinct ordinary token.")
            ids.append(encoded[0])
        self.identifiers, self.identifier_ids = identifiers, ids
        self.max_branch_tokens = max_branch_tokens

        rendered, self.template_source = render_prompt(model_dir, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'{{"state":{STATE_MARKER},"question":{QUESTION_MARKER}}}'},
        ])
        if rendered.count(STATE_MARKER) != 1 or rendered.count(QUESTION_MARKER) != 1:
            raise ValueError("The chat template changed the insertion boundaries.")
        if SYSTEM not in rendered:
            raise ValueError("The chat template dropped the system instructions.")
        before, remainder = rendered.split(STATE_MARKER)
        between, after = remainder.split(QUESTION_MARKER)
        self.before, self.between, self.after = [
            self.host_tokenizer.encode(part, add_special_tokens=False).ids
            for part in (before, between, after)
        ]
        self.template_sha256 = json_hash(rendered)

    def _data(self, value) -> list[int]:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        ids = self.data_tokenizer.encode(text, add_special_tokens=False).ids
        if any(token in self.reserved_ids for token in ids):
            raise ValueError("Untrusted data emitted a reserved host token.")
        return ids

    def _text(self, text: str) -> list[int]:
        ids = self.data_tokenizer.encode(text, add_special_tokens=False).ids
        if any(token in self.reserved_ids for token in ids):
            raise ValueError("Untrusted data emitted a reserved host token.")
        return ids

    def compile_detailed(self, state, question: Question):
        """(sequence, option_ids, candidate_end_positions) with per-candidate encoding."""
        count = len(question.labels)
        if count > len(self.identifier_ids):
            raise RequestLimitError("Too many choices for verified one-token identifiers.")

        def dumps(value):
            return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

        objects = [dumps({"identifier": self.identifiers[i], "meaning": description})
                   for i, description in enumerate(question.descriptions)]
        head = ('{"type":' + dumps(question.kind) + ',"instructions":'
                + dumps(question.instructions) + ',"candidates":[')
        payload = {"type": question.kind, "instructions": question.instructions,
                   "candidates": [json.loads(o) for o in objects]}
        if head + ",".join(objects) + "]}" != dumps(payload):
            raise ValueError("Piecewise candidate text differs from the payload JSON.")
        sequence = [*self.before, *self._data(state), *self.between, *self._text(head)]
        ends = []
        for index, text in enumerate(objects):
            if index:
                sequence += self._text(",")
            sequence += self._text(text)
            ends.append(len(sequence) - 1)
        sequence += [*self._text("]}"), *self.after]
        if len(sequence) + 1 > self.max_branch_tokens:
            raise RequestLimitError("State plus question exceeds this student's branch limit.")
        return sequence, self.identifier_ids[:count], ends

    def compile(self, state, question: Question) -> tuple[list[int], list[int]]:
        if self.piecewise:
            sequence, options, _ = self.compile_detailed(state, question)
            return sequence, options
        count = len(question.labels)
        if count > len(self.identifier_ids):
            raise RequestLimitError("Too many choices for verified one-token identifiers.")
        payload = {
            "type": question.kind, "instructions": question.instructions,
            "candidates": [{"identifier": self.identifiers[i], "meaning": description}
                           for i, description in enumerate(question.descriptions)],
        }
        sequence = [*self.before, *self._data(state), *self.between, *self._data(payload),
                    *self.after]
        if len(sequence) + 1 > self.max_branch_tokens:
            raise RequestLimitError("State plus question exceeds this student's branch limit.")
        return sequence, self.identifier_ids[:count]


class StudentScorer:
    def __init__(self, model_dir: Path, *, device_map: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        # Load straight onto the GPU so host RAM never holds a full copy
        # (Tri-7B stores FP32 weights).
        # "auto" splits a BF16 model that exceeds one GPU across the visible GPUs.
        options = dict(dtype=torch.bfloat16, device_map=device_map)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **options).eval()
            self.loader = "AutoModelForCausalLM"
        except ValueError:
            # Natively multimodal checkpoints (e.g. Qwen3.5+) are text-only here.
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_dir, **options).eval()
            self.loader = "AutoModelForImageTextToText"
        self.device = self.model.device

    def logits(self, sequence: list[int], option_ids: list[int]) -> tuple[list[float], float]:
        torch = self.torch
        with torch.inference_mode():
            ids = torch.tensor([sequence], device=self.device)
            output = self.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            last = output.logits[0, -1].float()
            log_probs = torch.log_softmax(last, dim=-1)
            chosen = last[option_ids]
            mass = torch.logsumexp(log_probs[option_ids], dim=0)
        return chosen.tolist(), float(mass)


def evaluate(rows: list[dict], compiler: StudentCompiler, scorer) -> tuple[list[dict], list]:
    results, failures = [], []
    for row in rows:
        started = time.perf_counter()
        state, questions = parse_request(row["request"])
        if list(questions[0].labels) != row["candidate_ids"]:
            raise RuntimeError(f"Presented candidate order differs from the row: {row['id']}")
        # Only an over-long request is a per-row failure. Anything else (a kernel,
        # loader or device error) aborts the run instead of becoming 3,188 "failures".
        try:
            sequence, option_ids = compiler.compile(state, questions[0])
        except RequestLimitError as error:
            failures.append({"id": row["id"], "task": row["task"], "family": row["family"],
                             "error": f"{type(error).__name__}: {error}"})
            continue
        logits, mass = scorer.logits(sequence, option_ids)
        if not all(math.isfinite(value) for value in logits):
            failures.append({"id": row["id"], "task": row["task"], "family": row["family"],
                             "error": "non-finite logits"})
            continue
        scored = scored_row({
            "id": row["id"], "group_id": row["group_id"], "task": row["task"],
            "family": row["family"], "language_origin": row["language_origin"],
            "review": row["review"], "counterfactual": row["counterfactual"],
            "candidate_ids": row["candidate_ids"], "target": row["target"],
            "supervision": "hard_label", "logits": logits, "tie_break": "request_order",
        })
        scored.update(input_tokens=len(sequence), candidate_log_mass=mass,
                      seconds=time.perf_counter() - started)
        results.append(scored)
    return results, failures


def summarize(results: list[dict], failures: list[dict], requested: int) -> dict:
    def block(items):
        metrics = {k: v for k, v in basic_metrics(items).items() if k != "calibration_bins"}
        if items:
            metrics["accuracy_95ci_by_component"] = grouped_interval(items)
        return metrics

    by_task, by_family = defaultdict(list), defaultdict(list)
    for row in results:
        by_task[row["task"]].append(row)
        by_family[row["family"]].append(row)
    failed = defaultdict(int)
    for row in failures:
        failed[row["task"]] += 1
    # Requests that failed count as wrong: they stay in the denominator.
    tasks = {}
    for task, items in sorted(by_task.items()):
        total = len(items) + failed[task]
        tasks[task] = {**block(items), "failed_requests": failed[task],
                       "accuracy_including_failures": sum(r["correct"] for r in items) / total}
    pairs = defaultdict(list)
    for row in results:
        if row["counterfactual"]:
            pairs[row["counterfactual"]["pair_id"], row["family"]].append(row)
    flips = [group for group in pairs.values()
             if len(group) > 1 and len({r["target"] for r in group}) > 1]
    return {
        "requested_rows": requested, "scored_rows": len(results), "failed_rows": len(failures),
        "overall": block(results),
        "task_macro_accuracy_including_failures": (
            sum(t["accuracy_including_failures"] for t in tasks.values()) / len(tasks)
            if tasks else None),
        "tasks": tasks,
        "families": {family: block(items) for family, items in sorted(by_family.items())},
        "counterfactual_groups_with_target_change": len(flips),
        "counterfactual_all_correct": (
            sum(all(r["correct"] for r in group) for group in flips) / len(flips)
            if flips else None),
        "input_tokens": {"total": sum(r["input_tokens"] for r in results),
                         "max": max((r["input_tokens"] for r in results), default=0)},
        "seconds_per_row_median": (sorted(r["seconds"] for r in results)[len(results) // 2]
                                   if results else None),
    }


def download(repo: str, revision: str, out: Path, patterns: list[str] | None = None) -> dict:
    """Fetch a pinned snapshot (optionally only `patterns`) and verify every fetched file
    against the Hub: LFS files by SHA256, other files by git blob SHA-1."""
    from fnmatch import fnmatch

    from huggingface_hub import snapshot_download

    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        meta = json.loads(response.read())
    if meta.get("sha") != revision:
        raise ValueError("The Hub returned a different revision.")
    snapshot_download(repo_id=repo, revision=revision, local_dir=out, allow_patterns=patterns)
    files = {}
    for sibling in meta["siblings"]:
        name, lfs = sibling["rfilename"], sibling.get("lfs")
        if patterns and not any(fnmatch(name, pattern) for pattern in patterns):
            continue
        path = out / name
        data_hash = file_hash(path)
        if lfs:
            ok = path.stat().st_size == lfs["size"] and data_hash == lfs["sha256"]
            method = "lfs sha256"
        else:
            blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode() + path.read_bytes())
            ok, method = blob.hexdigest() == sibling.get("blobId"), "git blob sha1"
        if not ok:
            raise ValueError(f"Checksum failed: {name}")
        files[name] = {"status": "ok", "bytes": path.stat().st_size, "sha256": data_hash,
                       "verified_against": method}
    return {"repo": repo, "revision": revision, "patterns": patterns, "files": files}


RELEASE_SCHEMA = "bobcat-release-manifest-v1"


def checked_split(data: Path, release: Path | None = None) -> tuple[dict, str]:
    """Development/calibration always; the sealed final only for a frozen release manifest
    committed before the run that names this exact final file."""
    manifest = json.loads((data.parent / "manifest.json").read_text())
    split = data.stem
    if manifest.get("schema") != "bobcat-product-eval-v2":
        raise ValueError("Use product eval v2; v1 files reordered Choice criteria.")
    if split == "final":
        frozen = json.loads(release.read_text()) if release is not None else {}
        if frozen.get("schema") != RELEASE_SCHEMA or frozen.get("status") != "frozen":
            raise ValueError("The final split opens only for a frozen release manifest.")
        if frozen["final_evaluation"]["data_sha256"] != manifest["files"]["final"]["sha256"]:
            raise ValueError("The release manifest names a different final split.")
    elif split not in ("dev", "calibration"):
        raise ValueError("Only development/calibration splits of the product eval are allowed.")
    if file_hash(data) != manifest["files"][split]["sha256"]:
        raise ValueError("The evaluation split does not match its manifest.")
    return manifest, split


def prepare(args: argparse.Namespace) -> dict:
    """Checks shared by `run`, `compile` and `score`: split, pinned files, identifiers."""
    manifest, split = checked_split(args.data, getattr(args, "release_manifest", None))
    receipt = json.loads((args.model_dir / "bobcat-download.json").read_text())
    if receipt["repo"] != args.repo or receipt["revision"] != args.revision:
        raise ValueError("The model directory does not hold the requested pinned revision.")
    pinned = receipt.get("files")
    if pinned is None:  # first receipts recorded LFS files only; use the survey pins
        survey = json.loads(args.survey.read_text())
        candidate = next(c for c in survey["candidates"] if c["repo"] == args.repo)
        if candidate["revision"] != args.revision:
            raise ValueError("The requested revision differs from the surveyed revision.")
        pinned = candidate["files"]
    glm = json.loads(args.identifiers.read_text())["identifiers"]
    host = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    reserved = {t["id"] for t in json.loads((args.model_dir / "tokenizer.json").read_text())
                .get("added_tokens", [])}
    identifiers = identifier_scheme(host, reserved, glm)
    shared = next((i for i, (a, b) in enumerate(zip(identifiers, glm, strict=True)) if a != b),
                  len(glm))
    compiler = StudentCompiler(args.model_dir, pinned, identifiers,
                               max_branch_tokens=args.max_branch_tokens,
                               piecewise=getattr(args, "piecewise", False))
    rows = [json.loads(line) for line in args.data.open()]
    if args.limit:
        rows = rows[: args.limit]
    return {"manifest": manifest, "split": split, "compiler": compiler, "rows": rows,
            "identifiers": identifiers, "shared": shared}


def write_outputs(args, ready: dict, results: list[dict], failures: list[dict], *,
                  loader: str, environment: dict, seconds: dict) -> dict:
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "rows.jsonl").open("x") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest, split, compiler = ready["manifest"], ready["split"], ready["compiler"]
    summary = {
        "schema": "bobcat-student-zero-shot-v1", "profile": PROFILE, "repo": args.repo,
        "revision": args.revision, "split": split,
        "data_sha256": manifest["files"][split]["sha256"],
        "eval_content_sha256": manifest["content_sha256"],
        "template_sha256": compiler.template_sha256, "template_source": compiler.template_source,
        "identifiers_sha256": json_hash(ready["identifiers"]),
        "identifiers_shared_with_glm_prefix": ready["shared"], "loader": loader,
        "generated_tokens": 0, "calibration": "none (raw zero-shot)",
        "environment": environment, "seconds": seconds,
        "failures": failures, **summarize(results, failures, len(ready["rows"])),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def run(args: argparse.Namespace) -> dict:
    ready = prepare(args)
    import torch
    import transformers

    started = time.time()
    scorer = StudentScorer(args.model_dir, device_map=args.device_map)
    loaded = time.time()
    results, failures = evaluate(ready["rows"], ready["compiler"], scorer)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "transformers": transformers.__version__,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "dtype": "bfloat16", "device_map": args.device_map,
                   "visible_gpus": torch.cuda.device_count()}
    return write_outputs(args, ready, results, failures, loader=scorer.loader,
                         environment=environment,
                         seconds={"load": loaded - started, "evaluate": time.time() - loaded})


def compile_split(args: argparse.Namespace) -> dict:
    """Token IDs for an external runtime (e.g. DeepSeek's TP8 reference code)."""
    ready = prepare(args)
    args.out.mkdir(parents=True, exist_ok=False)
    written, failed = 0, []
    with (args.out / "compiled.jsonl").open("x") as stream:
        for row in ready["rows"]:
            state, questions = parse_request(row["request"])
            if list(questions[0].labels) != row["candidate_ids"]:
                raise RuntimeError(f"Presented candidate order differs from the row: {row['id']}")
            try:
                if ready["compiler"].piecewise:
                    sequence, option_ids, ends = ready["compiler"].compile_detailed(
                        state, questions[0])
                else:
                    (sequence, option_ids), ends = ready["compiler"].compile(
                        state, questions[0]), None
            except RequestLimitError as error:
                failed.append({"id": row["id"], "error": str(error)})
                continue
            record = {"id": row["id"], "input_ids": sequence, "option_ids": option_ids,
                      "input_sha256": json_hash(sequence)}
            if ends is not None:
                record["candidate_ends"] = ends
            stream.write(json.dumps(record) + "\n")
            written += 1
    meta = {"repo": args.repo, "revision": args.revision, "split": ready["split"],
            "rows": written, "failed": failed,
            "compiled_sha256": file_hash(args.out / "compiled.jsonl"),
            "template_sha256": ready["compiler"].template_sha256,
            "template_source": ready["compiler"].template_source,
            "identifiers": ready["identifiers"], "shared": ready["shared"],
            "piecewise": ready["compiler"].piecewise}
    (args.out / "compile-manifest.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                               indent=2) + "\n")
    return meta


class PrecomputedScorer:
    """Logits produced elsewhere, matched to the exact compiled input by its hash."""

    def __init__(self, path: Path):
        self.rows = {}
        for line in path.open():
            record = json.loads(line)
            self.rows[record["input_sha256"]] = record

    def logits(self, sequence: list[int], option_ids: list[int]) -> tuple[list[float], float]:
        record = self.rows.get(json_hash(sequence))
        if record is None or record["option_ids"] != option_ids:
            raise RuntimeError("No runtime logits for this compiled input.")
        return record["logits"], record["candidate_log_mass"]


def score(args: argparse.Namespace) -> dict:
    ready = prepare(args)
    runtime = json.loads(args.runtime.read_text())
    results, failures = evaluate(ready["rows"], ready["compiler"], PrecomputedScorer(args.logits))
    return write_outputs(args, ready, results, failures, loader=runtime["loader"],
                         environment=runtime["environment"], seconds=runtime["seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--pattern", action="append", help="only fetch matching files")
    def common(command: argparse.ArgumentParser) -> argparse.ArgumentParser:
        command.add_argument("--repo", required=True)
        command.add_argument("--revision", required=True)
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--survey", type=Path,
                             default=Path("reports/2026-09-24-student-candidate-survey.json"))
        command.add_argument("--identifiers", type=Path,
                             default=Path("reports/2026-09-22-glm-readout-preflight.json"))
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--max-branch-tokens", type=int, default=16384)
        command.add_argument("--limit", type=int)
        command.add_argument("--piecewise", action="store_true",
                             help="per-candidate encoding with exact end positions")
        command.add_argument("--release-manifest", type=Path,
                             help="frozen release manifest; required for the final split")
        return command

    common(sub.add_parser("run")).add_argument("--device-map", default="cuda",
                                                choices=["cuda", "auto"])
    common(sub.add_parser("compile"))
    scoring = common(sub.add_parser("score"))
    scoring.add_argument("--logits", type=Path, required=True)
    scoring.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compile":
        meta = compile_split(args)
        print(json.dumps({k: meta[k] for k in ("rows", "compiled_sha256", "shared")}))
        return
    if args.command == "score":
        summary = score(args)
        print(json.dumps({k: summary[k] for k in ("repo", "scored_rows", "failed_rows",
                                                  "task_macro_accuracy_including_failures")}))
        return
    if args.command == "download":
        receipt = download(args.repo, args.revision, args.out, args.pattern)
        (args.out / "bobcat-download.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"verified_files": len(receipt["files"])}))
    else:
        summary = run(args)
        print(json.dumps({k: summary[k] for k in ("repo", "scored_rows", "failed_rows",
                                                  "task_macro_accuracy_including_failures")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
