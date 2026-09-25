"""Serve a trained student behind the closed decision contract and measure it (PLAN step 4).

`ServingStudent` wraps the pretrained student (plus an optional LoRA adapter, merged for
serving) with the piecewise student compiler used in training and evaluation. The HTTP app
is `serve.create_app`: the model returns only the offered identifiers' logits at the first
answer position and the host builds the Choice/Noul/Score JSON. Nothing is generated.

The questions of one request run as one right-padded batch; causal attention and the
causal linear-attention recurrence keep padding from reaching any read position. With
`shared_prefix`, the token prefix common to every question (template, state and the start
of the question object) is prefilled once and each question continues from a copy of the
hybrid cache: the full-attention KV and the Gated DeltaNet conv/recurrent states.

Commands
  probe  every stress case through HTTP (independent reply checks), the output-contract
         attack corpus, and determinism: repeats, batch composition, shared prefix and
         merged-vs-unmerged adapter logits
  bench  single-GPU latency for the SPEC W1-W4 workloads and batch throughput
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from tokenizers import Tokenizer

from bobcat.corpus import atomic_json
from bobcat.protocol import RequestLimitError, parse_request, probabilities
from bobcat.schema import file_hash, json_hash
from bobcat.student_readout import StudentCompiler, identifier_scheme

TOKEN_BUDGET = 131072  # padded tokens per forward chunk


def common_prefix(sequences: list[list[int]]) -> int:
    first, length = sequences[0], min(map(len, sequences))
    for index in range(length):
        if any(s[index] != first[index] for s in sequences[1:]):
            return index
    return length


def expand_cache(torch, cache, batch: int) -> None:
    """Repeat a batch-1 hybrid cache: attention K/V and linear-attention conv/recurrent states."""
    for layer in cache.layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if isinstance(tensor, torch.Tensor) and tensor.numel():
                setattr(layer, name, tensor.expand(batch, *tensor.shape[1:]).contiguous())
        for name in ("conv_states", "recurrent_states"):
            states = getattr(layer, name, None)
            if isinstance(states, dict):
                for key, tensor in states.items():
                    if tensor is not None:
                        states[key] = tensor.expand(batch, *tensor.shape[1:]).contiguous()


class ServingStudent:
    readout_mode = "first_position_logits"
    release_gate_passed = False

    def __init__(self, model_dir: Path, *, name: str, identifiers_path: Path,
                 adapter: Path | None = None, temperature: float = 1.0, merge: bool = True,
                 shared_prefix: bool = False, max_branch_tokens: int = 16384,
                 device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM

        receipt = json.loads((model_dir / "bobcat-download.json").read_text())
        glm = json.loads(identifiers_path.read_text())["identifiers"]
        host = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        reserved = {t["id"] for t in json.loads((model_dir / "tokenizer.json").read_text())
                    .get("added_tokens", [])}
        self.identifiers = identifier_scheme(host, reserved, glm)
        self.compiler = StudentCompiler(model_dir, receipt["files"], self.identifiers,
                                        max_branch_tokens=max_branch_tokens, piecewise=True)
        self.torch, self.device = torch, torch.device(device)
        # Same loader order as training, so adapter module names match.
        options = dict(dtype=torch.bfloat16, device_map={"": device})
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, **options)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_dir, **options)
        self.adapter_sha256 = None
        if adapter is not None:
            from peft import PeftModel

            self.adapter_sha256 = file_hash(adapter / "lora" / "adapter_model.safetensors")
            model = PeftModel.from_pretrained(model, adapter / "lora")
            if merge:
                model = model.merge_and_unload()
        self.model = model.eval()
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.backbone = base.get_decoder() if hasattr(base, "get_decoder") else base.model
        self.lm_weight = base.get_output_embeddings().weight
        self.model_name = name
        self.temperatures = {kind: temperature for kind in ("choice", "noul", "score")}
        self.limits = {"max_branch_tokens": max_branch_tokens,
                       "max_choices": len(self.identifiers)}
        self.shared_prefix = shared_prefix
        self.merged = adapter is not None and merge
        self.timing = {}

    # ---------------------------------------------------------------- forward paths

    def _padded(self, sequences, offset=0):
        torch = self.torch
        width = max(map(len, sequences))
        ids = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = torch.tensor(sequence)
        positions = torch.arange(offset, offset + width).expand(len(sequences), -1)
        return ids.to(self.device), positions.to(self.device)

    def _chunks(self, sequences):
        chunk, width = [], 0
        for index, sequence in enumerate(sequences):
            if chunk and max(width, len(sequence)) * (len(chunk) + 1) > TOKEN_BUDGET:
                yield chunk
                chunk, width = [], 0
            chunk.append(index)
            width = max(width, len(sequence))
        if chunk:
            yield chunk

    def _read(self, hidden_rows, option_ids):
        return [(self.lm_weight[options].float() @ row.float()).tolist()
                for row, options in zip(hidden_rows, option_ids, strict=True)]

    def last_hidden_full(self, sequences):
        torch = self.torch
        rows = [None] * len(sequences)
        for chunk in self._chunks(sequences):
            ids, positions = self._padded([sequences[i] for i in chunk])
            hidden = self.backbone(input_ids=ids, position_ids=positions,
                                   use_cache=False).last_hidden_state
            ends = torch.tensor([len(sequences[i]) - 1 for i in chunk], device=self.device)
            picked = hidden[torch.arange(len(chunk), device=self.device), ends]
            for slot, index in enumerate(chunk):
                rows[index] = picked[slot]
        return rows, sum(map(len, sequences))

    def last_hidden_shared(self, sequences):
        """Prefill the common prefix once, then run all suffixes from cache copies."""
        import copy

        torch = self.torch
        prefix = min(common_prefix(sequences), min(map(len, sequences)) - 1)
        if len(sequences) == 1 or prefix < 1:
            return self.last_hidden_full(sequences)
        head, _ = self._padded([sequences[0][:prefix]])
        parent = self.backbone(input_ids=head, use_cache=True).past_key_values
        suffixes = [s[prefix:] for s in sequences]
        rows = [None] * len(sequences)
        for chunk in self._chunks(suffixes):
            cache = copy.deepcopy(parent)
            expand_cache(torch, cache, len(chunk))
            ids, positions = self._padded([suffixes[i] for i in chunk], offset=prefix)
            hidden = self.backbone(input_ids=ids, position_ids=positions, past_key_values=cache,
                                   use_cache=True).last_hidden_state
            ends = torch.tensor([len(suffixes[i]) - 1 for i in chunk], device=self.device)
            picked = hidden[torch.arange(len(chunk), device=self.device), ends]
            for slot, index in enumerate(chunk):
                rows[index] = picked[slot]
        return rows, prefix + sum(map(len, suffixes))

    def logits(self, sequences, option_ids, *, shared=None):
        torch = self.torch
        shared = self.shared_prefix if shared is None else shared
        sync = (lambda: torch.cuda.synchronize(self.device)) if self.device.type == "cuda" \
            else (lambda: None)
        with torch.inference_mode():
            sync()
            started = time.perf_counter()
            rows, processed = (self.last_hidden_shared if shared
                               else self.last_hidden_full)(sequences)
            values = self._read(rows, option_ids)
            sync()
        self.timing = {"forward_seconds": time.perf_counter() - started,
                       "processed_tokens": processed, "shared_prefix": shared}
        return values, processed

    def compile(self, state, questions):
        compiled = [self.compiler.compile(state, question) for question in questions]
        return [c[0] for c in compiled], [c[1] for c in compiled]

    def score(self, state, questions):
        started = time.perf_counter()
        sequences, option_ids = self.compile(state, questions)
        compiled = time.perf_counter()
        values, processed = self.logits(sequences, option_ids)
        self.timing["compile_seconds"] = compiled - started
        return values, processed


# ---------------------------------------------------------------- probe

def http_client(server):
    from fastapi.testclient import TestClient

    from bobcat.serve import create_app

    return TestClient(create_app(server), raise_server_exceptions=False)


def run_cases(client, cases, name, out: Path) -> dict:
    from bobcat.output_contract_audit import inspect_reply

    statuses = {}
    with out.open("x") as stream:
        for case in cases:
            started = time.perf_counter()
            reply = client.post("/v1/systemone", json=case["payload"])
            seconds = time.perf_counter() - started
            try:
                inspect_reply(case["payload"], reply, expected_model=name)
                violation = None
            except (AssertionError, ValueError, TypeError, KeyError) as error:
                violation = str(error)
            try:
                body = reply.json()
            except ValueError:
                body = reply.text
            statuses[reply.status_code] = statuses.get(reply.status_code, 0) + 1
            stream.write(json.dumps({"id": case["id"], "status": reply.status_code,
                                     "body": body, "violation": violation,
                                     "seconds": seconds}, ensure_ascii=False) + "\n")
    return statuses


def _compare(a, b):
    """Max logit gap, max probability gap and argmax agreement between logit lists."""
    gaps, prob_gaps, same = [], [], []
    for x, y in zip(a, b, strict=True):
        gaps.append(max(abs(p - q) for p, q in zip(x, y, strict=True)))
        px, py = probabilities(x, 1.0), probabilities(y, 1.0)
        prob_gaps.append(max(abs(p - q) for p, q in zip(px, py, strict=True)))
        same.append(max(range(len(x)), key=x.__getitem__) == max(range(len(y)),
                                                                    key=y.__getitem__))
    return {"items": len(gaps), "max_logit_gap": max(gaps, default=0.0),
            "max_probability_gap": max(prob_gaps, default=0.0),
            "mean_probability_gap": statistics.fmean(prob_gaps) if prob_gaps else 0.0,
            "argmax_agreement": sum(same) / len(same) if same else None}


def determinism(server, cases, dev_rows, stored: Path | None) -> dict:
    single = [c for c in cases if c["suite"] == "permutation"
              and c["meta"]["variant"] == "original"][:64]
    general = [c for c in cases if c["suite"] == "general"][:32]
    result = {}
    # Same request, same batch, five times.
    first, gaps = {}, []
    for repeat in range(5):
        for case in general:
            state, questions = parse_request(case["payload"])
            values, _ = server.logits(*server.compile(state, questions), shared=False)
            if repeat == 0:
                first[case["id"]] = values
            else:
                gaps.append(_compare(first[case["id"]], values)["max_logit_gap"])
    result["repeat_same_batch"] = {"requests": len(general), "repeats": 5,
                                   "max_logit_gap": max(gaps, default=0.0)}
    # One question alone versus inside a batch of eight different questions.
    alone, batched = [], []
    compiled = [server.compile(*parse_request(c["payload"])) for c in single]
    for sequences, options in compiled:
        alone += server.logits(sequences, options, shared=False)[0]
    for start in range(0, len(compiled), 8):
        group = compiled[start:start + 8]
        batched += server.logits([s for g in group for s in g[0]],
                                 [o for g in group for o in g[1]], shared=False)[0]
    result["batch_of_8_vs_alone"] = _compare(alone, batched)
    # Shared prefix versus full sequences on multi-question requests: the general
    # multi-question probes, the attack corpus's three-question requests, and the four
    # candidate orders of one permutation base asked as one request.
    multi = [c for c in cases if len(c["payload"]["questions"]) > 1]
    perm = {}
    for case in cases:
        if case["suite"] == "permutation":
            (spec,) = case["payload"]["questions"].values()
            perm.setdefault(case["meta"]["base"], {"model": case["payload"]["model"],
                                                   "state": case["payload"]["state"],
                                                   "questions": {}})
            perm[case["meta"]["base"]]["questions"][case["meta"]["variant"]] = spec
    from bobcat.output_contract_audit import attack_cases

    payloads = ([c["payload"] for c in multi] + list(perm.values())[:100]
                + [c["payload"] for c in attack_cases()[::24]])
    full, shared, saved = [], [], []
    for payload in payloads:
        sequences, options = server.compile(*parse_request(payload))
        a, full_tokens = server.logits(sequences, options, shared=False)
        b, shared_tokens = server.logits(sequences, options, shared=True)
        full += a
        shared += b
        saved.append(1 - shared_tokens / full_tokens)
    result["shared_prefix_vs_full"] = {**_compare(full, shared), "requests": len(payloads),
                                       "mean_token_saving": statistics.fmean(saved)}
    # Serving path (merged adapter, batched) versus the evaluation path's stored logits.
    if stored is not None and stored.exists():
        records = {r["input_sha256"]: r for r in map(json.loads, stored.open())}
        mine, theirs = [], []
        for row in dev_rows:
            state, questions = parse_request(row["request"])
            sequences, options = server.compile(state, questions)
            record = records.get(json_hash(sequences[0]))
            if record is None or record["option_ids"] != options[0]:
                continue
            mine += server.logits(sequences, options, shared=False)[0]
            theirs.append(record["logits"])
        result["serving_vs_evaluation_path"] = {
            **_compare(theirs, mine), "stored": str(stored),
            "serving": "merged LoRA" if server.merged else "unmerged",
        }
    return result


def probe(args) -> dict:
    from bobcat.output_contract_audit import attack_cases, audit

    if args.out.exists():
        raise ValueError("Probe outputs are immutable; choose a new path.")
    args.out.mkdir(parents=True)
    started = time.time()
    server = ServingStudent(args.model_dir, name=args.name, identifiers_path=args.identifiers,
                            adapter=args.adapter, temperature=args.temperature,
                            merge=not args.unmerged)
    loaded = time.time()
    cases = [json.loads(line) for line in args.cases.open()]
    client = http_client(server)
    statuses = run_cases(client, cases, args.name, args.out / "responses.jsonl")
    attacks = audit(client, attack_cases(), model=args.name, out=args.out / "attack-audit",
                    scope=f"{args.name} served through serve.create_app")
    dev = [json.loads(line) for line in args.dev.open()] if args.dev else []
    rng = random.Random(7)
    dev_sample = rng.sample(dev, min(len(dev), 200)) if dev else []
    checks = determinism(server, cases, dev_sample, args.stored_logits)
    import torch

    report = {
        "schema": "bobcat-student-probe-v1", "model": args.name,
        "adapter_sha256": server.adapter_sha256, "merged": server.merged,
        "temperature": args.temperature, "cases_sha256": file_hash(args.cases),
        "statuses": {str(k): v for k, v in statuses.items()},
        "attack_audit": {k: attacks[k] for k in (
            "requests", "unique_payloads", "statuses", "typed_answers",
            "output_contract_violations", "observed_violation_rate")},
        "determinism": checks,
        "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30},
        "seconds": {"load": loaded - started, "total": time.time() - started},
    }
    atomic_json(args.out / "report.json", report)
    return report


# ---------------------------------------------------------------- bench

def text_pool(dev_rows) -> str:
    seen, parts = set(), []
    for row in sorted(dev_rows, key=lambda r: r["id"]):
        state = row["request"]["state"]
        for value in (state.values() if isinstance(state, dict) else [state]):
            if isinstance(value, str) and len(value) > 200 and value not in seen:
                seen.add(value)
                parts.append(value)
    return "\n".join(parts)


def sized_state(server, pool: str, questions_payload: dict, tokens: int, offset: int = 0):
    """A {"문서": text} state whose longest compiled question is about `tokens` long."""
    def length(chars):
        payload = {"model": "bobcat-latest", "state": {"문서": pool[offset:offset + chars]},
                   "questions": questions_payload}
        try:
            sequences, _ = server.compile(*parse_request(payload))
        except RequestLimitError:
            return math.inf, payload
        return max(map(len, sequences)), payload

    low, high = 1, min(len(pool) - offset, tokens * 4)
    while low < high:
        middle = (low + high + 1) // 2
        if length(middle)[0] <= tokens:
            low = middle
        else:
            high = middle - 1
    return length(low)


def workload_questions():
    topics = {f"t{i}": description for i, description in enumerate([
        "정치와 선거", "경제와 금융", "사회와 사건", "국제 관계", "과학과 기술", "문화와 예술",
        "스포츠", "생활과 건강"])}
    return {
        "w1": {"topic": {"type": "choice", "instructions": "문서의 주된 주제를 고르라.",
                         "criteria": topics}},
        "w2": {
            **{f"has_{i}": {"type": "noul", "instructions": f"문서가 {name}을(를) 다루는가?",
                            "criteria": {"true": "직접 다룬다", "false": "다루지 않는다"}}
               for i, name in enumerate(["인물", "날짜", "금액", "장소"])},
            "topic": {"type": "choice", "instructions": "문서의 주된 주제를 고르라.",
                      "criteria": topics},
            "tone": {"type": "choice", "instructions": "문서의 어조를 고르라.",
                     "criteria": {"neutral": "중립 보도", "critical": "비판적",
                                  "promotional": "홍보성", "personal": "개인 의견"}},
            "audience": {"type": "choice", "instructions": "주된 독자를 고르라.",
                         "criteria": {"general": "일반 독자", "expert": "전문가",
                                      "investor": "투자자", "official": "공무원·기관",
                                      "student": "학생", "consumer": "소비자"}},
            "quality": {"type": "score", "instructions": "정보의 구체성을 평가하라.",
                        "criteria": ["없음", "낮음", "보통", "높음", "매우 높음"]},
        },
        "w3": {"relevant": {"type": "choice", "instructions": "문서의 성격을 고르라.",
                            "criteria": {"news": "보도 기사", "essay": "칼럼·에세이",
                                         "notice": "공지·안내", "other": "그 밖의 글"}}},
    }


def timed(fn, repeats: int, warmup: int = 3):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return {"p50_ms": samples[len(samples) // 2],
            "p95_ms": samples[min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1)],
            "mean_ms": statistics.fmean(samples), "repeats": repeats}


def bench(args) -> dict:
    import torch

    if args.out.exists():
        raise ValueError("Benchmark outputs are immutable; choose a new path.")
    args.out.mkdir(parents=True)
    server = ServingStudent(args.model_dir, name=args.name, identifiers_path=args.identifiers,
                            adapter=args.adapter, temperature=args.temperature,
                            merge=not args.unmerged)
    client = http_client(server)
    dev = [json.loads(line) for line in args.dev.open()]
    pool = text_pool(dev)
    specs = workload_questions()
    results = {}

    def http(payload):
        def call():
            reply = client.post("/v1/systemone", json=payload)
            if reply.status_code != 200:
                raise RuntimeError(f"benchmark request failed: {reply.status_code}")
        return call

    def model_only(payload, shared):
        sequences, options = server.compile(*parse_request(payload))
        return lambda: server.logits(sequences, options, shared=shared)

    def record(name, payload, *, shared_modes=(False,), repeats=args.repeats):
        sequences, _ = server.compile(*parse_request(payload))
        entry = {"questions": len(sequences), "tokens_per_question": [len(s) for s in sequences],
                 "common_prefix_tokens": common_prefix(sequences)}
        for shared in shared_modes:
            label = "shared" if shared else "full"
            server.shared_prefix = shared
            entry[f"http_{label}"] = timed(http(payload), repeats)
            entry[f"model_{label}"] = timed(model_only(payload, shared), repeats)
        server.shared_prefix = False
        results[name] = entry
        print(json.dumps({name: {k: v for k, v in entry.items()
                                 if k != "tokens_per_question"}}), flush=True)

    _, w1 = sized_state(server, pool, specs["w1"], 512)
    record("W1_512tok_8choices_1q", w1)
    _, w2 = sized_state(server, pool, specs["w2"], 3072 + 120)
    record("W2_3k_state_8q", w2, shared_modes=(False, True))
    for tokens in (8192, 16384 - 64):
        _, w3 = sized_state(server, pool, specs["w3"], tokens)
        record(f"W3_{tokens}tok_1q", w3, repeats=max(5, args.repeats // 2))
    for k in (77, 255):
        row = next(r for r in sorted(dev, key=lambda r: r["id"])
                   if r["family"] == f"document_title_k{k}")
        record(f"W3_K{k}_titles", row["request"])
    # W4: independent 1k-token requests as one static batch (no queueing server).
    singles = [sized_state(server, pool, specs["w1"], 1024, offset=i * 4000)[1]
               for i in range(32)]
    compiled = [server.compile(*parse_request(p)) for p in singles]
    throughput = {}
    for batch in (1, 8, 16, 32):
        group = compiled[:batch]
        sequences = [s for g in group for s in g[0]]
        options = [o for g in group for o in g[1]]
        timing = timed(lambda s=sequences, o=options: server.logits(s, o, shared=False),
                       max(5, args.repeats // 2))
        tokens = sum(map(len, sequences))
        throughput[batch] = {**timing, "tokens": tokens,
                             "tokens_per_second": tokens / (timing["p50_ms"] / 1000)}
        print(json.dumps({"W4_batch": batch, **throughput[batch]}), flush=True)
    results["W4_1k_independent_static_batch"] = throughput
    best = max(throughput.values(), key=lambda t: t["tokens_per_second"])
    cost = None
    if args.gpu_hour_usd:
        per_mtok = args.gpu_hour_usd / (best["tokens_per_second"] * 3600) * 1e6
        w1_p50 = results["W1_512tok_8choices_1q"]["http_full"]["p50_ms"] / 1000
        cost = {"gpu_hour_usd": args.gpu_hour_usd, "price_basis": args.price_basis,
                "input_usd_per_mtok_at_best_batch": per_mtok,
                "reference_typesafe_usd_per_mtok": 0.042,
                "w1_usd_per_1000_serial_requests": args.gpu_hour_usd / 3600 * w1_p50 * 1000}
    report = {
        "schema": "bobcat-student-bench-v1", "model": args.name,
        "adapter_sha256": server.adapter_sha256, "merged": server.merged,
        "precision": "bfloat16", "attention": "sdpa (transformers default)",
        "linear_attention": "flash-linear-attention kernels; causal_conv1d not installed",
        "workloads": results, "cost": cost,
        "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                        "gpus_used": 1,
                        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30},
        "notes": ["W4 is a static batch on one process, not a queueing server under load.",
                  "HTTP timings use the in-process ASGI test client (no network).",
                  "No torch.compile, CUDA graphs or quantization."],
    }
    atomic_json(args.out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (sub.add_parser("probe"), sub.add_parser("bench")):
        command.add_argument("--model-dir", type=Path, required=True)
        command.add_argument("--name", required=True)
        command.add_argument("--adapter", type=Path)
        command.add_argument("--temperature", type=float, default=1.0)
        command.add_argument("--unmerged", action="store_true")
        command.add_argument("--identifiers", type=Path,
                             default=Path("reports/2026-09-22-glm-readout-preflight.json"))
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--dev", type=Path)
    probing = sub.choices["probe"]
    probing.add_argument("--cases", type=Path, required=True)
    probing.add_argument("--stored-logits", type=Path)
    timing = sub.choices["bench"]
    timing.add_argument("--repeats", type=int, default=20)
    timing.add_argument("--gpu-hour-usd", type=float)
    timing.add_argument("--price-basis", default="")
    args = parser.parse_args()
    report = probe(args) if args.command == "probe" else bench(args)
    print(json.dumps({k: report[k] for k in report if k not in ("workloads", "determinism")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
