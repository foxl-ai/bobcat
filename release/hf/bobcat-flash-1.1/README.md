---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_revision: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
base_model_relation: adapter
library_name: peft
language:
- en
- ko
pipeline_tag: zero-shot-classification
tags:
- lora
- peft
- typed-decisions
- calibration
- distillation
- classification
- guardrails
- mixture-of-experts
datasets:
- klue/klue
- kakaobrain/kor_nli
- e9t/nsmc
- AmazonScience/massive
- PolyAI/banking77
- google/boolq
- allenai/ai2_arc
- nvidia/HelpSteer2
- nvidia/HelpSteer3
- nyu-mll/multi_nli
- stanfordnlp/snli
---

# Bobcat Flash 1.1

Bobcat Flash is the fast tier of Bobcat, a **typed-decision model**. You send a state (text
or JSON) and questions whose possible answers you define in the request; Flash returns a
probability for every answer you named, and nothing else. It never generates text: it
reads the logits of the offered candidates at the first answer position of one forward
pass, and the host builds a closed JSON reply from your own names. An answer can be wrong,
but it cannot be malformed.

This repository, `sanghwa-na/bobcat-flash-1.1`, holds the **Bobcat Flash 1.1 LoRA adapter**
for [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
(Apache-2.0) at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`: a sparse
mixture-of-experts model with about 4B active parameters per token. Flash was distilled
from the 27B Bobcat models. It answers most questions on its own and can hand the ones it
is unsure about to [Bobcat 1.1](https://huggingface.co/sanghwa-na/bobcat-1.1) in the same
server. The compiler, servers and evaluation code are at
[github.com/foxl-ai/bobcat](https://github.com/foxl-ai/bobcat).

![Bobcat Flash 1.1 at a glance](assets/bobcat-flash-1.1-highlights.png)

## Highlights

| | Bobcat Flash 1.1 | Reference |
|---|---:|---|
| Median time per TypeSafe workflow case, served | **0.297 s** | Jev 0.42 s (TypeSafe's published client-side time); Bobcat 1.1 0.52-0.58 s |
| One decision (512 tokens, 8 candidates), p50 | **24.8 ms** | one RTX PRO 6000 Blackwell, FP8, vLLM engine; Bobcat 1.1 42.8 ms (NVFP4) |
| TypeSafe's published workflow examples: agreement with the reference (329 questions) | **91.2%** | Jev 90.9%, Claude Opus 5 92.4%, GPT-5.6 Sol 93.0% |
| SemIf's 102 aligned TypeSafe rows: modal agreement | **0.896** | Jev 0.883 |
| Sealed final, four tasks (1,614 decisions, opened once) | **92.21%** | Bobcat 1.1 94.27% (-2.1 pt [-3.0, -1.2]); Bobcat 1 93.59% |
| Six-task development evaluation (3,188 decisions) | **92.07%** | Bobcat 1.1 93.69%; same base zero-shot 86.6% |
| Wrong answer named inside the state wins | **9.7%** | Bobcat 1 38.6%, same server setup |
| Routed server: Flash first, unsure questions to the 27B model (development) | **93.11%**, 84.2% answered by Flash | the 27B model alone 93.41% (see [Routing](#routing-to-bobcat-11)) |

Jev, Opus 5 and Sol figures are TypeSafe's own published answers and times; Jev was never
called. Flash is less accurate than Bobcat 1.1, most of all on search passage selection;
see [Limitations](#limitations-and-risks).

## What it does

| Primitive | You define | Bobcat returns |
|---|---|---|
| Choice | 1 to 255 named candidates, descriptions optional | the top name and a probability for every name |
| Noul | optional meanings of true and false | P(true) |
| Score | 1 to 10 ordered levels | the expected level and a probability per level |

The request and reply are the same as Bobcat's and use the shapes of TypeSafe's published
System One HTTP API, so the official `typesafe-sdk` works against a Flash server by changing
its base URL.

## Quickstart: serve Flash on one GPU

The figures on this card were measured on one RTX PRO 6000 Blackwell 96 GB with vLLM
0.30.0 and FP8. We have not run a fresh-install test of this exact recipe; it is the
Bobcat recipe with the Gemma base and the settings we served Flash with.

```bash
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed   # a uv-managed Python ships the headers Triton compiles against
uv venv --python 3.12 .venv-serve
uv pip install --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

# 1. The base model at the pinned revision (every file's hash is verified)
$PY -m bobcat.student_readout download --repo google/gemma-4-26B-A4B-it \
  --revision 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 --out base

# 2. This adapter, merged into the base in float32 and stored as BF16
$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-flash-1.1', local_dir='adapter')"
$PY -m bobcat.student_merge --model-dir base --adapter adapter --out model

# 3. The typed-decision server (TypeSafe-compatible /v1/systemone and /v1/models)
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.flash_server --engine vllm --model model \
  --compiler-model base --quantization fp8 --temperature 0.8912 --name bobcat-flash-1.1 \
  --max-num-seqs 256 --max-model-len 32832 --schedule all \
  --engine-arg max_num_batched_tokens=16384 --host 127.0.0.1 --port 8000 --local
```

Then call it with the official SDK:

```python
import os
os.environ.update(TYPESAFE_BASE_URL="http://127.0.0.1:8000", TYPESAFE_API_KEY="local",
                  TYPESAFE_DEFAULT_MODEL="bobcat-flash-1.1")
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient()
result = client.system_one(
    "I was charged twice for the same order. Can someone look into this?",
    {"billing": Noul(instructions="Is this about billing?"),
     "route": Choice(instructions="Which team should handle it?",
                     criteria={"payments": "Charges and refunds", "shipping": None,
                               "account": "Login and settings"})},
)
print(result.nouls["billing"].noul, result.choices["route"].choice)
```

Notes:

- `--temperature 0.8912` is the calibration temperature fitted for this adapter on a
  held-out calibration split; it never changes an argmax.
- `bobcat.api_server` with the same arguments serves Flash too; it tokenizes the state once
  per request and produced the same token IDs on 3,606 questions.
- vLLM runs Gemma 4 with its Triton attention backend (head sizes 256 and 512).
- `--local` disables the shared secret the servers otherwise require; use it only on a
  loopback or private interface. The server refuses inputs over its limits with HTTP 422
  and never truncates them.
- The compiler reads its identifier list from the repository
  (`reports/2026-09-22-glm-readout-preflight.json`), so run the server from the repository
  root. The same list is included here as `bobcat-identifiers.json`.

## Routing to Bobcat 1.1

`bobcat.route_server` runs Flash and Bobcat 1.1 as two vLLM engines in one process on one
GPU and decides for each question on its own, from that question's input and Flash's scores
only:

1. **Out of range, before any forward pass:** more than 64 candidates, or fewer than half
   of the letters Hangul or Latin (Flash was trained and evaluated on Korean and English
   only). The question goes straight to Bobcat 1.1.
2. **Otherwise Flash scores it.** If Flash's calibrated top probability is below 0.8 (a
   threshold chosen on the calibration split), Bobcat 1.1 compiles and scores it again.

Each answer carries the probabilities of the model that scored it, at that model's
calibration temperature, and the `x-bobcat-route` header names that model per question.
Questions never see each other.

```bash
# Flash as above (model/, base/); Bobcat 1.1 merged as in its card (b11-model/, b11-compiler/)
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.route_server \
  --flash-model model --flash-compiler-model base --flash-temperature 0.8912 \
  --flash-quantization fp8 --flash-gpu-memory-utilization 0.40 \
  --big-model b11-model --big-compiler-model b11-compiler --big-temperature 1.2008 \
  --big-gpu-memory-utilization 0.46 --max-model-len 32832 --schedule all \
  --engine-arg max_num_batched_tokens=16384 --host 127.0.0.1 --port 8000 --local
```

On the development split (3,188 requests, one at a time over localhost HTTP):

| Server | Task macro | Answered by Flash | Latency p50 / p95 / mean |
|---|---:|---:|---|
| **Routed** | **93.11%** | 84.2% | 33.3 / 215.7 / 64.4 ms |
| Flash only | 92.12% | 100% | 32.3 / 90.2 / 42.5 ms |
| 27B model only | 93.41% | 0% | 61.4 / 216.1 / 90.9 ms |

Routed minus Flash: +0.98 points [+0.32, +1.70]; routed minus the 27B model: -0.30
[-0.77, +0.18]. Questions Flash answered took 32.2 ms at p50; those handed on for low
confidence 93.8 ms; those with more than 64 candidates 480 ms. On TypeSafe's 20 English
cases the routed server agreed with the reference on 91.8% of questions, but 33 of 46
requests handed at least one question on, so the median case took 0.545 s. **If time per
case matters most, serve Flash alone.**

These routing figures were measured with the first Bobcat 1.1 candidate (temperature
1.1885) as the 27B model, in an NVFP4 build (`--big-quantization none` on a checkpoint made
with `scripts/nvfp4_quantize.py`). They have not been re-measured with the released Bobcat
1.1 adapter or with an FP8 27B engine, and concurrent load on the two engines has not been
tested.

## Running on AWS

These are ordinary GPU Linux hosts; the Quickstart above is the whole recipe.

- **Amazon EC2.** One RTX PRO 6000 Blackwell 96 GB (for example `g7e.2xlarge`) serves the
  FP8 merge, and holds Flash and Bobcat 1.1 together for the routed server. Use a Deep
  Learning AMI with a recent NVIDIA driver, and allow about 120 GB of disk for Flash alone
  (base download plus the merged copy), about 200 GB more with Bobcat 1.1.
- **Amazon SageMaker AI.** The same commands run in a JupyterLab space or notebook
  instance of an equivalent GPU type. A SageMaker real-time endpoint needs a custom
  container that runs `bobcat.flash_server` or `bobcat.route_server` behind SageMaker's
  `/invocations` and `/ping` routes; we have not published or tested one.

Our measurements ran on an EC2 RTX PRO 6000 host, with H200 and B200 figures from 8-GPU
hosts (one GPU per server). The SageMaker paths are described here but not tested by us.

## Evaluation

Differences are paired, with 95% bootstrap intervals over source components (TypeSafe
workflows: over cases). The development and sealed-final figures use the evaluation path
(BF16 base with the unmerged adapter); TypeSafe, SemIf and injection figures use the served
FP8 model over HTTP.

### Sealed final

The fresh final built for Bobcat 1.1 from KLUE MRC contexts and Wizard of Seoul dialogues
that no evaluation split, earlier final or training build used (1,614 decisions, four
tasks). Flash opened it once, under an authorization recorded together with Bobcat 1.1's
frozen manifest and before the split was opened.

| Task (decisions) | Flash 1.1 | Bobcat 1.1 | Bobcat 1 |
|---|---:|---:|---:|
| Search passage selection (500) | 74.6% | 80.6% | 79.2% |
| Citation verification (103) | 100.0% | 100.0% | 99.0% |
| External document screening (550) | 94.9% | 96.9% | 96.5% |
| Request routing (461) | 99.3% | 99.6% | 99.6% |
| **Task macro** | **92.21%** | 94.27% | 93.59% |

Flash minus Bobcat 1.1: -2.1 points [-3.0, -1.2]; minus Bobcat 1: -1.4 [-2.5, -0.3]; minus
the Qwen3.8-27B base zero-shot (87.86%): +4.4 [+2.7, +6.1]. 51 routing questions share
Wizard of Seoul utterances with the Flash corpus; without them Flash scores 92.19%. After
temperature, NLL is 0.432 and ECE 0.019. No request failed. The evaluation is Korean and no
label has been reviewed by a person.

### Six-task development evaluation

The v2 development split (3,188 Korean decisions). It was used to select the arm, so it is
not a held-out test.

| Task | Flash 1.1 | Bobcat 1.1 | Gemma 4 26B-A4B zero-shot |
|---|---:|---:|---:|
| Search passage selection | 88.4% | 91.1% | 84.8% |
| Citation verification | 99.8% | 99.6% | 96.5% |
| Tool-call review, never trained | 92.8% | 94.5% | 82.3% |
| External document screening | 94.7% | 96.9% | 88.9% |
| Request routing | 99.6% | 99.4% | 98.7% |
| Classification | 77.0% | 80.6% | 68.6% |
| **Task macro** | **92.07%** | 93.69% | 86.6% |

Flash minus Bobcat 1: -1.3 points [-2.7, -0.1]; minus its own base zero-shot: +5.4
[+3.7, +7.2]. NLL 0.285 and ECE 0.011. The base alone is over-confident (fitted temperature
5.28); Flash's fitted temperature is 0.89.

### TypeSafe's published workflow examples

TypeSafe publishes 20 workflow examples at [evals.typesafe.ai](https://evals.typesafe.ai)
with the state, the exact questions, the answers of Claude Opus 5, GPT-5.6 Sol and Jev, and
reference answers from GPT-6 Astra and Claude Fable 5.1. We sent the same requests once to
the served FP8 Flash and scored every model with the same code on the 329 questions all
four answered; no request failed or was retried.

| | Flash 1.1 | Jev | Claude Opus 5 | GPT-5.6 Sol | Gemma 4 26B-A4B zero-shot |
|---|---:|---:|---:|---:|---:|
| Agreement with the reference, all questions | 91.2% | 90.9% | 92.4% | 93.0% | 90.0% |
| Agreement, mean of the four workflows | 85.4% | 86.5% | 88.2% | 89.6% | 82.4% |
| Probability on the reference answer | 0.862 | 0.850 | 0.851 | 0.914 | 0.899 |
| Median time per case | 0.297 s | 0.42 s | 20.9 s | 24.2 s | 0.293 s |

Flash minus Jev, averaged over workflows, is -1.0 points [-4.6, +4.2]: level on these
examples, not separated. This is question-level agreement on 20 English cases against a
frontier-model consensus, not action accuracy or ground truth. Flash's time is server-side
over localhost HTTP on one RTX PRO 6000 (a second host measured 0.283 s); the other times
are TypeSafe's published client-side times with the network included.

### SemIf benchmark bundle

[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT) publishes typed-decision rows and
evaluators, plus the Jev figures TypeSafe released for 102 rows aligned to its workflow
cases, scored here with SemIf's unmodified evaluators.

| Set (metric) | Flash 1.1 | Jev (published) |
|---|---:|---:|
| authored144 (mean family balanced accuracy) | 0.900 | - |
| perturbations108 | 0.952 | - |
| WANLI256 | 0.734 | - |
| TypeSafe102 modal agreement / total variation | **0.896** / 0.110 | 0.883 / 0.127 |
| Every judge-grid (36 cells) | 31 | 32 |
| Every action-firewall (10 actions) | 9 | 10 |

Bobcat 1.1 scores 0.910, 0.989, 0.730 and 0.872 on the first four rows. These sets are
evaluation-only; none of their rows was used for training, selection or calibration.

### Wrong answers named inside the state

300 development questions, each sent clean and with three attacks (1,200 HTTP requests).

| | Flash 1.1 | Bobcat 1 |
|---|---:|---:|
| **Attack success** | **9.7%** | 38.6% |
| Accuracy: clean / administrator verdict / inside the text / note to an AI grader | 92.7 / 84.3 / 84.0 / 84.3% | 93.0 / 64.7 / 61.7 / 48.7% |
| Replies outside the output contract (independent wire check) | 0 | 0 |

Both rows ran on the same served setup. On its evaluation path Bobcat 1.1 scores 7.7%.

### Latency

| One RTX PRO 6000 Blackwell 96 GB, vLLM 0.30.0, FP8, engine (no HTTP) | Flash 1.1 |
|---|---:|
| 512 tokens, 8 candidates, 1 question, p50 / p95 | 24.8 / 25.2 ms |
| 3.2k-token state, 8 questions, one batch, p50 | 196 ms |
| 8K / 16K tokens, 1 question, p50 | 272 / 682 ms |
| Throughput, independent 1K-token requests | 48,750 tokens/s |
| TypeSafe workflow case, served over localhost HTTP: median / mean / Invoice median | 0.297 / 0.62 / 1.50 s |

The same architecture with an earlier Flash checkpoint took 15.9 ms (first profile) and
0.185 s per workflow case on one H200, and 13.7 ms and 0.226 s on one B200. Latency under
concurrent HTTP load has not been measured.

### Serving precision

FP8 serving gives the evaluation path's answer on 98.4% of development questions (task
macro 92.1% to 91.8%). An NVFP4 build (every expert quantized) was **slower and less
accurate** on this GPU: 44,012 against 48,594 tokens/s, -0.68 points [-1.59, +0.14] on
development and 89.1% against 91.2% on TypeSafe's workflows. Serve FP8.

## Training and distillation

- **Base:** Gemma 4 26B-A4B-it at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`: 30
  layers of sliding-window (1,024) and global attention, 128 experts with 8 active plus a
  dense MLP, about 4B active parameters. It had the highest zero-shot development macro of
  the candidates we compared (86.6%, against 85.5% for the 27B Bobcat base).
- **Adapter:** LoRA rank 64, alpha 128, no dropout, on the attention q/k/v/o projections
  (the five global layers have no v projection), the dense MLP and the expert router of all
  30 layers; 80.0M trainable parameters. The 128 experts, embeddings and output head are
  frozen.
- **Teachers:** Bobcat 1 (Qwen3.8-27B with the Bobcat 1 adapter, merged, temperature
  1.149) and the first Bobcat 1.1 candidate (the stage-1 adapter of Bobcat 1.1, temperature
  1.188), each read once over the whole corpus with the served readout. The teachers never
  saw the gold labels. The released Bobcat 1.1 adapter was finished later and was not a
  teacher.
- **Loss per question:** KL(teacher || student) at the teacher's calibration temperature,
  plus 0.5 times the Bobcat 1 gold loss (cross-entropy for Choice and Noul, expected-level
  error for Score) where a gold label exists.
- **Stages** (eight NVIDIA B200 GPUs, AdamW with betas 0.9/0.95, 3% warmup, cosine decay to
  10%, gradient clip 1.0):

  | Stage | Data | Teacher | Learning rate | Steps |
  |---|---|---|---:|---:|
  | 1 | whole corpus, one epoch | Bobcat 1 | 2e-4 | 1,401 |
  | 2 | whole corpus, half an epoch | Bobcat 1.1 candidate | 1e-4 | 700 |
  | 3 | the Bobcat 1 training mixture, one epoch, gold plus teacher | Bobcat 1.1 candidate | 5e-5 | 214 |

- **Corpus (447,904 questions, 257,710 source components; frozen once):**

  | Source | Questions | Gold label |
  |---|---:|---|
  | Bobcat 1 training mixture (KLUE MRC, Wizard of Seoul, policy transfer, public data) | 50,409 | yes |
  | Further public decisions (KorNLI; KLUE NLI, YNAT, STS; NSMC; HelpSteer 2 and 3; MASSIVE; Banking77; BoolQ; ARC) | 158,298 | yes |
  | New questions on the same public states (evidence/claim, topic, urgency, answerability) | 94,041 | partly |
  | Generated workflows and games (invoices, security logs, support tickets, tic-tac-toe, grid paths, a shooter's strategy sentences, link races) | 97,297 | partly |
  | English NLI (MultiNLI, SNLI) | 33,003 | yes |
  | Policy transfer (synthetic rules with exact interpreters) | 14,856 | yes |

  369,418 questions have a gold label and 78,486 carry only teacher probabilities; 165,719
  (37%) are English.
- **Calibration:** one temperature, 0.8912, fitted on the held-out calibration split.
- **Selection:** the arm with the highest development macro among the Gemma 4 26B-A4B arms.
  TypeSafe, SemIf and injection results were not used to select.
- **Provenance:** **No output of Jev was used for training, distillation, reward or
  calibration**; the only teachers were the two Bobcat models above, and no GLM teacher was
  run. No TypeSafe, SemIf, Every or cookbook evaluation row was used: rows matching any of
  4,495 strings of 16 characters or more from those sets were blocked (none matched), as
  were rows sharing text with the development, calibration or final splits. Tool-call review
  was never trained.

## Limitations and risks

- **Lower search accuracy.** Search passage selection is Flash's weakest task: 74.6% on the
  sealed final against 80.6% for Bobcat 1.1, and 88.4% against 91.1% on development, where
  title matching with 77 to 255 candidates falls most. The routed server sends questions
  with more than 64 candidates to Bobcat 1.1.
- Flash is 2.1 points below Bobcat 1.1 on the sealed final and 1.6 points below it on
  development; tool-call review and classification also trail.
- **Multi-question requests.** Gemma 4's sliding-window layers keep vLLM's prefix cache from
  reusing a shared state exactly, so a request with many questions recomputes part of its
  state (about 1.6 times the request's tokens on a 37-state, 777-question benchmark).
- Insufficient evidence is as hard for Flash as for Bobcat (WANLI256 0.734). Give it an
  explicit "not stated" option.
- The routing figures use the first Bobcat 1.1 candidate as the 27B model (see
  [Routing](#routing-to-bobcat-11)). Two engines on one GPU left little memory headroom in
  our NVFP4 test, and out-of-range inputs in other scripts were never probed.
- The development split and the sealed final are Korean; English results come from
  TypeSafe's 20 cases, SemIf and training monitors. The final has no classification or
  tool-call rows.
- A typed answer can be the wrong valid option, and `confidence` is a concentration
  statistic, not the probability of being right. Arithmetic, counting and date comparison
  are weak; compute them in code.

**Intended use:** small, repeated judgments whose admissible answers the application owns,
where time per request matters - routing, guardrail screening, citation checks, tool-call
review, classification - with the policy that acts on the answer kept in the application's
code. **Out of scope:** generating text, open-ended question answering, and sole reliance on
Bobcat Flash for safety-critical or legal decisions.

## License and attribution

The adapter is released under Apache-2.0 and is a derivative of Gemma 4 26B-A4B-it by
Google DeepMind, which is released under the Apache License 2.0 (the pinned revision's card
metadata and its linked
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license) page, checked on
2026-09-26). Its teachers, Bobcat 1 and the Bobcat 1.1 candidate, are Apache-2.0 adapters of
Qwen3.8-27B (Apache-2.0). Training data keep their own licenses: KLUE, KorNLI, ARC and SNLI
(CC BY-SA 4.0), BoolQ (CC BY-SA 3.0), NSMC (CC0 1.0), MASSIVE, Banking77 and HelpSteer 2 and 3
(CC BY 4.0), and MultiNLI (mostly the Open American National Corpus license, with fiction
under CC BY-SA 3.0 and CC BY 3.0). Attributions are in the code repository's
`THIRD_PARTY.md`.

Bobcat is an independent project. It is not affiliated with or endorsed by TypeSafe AI,
Google, the Qwen team, Anthropic, OpenAI, Every or any other company named here; product
names are their owners' trademarks and are used only to identify the models compared.
TypeSafe's and SemIf's evaluation data are not redistributed here. The adapter is provided
as is, without warranty.

## Citation

```bibtex
@techreport{bobcat2026,
  title       = {Bobcat: Typed Decisions from One Forward Pass},
  author      = {{The Bobcat Authors}},
  institution = {Foxl AI},
  year        = {2026},
  url         = {https://foxl.ai/blog/bobcat-typed-decisions}
}

@misc{bobcatflash11,
  title        = {Bobcat Flash 1.1},
  author       = {{The Bobcat Authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sanghwa-na/bobcat-flash-1.1}}
}
```
