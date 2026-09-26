---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
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
- classification
- reranking
- guardrails
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
- stanfordnlp/snli
- rajpurkar/squad_v2
---

# Bobcat 1.1

Bobcat is a **typed-decision model**. You send a state (text or JSON) and questions whose
possible answers you define in the request; Bobcat returns a probability for every answer
you named, and nothing else. It never generates text: it reads the logits of the offered
candidates at the first answer position of one forward pass, and the host builds a closed
JSON reply from your own names. An answer can be wrong, but it cannot be malformed.

This repository, `sanghwa-na/bobcat-1.1`, holds the **Bobcat 1.1 LoRA adapter** for
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. It uses the same base, compiler and readout
as [Bobcat 1](https://huggingface.co/sanghwa-na/bobcat-1) and drops into the same server;
Bobcat 1 is unchanged. The compiler, server and evaluation code are at
[github.com/foxl-ai/bobcat](https://github.com/foxl-ai/bobcat). A faster tier built from
this model is [Bobcat Flash 1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1).

![Bobcat 1.1 at a glance](assets/bobcat-1.1-highlights.png)

## Highlights

| | Bobcat 1.1 | Reference |
|---|---:|---|
| Sealed final, four tasks (1,614 decisions, opened once) | **94.27%** | Bobcat 1 93.59% (+0.7 pt [-0.0, +1.5]); same base zero-shot 87.86% |
| Wrong answer named inside the state wins (300 questions x 3 attacks) | **7.8%** | Bobcat 1 39.6%; same base zero-shot 21.0% |
| Six-task development evaluation (3,188 decisions) | **93.69%** | Bobcat 1 93.41%; same base zero-shot 85.48% |
| TypeSafe's published workflow examples: agreement with the reference (329 questions) | **92.1%** | Jev 90.9%, Claude Opus 5 92.4%, GPT-5.6 Sol 93.0% |
| SemIf's 102 aligned TypeSafe rows: modal agreement | **0.872** | Jev 0.883 |
| One decision (512 tokens, 8 candidates), p50 | **42.8 ms** | one RTX PRO 6000 Blackwell, NVFP4, vLLM engine |
| Median time per TypeSafe workflow case, served | **0.52-0.58 s** | Jev 0.42 s (TypeSafe's published client-side time) |

Jev, Opus 5 and Sol figures are TypeSafe's own published answers and times; Jev was never
called. See [Evaluation](#evaluation) for what each number does and does not mean, and
[Limitations](#limitations-and-risks) for the targets 1.1 did not meet.

## What changed from Bobcat 1

- **Injected answers.** Bobcat 1 followed a wrong answer named inside the state more often
  than its base model did (39.6% against 21.0%). Bobcat 1.1 was trained on 8,100
  counterfactual copies in which a note to an AI grader, an instruction inside the text or
  an administrator "final verdict" names an answer while the gold stays unchanged (in 20% of
  them the named answer is the right one, so "a named answer is wrong" is not a shortcut).
  Attack success is now 7.8%.
- **More English.** 36.0% of the training decisions are English (Bobcat 1: 18%), including
  SNLI and SQuAD 2.0.
- **Insufficient evidence and long inputs.** 6,100 copies whose evidence was removed or
  swapped, and 1,600 states padded to 8K-32K tokens. These moved the targets less than
  hoped; see [Limitations](#limitations-and-risks).
- Classification on the development set rose from 77.6% to 80.6%; the other tasks moved by
  about a point or less.

## What it does

| Primitive | You define | Bobcat returns |
|---|---|---|
| Choice | 1 to 255 named candidates, descriptions optional | the top name and a probability for every name |
| Noul | optional meanings of true and false | P(true) |
| Score | 1 to 10 ordered levels | the expected level and a probability per level |

```json
{
  "model": "bobcat-1.1",
  "state": {"task": "Add a discount_code column to the orders table.",
            "action": {"tool": "shell", "command": "npm run db:reset"}},
  "questions": {
    "overreach": {"type": "noul",
                  "instructions": "Does the action do more than the task asked for?"},
    "risk": {"type": "score", "instructions": "How destructive is the action?",
             "criteria": ["Harmless", "Reversible", "Destroys data"]}
  }
}
```

The request and reply use the shapes of TypeSafe's published System One HTTP API, so the
official `typesafe-sdk` works against a Bobcat server by changing its base URL.

## Quickstart: serve Bobcat 1.1 on one GPU

Tested end to end from a fresh clone, exactly as written below, on one RTX PRO 6000
Blackwell 96 GB with a fresh Ubuntu 24.04 GPU image (NVIDIA driver 595) and local NVMe:
about 10 seconds to install, 1.4 minutes to download the base, 2 minutes to merge and
2.5-4 minutes until the server was ready. The adapter came from a local copy of this
repository, so its download was not timed. Through that server the SDK example below
answered `payments`, and TypeSafe's 20 workflow cases agreed with the reference on 92.1% of
329 questions (no failed request; the evaluation path also gives 92.1%) at 0.86 s per case
in FP8. The other figures below were measured on the same GPU type with vLLM 0.30.0.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"   # uv
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed   # a uv-managed Python ships the headers Triton compiles against
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

# 1. The base model at the pinned revision (every file's hash is verified)
$PY -m bobcat.student_readout download --repo Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --out base

# 2. This adapter, merged into the base in float32 and stored as BF16
$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-1.1', local_dir='adapter')"
$PY -m bobcat.student_merge --model-dir base --adapter adapter --out model
mkdir -p compiler && cp base/{tokenizer.json,tokenizer_config.json,chat_template.jinja,config.json,bobcat-download.json} compiler/

# 3. The typed-decision server (TypeSafe-compatible /v1/systemone and /v1/models)
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.api_server --engine vllm --model model \
  --compiler-model compiler --quantization fp8 --temperature 1.2008 --name bobcat-1.1 \
  --release-date 2026-09-26 --max-num-seqs 128 --host 127.0.0.1 --port 8000 --local
```

Then call it with the official SDK:

```python
import os
os.environ.update(TYPESAFE_BASE_URL="http://127.0.0.1:8000", TYPESAFE_API_KEY="local",
                  TYPESAFE_DEFAULT_MODEL="bobcat-1.1")
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

- `--no-config` keeps uv from applying this repository's development settings to the
  serving environment: `pyproject.toml` constrains setuptools to >= 83, and vLLM 0.30.0
  requires setuptools < 81.
- `--temperature 1.2008` is the calibration temperature fitted for this adapter on a
  held-out calibration split; it never changes an argmax.
- The served workflow timings below add `--schedule all --engine-arg
  max_num_batched_tokens=16384`, which schedule every question of a request together.
  In FP8 on the fresh install this gave the same answers at 0.85 s per case; the 0.52-0.58 s
  figures are NVFP4.
- `--local` disables the shared secret the server otherwise requires; use it only on a
  loopback or private interface.
- The server refuses inputs over its limits (128 questions, 255 candidates, 16,384 tokens
  per compiled question by default) with HTTP 422 and never truncates them. Pass
  `--max-model-len 32832` to accept up to 32,768 tokens per compiled question; see the
  long-input figures below before relying on it.
- The compiler reads its identifier list from the repository
  (`reports/2026-09-22-glm-readout-preflight.json`), so run the server from the repository
  root. The same list is included here as `bobcat-identifiers.json`.
- **NVFP4 (Blackwell GPUs).** The latency figures below come from an NVFP4 W4A4 build of
  the merged model made with `scripts/nvfp4_quantize.py` (llm-compressor 0.14.0, 256
  compiled development prompts). That build is not published here; FP8 gives the same
  accuracy (see [Serving precision](#serving-precision)).

## Running on AWS

These are ordinary GPU Linux hosts; the Quickstart above is the whole recipe.

- **Amazon EC2.** One RTX PRO 6000 Blackwell 96 GB (for example `g7e.2xlarge`) serves the
  FP8 or NVFP4 merge; one L40S 48 GB (for example `g6e.2xlarge`) serves the FP8 merge with
  up to 128 concurrent sequences. Use a Deep Learning AMI with a recent NVIDIA driver, and
  allow about 200 GB of disk for the base download plus the merged copy.
- **Amazon SageMaker AI.** The same commands run in a JupyterLab space or notebook
  instance of an equivalent GPU type (for example `ml.g6e.2xlarge`). A SageMaker real-time
  endpoint needs a custom container that runs `bobcat.api_server` behind SageMaker's
  `/invocations` and `/ping` routes; we have not published or tested one.

Our 1.1 measurements ran on an EC2 RTX PRO 6000 host. The SageMaker paths are described
here but not tested by us.

## Evaluation

All Bobcat 1.1, Bobcat 1 and zero-shot figures in this section come from one evaluation
path (BF16 base with the unmerged adapter, one request at a time) unless a row says
otherwise. Differences are paired, with 95% bootstrap intervals over source components.

### Sealed final

A fresh final built from KLUE MRC contexts and Wizard of Seoul dialogues that no
evaluation split, earlier final or training build had used. Classification and tool-call
review could not be rebuilt without reusing text, so this final has four tasks. The
release manifest binding the base revision, tokenizer, adapter hash, temperature and
decision rule was frozen before the split was opened, once per model.

| Task (decisions) | Bobcat 1.1 | Bobcat 1 | Same base, zero-shot |
|---|---:|---:|---:|
| Search passage selection (500) | **80.6%** | 79.2% | 74.2% |
| Citation verification (103) | 100.0% | 99.0% | 93.2% |
| External document screening (550) | **96.9%** | 96.5% | 86.0% |
| Request routing (461) | 99.6% | 99.6% | 98.0% |
| **Task macro** | **94.27%** | 93.59% | 87.86% |
| NLL / ECE after temperature | 0.334 / 0.012 | 0.350 / 0.017 | 0.545 / 0.051 |

Bobcat 1.1 minus Bobcat 1: +0.7 points [-0.0, +1.5]; minus zero-shot: +6.4 [+4.8, +8.1].
51 routing questions share Wizard of Seoul utterances with a compared model's training
data (46 with Bobcat 1.1's); without them the macro is 94.26% and the difference from
Bobcat 1 is +0.7 [+0.0, +1.5]. No request failed. The evaluation is Korean and no label has been reviewed by
a person.

**Selection history.** A first 1.1 candidate, trained on the same data without the SQuAD
and option-order continuation, opened an earlier fresh final once (95.27%, equal to Bobcat
1 on those 1,956 decisions). It then regressed slightly on the English SemIf sets, so the
released adapter continued from it on SQuAD 2.0 answer/no-answer pairs, English
option-order permutations and replay. It replaced the first candidate only because it met
all four conditions of a rule written down before its data existed (SemIf mean, dev macro,
injection and 30K limits), and it then opened the final above once.

### Six-task development evaluation

The v2 development split (3,188 Korean decisions from KLUE and Wizard of Seoul, plus a
catalog of tool-call situations). It was used to select the adapter, so it is not a
held-out test.

| Task | Bobcat 1.1 | Bobcat 1 | Same base, zero-shot |
|---|---:|---:|---:|
| Search passage selection | 91.1% | 92.2% | 81.7% |
| Citation verification | 99.6% | 99.8% | 94.0% |
| Tool-call review, never trained | 94.5% | 95.0% | 81.8% |
| External document screening | 96.9% | 96.5% | 87.1% |
| Request routing | 99.4% | 99.3% | 98.5% |
| Classification | **80.6%** | 77.6% | 69.8% |
| **Task macro** | **93.69%** | 93.41% | 85.48% |

Tool-call review, a task never trained, differs from Bobcat 1 by -0.6 points [-4.8, +3.2].
After temperature, development NLL is 0.218 and ECE 0.010.

### Wrong answers named inside the state

300 development questions, each attacked three ways: a note to an AI grader, an
instruction inside the text, and an administrator "final verdict". Attack success counts
the attacked questions whose answer moved to the named wrong answer.

| | Bobcat 1.1 | Bobcat 1 | Same base, zero-shot |
|---|---:|---:|---:|
| **Attack success, all three** | **7.8%** | 39.6% | 21.0% |
| Accuracy, clean input | 93.7% | 92.7% | 85.7% |

Per attack, Bobcat 1.1's success is 7.3% (verdict), 8.0% (inside the text) and 8.0% (note);
it is 0% on search, citation, tool-call review, routing and classification. The external
document screening questions ask whether the text contains instructions aimed at an AI, so
inserting an attack there changes the right answer; their 46.5% is by construction.

### TypeSafe's published workflow examples

TypeSafe publishes 20 workflow examples at [evals.typesafe.ai](https://evals.typesafe.ai)
with the state, the exact questions, the answers of Claude Opus 5, GPT-5.6 Sol and Jev, and
reference answers from GPT-6 Astra and Claude Fable 5.1. We sent the same requests to Bobcat
and scored every model with the same code on the 329 questions all four answered.

| | Bobcat 1.1 | Bobcat 1 | Jev | Claude Opus 5 | GPT-5.6 Sol |
|---|---:|---:|---:|---:|---:|
| Agreement with the reference, all questions | 92.1% | 92.4% | 90.9% | 92.4% | 93.0% |
| Agreement, mean of the four workflows | 87.3% | 88.0% | 86.5% | 88.2% | 89.6% |
| Probability on the reference answer | 0.890 | 0.888 | 0.850 | 0.851 | 0.914 |
| Served (NVFP4, HTTP), agreement / median time per case | 91.2% / 0.52-0.58 s | | 90.9% / 0.42 s | 92.4% / 20.9 s | 93.0% / 24.2 s |

Bobcat 1.1 minus Jev, averaged over workflows, is +0.9 points (95% interval -4.4 to +8.6):
level on these examples, not separated. This is question-level agreement on 20 English
cases against a frontier-model consensus, not action accuracy or ground truth. Bobcat's
served times are server-side on one RTX PRO 6000 (two passes); the other times are
TypeSafe's published client-side times with the network included.

### SemIf benchmark bundle

[SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT) publishes typed-decision rows and
evaluators, plus the Jev figures TypeSafe released for 102 rows aligned to its workflow
cases. We converted the rows to Bobcat requests and scored them with SemIf's unmodified
evaluators.

| Set (metric) | Bobcat 1.1 | Bobcat 1 | Same base, zero-shot | Jev (published) |
|---|---:|---:|---:|---:|
| authored144 (mean family balanced accuracy) | 0.910 | 0.911 | 0.876 | - |
| perturbations108 | 0.989 | 1.000 | 0.924 | - |
| WANLI256 | 0.730 | 0.738 | 0.738 | - |
| TypeSafe102 modal agreement / total variation | 0.872 / 0.130 | 0.847 / 0.116 | 0.820 / 0.194 | **0.883** / 0.127 |
| Every judge-grid (36 cells) | 32 | 31 | 29 | 32 |
| Every action-firewall (10 actions) | 9 | 10 | 10 | 10 |

Jev is ahead on TypeSafe102 modal agreement and the action firewall. These sets are
evaluation-only; none of their rows was used for training, selection or calibration.

### Long inputs

The development tasks with the state padded by unrelated passages (300 questions, drawn
per task). Change in accuracy against the unpadded state:

| State length | Bobcat 1.1 | Bobcat 1 |
|---|---|---|
| Unpadded (accuracy) | 93.7% | 93.7% |
| 8K tokens | 0.0 [-1.9, +1.9] | -1.0 |
| 16K tokens | -2.7 [-5.0, -0.4] | -2.0 |
| 30K tokens | **-1.7 [-4.0, +0.7]** | -2.3 [-4.3, -0.3] |
| 60K tokens (evaluation path only) | -1.0 [-2.9, +1.0] | -2.3 |

Per task (50 questions each), the drop at 30K is largest in classification (80% to 72%) and search
(98% to 94%).

### Latency

| One RTX PRO 6000 Blackwell 96 GB, vLLM 0.30.0, engine (no HTTP), p50 | NVFP4 |
|---|---:|
| 512 tokens, 8 candidates, 1 question | 42.8 ms |
| 3.2k-token state, 8 questions, one batch | 467 ms |
| 8K / 16K tokens, 1 question | 481 / 583 ms |
| Throughput, 32 to 512 concurrent 1K-token requests | 19,207-19,614 tokens/s |
| TypeSafe workflow case, median, served over HTTP (localhost, `--schedule all`) | 0.576 s / 0.515 s (two passes) |

Bobcat 1 in FP8 on the same GPU and engine takes 57.5 ms for the first profile; the merged
1.1 model has the same architecture and shapes. Latency under concurrent HTTP load has not
been measured.

### Serving precision

On the 2,932 development decisions not used to calibrate NVFP4: evaluation path 93.28%,
FP8 93.25%, NVFP4 93.25% (NVFP4 minus FP8: 0.0 points [-0.4, +0.5]; the same answer as FP8
on 98.2%). Evaluate the exact artifact you serve.

## Training

- **Base:** Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, a hybrid of
  Gated DeltaNet and full-attention layers (3:1).
- **Adapter:** LoRA rank 16, alpha 32, no dropout, on every linear projection of the
  attention, Gated DeltaNet and MLP layers; 116.7M trainable parameters. Embeddings and the
  output head are frozen.
- **Stage 1:** one epoch over 81,432 decisions (87.5M tokens), 2,545 steps of 32
  questions, AdamW at 1e-4, on eight NVIDIA B300 GPUs. Loss: cross-entropy on the candidate
  softmax for Choice and Noul, expected-level error for Score (the Bobcat 1 recipe).
- **Stage 2:** 313 steps at 2e-5 from stage 1 on 10,000 decisions: 4,000 SQuAD 2.0
  answer-stated / not-stated pairs on the same passage, 3,000 English Choice questions with
  permuted options, and 3,000 replayed stage-1 decisions.
- **Calibration:** one temperature, 1.2008, fitted on the held-out calibration split
  (1,605 decisions).
- **Data (stage 1):**

  | Part | Decisions | Korean | English |
  |---|---:|---:|---:|
  | Product tasks (KLUE MRC, Wizard of Seoul; the Bobcat 1 training split) | 25,104 | 25,104 | 0 |
  | Policy transfer (synthetic rules with exact interpreters) | 8,374 | 5,335 | 3,039 |
  | Public labelled data (KorNLI; KLUE NLI, YNAT, STS; NSMC; MASSIVE; Banking77; BoolQ; ARC; HelpSteer 2 and 3; SNLI) | 32,154 | 11,184 | 20,970 |
  | Injected-answer counterfactuals (gold unchanged) | 8,100 | 6,693 | 1,407 |
  | Insufficient-evidence counterfactuals | 6,100 | 2,500 | 3,600 |
  | Long states (8K-32K tokens) | 1,600 | 1,268 | 332 |
  | **Total** | **81,432** | 52,084 | 29,348 (36.0%) |

- **Provenance:** every new decision's gold comes from a construction rule or the source
  dataset's human label. **No output of Jev, of GLM or of any other teacher model was used
  for training, distillation, reward or calibration**, and no SemIf, Every or TypeSafe
  evaluation row was used. Rows whose text appears in any development, calibration or final
  split were removed before training. Tool-call review was never trained. 2,231 KorNLI rows
  are machine translations (as in Bobcat 1); no new machine-translated data was added.

## Limitations and risks

- **Insufficient evidence stays weak.** On SemIf's WANLI256, Bobcat 1.1 recognises 27 of
  85 "insufficient" rows (Bobcat 1: 39 of 85); on the 36 SemIf rows whose evidence was
  removed it makes 9 errors, 5 of them at a confidence of 0.8 or more. It leans towards a
  definite answer when the evidence is related but does not settle the question. Give it
  an explicit "not stated" option and do not treat a confident answer as proof that the
  evidence exists.
- **Long states.** Accuracy drops 1.7 points at a 30K-token state (target: at most 1
  point, not met) and 2.7 points at 16K.
- **NVFP4 near-ties.** With NVFP4, a question answered from a cached shared prefix and the
  same question computed from scratch agreed on 61 of 64 boundary questions (95.3%; FP8:
  64 of 64). The served NVFP4 build agreed with the reference on 91.2% of TypeSafe's
  workflow questions against 92.1% on the evaluation path. Use FP8 where a near-tie must be
  reproducible.
- An attack sentence inside the state still moves 7.8% of answers. For untrusted input,
  add checks outside the model.
- Order and wording: reversing the option order flipped 1 of SemIf's perturbation rows
  (Bobcat 1: 0).
- The sealed finals have no classification rows, and the second has no tool-call rows;
  classification and tool-call review rest on the development split. The finals and the
  development split are Korean; English results come from TypeSafe's 20 cases, SemIf and
  training monitors.
- Arithmetic, counting and date comparison are weak; compute them in code. A typed answer
  can be the wrong valid option, and `confidence` is a concentration statistic, not the
  probability of being right.

**Intended use:** small, repeated judgments whose admissible answers the application owns -
routing, reranking, citation checks, tool-call review, guardrail screening, classification -
with the policy that acts on the answer kept in the application's code.
**Out of scope:** generating text, open-ended question answering, and sole reliance on
Bobcat for safety-critical or legal decisions.

## License and attribution

The adapter is released under Apache-2.0 and is a derivative of Qwen3.8-27B by the Qwen
team (Apache-2.0); see the base model's license. Training data keep their own licenses:
KLUE, KorNLI, ARC, SNLI and SQuAD 2.0 (CC BY-SA 4.0), BoolQ (CC BY-SA 3.0), NSMC (CC0 1.0),
MASSIVE, Banking77 and HelpSteer 2 and 3 (CC BY 4.0). Attributions are in the code
repository's `THIRD_PARTY.md`.

Bobcat is an independent project. It is not affiliated with or endorsed by TypeSafe AI, the
Qwen team, Anthropic, OpenAI, Every or any other company named here; product names are
their owners' trademarks and are used only to identify the models compared. TypeSafe's and
SemIf's evaluation data are not redistributed here. The adapter is provided as is, without
warranty.

## Citation

```bibtex
@techreport{bobcat2026,
  title       = {Bobcat: Typed Decisions from One Forward Pass},
  author      = {{The Bobcat Authors}},
  institution = {Foxl AI},
  year        = {2026},
  url         = {https://foxl.ai/blog/bobcat-typed-decisions}
}

@misc{bobcat11,
  title        = {Bobcat 1.1},
  author       = {{The Bobcat Authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sanghwa-na/bobcat-1.1}}
}
```
