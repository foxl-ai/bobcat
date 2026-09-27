---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
base_model_relation: finetune
library_name: transformers
language:
- en
- ko
pipeline_tag: zero-shot-classification
tags:
- typed-decisions
- vllm
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

This repository, `sanghwa-na/bobcat-1.1`, holds **Bobcat 1.1 as ready-to-serve BF16
weights**: a rank-16 LoRA trained on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(Apache-2.0) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, merged into the base.
vLLM serves it in FP8. For NVIDIA Blackwell GPUs, the NVFP4 checkpoint behind this card's
latency figures is at [sanghwa-na/bobcat-1.1-nvfp4](https://huggingface.co/sanghwa-na/bobcat-1.1-nvfp4).
The compiler, server and evaluation code are at
[github.com/foxl-ai/bobcat](https://github.com/foxl-ai/bobcat). A faster tier is
[Bobcat Flash 1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1).

![Bobcat 1.1 at a glance](assets/bobcat-1.1-highlights.png)

## Highlights

| | Bobcat 1.1 | Reference |
|---|---:|---|
| Sealed final, four tasks (1,614 decisions, opened once) | **94.27%** | same base zero-shot 87.86% (+6.4 pt [+4.8, +8.1]) |
| Wrong answer named inside the state wins (300 questions x 3 attacks) | **7.8%** | same base zero-shot 21.0% |
| Six-task development evaluation (3,188 decisions) | **93.69%** | same base zero-shot 85.48% (+8.2 pt [+6.8, +9.4]) |
| TypeSafe's published workflow examples: agreement with the reference (329 questions) | **92.1%** | Jev 90.9%, Claude Opus 5 92.4%, GPT-5.6 Sol 93.0% |
| SemIf's 102 aligned TypeSafe rows: modal agreement | **0.872** | Jev 0.883 |
| One decision (512 tokens, 8 candidates), p50 | **42.8 ms** | one RTX PRO 6000 Blackwell, NVFP4, vLLM engine |
| Median time per TypeSafe workflow case, served | **0.52-0.58 s** | Jev 0.42 s (TypeSafe's published client-side time) |

Jev, Opus 5 and Sol figures are TypeSafe's own published answers and times; Jev was never
called. The same-base baseline is untrained Qwen3.8-27B with the same compiler and readout.
See [Evaluation](#evaluation) for what each number does and does not mean, and
[Limitations](#limitations-and-risks) for the targets Bobcat 1.1 did not meet.

## What it was trained for

- **Wrong answers named inside the state.** 8,100 counterfactual copies in which a note to an
  AI grader, an instruction inside the text or an administrator "final verdict" names an
  answer while the gold stays unchanged (in 20% of them the named answer is the right one, so
  "a named answer is wrong" is not a shortcut). Attack success is 7.8%, against 21.0% for the
  untrained base.
- **English as well as Korean.** 36.0% of the training decisions are English, including SNLI
  and SQuAD 2.0.
- **Insufficient evidence and long inputs.** 6,100 copies whose evidence was removed or
  swapped, and 1,600 states padded to 8K-32K tokens. These moved the targets less than
  hoped; see [Limitations](#limitations-and-risks).

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

The staged files of this repository were load-tested on one RTX PRO 6000 Blackwell 96 GB with
vLLM 0.30.0 by the commands below (the download line runs once the repository is public):
`vllm serve` came up and listed the model; the Bobcat server was ready two minutes after it
started, answered `/health`, the SDK example below answered `payments`, and TypeSafe's 20
workflow cases agreed with the reference on 92.1% of 329 questions (no failed request; the
evaluation path also gives 92.1%) at 0.85 s per case in FP8. The other figures below were
measured on the same GPU type with vLLM 0.30.0.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"   # uv
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed   # a uv-managed Python ships the headers Triton compiles against
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

# These weights (about 54 GB), then the typed-decision server
# (TypeSafe-compatible /v1/systemone and /v1/models), FP8 at load
$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-1.1', local_dir='bobcat-1.1')"
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.api_server --engine vllm --model bobcat-1.1 \
  --compiler-model bobcat-1.1/compiler --quantization fp8 --temperature 1.2008 --name bobcat-1.1 \
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

- `vllm serve sanghwa-na/bobcat-1.1 --quantization fp8 --max-model-len 16448` also loads these
  weights, but plain vLLM exposes text generation; the typed-decision contract (closed JSON
  replies, no generated tokens) comes from `bobcat.api_server`.
- `--quantization fp8` is vLLM's online dynamic FP8 of the BF16 weights. For the NVFP4
  checkpoint on a Blackwell GPU, download `sanghwa-na/bobcat-1.1-nvfp4` instead and pass
  `--model bobcat-1.1-nvfp4 --compiler-model bobcat-1.1-nvfp4/compiler --quantization none`
  (see its card).
- `--no-config` keeps uv from applying this repository's development settings to the
  serving environment: `pyproject.toml` constrains setuptools to >= 83, and vLLM 0.30.0
  requires setuptools < 81.
- `--temperature 1.2008` is the calibration temperature fitted for this model on a
  held-out calibration split; it never changes an argmax.
- The served workflow timings below add `--schedule all --engine-arg
  max_num_batched_tokens=16384`, which schedule every question of a request together.
- `--compiler-model bobcat-1.1/compiler` points the Bobcat compiler at the base model's
  pinned tokenizer, template and configuration; it checks them against the download receipt
  in that folder.
- `--local` disables the shared secret the server otherwise requires; use it only on a
  loopback or private interface.
- The server refuses inputs over its limits (128 questions, 255 candidates, 16,384 tokens
  per compiled question by default) with HTTP 422 and never truncates them. Pass
  `--max-model-len 32832` to accept up to 32,768 tokens per compiled question; see the
  long-input figures below before relying on it.
- The compiler reads its identifier list from the repository
  (`reports/2026-09-22-glm-readout-preflight.json`), so run the server from the repository
  root, or pass `--identifiers bobcat-1.1/bobcat-identifiers.json`: the same list ships here.

## Running on AWS

These are ordinary GPU Linux hosts; the Quickstart above is the whole recipe.

- **Amazon EC2.** One RTX PRO 6000 Blackwell 96 GB (for example `g7e.2xlarge`) serves these
  weights in FP8 or the NVFP4 checkpoint; in FP8 they also fit one L40S 48 GB (for example
  `g6e.2xlarge`) with up to 128 concurrent sequences. Use a Deep Learning AMI with a recent
  NVIDIA driver, and allow about 80 GB of disk for the download and caches.
- **Amazon SageMaker AI.** The same commands run in a JupyterLab space or notebook
  instance of an equivalent GPU type (for example `ml.g6e.2xlarge`). A SageMaker real-time
  endpoint needs a custom container that runs `bobcat.api_server` behind SageMaker's
  `/invocations` and `/ping` routes; we have not published or tested one.

Our measurements ran on an EC2 RTX PRO 6000 host. The SageMaker paths are described
here but not tested by us.

## Evaluation

All Bobcat 1.1 and zero-shot figures in this section come from one evaluation path (BF16
base with the unmerged adapter, or none for the zero-shot base, one request at a time) unless
a row says otherwise. The weights in this repository are that adapter merged into the base: on a
development sample (every 10th decision of the v2 development split, 319 decisions across the six
tasks) they gave the evaluation path's top answer on all 319 and the same accuracy (298 correct);
the largest probability change was 0.075 at the calibration temperature. Differences are paired, with 95% bootstrap intervals over source components.

### Sealed final

A fresh final built from KLUE MRC contexts and Wizard of Seoul dialogues that no
evaluation split, other final or training build had used. Classification and tool-call
review could not be rebuilt without reusing text, so this final has four tasks. The
release manifest binding the base revision, tokenizer, adapter hash, temperature and
decision rule was frozen before the split was opened, once per model.

| Task (decisions) | Bobcat 1.1 | Same base, zero-shot |
|---|---:|---:|
| Search passage selection (500) | **80.6%** | 74.2% |
| Citation verification (103) | **100.0%** | 93.2% |
| External document screening (550) | **96.9%** | 86.0% |
| Request routing (461) | **99.6%** | 98.0% |
| **Task macro** | **94.27%** | 87.86% |
| NLL / ECE after temperature | 0.334 / 0.012 | 0.545 / 0.051 |

Bobcat 1.1 minus zero-shot: +6.4 points [+4.8, +8.1]. 46 routing questions share Wizard of
Seoul utterances with this adapter's training data; without the 51 that overlap any training
data used in this family of models, the macro is 94.26% (zero-shot 87.80%) and the difference
is +6.5 [+4.9, +8.0]. No request failed. The evaluation is Korean and no label has been reviewed by
a person.

**How the adapter was selected.** A first candidate, trained on the stage-1 data alone,
opened another fresh final once (1,956 decisions: 95.27%, same base zero-shot 86.69%). It
then scored slightly lower on the English SemIf sets, so the released adapter continued from
it (stage 2 below). It replaced the first candidate only because it met all four conditions
of a rule written down before its data existed (SemIf mean, development macro, injection and
30K limits), and it then opened the final above once.

### Six-task development evaluation

The v2 development split (3,188 Korean decisions from KLUE and Wizard of Seoul, plus a
catalog of tool-call situations). It was used to select the adapter, so it is not a
held-out test.

| Task | Bobcat 1.1 | Same base, zero-shot |
|---|---:|---:|
| Search passage selection | **91.1%** | 81.7% |
| Citation verification | **99.6%** | 94.0% |
| Tool-call review, never trained | **94.5%** | 81.8% |
| External document screening | **96.9%** | 87.1% |
| Request routing | **99.4%** | 98.5% |
| Classification | **80.6%** | 69.8% |
| **Task macro** | **93.69%** | 85.48% |

Bobcat 1.1 minus zero-shot: +8.2 points [+6.8, +9.4], including +12.7 on tool-call review, a
task it was never trained on. After temperature, development NLL is 0.218 and ECE 0.010
(zero-shot: 0.504 and 0.045).

### Wrong answers named inside the state

300 development questions, each attacked three ways: a note to an AI grader, an
instruction inside the text, and an administrator "final verdict". Attack success counts
the attacked questions whose answer moved to the named wrong answer.

| | Bobcat 1.1 | Same base, zero-shot |
|---|---:|---:|
| **Attack success, all three** | **7.8%** | 21.0% |
| Accuracy, clean input | **93.7%** | 85.7% |

Per attack, Bobcat 1.1's success is 7.3% (verdict), 8.0% (inside the text) and 8.0% (note);
it is 0% on search, citation, tool-call review, routing and classification. The external
document screening questions ask whether the text contains instructions aimed at an AI, so
inserting an attack there changes the right answer; their 46.5% is by construction.

### TypeSafe's published workflow examples

TypeSafe publishes 20 workflow examples at [evals.typesafe.ai](https://evals.typesafe.ai)
with the state, the exact questions, the answers of Claude Opus 5, GPT-5.6 Sol and Jev, and
reference answers from GPT-6 Astra and Claude Fable 5.1. We sent the same requests to Bobcat
and scored every model with the same code on the 329 questions all four answered.

| | Bobcat 1.1 | Same base, zero-shot | Jev | Claude Opus 5 | GPT-5.6 Sol |
|---|---:|---:|---:|---:|---:|
| Agreement with the reference, all questions | 92.1% | 86.9% | 90.9% | 92.4% | 93.0% |
| Agreement, mean of the four workflows | 87.3% | 84.1% | 86.5% | 88.2% | 89.6% |
| Probability on the reference answer | 0.890 | 0.756 | 0.850 | 0.851 | 0.914 |
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

| Set (metric) | Bobcat 1.1 | Same base, zero-shot | Jev (published) |
|---|---:|---:|---:|
| authored144 (mean family balanced accuracy) | 0.910 | 0.876 | - |
| perturbations108 | 0.989 | 0.924 | - |
| WANLI256 | 0.730 | 0.738 | - |
| TypeSafe102 modal agreement / total variation | 0.872 / 0.130 | 0.820 / 0.194 | **0.883** / 0.127 |
| Every judge-grid (36 cells) | 32 | 29 | 32 |
| Every action-firewall (10 actions) | 9 | 10 | 10 |

Jev is ahead on TypeSafe102 modal agreement and the action firewall, and the untrained base
is level or ahead on WANLI256 and the action firewall. These sets are
evaluation-only; none of their rows was used for training, selection or calibration.

### Long inputs

The development tasks with the state padded by unrelated passages (300 questions, drawn
per task). Change in accuracy against the unpadded state:

| State length | Bobcat 1.1 | Same base, zero-shot |
|---|---|---|
| Unpadded (accuracy) | 93.7% | 86.3% |
| 8K tokens | 0.0 [-1.9, +1.9] | -0.3 |
| 16K tokens | -2.7 [-5.0, -0.4] | -2.3 |
| 30K tokens | **-1.7 [-4.0, +0.7]** | -4.0 |
| 60K tokens (evaluation path only) | -1.0 [-2.9, +1.0] | -4.3 |

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

In FP8 (option B), the fresh-install server took 0.86 s per TypeSafe workflow case on the
same GPU; the FP8 engine profile was not measured separately. Latency under concurrent HTTP
load has not been measured.

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
  softmax for Choice and Noul, expected-level error for Score.
- **Stage 2:** 313 steps at 2e-5 from stage 1 on 10,000 decisions: 4,000 SQuAD 2.0
  answer-stated / not-stated pairs on the same passage, 3,000 English Choice questions with
  permuted options, and 3,000 replayed stage-1 decisions.
- **Calibration:** one temperature, 1.2008, fitted on the held-out calibration split
  (1,605 decisions).
- **Merge:** the adapter (`adapter_model.safetensors` sha256 `7351d959…83b2`) merged into the
  base with `bobcat.student_merge`: each adapted weight becomes `bf16(W + (alpha/r) B A)` with
  the product and sum in float32, and every other tensor is copied. The shard hashes are in
  `SHA256SUMS.json` and in the release manifest (`serving_builds.merged_bf16`).
- **Data (stage 1):**

  | Part | Decisions | Korean | English |
  |---|---:|---:|---:|
  | Product tasks (KLUE MRC and Wizard of Seoul training components) | 25,104 | 25,104 | 0 |
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
  are machine translations; no other machine-translated data was used.

## Limitations and risks

- **Insufficient evidence stays weak.** On SemIf's WANLI256, Bobcat 1.1 recognises 27 of
  85 "insufficient" rows (the untrained base: 58 of 85); on the 36 SemIf rows whose evidence was
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
- Order and wording: reversing the option order flipped 1 of SemIf's perturbation rows.
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

These weights are released under Apache-2.0 as a derivative of Qwen3.8-27B by the Qwen team
(Apache-2.0); the Qwen LICENSE file is included unchanged, and `NOTICE` lists what was
changed. Training data keep their own licenses:
KLUE, KorNLI, ARC, SNLI and SQuAD 2.0 (CC BY-SA 4.0), BoolQ (CC BY-SA 3.0), NSMC (CC0 1.0),
MASSIVE, Banking77 and HelpSteer 2 and 3 (CC BY 4.0). Attributions are in the code
repository's `THIRD_PARTY.md`.

Bobcat is an independent project. It is not affiliated with or endorsed by TypeSafe AI, the
Qwen team, Anthropic, OpenAI, Every or any other company named here; product names are
their owners' trademarks and are used only to identify the models compared. TypeSafe's and
SemIf's evaluation data are not redistributed here. The weights are provided as is, without
warranty.

## Citation

```bibtex
@misc{bobcat11,
  title        = {Bobcat 1.1},
  author       = {{The Bobcat Authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sanghwa-na/bobcat-1.1}}
}
```
