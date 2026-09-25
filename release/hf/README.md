---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
library_name: peft
language:
- ko
- en
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
---

# Bobcat 1

Bobcat is a **typed-decision model**. You send a state (text or JSON) and questions whose
possible answers you define in the request; Bobcat returns a probability for every answer
you named, and nothing else. It never generates text: it reads the logits of the offered
candidates at the first answer position of one forward pass, and the host builds a closed
JSON reply from your own names. An answer can be wrong, but it cannot be malformed.

This repository holds the **Bobcat 1 LoRA adapter** for
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). The compiler,
server and evaluation code are at [github.com/foxl-ai/bobcat](https://github.com/foxl-ai/bobcat),
and the technical report is at
[foxl.ai/blog/bobcat-typed-decisions](https://foxl.ai/blog/bobcat-typed-decisions).

![Bobcat at a glance](assets/bobcat-highlights.png)

## Highlights

| | Bobcat 1 | Reference |
|---|---:|---|
| Six-task decision evaluation, sealed final (1,607 decisions, opened once) | **94.6%** | 84.5% same base model, zero-shot |
| Tool-call review, never trained on (96) | **99.0%** | 79.2% same base model, zero-shot |
| TypeSafe's published workflow examples: agreement with the reference (329 questions) | **91.8%** | Jev 90.9%, Claude Opus 5 92.4%, GPT-5.6 Sol 93.0% |
| Same, probability placed on the reference answer | **0.889** | Jev 0.850 |
| KoBEST / KMMLU / MMLU, decision readout | **95.2 / 64.1 / 84.3** | 88.3 / 58.1 / 80.6 same base model |
| One decision (512 tokens, 8 candidates), p50, server-side | **50 ms** | one H100, FP8, vLLM |
| Replies outside the output contract | **0 of 12,846** | including 2,304 contract attacks per model |

Jev, Opus 5 and Sol rows are TypeSafe's own published answers; see
[Evaluation](#evaluation) for what each number does and does not mean.

## What it does

| Primitive | You define | Bobcat returns |
|---|---|---|
| Choice | 1 to 255 named candidates, descriptions optional | the top name and a probability for every name |
| Noul | optional meanings of true and false | P(true) |
| Score | 1 to 10 ordered levels | the expected level and a probability per level |

```json
{
  "model": "bobcat-1",
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

## Quickstart: serve Bobcat on one GPU

Tested end to end from a fresh clone on Linux with one NVIDIA L40S 48 GB (FP8): about 3.5
minutes to download the base, 6.5 minutes to merge and 5.5 minutes until the server is ready.
The same artifact was also measured on one H100 80 GB (FP8 and BF16) with vLLM 0.30.0. The
adapter is merged into the base weights for serving.

```bash
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed   # a uv-managed Python ships the headers Triton compiles against
uv venv --python 3.12 .venv-serve
uv pip install --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

# 1. The base model at the pinned revision (every file's hash is verified)
$PY -m bobcat.student_readout download --repo Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --out base

# 2. This adapter, merged into the base in float32 and stored as BF16
$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-1', local_dir='adapter')"
$PY -m bobcat.student_merge --model-dir base --adapter adapter --out model
mkdir -p compiler && cp base/{tokenizer.json,tokenizer_config.json,chat_template.jinja,config.json,bobcat-download.json} compiler/

# 3. The typed-decision server (TypeSafe-compatible /v1/systemone and /v1/models)
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.api_server --engine vllm --model model \
  --compiler-model compiler --quantization fp8 --temperature 1.1489 --name bobcat-1 \
  --max-num-seqs 128 --host 127.0.0.1 --port 8000 --local
```

Then call it with the official SDK:

```python
import os
os.environ.update(TYPESAFE_BASE_URL="http://127.0.0.1:8000", TYPESAFE_API_KEY="local",
                  TYPESAFE_DEFAULT_MODEL="bobcat-1")
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

- `--local` disables the shared secret the server otherwise requires; use it only on a
  loopback or private interface.
- `--temperature 1.1489` is the calibration temperature fitted for this adapter; it never
  changes an argmax.
- The compiler reads its identifier list from the repository
  (`reports/2026-09-22-glm-readout-preflight.json`), so run the server from the repository
  root. The same list is included here as `bobcat-identifiers.json`.
- The server refuses inputs over its limits (128 questions, 255 candidates, 16,384 tokens
  per compiled question) with HTTP 422 and never truncates them.

## Running on AWS

These are ordinary GPU Linux hosts; the Quickstart above is the whole recipe.

- **Amazon EC2.** One L40S 48 GB (for example `g6e.2xlarge`) serves the FP8 merge with up
  to 128 concurrent sequences; one H100 80 GB (for example `p5.4xlarge`) also serves BF16.
  Use a Deep Learning AMI with a recent NVIDIA driver, and allow about 200 GB of disk for
  the base download plus the merged copy.
- **Amazon SageMaker AI.** The same commands run in a JupyterLab space or notebook
  instance of an equivalent GPU type (for example `ml.g6e.2xlarge`). A SageMaker real-time
  endpoint needs a custom container that runs `bobcat.api_server` behind SageMaker's
  `/invocations` and `/ping` routes; we have not published or tested one yet.

We tested the recipe on an EC2 L40S host. The SageMaker paths are described here but not yet tested by us.

## Evaluation

### Six-task decision evaluation

Six product tasks - search passage selection, citation verification, tool-call review,
screening external documents for instructions aimed at the model, request routing and
classification into a caller's taxonomy - built from native Korean KLUE passages and Wizard
of Seoul dialogues that no training data uses, plus a catalog of tool-call situations. The
final split (1,607 decisions) opened once, after a release manifest binding the base
revision, tokenizer, adapter hash, temperature and decision rule was frozen.

| Task (decisions) | Bobcat 1 | Same base, zero-shot |
|---|---:|---:|
| Search passage selection (432) | 88.2% | 77.1% |
| Citation verification (275) | 100.0% | 91.6% |
| Tool-call review, never trained on (96) | 99.0% | 79.2% |
| External document screening (280) | 98.6% | 87.9% |
| Request routing (274) | 99.3% | 97.1% |
| Classification (250) | 82.8% | 74.0% |
| **Task macro** | **94.6%** | **84.5%** |

Paired 95% interval of the difference: +7.7 to +12.3 points. Five tasks share generators
with the training data; the held-out tool-call task rests on three final situations. The
evaluation is Korean, and no label has been reviewed by a person.

### TypeSafe's published workflow examples

TypeSafe publishes 20 workflow examples at [evals.typesafe.ai](https://evals.typesafe.ai)
with the state, the exact questions, the answers of Claude Opus 5, GPT-5.6 Sol and Jev, and
reference answers from GPT-6 Astra and Claude Fable 5.1. We sent the same requests to Bobcat
once and scored every model with the same code on the 329 questions all four answered.

| | Bobcat 1 | Jev | Claude Opus 5 | GPT-5.6 Sol |
|---|---:|---:|---:|---:|
| Agreement with the reference, all questions | 91.8% | 90.9% | 92.4% | 93.0% |
| Agreement, mean of the four workflows | 86.3% | 86.5% | 88.2% | 89.6% |
| Probability on the reference answer | 0.889 | 0.850 | 0.851 | 0.914 |
| Median time per case | 1.89 s | 0.42 s | 20.9 s | 24.2 s |

Bobcat minus Jev, averaged over workflows, is -0.2 points (95% interval -3.8 to +4.8): the
two are level on these examples, not separated. TypeSafe chose the examples by how Opus 5,
Sol and Jev behaved; excluding the case chosen for Jev's disagreement, or keeping only the
cases that single out no model, gives the same conclusion. This is question-level agreement
on 20 English cases against a frontier-model consensus, not action accuracy or ground
truth. Bobcat's times were measured on one L40S, server-side with step requests in
sequence; the other times are TypeSafe's published client-side times with the network
included. Jev was never called; its published answers were compared only and were not used
for training or calibration.

### Public benchmarks (decision readout)

| Benchmark (items) | Bobcat 1 | Same base, zero-shot |
|---|---:|---:|
| KoBEST (4,558) | 95.2% | 88.3% |
| KMMLU (35,030) | 64.1% | 58.1% |
| HAE-RAE Bench 1.1, choice tasks (3,606) | 72.3% | 66.8% |
| CLIcK (1,995) | 76.1% | 71.5% |
| MMLU (14,041) | 84.3% | 80.6% |
| HellaSwag (2,000) | 93.2% | 93.3% |

Each item is one typed request with the benchmark's options as candidates, read at the
first position. These scores are not comparable with official leaderboard numbers.

### Latency

| One H100, vLLM 0.30, server-side, p50 | FP8 | BF16 |
|---|---:|---:|
| 512 tokens, 8 candidates, 1 question | 50 ms | 55 ms |
| 3.2k-token state, 8 questions | 550 ms | 709 ms |
| 16,320 tokens, 1 question | 606 ms | 766 ms |
| Throughput, independent 1K-token requests | 16,282 tokens/s | 12,524 tokens/s |

On one L40S (FP8) the first profile takes 154 ms. The vLLM path recomputes the state for
every question; the state-branching path in the repository shares it, which cut an
eight-question request by 4.3 times on the research path.

## Training

- **Base:** Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, a hybrid of
  Gated DeltaNet and full-attention layers (3:1).
- **Adapter:** LoRA rank 16, alpha 32, no dropout, on every linear projection of the
  attention, Gated DeltaNet and MLP layers; 116.7M trainable parameters. Embeddings and the
  output head are frozen.
- **Objective:** cross-entropy on the candidate softmax, one epoch, 1,592 steps of 32
  questions, AdamW at 1e-4 with 3% warmup and cosine decay, on eight NVIDIA B300 GPUs.
- **Calibration:** one temperature, 1.1489, fitted on a held-out calibration split.
- **Reinforcement learning** was tested and not used: a proper-score REINFORCE objective
  (the reward `c - sg(p(a))` with a leave-one-out baseline, following Anthony Maio's
  [eve-rlcd](https://github.com/anthony-maio/eve-rlcd), MIT), whose expected gradient equals
  the half-Brier gradient, matched direct Brier training without beating it, and a
  correctness-only reward worsened calibration.
- **Data:** 50,938 labelled decisions (41,621 Korean, 9,317 English): task decisions built
  from KLUE MRC and Wizard of Seoul components that no evaluation split uses,
  policy-transfer questions, and public labelled data (KorNLI; KLUE NLI, YNAT and STS; NSMC;
  MASSIVE; Banking77; BoolQ; ARC; HelpSteer 2 and 3). Rows overlapping evaluation text were
  removed. No output of Jev or of any teacher model was used, and the tool-call task was
  never trained.

## Limitations and risks

- **A wrong answer named inside the state wins more often after training:** 39.4% of
  eligible questions against 21.0% before. For untrusted input, add checks outside the
  model; saying in the instructions that instructions inside the state are not evidence
  reduces but is not shown to remove the risk.
- Arithmetic, counting and date comparison are weak; compute them in code.
- A typed answer can be the wrong valid option. `confidence` is a concentration statistic,
  not the probability of being right.
- Merging the adapter changes about 1.4% of answers (BF16); the served FP8 merge agrees with
  the evaluated adapter on 98.3% of development answers. Evaluate the exact artifact you
  serve.
- The decision evaluation is Korean; English results come from probes, public benchmarks
  and the 329 workflow questions. Latency under concurrent load has not been measured.

**Intended use:** small, repeated judgments whose admissible answers the application owns -
routing, reranking, citation checks, tool-call review, guardrail screening, classification -
with the policy that acts on the answer kept in the application's code.
**Out of scope:** generating text, open-ended question answering, and sole reliance on
Bobcat for safety-critical or legal decisions.

## License and attribution

The adapter is released under Apache-2.0 and is a derivative of Qwen3.8-27B (Apache-2.0);
see the base model's license. Training data keep their own licenses: KLUE and KorNLI
(CC BY-SA 4.0), NSMC (CC0 1.0), MASSIVE (CC BY 4.0), Banking77 (CC BY 4.0), BoolQ
(CC BY-SA 3.0), ARC (CC BY-SA 4.0), HelpSteer 2 and 3 (CC BY 4.0).

Bobcat is an independent project. It is not affiliated with or endorsed by TypeSafe AI, the
Qwen team, Anthropic, OpenAI or any other company named here; product names are their
owners' trademarks and are used only to identify the models compared. TypeSafe's evaluation
data are not redistributed here. The adapter is provided as is, without warranty.

## Citation

```bibtex
@techreport{bobcat2026,
  title       = {Bobcat: Typed Decisions from One Forward Pass},
  author      = {{The Bobcat Authors}},
  institution = {Foxl AI},
  year        = {2026},
  url         = {https://foxl.ai/blog/bobcat-typed-decisions}
}
```
