<p align="center">
  <a href="https://foxl.ai"><img src="assets/readme/foxl.svg" width="64" height="64" alt="Foxl" /></a>
</p>

<h1 align="center">Bobcat</h1>

<p align="center">
  <strong>A typed-decision model.</strong><br />
  State in, typed decisions out: Choice, Noul and Score, with a probability for every answer you name.<br />
  No generated text, no per-task fine-tuning.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" /></a>
  <a href="https://huggingface.co/sanghwa-na/bobcat-1.1"><img alt="Hugging Face model: bobcat-1.1" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-bobcat--1.1-ffc107" /></a>
  <a href="https://huggingface.co/sanghwa-na/bobcat-1.1-nvfp4"><img alt="Hugging Face model: bobcat-1.1-nvfp4" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-bobcat--1.1--nvfp4-ffc107" /></a>
  <a href="https://huggingface.co/sanghwa-na/bobcat-flash-1.1"><img alt="Hugging Face model: bobcat-flash-1.1" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-bobcat--flash--1.1-ffc107" /></a>
  <a href="https://huggingface.co/spaces/sanghwa-na/bobcat"><img alt="Hugging Face Space: bobcat" src="https://img.shields.io/badge/%F0%9F%A4%97%20Space-bobcat-ffc107" /></a>
  <a href="https://foxl.ai/blog/bobcat-typed-decisions"><img alt="Blog: foxl.ai" src="https://img.shields.io/badge/blog-foxl.ai-c2410c" /></a>
  <a href="pyproject.toml"><img alt="Python 3.12 to 3.14" src="https://img.shields.io/badge/python-3.12%E2%80%933.14-3776ab" /></a>
</p>

<p align="center">
  <a href="https://foxl.ai/blog/bobcat-typed-decisions">Announcement and technical write-up</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/spaces/sanghwa-na/bobcat">Try it in the browser</a> &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a>
</p>

---

![Bobcat at a glance. Left, accuracy against the same base model zero-shot on identical inputs: four-task sealed final 94.3% against 87.9%, six-task development 93.7% against 85.5%, TypeSafe's workflow examples 92.1% against 86.9%, SemIf TypeSafe 102 87.2% against 82.0%, SemIf authored 144 91.0% against 87.6%. Right, agreement with TypeSafe's reference against median time per case on TypeSafe's 20 published workflow examples, log time axis: Jev 90.9% at 0.42 s client-side as published, Bobcat 92.1% agreement from the evaluation path at 0.56 s client-side from the same datacenter on the same architecture before the released checkpoint, Claude Opus 5 92.4% at 20.9 s, GPT-5.6 Sol 93.0% at 24.2 s. One decision takes 42.8 ms at the median server-side on one RTX PRO 6000 in NVFP4.](assets/readme/bobcat-at-a-glance.png)

<sub>Left: Bobcat vs Qwen3.8-27B zero-shot, identical inputs. Right: agreement with the mean of GPT-6 Astra and Claude
Fable 5.1 (329 questions); Jev, Opus 5 and Sol: TypeSafe's published answers and client-side times. Bobcat: the
released model on the BF16 evaluation path; time client-side, same datacenter, on the same architecture and NVFP4
setup before the released checkpoint. The released NVFP4 server: 91.2% at 0.58 s per case, server-side.</sub>

Bobcat reads a state (text or JSON), your questions and the answers you allow, and returns
typed decisions your code can threshold. It reads the logits of the offered candidates at
the first answer position of one forward pass; the host builds a closed JSON reply out of
your own names, and an independent validator refuses anything outside that contract. An
answer can be wrong, but it cannot be malformed.

The request and reply use the shapes of TypeSafe's published System One HTTP API, so the
official `typesafe-sdk` works against a Bobcat server by changing its base URL.

## Models

| Model | Weights | Base | Sealed final (1,614 decisions) | Same base, zero-shot | TypeSafe workflow agreement (329 questions) | Median time per workflow case | One decision, p50 |
|---|---|---|---:|---:|---:|---:|---:|
| **Bobcat 1.1** | [bobcat-1.1](https://huggingface.co/sanghwa-na/bobcat-1.1) (BF16), [bobcat-1.1-nvfp4](https://huggingface.co/sanghwa-na/bobcat-1.1-nvfp4) | Qwen3.8-27B | **94.27%** | 87.86% | 92.1% (served NVFP4: 91.2%) | 0.52-0.58 s (NVFP4) | 42.8 ms (NVFP4) |
| **Bobcat Flash 1.1** | [bobcat-flash-1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1) (BF16) | Gemma 4 26B-A4B | 92.21% | not run | 91.2% (served FP8) | **0.283 s** (FP8) | **24.8 ms** (FP8) |

Measured on one RTX PRO 6000 Blackwell 96 GB with vLLM 0.30.0: times per case are
server-side over localhost HTTP (0.56 s on the same architecture and serving setup from a
client in the same datacenter, measured before the released checkpoint), and one decision is 512 tokens with 8 candidates in the vLLM engine. On the same
20 workflow examples Jev agrees with the reference on 90.9% at 0.42 s per case (TypeSafe's
published client-side time; Jev was never called). A wrong answer named inside the state wins
7.8% of attacks on Bobcat 1.1 and 21.0% on its untrained base. Flash answers most questions
itself; in one server (`bobcat.route_server`) it hands the ones it is unsure about, and every
question longer than 2,048 Flash tokens with its state, to Bobcat 1.1.

The weights are ready to serve: each repository holds the trained model merged into its base
(BF16), which vLLM serves in FP8, and Bobcat 1.1 also comes as the NVFP4 checkpoint behind
its latency figures (Blackwell GPUs). The model cards give every number with its conditions
and limits, including the targets that were not met and the negative results.

- **Technical report:** see the [blog post](https://foxl.ai/blog/bobcat-typed-decisions).
- **Try it:** both models run in the [Bobcat Space](https://huggingface.co/spaces/sanghwa-na/bobcat), in the browser, with no key.
- **Demos:** see the [blog post](https://foxl.ai/blog/bobcat-typed-decisions).

## Quickstart

On one Linux GPU host with the weights of [sanghwa-na/bobcat-1.1](https://huggingface.co/sanghwa-na/bobcat-1.1):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"   # uv
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed   # a uv-managed Python ships the headers Triton compiles against
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

# The weights (about 56 GB), then the typed-decision server on vLLM
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

- On a Blackwell GPU, download [sanghwa-na/bobcat-1.1-nvfp4](https://huggingface.co/sanghwa-na/bobcat-1.1-nvfp4)
  instead and pass `--model bobcat-1.1-nvfp4 --compiler-model bobcat-1.1-nvfp4/compiler
  --quantization none`; the settings behind the latency figures are on
  [its card](https://huggingface.co/sanghwa-na/bobcat-1.1-nvfp4).
- `vllm serve sanghwa-na/bobcat-1.1-nvfp4` (or `sanghwa-na/bobcat-1.1 --quantization fp8`)
  also loads the weights, but plain vLLM exposes text generation; the typed-decision contract
  (closed JSON replies, no generated tokens) comes from `bobcat.api_server`, which runs vLLM
  in-process.
- Bobcat Flash 1.1 has its own server settings and the routing recipe in
  [its card](https://huggingface.co/sanghwa-na/bobcat-flash-1.1).
- `--no-config` keeps uv from applying this repository's development constraint
  (setuptools >= 83; vLLM 0.30.0 needs < 81) to the serving environment.

## Repository

| Path | What |
|---|---|
| `src/bobcat/protocol.py` | the closed request/response contract, limits and validator |
| `src/bobcat/student_readout.py` | the compiler (single-token identifiers, piecewise encoding) and the first-position readout |
| `src/bobcat/api_server.py` | the TypeSafe-compatible server (vLLM or transformers engine), with the state tokenized once per request |
| `src/bobcat/flash_server.py`, `route_server.py` | the Flash server, and one server running Flash with a fallback to Bobcat 1.1 |
| `src/bobcat/student_train.py`, `student_merge.py` | LoRA training (cross-entropy, Brier, proper-score RL, resumable) and merging |
| `src/bobcat/student_data_v11.py` | the Bobcat 1.1 mixture: injected-answer, insufficient-evidence and long-state counterfactuals |
| `src/bobcat/flash_data.py`, `flash_teacher.py`, `flash_train.py` | the Flash corpus, teacher probabilities and distillation |
| `src/bobcat/product_eval.py`, `korean_bench.py` | the six-task evaluation builder (including fresh finals) and the public benchmark sets |
| `scripts/` | calibration, scoring, the TypeSafe workflow and SemIf comparisons, NVFP4 quantization, release staging |
| `release/` | the release manifests, the model cards and their figures |
| `configs/`, `tests/` | pinned data and model sources, and the test suite |

## Development

```bash
uv sync --all-extras   # pytest, ruff and the server packages are extras
uv run python -m pytest
uv run ruff check src tests scripts
```

## License

The code is released under the [Apache License 2.0](LICENSE). The Bobcat 1.1 weights and
their NVFP4 checkpoint are released under Apache-2.0 as derivatives of Qwen3.8-27B, and the
Bobcat Flash 1.1 weights under Apache-2.0 as a derivative of Gemma 4 26B-A4B-it (Apache-2.0).
Datasets keep their own licenses; third-party notices are in [THIRD_PARTY.md](THIRD_PARTY.md)
and [licenses/](licenses/).

Bobcat is an independent project. It is not affiliated with or endorsed by TypeSafe AI,
Google, the Qwen team or any other company named here; product names are their owners'
trademarks and are used only to identify the models compared.

## Citation

```bibtex
@misc{bobcat11,
  title        = {Bobcat 1.1},
  author       = {{The Bobcat Authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sanghwa-na/bobcat-1.1}}
}
```
