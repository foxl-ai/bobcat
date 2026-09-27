---
license: apache-2.0
base_model: sanghwa-na/bobcat-1.1
base_model_relation: quantized
library_name: transformers
language:
- en
- ko
pipeline_tag: zero-shot-classification
tags:
- typed-decisions
- nvfp4
- compressed-tensors
- vllm
- calibration
- classification
- guardrails
---

# Bobcat 1.1 NVFP4

A ready-to-serve checkpoint of [Bobcat 1.1](https://huggingface.co/sanghwa-na/bobcat-1.1),
the typed-decision model: you send a state and questions whose answers you name, and it
returns a probability for every answer you named, read from the offered candidates' logits
at the first answer position; it never generates text. **Evaluation, training data,
limitations and the full license notes are on the main card,
[sanghwa-na/bobcat-1.1](https://huggingface.co/sanghwa-na/bobcat-1.1).**

This repository, `sanghwa-na/bobcat-1.1-nvfp4`, is the exact NVFP4 checkpoint we measured and
served: the Bobcat 1.1 adapter merged into Qwen3.8-27B, then quantized to NVFP4 (4-bit
weights and activations). It needs an NVIDIA Blackwell GPU (FP4 tensor cores); we measured it
on one RTX PRO 6000 Blackwell 96 GB with vLLM 0.30.0. For other GPUs, merge the adapter and
serve it in FP8 as the main card's quickstart describes.

## Serve

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"   # uv
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-1.1-nvfp4', local_dir='bobcat-1.1-nvfp4')"
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.api_server --engine vllm --model bobcat-1.1-nvfp4 \
  --compiler-model bobcat-1.1-nvfp4/compiler --quantization none --temperature 1.2008 \
  --name bobcat-1.1 --release-date 2026-09-26 --max-num-seqs 128 --schedule all \
  --engine-arg max_num_batched_tokens=16384 --host 127.0.0.1 --port 8000 --local
```

Then call it with the official `typesafe-sdk` as on the main card
(`TYPESAFE_BASE_URL=http://127.0.0.1:8000`, `TYPESAFE_DEFAULT_MODEL=bobcat-1.1`).

- `--quantization none`: vLLM reads the NVFP4 scheme from `config.json` (compressed-tensors).
- `--compiler-model .../compiler` points the Bobcat compiler at the base model's pinned
  tokenizer, template and configuration; it checks them against the receipt in that folder.
  Run the server from the repository root, where the identifier list lives
  (`reports/2026-09-22-glm-readout-preflight.json`), or pass
  `--identifiers bobcat-1.1-nvfp4/bobcat-identifiers.json`: the same list ships here.
- `vllm serve bobcat-1.1-nvfp4 --max-model-len 16448` also loads the checkpoint, but plain
  vLLM exposes text generation; the typed-decision contract (closed JSON replies, no
  generated tokens) comes from `bobcat.api_server`.
- The limits are those of the main card: 128 questions, 255 candidates and 16,384 tokens per
  compiled question by default (`--max-model-len 32832` for 32,768), refused with HTTP 422,
  never truncated.

**Load-tested as staged.** The staged folder of this repository (these files, before upload)
was served on one RTX PRO 6000 with vLLM 0.30.0 by the command above: `vllm serve` also came
up and listed the model; the Bobcat server answered `/health`, the main card's SDK example
answered `payments`, and TypeSafe's 20 workflow cases agreed with the reference on 90.9% of
329 questions (no failed request) at 0.61 s per case on a freshly started server. The earlier
measured NVFP4 runs gave 91.2%; NVFP4 answers on near-ties move between runs (see
[Precision](#precision)).

## Precision

| Development decisions not used for calibration (2,932) | Accuracy | Same answer as FP8 |
|---|---:|---:|
| Evaluation path (BF16 base, unmerged adapter) | 93.28% | 99.3% |
| Merged, FP8 at load | 93.25% | - |
| **This checkpoint (NVFP4)** | **93.25%** | 98.2% |

NVFP4 minus FP8: 0.0 points [-0.4, +0.5]. On one RTX PRO 6000 in the vLLM engine, one decision
(512 tokens, 8 candidates) takes 42.8 ms at p50, a 3.2k-token state with 8 questions 467 ms,
and independent 1K-token requests run at 19,207-19,614 tokens/s; TypeSafe's 20 workflow cases
took 0.52-0.58 s per case served over localhost HTTP.

**Near-ties.** NVFP4 moves answers whose top two candidates are close. A question answered
from a cached shared prefix and the same question computed from scratch agreed on 61 of 64
boundary questions (95.3%; FP8: 64 of 64), and the served NVFP4 model agreed with TypeSafe's
reference on 91.2% of the workflow questions against 92.1% on the evaluation path. Where a
near-tie must be reproducible, serve the adapter merged in FP8 (main card).

## Provenance

- **Adapter:** Bobcat 1.1, `adapter_model.safetensors` sha256 `7351d959…83b2`, released at
  [sanghwa-na/bobcat-1.1](https://huggingface.co/sanghwa-na/bobcat-1.1).
- **Base:** Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- **Merge:** `bobcat.student_merge`: each adapted weight becomes `bf16(W + (alpha/r) B A)`
  with the product and sum in float32; every other tensor is copied.
- **Quantization:** `scripts/nvfp4_quantize.py` with llm-compressor 0.14.0: NVFP4 W4A4,
  group 16, FP8 scales; `lm_head`, the vision tower and the Gated DeltaNet a/b projections
  stay BF16; the fused q/k/v, gate/up and Gated DeltaNet qkv/z groups share one global scale.
  Calibration used 256 compiled development prompts (up to 4,096 tokens), which were then
  left out of the accuracy comparison above; their ids are in `bobcat-nvfp4.json`.
- **Hashes:** `model.safetensors` sha256
  `2ea6716e36673bc73995bc5e06bd3a032ed8be19f4fff8dd544ce9a40bb173c9` and
  `model_mtp.safetensors` `1d8268aa…9da9fe`, as recorded in the release manifest
  (`bobcat-release-manifest.json`, `serving_builds.nvfp4`); every file is listed in
  `SHA256SUMS.json`.
- No output of Jev or of any other teacher model was used to train the adapter; see the main
  card.

## License and attribution

Apache-2.0. This checkpoint is a derivative of Qwen3.8-27B by the Qwen team (Apache-2.0); the
Qwen LICENSE file is included unchanged, and `NOTICE` lists what was changed. Training data
keep their own licenses (main card and `THIRD_PARTY.md` in
[github.com/foxl-ai/bobcat](https://github.com/foxl-ai/bobcat)). Bobcat is an independent
project, not affiliated with or endorsed by the Qwen team, TypeSafe AI or any other company
named here. Provided as is, without warranty.

## Citation

```bibtex
@misc{bobcat11,
  title        = {Bobcat 1.1},
  author       = {{The Bobcat Authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sanghwa-na/bobcat-1.1}}
}
```
