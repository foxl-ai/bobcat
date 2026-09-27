---
license: apache-2.0
base_model: sanghwa-na/bobcat-flash-1.1
base_model_relation: quantized
library_name: transformers
language:
- en
- ko
pipeline_tag: zero-shot-classification
tags:
- typed-decisions
- fp8
- compressed-tensors
- vllm
- mixture-of-experts
- classification
- guardrails
---

# Bobcat Flash 1.1 FP8

Technical write-up: **[foxl.ai/blog/bobcat-typed-decisions](https://foxl.ai/blog/bobcat-typed-decisions)**

[Bobcat Flash 1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1) · [Try it in the Space](https://huggingface.co/spaces/sanghwa-na/bobcat-flash) · [Code on GitHub](https://github.com/foxl-ai/bobcat)

A pre-quantized FP8 checkpoint of [Bobcat Flash 1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1), the fast tier of the typed-decision model Bobcat, for short states: you send a state and
questions whose answers you name, and it returns a probability for every answer you named; it
never generates text.
**Evaluation, training data, limitations and the full license notes are on the main card,
[sanghwa-na/bobcat-flash-1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1).**

This repository, `sanghwa-na/bobcat-flash-1.1-fp8`, is the BF16 weights of
[sanghwa-na/bobcat-flash-1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1) quantized to FP8 (27.2 GB, FP8_DYNAMIC, every expert included):
float8_e4m3fn weights with one scale per output channel, activations quantized per token at run
time, in the compressed-tensors format that vLLM loads as it is. It is the checkpoint the
[Bobcat Flash Space](https://huggingface.co/spaces/sanghwa-na/bobcat-flash) serves, read as stored with no conversion.

## Serve

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"   # uv
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-flash-1.1-fp8', local_dir='bobcat-flash-1.1-fp8')"
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.flash_server --engine vllm --model bobcat-flash-1.1-fp8 \
  --compiler-model bobcat-flash-1.1-fp8/compiler --quantization none --temperature 0.8912 \
  --name bobcat-flash-1.1 --release-date 2026-09-26 --max-num-seqs 256 --max-model-len 32832 \
  --schedule all --engine-arg max_num_batched_tokens=16384 --host 127.0.0.1 --port 8000 --local
```

Then call it with the official `typesafe-sdk` as on the main card
(`TYPESAFE_BASE_URL=http://127.0.0.1:8000`, `TYPESAFE_DEFAULT_MODEL=bobcat-flash-1.1`).

- `--quantization none`: vLLM reads the FP8 scheme from `config.json` (compressed-tensors).
- `--compiler-model .../compiler` points the Bobcat compiler at the base model's pinned
  tokenizer, template and configuration; run the server from the repository root, where the
  identifier list lives, or pass `--identifiers bobcat-flash-1.1-fp8/bobcat-identifiers.json`.
- `vllm serve sanghwa-na/bobcat-flash-1.1-fp8` also loads the checkpoint, but plain vLLM exposes text generation; the
  typed-decision contract (closed JSON replies, no generated tokens) comes from the Bobcat server.

## Precision

| Development decisions (3,188), one RTX PRO 6000, vLLM 0.30.0 | Accuracy | Task macro | Same answer as BF16 |
|---|---:|---:|---:|
| Evaluation path (BF16 base, unmerged adapter) | 91.84% | 92.07% | - |
| **This checkpoint (FP8)** | **91.94%** | **92.22%** | 98.8% |

FP8 minus BF16: +0.09 points [-0.28, +0.47]; 39 answers changed, mostly near-ties. Served by
`bobcat.flash_server` over localhost HTTP, TypeSafe's 20 workflow cases agreed with the reference
on 90.9% of 329 questions (no failed request; the served BF16 weights in FP8 give 91.2%). One
decision (512 tokens, 8 candidates) takes 24.6 ms at p50 in the vLLM engine. States longer than
2,048 tokens are the strength of [Bobcat 1.1](https://huggingface.co/sanghwa-na/bobcat-1.1); the
main card gives Flash's long-input figures and the routing recipe.

## Provenance

- **Weights:** the BF16 checkpoint of [sanghwa-na/bobcat-flash-1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1)
  (the Bobcat Flash 1.1 LoRA merged into Gemma 4 26B-A4B-it at revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`), downloaded and checked against its `SHA256SUMS.json`.
- **Quantization:** `scripts/fp8_quantize.py` with llm-compressor 0.14.0 (FP8_DYNAMIC, no
  calibration data): the 30 fused expert blocks are linearized into one Linear per expert, and
  205 dense and 11,520 expert projections are in FP8; `lm_head`, the embeddings, the 30 routers
  and the vision tower stay BF16.
- **Hashes:** `model-00001-of-00002.safetensors` sha256 `8e718d94…23efc`,
  `model-00002-of-00002.safetensors` `e0523d3c…a414`, as recorded in the release manifest
  (`bobcat-release-manifest.json`, `serving_artifacts.fp8`); every file is listed in
  `SHA256SUMS.json`.
- No output of Jev or of any other teacher model was used to train the model; see the main card.

## License and attribution

Apache-2.0. This checkpoint is a derivative of Gemma 4 26B-A4B-it by Google DeepMind
(Apache-2.0); the license text is included as LICENSE, and `NOTICE` lists what was changed.
Training data keep their own licenses (main card and `THIRD_PARTY.md` in
[github.com/foxl-ai/bobcat](https://github.com/foxl-ai/bobcat)). Bobcat is an independent
project, not affiliated with or endorsed by Google, the Qwen team, TypeSafe AI or any other
company named here. Provided as is, without warranty.

## Citation

```bibtex
@misc{bobcatflash11,
  title        = {Bobcat Flash 1.1},
  author       = {{The Bobcat Authors}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sanghwa-na/bobcat-flash-1.1}}
}
```
