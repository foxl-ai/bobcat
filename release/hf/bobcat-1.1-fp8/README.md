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
- fp8
- compressed-tensors
- vllm
- calibration
- classification
- guardrails
---

# Bobcat 1.1 FP8

Technical write-up: **[foxl.ai/blog/bobcat-typed-decisions](https://foxl.ai/blog/bobcat-typed-decisions)**

[Bobcat 1.1](https://huggingface.co/sanghwa-na/bobcat-1.1) · [Try it in the Space](https://huggingface.co/spaces/sanghwa-na/bobcat) · [Code on GitHub](https://github.com/foxl-ai/bobcat)

A pre-quantized FP8 checkpoint of [Bobcat 1.1](https://huggingface.co/sanghwa-na/bobcat-1.1), the typed-decision model: you send a state and questions whose answers you name, and it
returns a probability for every answer you named, read from the offered candidates' logits at
the first answer position; it never generates text.
**Evaluation, training data, limitations and the full license notes are on the main card,
[sanghwa-na/bobcat-1.1](https://huggingface.co/sanghwa-na/bobcat-1.1).**

This repository, `sanghwa-na/bobcat-1.1-fp8`, is the BF16 weights of
[sanghwa-na/bobcat-1.1](https://huggingface.co/sanghwa-na/bobcat-1.1) quantized to FP8 (31.3 GB, FP8_DYNAMIC):
float8_e4m3fn weights with one scale per output channel, activations quantized per token at run
time, in the compressed-tensors format that vLLM loads as it is. It is the checkpoint the
[Bobcat Space](https://huggingface.co/spaces/sanghwa-na/bobcat) serves, read as stored with no conversion.

## Serve

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"   # uv
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-1.1-fp8', local_dir='bobcat-1.1-fp8')"
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.api_server --engine vllm --model bobcat-1.1-fp8 \
  --compiler-model bobcat-1.1-fp8/compiler --quantization none --temperature 1.2008 \
  --name bobcat-1.1 --release-date 2026-09-26 --max-num-seqs 128 --schedule all \
  --engine-arg max_num_batched_tokens=16384 --host 127.0.0.1 --port 8000 --local
```

Then call it with the official `typesafe-sdk` as on the main card
(`TYPESAFE_BASE_URL=http://127.0.0.1:8000`, `TYPESAFE_DEFAULT_MODEL=bobcat-1.1`).

- `--quantization none`: vLLM reads the FP8 scheme from `config.json` (compressed-tensors).
- `--compiler-model .../compiler` points the Bobcat compiler at the base model's pinned
  tokenizer, template and configuration; run the server from the repository root, where the
  identifier list lives, or pass `--identifiers bobcat-1.1-fp8/bobcat-identifiers.json`.
- `vllm serve sanghwa-na/bobcat-1.1-fp8` also loads the checkpoint, but plain vLLM exposes text generation; the
  typed-decision contract (closed JSON replies, no generated tokens) comes from the Bobcat server.

## Precision

| Development decisions (3,188), one RTX PRO 6000, vLLM 0.30.0 | Accuracy | Task macro | Same answer as BF16 |
|---|---:|---:|---:|
| Evaluation path (BF16 base, unmerged adapter) | 93.54% | 93.69% | - |
| **This checkpoint (FP8)** | **93.48%** | **93.72%** | 99.5% |

FP8 minus BF16: -0.06 points [-0.29, +0.16]; the 17 changed answers are mostly near-ties
(top two within 0.1: 50 rows, 74% unchanged). Served by `bobcat.api_server` over localhost HTTP,
TypeSafe's 20 workflow cases agreed with the reference on 91.5% of 329 questions (no failed
request; the evaluation path gives 92.1%). One decision (512 tokens, 8 candidates) takes 56 ms at
p50 in the vLLM engine; the main card's latency figures are for the NVFP4 checkpoint. In the
Bobcat Space's own FP8 engine the weights take 27.7 GiB and a 27K-token request peaks at 42.1 GiB.

## Provenance

- **Weights:** the BF16 checkpoint of [sanghwa-na/bobcat-1.1](https://huggingface.co/sanghwa-na/bobcat-1.1)
  (the Bobcat 1.1 LoRA merged into Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`),
  downloaded and checked against its `SHA256SUMS.json`.
- **Quantization:** `scripts/fp8_quantize.py` with llm-compressor 0.14.0 (FP8_DYNAMIC, no
  calibration data): 400 Linear layers in FP8; `lm_head`, the embeddings, the vision tower and
  the Gated DeltaNet a/b projections stay BF16. `model_mtp.safetensors` is the BF16 MTP file,
  unchanged.
- **Hashes:** `model-00001-of-00002.safetensors` sha256 `a3b019df…4bd3b3`,
  `model-00002-of-00002.safetensors` `5891be03…57c2ce`, as recorded in the release manifest
  (`bobcat-release-manifest.json`, `serving_builds.fp8`); every file is listed in
  `SHA256SUMS.json`.
- No output of Jev or of any other teacher model was used to train the model; see the main card.

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
