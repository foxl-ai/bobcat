---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_revision: 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
base_model_relation: finetune
library_name: transformers
language:
- en
- ko
pipeline_tag: zero-shot-classification
tags:
- typed-decisions
- mixture-of-experts
- vllm
- calibration
- classification
- guardrails
---

# Bobcat Flash 1.1, merged

A ready-to-serve checkpoint of [Bobcat Flash 1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1),
the fast tier of the Bobcat typed-decision model: you send a state and questions whose answers
you name, and it returns a probability for every answer you named, read from the offered
candidates' logits at the first answer position; it never generates text. **Evaluation,
distillation, the routing server, limitations and the full license notes are on the main
card, [sanghwa-na/bobcat-flash-1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1).**

This repository, `sanghwa-na/bobcat-flash-1.1-merged`, is the Bobcat Flash 1.1 LoRA adapter
merged into Gemma 4 26B-A4B-it at the pinned revision and stored in BF16. It is the checkpoint
we served: its shards are byte-identical to the merge behind every served Flash figure on the
main card. As measured, vLLM quantizes it to FP8 when it loads (`--quantization fp8`); we
measured it on one RTX PRO 6000 Blackwell 96 GB with vLLM 0.30.0.

## Serve

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"   # uv
git clone https://github.com/foxl-ai/bobcat && cd bobcat
export UV_PYTHON_PREFERENCE=only-managed
uv venv --python 3.12 .venv-serve
uv pip install --no-config --python .venv-serve/bin/python vllm==0.30.0 fastapi uvicorn scipy jinja2 \
  "tokenizers>=0.21" huggingface_hub typesafe-sdk==0.7.1
export PYTHONPATH=$PWD/src PY=.venv-serve/bin/python

$PY -c "from huggingface_hub import snapshot_download as s; s('sanghwa-na/bobcat-flash-1.1-merged', local_dir='bobcat-flash-1.1-merged')"
VLLM_USE_FLASHINFER_SAMPLER=0 $PY -m bobcat.flash_server --engine vllm --model bobcat-flash-1.1-merged \
  --compiler-model bobcat-flash-1.1-merged/compiler --quantization fp8 --temperature 0.8912 \
  --name bobcat-flash-1.1 --release-date 2026-09-26 --max-num-seqs 256 --max-model-len 32832 \
  --schedule all --engine-arg max_num_batched_tokens=16384 --host 127.0.0.1 --port 8000 --local
```

Then call it with the official `typesafe-sdk` as on the main card
(`TYPESAFE_BASE_URL=http://127.0.0.1:8000`, `TYPESAFE_DEFAULT_MODEL=bobcat-flash-1.1`). For
the routed server that sends unsure and long questions to Bobcat 1.1, pass this folder as
`--flash-model` to `bobcat.route_server` (main card, "Routing to Bobcat 1.1").

- `--quantization fp8` is vLLM's online dynamic FP8 of these BF16 weights, the served
  configuration; the download is about 52 GB. vLLM runs Gemma 4 with its Triton attention
  backend.
- `--compiler-model .../compiler` points the Bobcat compiler at the base model's pinned
  tokenizer, template and configuration; it checks them against the receipt in that folder.
  Run the server from the repository root, where the identifier list lives
  (`reports/2026-09-22-glm-readout-preflight.json`), or pass
  `--identifiers bobcat-flash-1.1-merged/bobcat-identifiers.json`: the same list ships here.
- `vllm serve bobcat-flash-1.1-merged --quantization fp8 --max-model-len 32832` also loads the
  checkpoint, but plain vLLM exposes text generation; the typed-decision contract (closed JSON
  replies, no generated tokens) comes from the Bobcat servers.
- `bobcat.api_server` with the same arguments serves it too (the same token IDs).

**Load-tested as staged.** The staged folder of this repository (these files, before upload)
was served on one RTX PRO 6000 with vLLM 0.30.0 by the command above: `vllm serve` also came
up and listed the model; the Bobcat server answered `/health`, the main card's SDK example
answered `payments`, and TypeSafe's 20 workflow cases agreed with the reference on 91.2% of
329 questions (no failed request), the figure on the main card, at 0.28 s per case.

## Precision

**Same answers as the adapter.** On a development sample (every 10th decision of the v2
development split, 319 decisions across the six tasks), this checkpoint and the evaluation
path (BF16 base with the unmerged adapter, the same code, one RTX PRO 6000) gave the same top
answer on 314 (98.4%) and the same accuracy (291 of 319 correct, 91.2%). The five changed
answers were all search near-ties, with the top two candidates within 0.08 on the adapter
path; the mean largest probability change was 0.006 at the calibration temperature. Merging
rounds `W + (alpha/r) B A` to BF16 once, so a merged checkpoint is a slightly different
numerical artifact from the adapter; its served figures on the main card were measured with
these exact shards.

Served in FP8, Flash gives the evaluation path's answer on 98.4% of development questions (task
macro 92.1% to 91.8%), and near-ties move between server runs: two fresh installs scored 90.9%
and 91.8% on TypeSafe's 329 workflow questions, with the 6 changed answers all at a top
probability of 0.70 or less. Where an answer must be reproducible, send questions below Flash's
0.8 threshold to Bobcat 1.1, as `bobcat.route_server` does. An NVFP4 build was slower and less
accurate than FP8 on this GPU and is not published.

## Provenance

- **Adapter:** Bobcat Flash 1.1, `adapter_model.safetensors` sha256 `2304bd8e…9ca3`, released
  at [sanghwa-na/bobcat-flash-1.1](https://huggingface.co/sanghwa-na/bobcat-flash-1.1).
- **Base:** Gemma 4 26B-A4B-it at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, every
  file checked against the Hub's hashes at download.
- **Merge:** `bobcat.student_merge` in the serving environment (vLLM 0.30.0, torch 2.13.0):
  each of the 235 adapted weights becomes `bf16(W + (alpha/r) B A)` (alpha/r = 2.0) with the
  product and sum in float32; the 128 experts and every other tensor and file are copied.
- **Hashes:** `model-00001-of-00002.safetensors` sha256
  `6847aa860e793d33c308683308efc6815ea420604d2d805abab8a416acfd14a7` and
  `model-00002-of-00002.safetensors`
  `d72e43850d618d452c6ceabf0c38f29c030bdb7e31e2a6a2149820edcf5bb72a`, recorded in the release
  manifest (`bobcat-release-manifest.json`, `serving_artifacts.merged_bf16`) and identical to
  the merged shards the served figures and the NVFP4 study used
  (`serving_artifacts.nvfp4.derived_from`); every file is listed in `SHA256SUMS.json`.
- No output of Jev was used for training, distillation, reward or calibration; see the main
  card.

## License and attribution

Apache-2.0. This checkpoint is a derivative of Gemma 4 26B-A4B-it by Google DeepMind, released
under the Apache License 2.0 ([Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license));
the license text is included as `LICENSE`, and `NOTICE` lists what was changed. Its teachers,
Bobcat 1 and a Bobcat 1.1 candidate, are Apache-2.0 derivatives of Qwen3.8-27B by the Qwen
team. Training data keep their own licenses (main card and `THIRD_PARTY.md` in
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
