# Third-party notices

## TypeSafe public adapter confidence

The Choice and Score confidence functions in `src/bobcat/protocol.py` follow the
formulas in TypeSafe AI's `system-one-adapter-python`,
`src/system_one_adapter/_utils/confidence_metrics.py`, revision
`fb52b1030b7fc1f4f1cf39910afa5da54f9835e3`.

Source: https://github.com/typesafe-ai/system-one-adapter-python/tree/fb52b1030b7fc1f4f1cf39910afa5da54f9835e3

The implementation adds input validation and numerical-bound handling. The
original MIT notice is preserved in
[`licenses/typesafe-system-one-adapter-MIT.txt`](licenses/typesafe-system-one-adapter-MIT.txt).

This attribution does not imply TypeSafe affiliation, endorsement, access to
Jev weights, or identical Jev probabilities. Bobcat identifies the confidence
profile separately from the model that produced the probability distribution.

## GLM-5.3-Flash base weights

The original `zai-org/GLM-5.3-Flash` weights, revision
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, have been downloaded and checksum
verified for the reference experiments. Bobcat 1.1 and Bobcat Flash 1.1 are not derived
from GLM weights; they served as a quality reference and in comparison experiments only.

The exact upstream MIT notice is preserved in
[`licenses/GLM-5.3-Flash-MIT.txt`](licenses/GLM-5.3-Flash-MIT.txt).
The file identities and byte counts are pinned in
[`configs/glm-5.3-flash-source.json`](configs/glm-5.3-flash-source.json).
Any adapted release must retain its base-weight provenance and notice.

Research comparisons with other models do not grant a right to redistribute or
combine their weights. In particular, `openjev/openjev`'s CC BY-NC 4.0 weights
are not a base for Bobcat, and no openjev weights or outputs were used.

## KLUE Korean judgment data

Sungjoon Park et al., *KLUE: Korean Language Understanding Evaluation* (2021),
arXiv:2105.09680. Upstream project: `KLUE-benchmark/KLUE`, revision
`3efd98708a40ff49251fddde35453f8fbb11f536`; dataset: `klue/klue`, revision
`349481ec73fff722f88e0453ca05c77a447d967c`.

Source: https://github.com/KLUE-benchmark/KLUE/tree/3efd98708a40ff49251fddde35453f8fbb11f536

The project and dataset identify the data as **CC BY-SA 4.0**. Its full notice is
preserved in [`licenses/KLUE-CC-BY-SA-4.0.txt`](licenses/KLUE-CC-BY-SA-4.0.txt).
The derived Bobcat dataset retains that license and attribution. Changes are
NFC normalization, typed Korean Choice/Noul instructions, exact-input/article
grouping, duplicate quarantine and additional training/calibration partitions.
The original validation set is public development data, not a fresh final test.

[`configs/korean-decisions-klue-v1.json`](configs/korean-decisions-klue-v1.json)
pins four NLI/YNAT Parquet files and the class-label mapping.
[`src/bobcat/klue.py`](src/bobcat/klue.py) does not pass source IDs, labels,
article URLs or annotation metadata to a model. NLI yes/no views are deterministic
reformulations of the original annotation, not independent human judgments or
empirical uncertainty distributions. No Jev predictions are used.

This data notice is not a blanket license determination for future trained
weights; their actual dependencies and release terms must be recorded separately.

## NVIDIA NeMo AutoModel research dependency

The CPU architecture/adapter probes import NVIDIA NeMo AutoModel revision
`2a372db01d5b98aaa01b15747887852f074667ce`, including its `glm5_next` model,
native LoRA implementation, and small random test fixture. The regular source
files were checked against the pinned Git blobs. The source checkout and
separate probe dependencies remain local research artifacts, rather than being
vendored into Bobcat's model implementation.

The upstream [Apache 2.0 license](licenses/NVIDIA-AutoModel-Apache-2.0.txt)
is preserved. Relevant source files identify NVIDIA CORPORATION as copyright
holder. The Bobcat probe adds frozen-backbone controls, explicit adapter targets,
parameter/buffer checks, and checkpoint/optimizer/RNG comparisons.
This use does not imply NVIDIA endorsement or successful CUDA, large-weight,
distributed-training, or model-quality validation.

## Public Korean/English decision expansion

[`configs/public-decisions-v1.json`](configs/public-decisions-v1.json) pins the
original source files, dataset cards, revisions, sizes and SHA256 hashes.
[`src/bobcat/public_decisions.py`](src/bobcat/public_decisions.py) normalizes input
text, adds typed instructions, keeps all offered choices, groups shared source
inputs, and creates additional training/calibration/development partitions.
Validation and test annotations are public development diagnostics only.

- **KLUE STS:** Park et al. (2021), the same KLUE revisions and CC BY-SA 4.0 notice
  above. `labels.real-label` is retained as an observed annotator mean. Rounded
  labels and binary labels do not become invented categorical vote distributions.
- **Banking77:** Casanueva et al. (2020), *Efficient Intent Detection with Dual
  Sentence Encoders*, PolyAI Ltd. Publisher repository
  `PolyAI-LDN/task-specific-datasets@57ec275d8078af65b7731c2a98be812d844a6d6b`.
  All 77 original categories are retained. The publisher's
  [CC BY 4.0 notice](licenses/Banking77-CC-BY-4.0.txt) is preserved.
- **BoolQ:** Clark et al. (2019), *BoolQ: Exploring the Surprising Difficulty of
  Natural Yes/No Questions*. `google/boolq@35b264d03638db9f4ce671b711558bf7ff0f80d5`.
  The publisher's card declares CC BY-SA 3.0; the
  [license text](licenses/BoolQ-CC-BY-SA-3.0.txt) is preserved. Passage and question
  are model inputs; the original boolean answer is supervision only.
- **ARC Easy/Challenge:** Clark et al. (2018), *Think you have Solved Question
  Answering? Try ARC, the AI2 Reasoning Challenge*.
  `allenai/ai2_arc@210d026faf9955653af8916fad021475a3f00453`.
  The publisher's card YAML declares CC BY-SA 4.0; its body licensing subsection
  is unspecified. This distinction is recorded in the source manifest.
  The [CC BY-SA 4.0 text](licenses/KLUE-CC-BY-SA-4.0.txt) applies to that declared
  license; the attribution here remains ARC's own authorship.

The combined data retains per-source terms and attribution. It is not
relicensed as a single Apache/MIT dataset, and it is not a blanket determination
of the license for future trained weights. No Jev predictions or inferred
annotator vote histograms are used as targets.

## KoBEST Korean external evaluation

Kim et al. (2022), *KoBEST: Korean Balanced Evaluation of Significant Tasks*,
arXiv:2204.04541, SKT-LSL. Dataset `skt/kobest_v1`, revision
`a5ea15e3ac77ed694b79f6204eb31889a2ba989f`.

The publisher's pinned dataset card declares **CC BY-SA 4.0**.
[`configs/kobest-evaluation-v1.json`](configs/kobest-evaluation-v1.json)
records source URLs, full file hashes, attribution and card identity.
The [CC BY-SA 4.0 text](licenses/KLUE-CC-BY-SA-4.0.txt) is the same license text;
KoBEST's attribution remains its own authorship.

[`src/bobcat/kobest_eval.py`](src/bobcat/kobest_eval.py) creates a derived,
CC BY-SA 4.0 diagnostic subset with normalized text, Korean typed instructions,
source-component grouping and deterministic sampling. Original labels and
candidate meanings are preserved. Test data is used only for evaluation,
never optimization or calibration. The `test_originated` source training
reviews are excluded. This subset is not the full official benchmark or a
fresh release test, and no absence of language-pretraining exposure is claimed.

## Calibrated decision reinforcement learning

Anthony Maio's **eve-rlcd**, revision
`57a179b7b1bedc80f65bf42ccda129dd1888272f`, is the method reference for
Bobcat's sampled-correctness reward `c - stop_gradient(p(a))`, IID action
groups, and leave-one-out REINFORCE. The upstream
[MIT notice](licenses/Eve-RLCD-MIT.txt) is preserved. Bobcat's GLM attention
adapters, separate retention objective, distributed execution, and evaluation
constitute a different training setup. No Eve checkpoint or Jev-generated
training labels are imported.

Research references also include Bani-Harouni et al., *Rewarding Doubt*
(`arXiv:2503.02623v6`), and Damani et al., *Beyond Binary Rewards*
(`arXiv:2507.16806v2`). Both study verbalized confidence, which differs from
Bobcat's direct categorical probabilities. These references do not establish
TypeSafe's proprietary architecture, reward function, or training loop.

## Qwen3.8-27B base model

Bobcat 1.1 is a LoRA adapter trained on `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, released by the Qwen team under the Apache
License 2.0, and is published merged into that model (BF16, and an NVFP4 build). The
weights are derivatives of the base model and are released under the same license, with
the upstream LICENSE file and a NOTICE in each Hugging Face repository.

## Gemma 4 26B-A4B base model

Bobcat Flash 1.1 is a LoRA adapter trained on `google/gemma-4-26B-A4B-it`, revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, released by Google DeepMind. That revision's
model card declares `license: apache-2.0` and links the
[Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license) page, which is the
Apache License 2.0 text (checked 2026-09-26). It is published merged into that model (BF16);
the weights are a derivative of the base model and are released under the same license, with
the license text and a NOTICE in the Hugging Face repository. Bobcat Flash was distilled from
Bobcat adapters of Qwen3.8-27B (Apache-2.0); no Jev output was used.

## Further training data

In addition to the sources above, Bobcat 1.1's training decisions draw on the
following public data. Each keeps its own license and attribution; the derived
decisions are not relicensed.

- **KLUE NLI, YNAT, MRC and Wizard of Seoul:** Park et al. (2021), CC BY-SA 4.0
  (notice above).
- **KorNLI:** Ham et al. (2020), *KorNLI and KorSTS: New Benchmark Datasets for Korean
  Natural Language Understanding*, Kakao Brain, CC BY-SA 4.0.
- **NSMC:** Lucy Park, *Naver Sentiment Movie Corpus*, CC0 1.0.
- **MASSIVE:** FitzGerald et al. (2022), *MASSIVE: A 1M-Example Multilingual Natural
  Language Understanding Dataset*, CC BY 4.0.
- **HelpSteer 2 and 3:** Wang et al. (2024, 2025), NVIDIA, CC BY 4.0. Only the mean
  attribute ratings are used, as observed means.

Also used for Bobcat 1.1 (derived decisions keep each source's license and attribution):

- **SNLI:** Bowman et al. (2015), *A large annotated corpus for learning natural language
  inference*, Stanford NLP, CC BY-SA 4.0 (`stanfordnlp/snli`, revision `cdb5c3d5`).
  Premise/hypothesis pairs as relation and evidence/claim decisions, and unrelated-premise
  copies whose gold is the abstain label.
- **SQuAD 2.0:** Rajpurkar, Jia and Liang (2018), *Know What You Don't Know: Unanswerable
  Questions for SQuAD*, CC BY-SA 4.0 (`rajpurkar/squad_v2`, revision `3ffb306f`).
  Answerable and unanswerable questions on the same Wikipedia passage as "is the answer
  stated?" decisions.

Bobcat Flash 1.1's training corpus draws on the sources above (except SQuAD 2.0) and adds:

- **MultiNLI:** Williams, Nangia and Bowman (2018), *A Broad-Coverage Challenge Corpus for
  Sentence Understanding through Inference* (`nyu-mll/multi_nli`, revision `da70db2a`).
  The dataset card states that most of the corpus is released under the Open American
  National Corpus license; in the fiction section, *Seven Swords* is under CC BY-SA 3.0,
  *Living History* and *Password Incorrect* are under CC BY 3.0, and the remaining works are
  in the public domain in the United States. Premise/hypothesis training pairs as
  evidence/claim decisions.
- **SNLI** (above) training pairs, the same way.

## Evaluation-only data

KoBEST (above), KMMLU, HAE-RAE Bench 1.1, CLIcK, MMLU and HellaSwag were used only
to evaluate Bobcat and are not redistributed in this repository; see each dataset's
card for its license. The comparison with Jev uses the workflow examples TypeSafe
publishes at evals.typesafe.ai; those files are fetched at run time by
`scripts/typesafe_workflow_eval.py`, verified by hash, and not redistributed. The SemIf
benchmark bundle (`TheoLeeCJ/SemIf-OpenJev`, MIT), including its Every lab rows and the
Jev figures TypeSafe released for 102 aligned rows, was used only for evaluation through
`scripts/semif_bench.py` with SemIf's own evaluators; its rows are not redistributed.
