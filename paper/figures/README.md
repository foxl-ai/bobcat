# Figures for the Bobcat technical report

The PDFs in this directory are rendered by `scripts/paper_figures.py` from the fields listed
below (it also writes 2400x1350 PNG copies to `artifacts/paper/png/`). `main.tex` shows a
framed placeholder for any figure that is missing, so the paper compiles without them.

Render the figures with `scripts/paper_figures.py`. Save each as a vector PDF with exactly
the file name below. The exception is `highlights.pdf`, the model card's figure
(`release/hf/assets/bobcat-highlights.png`) wrapped as a PDF. Use only the JSON fields listed; do not type numbers by hand. Paths are relative to the repository root.

Sources used below:

- `ZS` = `reports/2026-09-24-student-zero-shot-comparison.json` (a list, one object per backbone)
- `DS` = `reports/2026-09-24-deepseek-v41-flash-zero-shot.json`
- `TR` = `reports/2026-09-25-bobcat-student-training.json` (a list, one object per arm)
- `FS` = `reports/2026-09-25-bobcat-exp1-final-and-stress.json`

## figures/leaderboard.pdf (Figure 1, Section 5.2)

Horizontal bar chart of development task-macro accuracy (failures counted wrong) for the
eight zero-shot backbones, sorted descending.

- Seven backbones: `ZS[*].repo` (label), `ZS[*].macro` (bar length).
- DeepSeek-V4.1-Flash: `DS.repo`, `DS.task_macro_accuracy_including_failures`.
- Annotate each bar with ECE: `ZS[*].ece`, and for DeepSeek `DS.overall.ece_10_equal_width_bins`.
- Reference marker (not a bar): `bobcat-exp1` dev macro, the object in `TR` with
  `arm == "sft_vocab"`, field `macro`. Label it "bobcat-exp1 (trained, piecewise encoding)".
  Also mark the piecewise zero-shot baseline, `TR` object `arm == "zero_shot_piecewise"`,
  field `macro`, to show that the two encodings differ (85.8% vs 85.5% for Qwen3.8-27B).
- No per-model error bars: the sources give only paired differences.
- Caption note: development split, 3,188 decisions; not the sealed final.

## figures/k_scaling.pdf (Figure 2, Section 5.2)

Line plot of title-selection accuracy against the number of candidates K in {8, 77, 200,
255} (categorical x axis).

- Seven backbones: `ZS[*].title_k.k8`, `.k77`, `.k200`, `.k255`.
- DeepSeek: `DS.families.document_title_k8.accuracy`, `..._k77.accuracy`,
  `..._k200.accuracy`, `..._k255.accuracy`; annotate that these are over finite rows only
  (`DS.families.document_title_k77.count` is 70, not 75).
- Piecewise zero-shot and bobcat-exp1: `TR` objects `zero_shot_piecewise` and `sft_vocab`,
  field `title_k.*`; draw them dashed.
- Shade K = 8 (all models share GLM's identifiers when K <= 62); mark that for K >= 77 Qwen
  and Gemma use extension letters while DeepSeek, A.X, Phi and Tri use multi-digit numerals.
- 75 questions per K (development split).

## figures/final_tasks.pdf (Figure 3, Section 5.4)

Grouped bar chart: one group per task, two bars (bobcat-exp1, same-input zero-shot), sealed
final split.

- bobcat-exp1: `FS.final.release_metrics.tasks.{search, citation, tool_call, injection,
  routing, classification}`.
- Zero-shot: `FS.final.baseline_metrics.tasks.{...}` (same keys).
- X labels with the number of decisions: search 432, citation 275, tool call 96 (held out),
  external document screening 280 (key `injection`), routing 274, classification 250
  (counts from `reports/2026-09-24-korean-product-evaluation-v2.json`,
  `summary.task_split["product_*|final"]`).
- Title or inset: task macro `FS.final.release_metrics.task_macro_accuracy_including_failures`
  vs `FS.final.baseline_metrics.task_macro_accuracy_including_failures`, and the paired
  difference `FS.final.release_minus_baseline` = [point, low, high] (+10.2 [+7.7, +12.3]).
- Mark the tool-call group as "never trained; 3 situations" and show its paired difference
  `FS.final.release_minus_baseline_tool_call`.

## figures/injection.pdf (Figure 5, Section 6.5)

Two panels, both models (bobcat-exp1 = `FS.stress.bobcat_exp1`, zero-shot =
`FS.stress.zero_shot`), 300 development questions.

- Left: attack success by variant, grouped bars for `injection.attack_success_by_variant.
  {state_note, inside_text, authority}`, plus a fourth group for
  `injection.attack_success_all`. Label the variants "system note", "inside the text",
  "authority". Optionally overlay accuracy as dots from
  `injection.accuracy_by_variant.{clean, state_note, inside_text, authority}`.
- Right: attack success by task, `injection.attack_success_by_task.product_{classification,
  injection, routing, citation, search, tool_call}`, sorted by the bobcat-exp1 value; label
  `product_injection` as "external document screening".
- Definition in the caption: among questions whose clean answer was not the named wrong
  answer X, the fraction whose answer became X.

## figures/latency.pdf (Figure 6, Section 7)

Two panels, one B300, BF16, merged LoRA (`FS.serving`).

- Left: bars of HTTP p50 with a cap or dot at p95, in ms, for
  `FS.serving.workloads.W1_512tok_8choices_1q.http_full`,
  `W2_3k_state_8q.http_full`, `W2_3k_state_8q.http_shared`, `W3_8192tok_1q.http_full`,
  `W3_16320tok_1q.http_full`, `W3_K77_titles.http_full`, `W3_K255_titles.http_full`
  (fields `p50_ms`, `p95_ms`). Log y axis. Dashed horizontal lines at 150 ms and 300 ms
  labelled "W1 targets (p50, p95)"; state that they apply to W1 only.
- Right: throughput against static batch size, x = 1, 8, 16, 32, y =
  `FS.serving.workloads.W4_1k_independent_static_batch["1"|"8"|"16"|"32"].tokens_per_second`;
  annotate each point with its `p50_ms`.
- Caption notes: warm, 20 timed repetitions (10 for W3 8K/16K and W4), in-process HTTP
  client (no network), one fixed request per profile for W1 to W3.

## figures/fixedness.pdf (Figure 4, Section 6.2)

Dot plot, log y axis, of the largest and the mean probability gap for each execution-path
comparison, both models (`FS.stress.bobcat_exp1.determinism`, `FS.stress.zero_shot.determinism`).

- Comparisons: `batch_of_8_vs_alone`, `shared_prefix_vs_full`, `serving_vs_evaluation_path`;
  fields `max_probability_gap` (filled marker) and `mean_probability_gap` (open marker).
- Print `argmax_agreement` next to each comparison.
- `repeat_same_batch.max_logit_gap` is exactly 0 for both models and the zero-shot
  `serving_vs_evaluation_path` gaps are exactly 0; these cannot be drawn on a log axis, so
  annotate them as "0 (bit-identical)" instead of plotting.
- Optional second panel: permutation stability, `permutation.same_answer_all_orders` and
  `permutation.mean_tv_vs_original` for both models.

## figures/korean.pdf (Figure 7, Section 8)

Grouped bars, bobcat-exp1 vs same-input zero-shot, from
`KB` = `reports/2026-09-25-korean-external-benchmarks.json`, for `kobest`, `kmmlu`, `haerae`,
`click` (Korean) and `mmlu`, `hellaswag_en` (English controls).

- Bars: `KB.benchmarks.<name>.bobcat_exp1.accuracy` and `.zero_shot.accuracy`; x labels with
  `.language` and `.bobcat_exp1.n`.
- Above each pair: the paired difference `KB.benchmarks.<name>.bobcat_exp1_minus_zero_shot`
  = [point, low, high] (95% cluster bootstrap).
- Chance level: `KB.benchmarks.<name>.bobcat_exp1.chance` (mean 1/K), dotted.

## Numbering

LaTeX numbers figures in order of appearance: leaderboard (1), k_scaling (2),
final_tasks (3), fixedness (4), injection (5), latency (6), korean (7). The headings above
give that number for each file.
