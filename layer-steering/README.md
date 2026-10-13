# Layer-Dependent Activation Steering in Qwen2.5-1.5B-Instruct

Research code for two questions about activation steering:

1. **How does steering effectiveness vary across the 28 transformer layers?**
2. **How well does a direction built at one layer transfer to another?**

Built on [Causalab](https://github.com/goodfire-ai/causalab), pinned at commit
`8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e` (2026-09-30).

> **Status: preliminary.** Every number in this README was produced by the
> committed code on a single model and a single task. Section
> [20. Current status](#20-current-status) states precisely what has and has not
> been run.

> **Licensing.** Upstream Causalab ships **no license**. This directory
> therefore has none either. See [`LICENSE_STATUS.md`](LICENSE_STATUS.md) —
> read it before redistributing anything here.

---

## Table of contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
3. [Research questions](#3-research-questions)
4. [Hypotheses](#4-hypotheses)
5. [Methodology](#5-methodology)
6. [Model and framework](#6-model-and-framework)
7. [Dataset and splits](#7-dataset-and-splits)
8. [Activation harvesting](#8-activation-harvesting)
9. [Direction construction](#9-direction-construction)
10. [Intervention semantics](#10-intervention-semantics)
11. [Layer selection](#11-layer-selection)
12. [Strength calibration](#12-strength-calibration)
13. [Evaluation metrics](#13-evaluation-metrics)
14. [Statistical analysis](#14-statistical-analysis)
15. [Cross-layer transfer](#15-cross-layer-transfer)
16. [Reproduction](#16-reproduction)
17. [Runtime and hardware](#17-runtime-and-hardware)
18. [Results](#18-results)
19. [Limitations](#19-limitations)
20. [Ethics and safety](#20-ethics-and-safety)
21. [Citation](#21-citation)
22. [Current status](#22-current-status)

---

## 1. Overview

This project measures how the depth of an activation-steering intervention
changes its behavioural effect, and how well a direction built at one layer
works when applied at a different one.

Everything is expressed as **Causalab protocol documents** — JSON files
describing an experiment, executed by the Causalab CLI. No custom forward
hooks, no reimplementation of the intervention engine. The research code here
does three things Causalab does not: build datasets, construct directions, and
analyse results.

The intervention itself is Causalab's `add_scaled`:

```
h'_l = h_l + alpha * v_l
```

Applied at `block_output` (the output of a whole transformer block), at the
final prompt token, on a 2-way multiple-choice task.

## 2. Motivation

Steering vectors are usually reported at one layer, or averaged over layers. That
hides two things a practitioner actually needs:

* **Dose.** A raw steering coefficient is not comparable across depth. Residual
  norms grow substantially with depth, so the same `alpha` is a much smaller
  perturbation in late layers. Any comparison using a shared `alpha` is
  comparing unequal interventions.
* **Layer identity.** Whether a direction works *everywhere*, or only where it
  was built, changes how it should be used — and whether a failure at one layer
  says anything about the feature.

A first pilot here returned a clean null at every layer. The cause was dose, not
direction: at `alpha = 2` the perturbation reached only ~0.05σ. That forced the
z-score calibration in [§12](#12-strength-calibration), after which the effect
appeared. This is the methodological spine of the project.

## 3. Research questions

* **RQ1.** How does the behavioural effect of an additive steering intervention
  vary across the 28 layers of Qwen2.5-1.5B-Instruct?
* **RQ2.** How does that effect vary with intervention strength, and is the
  relationship monotone?
* **RQ3.** Does positive and negative steering produce distinguishable effects?
* **RQ4.** Does a direction constructed on held-out data generalise to unseen
  prompts?
* **RQ5.** How well does a direction built at source layer *i* transfer to target
  layer *j*, at matched effective dose?

## 4. Hypotheses

Stated as hypotheses. Results in [§18](#18-results) are separate.

| ID | Hypothesis | Outcome |
|---|---|---|
| **H1** | Steering effects vary systematically with intervention layer. | **Supported** (Friedman p = 1.7e-107) |
| **H2** | Positive and negative steering produce distinguishable effects. | **Not supported on held-out** — significant at 15/28 layers individually, but no overall asymmetry (sign test p = 0.185). Did not replicate from calibration. |
| **H3** | Steering generalises to prompts not used to build the direction. | **Partially evaluated** — evaluation split run once, but target-answer rate ≈ 0 everywhere, so behavioural generalisation is not demonstrated. |
| **H4** | A direction built at one layer behaves differently when applied at another. | **Supported** — same-layer mean \|effect\| 0.472 vs cross-layer 0.074. |

## 5. Methodology

```
                 ┌─ construction split (189) ─┐
seed ───────────▶│  harvest block_output       │
                 │  build v_l per layer        │
                 └──────────────┬──────────────┘
                                │
                 ┌─ calibration split (74) ───┐
                 │  calibrate alpha (z-score) │
                 │  rank layers               │
                 └──────────────┬──────────────┘
                                │  configuration frozen
                 ┌─ evaluation split (118) ───┐
                 │  run ONCE, both arms       │
                 └────────────────────────────┘
```

Splits are group-disjoint by **colour**, enforced and asserted, so a colour
word seen while building a direction is never seen at evaluation.

## 6. Model and framework

| | |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, revision `main` |
| Layers | 28 |
| `hidden_size` | 1536 |
| dtype | bfloat16 |
| Framework | Causalab @ `8e8d5d1`, engine `pytorch_hooks` |
| Component | `block_output` — output of a whole transformer block |
| Token position | `{"index": -1}` — final prompt token |
| Task | 2-way multiple choice (MCQA, shipped with Causalab) |

Verified environment:

```
python     3.10.18
torch      2.9.0+cu128
transformers 5.16.1
numpy      2.2.6
scipy      1.15.3
CUDA       12.8
GPU        NVIDIA A100-PCIE-40GB
platform   Linux-5.15.0-139-generic-x86_64, glibc 2.31
```

`block_output` is `Tap("block")` in Causalab's registry
(`causalab/protocol/registry/families.py:587`), and Causalab records the identity
`block_output == attention_output + mlp_output + block_input`. The hook was
measured directly: a forward hook on `model.model.layers[L]` returns
`(batch, padded_seq_len, 1536)` in bfloat16.

## 7. Dataset and splits

Generated from Causalab's `generate_split_dataset`, which partitions **unique
inputs** into disjoint groups and forms counterfactual pairs *within* each split.

| Split | Rows | Role |
|---|---|---|
| construction | 189 | harvest activations, build directions |
| calibration | 74 | calibrate strength, rank layers |
| evaluation | 118 | run once, after the configuration was frozen |

* **381 unique prompts**, group key `color`, 10 colour groups split 5/2/3.
* `require_label_change=True` — pairs whose counterfactual does not change the
  answer are dropped, so accuracy is not inflated for free.
* **Leakage is asserted, not assumed.** `build_dataset.py` fails loudly if a
  prompt or a colour value crosses a split. This check *caught* an earlier
  configuration: grouping by prompt alone let objects span splits, which is why
  the shipped config groups by colour.
* Record in `data/build_report.json`.

**Known confound.** MCQA draws a fresh symbol pair per example, so the symbol
inventory is deliberately unbalanced (7–24 uses per letter, recorded in the build
report). A mean-difference direction over such pairs can encode letter frequency.
This is flagged, not corrected — correcting it needs a dataset the task does not
offer.

## 8. Activation harvesting

Read at `block_output`, position `{"index": -1}`, one tensor per swept layer:
`(189, 1, 1536)` bfloat16.

```bash
uv run causalab run research/protocols/harvest_dense.json \
  --engine auto --data-root research/data \
  --out research/results/raw/harvest_dense
```

**Padding is a real hazard, and it is handled.** In a ragged batch the tensor is
padded to the widest row and **padded positions carry a non-zero residual norm**
(measured ≈ 53 against ≈ 630 for real tokens). A hard-coded `-1` would address
padding, not the answer slot. Causalab resolves positions against each row's
*first real* token (`token_index = index - first`,
`causalab/protocol/positions/resolve.py:404`), so the pilot uses a position spec
and never a precomputed index.

## 9. Direction construction

For each layer, on the **construction split only**:

```
v_l = mean(h_l | group A) - mean(h_l | group B)
```

Groups are defined by answer position: group A is rows whose answer is the
*second* option's symbol, group B the first. Accumulation is in float64 (a bf16
mean over many rows loses the bits the difference is made of), then cast to
float32.

Two properties are recorded per layer in `directions_manifest.json`:

* `sigma_within` — pooled within-group standard deviation along `v_l`
* `raw_norm` / `norm_ratio` — the direction's size relative to the local
  activation norm

**Measured quality (Cohen's *d* along the steering axis) shows three regimes:**

| Layers | Cohen's *d* |
|---|---|
| 1–9 | 0.28 – 0.57 |
| 10–15 | 0.62 – 1.91 |
| **16** | **4.88** ← step |
| 17–27 | 3.75 – 7.54 |

The jump between L15 and L16 is discontinuous, not a smooth gradient. Layer 0 is
degenerate (`sigma_within = 0.053`), so its *d* is inflated by a near-zero
denominator and is not informative.

## 10. Intervention semantics

Steering is Causalab's `add_scaled`, verified in the executor:

* `causalab/neural/shared/mechanisms.py:142` returns `alpha * op`
* `causalab/neural/shared/executor/writes.py:707` writes `f = f0 + delta`

So the write is **additive, never a replacement**. Upstream pins this
numerically in `tests/neural/engines/pytorch_hooks/test_write_oracle.py:169`.

**Verified directly on the A100:**

| Check | Result |
|---|---|
| `alpha = 0` vs unmodified baseline | identical (max deviation 0.0026, one bf16 ulp) |
| no-op hook | logit change **0.0** |
| `alpha = 2` | logit change **0.359** |
| repeat run | logit change **0.0** (bitwise deterministic) |

The residual 0.0026 is bf16 rounding on near-tied prompts — float reduction
order, not intervention leakage. Accuracy at `alpha = 0` is exact in every cell.

## 11. Layer selection

No layer is chosen by searching the evaluation split.

* Directions and sweeps cover **all 28 layers** (`configs/dense.json` sets
  `allow_full_sweep: true`; the earlier 4-layer pilot set it false).
* Strength calibration and effect ranking used the **calibration** split.
* The evaluation split was run **once**, after the configuration was frozen.
* `results/evaluation/layer_selection.json` records the calibration leader and
  the evaluation leader separately.

**They disagree.** The calibration leader (by largest-magnitude paired margin
change) was **L21 at −3.370**; the evaluation leader is **L16 at −0.518**. Both
are recorded in `results/evaluation/layer_selection.json`. The disagreement is
reported rather than resolved by re-picking, and it is the clearest evidence
that a per-layer claim is not stable across splits at this sample size. The
magnitudes are also not directly comparable: calibration was swept to `z = ±12`
and evaluation only to `z = ±8`.

## 12. Strength calibration

**This is the methodological core.** Residual norms grow with depth, so a shared
`alpha` is not a shared intervention. Strength is therefore

```
alpha(z) = z * sigma_within
```

where `sigma_within` is the pooled within-group standard deviation of the
activation **projected onto that layer's steering direction**. `z` is then a
within-group z-score, and the same `z` means the same relative dose at every
layer.

Measured justification — at a shared `alpha = 2` on a unit direction, the
perturbation was 0.7 %–5.5 % of the local residual norm, and **7.5× less at L26
than at L4**:

| Layer | mean ‖h‖ | perturbation at α=2 |
|---|---|---|
| 4 | 36.5 | 5.5 % |
| 12 | 48.2 | 4.2 % |
| 20 | 80.5 | 2.5 % |
| 26 | 273.2 | **0.7 %** |

The uncalibrated sweep returned a **null at every layer**; the calibrated sweep
returns significant effects at 22/28. Same directions, same model, same data —
only the dose convention changed.

Grids: `z ∈ {0, ±2, ±4, ±6, ±8, ±12}` (layer-wise), `z ∈ {0, ±4, ±8}`
(held-out and transfer). `z = 0` is the mandatory zero-strength control and the
config refuses a grid without it.

## 13. Evaluation metrics

Scoring only — **no tokens are generated**, so results are a deterministic
function of the weights. Decoding configuration is stamped into every results
directory.

| Metric | Definition | Denominator |
|---|---|---|
| Baseline accuracy | `mean[match(baseline, base_answer)]` | n prompts |
| Original-answer rate | `mean[match(steered, base_answer)]` | n prompts |
| Target-answer rate | `mean[match(steered, cf_answer)]` | n prompts |
| **Preference change** | `mean[margin(steered) − margin(baseline)]`, `margin = logit(base) − logit(cf)` | n prompts, paired |
| **Answer-flip rate** | `mean[correct(baseline) ∧ match(steered, cf_answer)]` | n prompts; also over correct-at-baseline |
| Output validity | fraction of rows with `eligible: true` | n prompts |

Confidence intervals: **paired bootstrap** over prompts (10 000 resamples,
percentile) for every effect, and **Wilson** intervals for all rates.

## 14. Statistical analysis

* **Unit of analysis is the prompt.** Every comparison is paired by prompt.
* **Within a layer:** Wilcoxon signed-rank on paired per-prompt differences
  against `z = 0`. Non-parametric because margins are bounded and skewed.
* **Across layers:** Friedman omnibus on per-prompt deltas at matched `z`, then
  pairwise Wilcoxon with **Holm-Bonferroni** across layer pairs.
* **Across strength:** paired bootstrap CIs per cell.
* **Selection discipline:** layers and strengths chosen on calibration; held-out
  touched once.

## 15. Cross-layer transfer

Run as a **separate experiment** (`research/transfer.py`), never pooled with the
layer-wise results. Same-layer cells exist only as references *within* this grid.

`check_dimension_compatibility` verifies all directions share one width —
necessary because Causalab would otherwise **silently broadcast** a
1536-wide direction into any layer.

**Equal width ≠ equal coordinates.** Measured on the construction split:

```
cos(v₄,v₁₂)=0.13   cos(v₄,v₂₀)=0.01   cos(v₄,v₂₆)=0.01
cos(v₁₂,v₂₀)=0.05  cos(v₁₂,v₂₆)=0.02  cos(v₂₀,v₂₆)=0.45
```

and the target layer's spread along a *foreign* direction is 2–20× smaller than
along its own (σ table in `transfer_results.json`). Dose is therefore calibrated
**per cell**: `alpha_ij = z * sigma_ij`.

Outcome classification is deliberately conservative, and the language is
enforced in code (`OUTCOMES` in `analyze_transfer.py`):

* `successful_transfer` — CI excludes 0 and effect ≥ half the same-layer reference
* `failed_transfer_possible_misalignment` — CI includes 0 at calibrated dose
* `intervention_induced_degradation` — accuracy falls, target-answer rate does not rise
* `inconclusive` — everything else

**A failed transfer does not show a feature is absent from a layer.** A
successful transfer shows behavioural effect, not a uniquely interpretable
feature.

## 16. Reproduction

### Environment setup

```bash
# Rust 1.98.1 — must match upstream rust-toolchain.toml (builds the C extension)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --default-toolchain 1.98.1 --profile minimal --no-modify-path

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
cd causalab && git checkout 8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e

uv sync          # ~8.1 GB, dominated by CUDA wheels
```

Verify:

```bash
uv run python -c "import torch, causalab, causalab.io.fastersafetensors._core as c; \
  print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# torch 2.9.0+cu128 True NVIDIA A100-PCIE-40GB
```

### Tests and smoke test

```bash
uv run pytest research/tests -q        # 153 passed
uv run pytest tests/tasks tests/demos -q   # upstream, unmodified

# protocol validation needs no weights
uv run causalab validate research/protocols/harvest_construction.json \
  --engine auto --data-root research/data --tokenizer
# OK: — 4 points, digest 5df649f41cab6c3b…
```

### Full pipeline

```bash
# 1. splits (asserts leakage checks)
uv run python -m research.scripts.build_dataset \
  --config research/configs/dense.json --pool-size 400

# 2. harvest all 28 layers
uv run causalab run research/protocols/harvest_dense.json \
  --engine auto --data-root research/data --out research/results/raw/harvest_dense

# 3. directions + norm-matched random controls
uv run python -m research.scripts.build_directions \
  --config research/configs/dense.json \
  --harvest-dir research/results/raw/harvest_dense \
  --out-dir research/directions_dense

# 4. calibrate alpha (arithmetic only, no GPU)
uv run python -m research.scripts.calibrate_alpha \
  --config research/configs/dense.json \
  --harvest-dir research/results/raw/harvest_dense \
  --directions-dir research/directions_dense \
  --z-grid=-12,-8,-6,-4,-2,0,2,4,6,8,12 \
  --out research/results/tables_dense/dose_calibration.json

# 5. analysis (tables, statistics, figures)
uv run python -m research.scripts.run_analysis --config research/configs/dense.json
uv run python -m research.scripts.run_transfer_analysis \
  --config research/configs/dense.json --z 8
```

Held-out and transfer runs are one document per cell and are independently
resumable — a cell whose `run_config.json` exists is skipped.

### Expected output structure

```
research/
├── results/
│   ├── evaluation/
│   │   ├── evaluation_table.csv        280 rows: one per (layer, α, arm)
│   │   ├── evaluation_summary.json     Friedman, Holm, zero-strength audit
│   │   ├── layer_selection.json        calibration vs evaluation leader
│   │   └── figures/A…E*.png            five publication figures
│   └── transfer_analysis/
│       ├── transfer_table.csv          16 rows: one per (source, target)
│       ├── transfer_summary.json       outcomes, asymmetry, caveats
│       └── figures/transfer_heatmap.png
├── *_results.json                      per-stage machine-readable findings
└── data/build_report.json              split sizes, leakage checks, symbol counts
```

Raw per-example tables (251 MB) and swept activation tensors are **gitignored**;
they are regenerated exactly by the commands above.

## 17. Runtime and hardware

Measured on one A100-PCIE-40GB (spans computed from artifact timestamps):

| Stage | Measured |
|---|---|
| Dense harvest, 28 layers × 189 prompts | ~4 min |
| Calibration sweep, 28 layers × 11 z | **32 min** |
| Held-out sweep, 28 layers × 2 arms × 5 z | **60 min** |
| Transfer grid, 16 cells × 5 z | **15 min** |
| Analysis + figures | < 2 min |
| `uv sync` (first time) | ~50 min (8.1 GB over ~2 MB/s) |

Requirements: **1× GPU with ≥8 GB** for Qwen2.5-1.5B in bf16; the measured host
used 40 GB but peak usage was far lower. CPU-only is not supported by the
upstream default configuration.

## 18. Results

All from the committed code on Qwen2.5-1.5B-Instruct.

### Layer dependence (held-out, 118 prompts)

* **Friedman across 28 layers at matched `z`: p = 1.69e-107**, n = 118 paired prompts.
* **164 of 378** pairwise layer comparisons significant after Holm correction.
* **21 of 28 layers** have 95 % CIs excluding zero at `z = +8`.
* **Control.** Against a norm-matched random direction at the same layer and the
  same effective dose (`z = +8`), the real direction has the larger absolute
  effect at **15/28 layers** and pushes further in the same direction at
  **18/28**. Strongest separation at L16 (real −0.518 vs control −0.152).
  This is partial evidence that the effect is attributable to the *direction*
  rather than to perturbing the residual stream by that magnitude; the control
  is not negligible at several layers, so the effect is not cleanly separable
  at this sample size.
* Same-layer diagonal effects: L4 −0.201, L12 −0.152, L20 −0.910, L26 −0.624.

### Cross-layer transfer (4×4 grid, 74 prompts, matched z = +8)

| source \ target | t4 | t12 | t20 | t26 |
|---|---|---|---|---|
| **4** | −0.201* | −0.079 | −0.003 | +0.029 |
| **12** | −0.051 | −0.152* | −0.061 | +0.058 |
| **20** | −0.008 | +0.038 | −0.910* | +0.212 |
| **26** | +0.004 | +0.076 | −0.274 | −0.624* |

`*` same-layer reference. Same-layer mean |effect| **0.472**; cross-layer
**0.074**.

Outcomes: **3** successful (the diagonal) · **4** failed-possible-misalignment
· **1** degradation (20→20) · **8** inconclusive.

The 20↔26 pair is the only one with meaningful cosine (0.45) and the only one
with off-diagonal structure — transferring in **opposite signs**
(+0.212 / −0.274), consistent with shared structure in a rotated basis rather
than a shared coordinate.

### Preliminary observations

* **Dose dominates everything.** The same experiment returned a uniform null
  before calibration and a clear result after.
* **Layer identity matters, and the transition is abrupt** — Cohen's *d* steps
  from 1.91 to 4.88 between L15 and L16.
* **Margin and behaviour come apart.** Logit margins move by up to −3.37, but the
  counterfactual-answer rate never exceeded 0.068. Steering *disrupts* the
  computation without *redirecting* it toward the counterfactual. **The headline
  result is about representation, not about making the model say the wrong
  answer.**

## 19. Limitations

1. **Single model, single task.** One 1.5B model, one 2-way MCQA task, one
   component, one token position. Nothing here shows generality.
2. **Sample size.** 118 evaluation prompts against effects of 0.05–0.5 logits.
   A 0.014 accuracy granularity is *one prompt*.
3. **No stable best layer.** Calibration said L20, evaluation says L16.
4. **H2 did not replicate** from calibration to held-out.
5. **Behavioural transfer not observed.** Target-answer rate ≈ 0 in every cell.
6. **Coarse holdout.** 10 colour groups split 5/2/3.
7. **Symbol inventory unbalanced** (7–24 per letter) — flagged, not corrected.
8. **Layer 0 is degenerate** and excluded from interpretation.
9. **No control arm in the transfer experiment**, unlike the layer-wise one, so
   cross-layer effects cannot yet be separated from generic perturbation.
10. **The direction's causal validity is assumed, not established.** It is a
    mean difference between two answer positions.

## 20. Ethics and safety

* **This is interpretability research, not a capability study.** The objective is
  to understand where information lives in a model, not to make a model behave
  badly. No jailbreak, no harmful-task evaluation, no red-teaming.
* **The model is small and open.** Qwen2.5-1.5B-Instruct is used unmodified,
  with weights downloaded from its official source and **not redistributed here**.
* **Steering is a dual-use technique.** The same method that localises a model's
  internal representations can also be used to steer them toward undesirable
  behaviour. Results here are deliberately reported as **margin** effects rather
  as behavioural control, and the failure to flip answers is documented rather
  than engineered around.
* **Negative results are reported.** The uncalibrated null, the failed H2
  replication, the split instability in the best layer, and two implementation
  bugs that produced wrong numbers are all in the record — see
  `GAP_ANALYSIS.md` and `*_results.json`.
* **Honest reporting.** No result in this README was produced by a run that did
  not happen. Where something is untested, it is labelled untested.
* **Not peer reviewed.** These are preliminary results and should not be cited
  as established findings.

## 21. Citation

This project has **no publication** and no bibliographic metadata to cite. A
`CITATION.cff` is deliberately **not** provided: inventing one would fabricate a
DOI or author list that does not exist.

**Citing the method** — cite Causalab, which supplies the intervention engine,
task, and execution machinery:

> Geiger, A., Loeffler, M., and contributors. *Causalab: Causal analysis of
> neural networks.* <https://github.com/goodfire-ai/causalab>
> (pinned at `8e8d5d1f8f9ca8f42bfce7c6b5eef194f012660e`).

**Citing the model** — see the official model card for its license and terms:

> Qwen2.5-1.5B-Instruct, Alibaba Cloud.

## 22. Current status

**Implemented and tested**

* Dataset construction with group-disjoint splits and enforced leakage checks.
* Activation harvesting across all 28 layers.
* Per-layer direction construction with a norm-matched random control.
* z-score strength calibration.
* Additive steering via Causalab `add_scaled`, at 28 layers × 11 strengths.
* Held-out evaluation, both arms, run once.
* Cross-layer transfer grid, 4×4, per-cell dose calibration.
* Paired statistics with Holm correction; paired-bootstrap and Wilson CIs.
* Figures A–E plus the transfer heatmap.
* **153 unit tests**, including regression guards for both bugs described below.

**Tests executed**

```
research/tests                 153 passed
tests/tasks + tests/demos     1514 passed, 18 skipped   (upstream, unmodified)
```

**Experiments executed**

| Experiment | Status |
|---|---|
| Uncalibrated 4-layer pilot | ✅ complete (null) |
| Calibrated 4-layer pilot | ✅ complete |
| Calibrated 28-layer sweep | ✅ complete |
| Held-out evaluation, 28 layers × 2 arms | ✅ complete, 25/25 cells each |
| Cross-layer transfer 4×4 | ✅ complete, 16/16 cells |
| 28×28 exhaustive transfer | ❌ **deliberately not run** |
| Second model | ❌ **not run** |
| H4 cross-layer at full depth | ❌ not run |

**Two implementation bugs found and fixed**, both documented in
`GAP_ANALYSIS.md` and `evaluation_results.json`:

1. **Write collision** — real and control writes in one document at the same
   module; the control's alpha sweep silently never bound. Fixed by one arm per
   document; regression-tested.
2. **Baseline cell mislookup** — the reference cell was keyed on `min(alpha)`
   instead of `0.0`, making the zero-strength control appear violated by 0.53
   logits. Fixed; regression-tested.

**Not done:** peer review, second model, larger prompt sets, control arm for
transfer, exhaustive transfer grid.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | Validated configuration for the layer-wise experiment |
| `steering.py` | Direction construction and reference additive arithmetic |
| `calibration.py` | z-score dose calibration primitives |
| `evaluation.py` | Held-out protocol documents and run-config stamping |
| `transfer.py` | Cross-layer transfer config, compatibility checks, documents |
| `analyze_transfer.py` | Heatmap, outcome classification, asymmetry |
| `figures.py` | Figures A–E and the across-layer tests |
| `sign_asymmetry.py` | Positive-vs-negative paired test |
| `results_table.py` | Compact results table with Wilson/bootstrapped CIs |
| `scripts/build_dataset.py` | Group-disjoint splits, leakage assertions |
| `scripts/emit_protocols.py` | Harvest document generation |
| `scripts/build_directions.py` | Per-layer directions + random controls |
| `scripts/calibrate_alpha.py` | Per-layer dose table |
| `scripts/aggregate.py` | Per-(layer, α) tables + zero-strength audit |
| `scripts/analyze.py` | Paired statistics for the calibration sweep |
| `scripts/run_analysis.py` | Held-out table, statistics, figures A–E |
| `scripts/prepare_transfer.py` | Transfer grid: sigma table + documents |
| `scripts/run_transfer_analysis.py` | Transfer table, summary, heatmap |
| `tests/` | 153 unit tests |
| `configs/pilot.json` | 4-layer pilot configuration |
| `configs/dense.json` | 28-layer configuration |
| `GAP_ANALYSIS.md` | Gap analysis between design and implementation |
| `LICENSE_STATUS.md` | Upstream licensing findings — read before redistributing |
| `*_results.json` | Machine-readable findings per stage |