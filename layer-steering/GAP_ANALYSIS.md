# Gap analysis: current state → target specification

Written before implementing, so the delta is explicit rather than implied.

## Already satisfied

| Requirement | Status |
|---|---|
| Separate direction per layer, construction split only | ✅ `build_directions.py`, reads `construction.json` only |
| Each direction applied at its own layer | ✅ one document per layer |
| Common documented strength calibration | ✅ `alpha = z * sigma_within`, `calibration.py` |
| Baseline condition | ✅ `intervened_models.baseline` |
| Zero-strength condition | ✅ z=0 in every grid; exact at all 28 layers |
| Positive steering | ✅ z>0 arm |
| Negative steering | ✅ z<0 arm |
| Construction / calibration separation | ✅ group-disjoint by colour, asserted |
| Optional all-28-layer sweep | ✅ done, 44/44 cells |
| Deterministic decoding | ✅ greedy-free scoring; `match`/`logit_diff` are argmax comparisons. Repeat-run delta measured 0.0 |
| Compact artifacts | ✅ CSVs + JSON summaries; raw kept out of tables |
| Per-layer resumability | ✅ one run dir per layer, independent documents |
| Directions not recomputed unnecessarily | ✅ cached `.safetensors`, reused across runs |
| Targeted tests | ✅ 71 pass |

## Gaps to close

| # | Gap | Why it matters | Fix |
|---|---|---|---|
| **G1** | **Held-out evaluation never run.** Every result so far is calibration-split. | Requirement 5; and selection was made on calibration, so nothing is independently validated. Running evaluation now is exactly the right time — the configuration is frozen. | New evaluation-split protocols, one per layer. |
| **G2** | **Control direction generated but never applied.** `control_L*.safetensors` exist; no run used them. | Requirement 7 ("justified control direction"). Without it, a large effect can't be attributed to the *direction* rather than to any perturbation of that size. | Control arm per layer in the same document as the real direction. |
| **G3** | **Answer-flip rate not computed.** | Required results column. Flip ≠ target-answer rate: a flip needs the baseline to have been correct *and* the steered model to give the counterfactual. Needs a per-example join across three tables. | `results_table.py` joins `baseline_accuracy`, `original_answer_rate`, `cf_answer_rate` by `example_id` within a `(layer, strength)` cell. |
| **G4** | **No uncertainty estimates.** Only p-values. | Required column. p-values are not effect sizes. | **Paired bootstrap** over prompts for every rate and margin, 10k resamples, percentile CI. Plus Wilson CI for the rates. |
| **G5** | **Relative network depth not reported.** | Required column. | `layer / (n_layers - 1)`. |
| **G6** | **No figures.** `figures/` is empty; analyses A–E absent. | Required deliverables. | `figures.py` producing A–E. |
| **G7** | **`analyze.py` hardcodes 4 layers** — cannot regenerate the 28-layer statistics. | Reproducibility: committed code does not produce the headline result. | Parameterize by config. |
| **G8** | **Decoding configuration not recorded** in artifacts. | Requirement 9. | Stamp engine, dtype, revision, greedy flag, tokenizer into every results directory. |
| **G9** | **Results not excluded from version control.** `research/` is untracked wholesale; 43 MB of raw tensors would land in git. | Requirement: keep generated results out of VCS. | `.gitignore` for raw activations and safetensors; keep only tables/figures/JSON summaries. |

## Deliberately not done

* **G1 for a second model** — out of scope, single-model spec.
* **k-fold over seeds** — 74 calibration prompts is the binding constraint; more prompts beats more folds here. Noted as a limitation.
* **Layer 0** — degenerate (`sigma_within = 0.053`, so d is inflated by a near-zero denominator). Reported but flagged as uninformative rather than silently dropped.

## Ordering rationale

G8 → G1/G2 must come first, because the held-out run writes the artifacts that G3/G4 read and G6 plots. Computing the table before the run would be computing it on the wrong split — which is precisely the error requirement 8 forbids.