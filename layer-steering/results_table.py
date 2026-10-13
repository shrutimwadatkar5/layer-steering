"""Build the compact per-condition results table for the held-out evaluation.

Every column requested by the specification, with the denominator stated:

    layer, relative_depth, alpha, n_prompts,
    baseline_accuracy, target_answer_rate, preference_change, flip_rate,
    output_validity, and confidence intervals for each.

Definitions
-----------
Let ``b`` = baseline prediction, ``s`` = steered prediction, over the ``n``
evaluation prompts. ``correct(x)`` is ``match(x, base_answer)``.

* **baseline_accuracy** = mean[correct(b)]
* **target_answer_rate** = mean[match(s, cf_answer)] — the counterfactual-answer
  rate under steering. Denominator ``n``.
* **preference_change** = mean[margin(s) - margin(b)], where
  ``margin = logit(base_answer) - logit(cf_answer)``. A positive value means
  steering moved probability mass *away* from the original answer. Paired by
  prompt. Denominator ``n``.
* **flip_rate** = mean[correct(b) AND match(s, cf_answer)] — the strictest and
  most interpretable behavioural measure: the prompt was answered correctly and
  steering made the model give the counterfactual. Denominator ``n``, and the
  count over ``correct(b)`` is reported alongside as ``flip_rate_of_correct``.
* **output_validity** = fraction of rows with ``eligible: true``. A row is
  ineligible when the engine could not resolve the metric for it, so this is the
  check that no condition silently lost prompts.

Control direction
-----------------
The same columns are computed for the norm-matched random direction, so a
reader can subtract the two. ``real_minus_control`` on the margin is the direct
estimate of how much the *direction* contributes as opposed to any perturbation
of that size.

Uncertainty
-----------
Paired bootstrap over prompts, 10,000 resamples, percentile intervals. Wilson
intervals for the rates. Both are computed from the per-prompt vectors, so they
respect the pairing; an unpaired interval would be wider and wrong here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "ALPHA_KEY",
    "wilson_ci",
    "bootstrap_ci_paired",
    "build_table",
]

ALPHA_KEY = "writes.push.do.add_scaled.alpha"

_RNG_SEED = 20240917
_BOOTSTRAP = 10000


def wilson_ci(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a proportion. Stable at 0 and 1, unlike Wald."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci_paired(
    a: list[float], b: list[float], *, seed: int = _RNG_SEED
) -> tuple[float, float, float]:
    """Percentile CI for ``mean(a - b)``, resampling prompts with replacement.

    The same resampled prompt indices are used for both arms, which is what makes
    this *paired* -- the resampling respects prompt identity.
    """
    import numpy as np

    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = diff.size
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(_BOOTSTRAP, n))
    means = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _load(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else [data]


def _series(rows: list[dict[str, Any]]) -> dict[float, dict[str, float]]:
    """``{alpha: {example_id: value}}`` from one result table."""
    out: dict[float, dict[str, float]] = {}
    for row in rows:
        if ALPHA_KEY not in row:
            continue
        out.setdefault(float(row[ALPHA_KEY]), {})[str(row["example_id"])] = float(
            row["value"]
        )
    return out


def _validity(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r.get("eligible", True)) / len(rows)


def build_table(
    run_dir: Path, layer: int, n_layers: int = 28, *, seed: int = _RNG_SEED
) -> list[dict[str, Any]]:
    """One row per (layer, alpha, arm) for one evaluation run directory."""
    import numpy as np

    run_dir = Path(run_dir)
    base_acc = _series(_load(run_dir / "baseline_accuracy.json"))
    base_margin = _series(_load(run_dir / "baseline_margin.json"))
    arms: dict[str, tuple] = {}
    for arm, prefix in (("real", "steered"), ("control", "control")):
        raw_rows = _load(run_dir / f"{prefix}_original_rate.json")
        if not raw_rows:
            continue
        arms[arm] = (
            _series(raw_rows),
            _series(_load(run_dir / f"{prefix}_cf_rate.json")),
            _series(_load(run_dir / f"{prefix}_margin.json")),
            _validity(raw_rows),
            len(raw_rows),
        )

    if not base_acc:
        return []

    rows: list[dict[str, Any]] = []
    # The baseline is repeated identically at every swept alpha, so its alpha
    # coordinate is a label, not a condition. Pin it to the genuine zero rather
    # than min(): `min` returns the most negative alpha, which mislabels the
    # reference cell and makes the alpha=0 row of an arm compare against the
    # wrong baseline entry.
    zero_alpha = 0.0 if 0.0 in base_acc else min(base_acc)

    for arm, (orig, cf, margin, validity, n_rows_raw) in arms.items():
        if not orig:
            continue
        for alpha in sorted(orig):
            ids = sorted(
                set(orig[alpha])
                & set(cf.get(alpha, {}))
                & set(margin.get(alpha, {}))
                & set(base_acc.get(zero_alpha, {}))
                & set(base_margin.get(zero_alpha, {}))
            )
            n = len(ids)
            if n == 0:
                continue

            base_ok = np.array(
                [base_acc[zero_alpha][i] for i in ids], dtype=float
            )
            base_mg = np.array(
                [base_margin[zero_alpha][i] for i in ids], dtype=float
            )
            orig_ok = np.array([orig[alpha][i] for i in ids], dtype=float)
            target = np.array([cf[alpha][i] for i in ids], dtype=float)
            mg = np.array([margin[alpha][i] for i in ids], dtype=float)

            flip = base_ok * target
            correct = base_ok.sum()
            n_correct = int(round(float(correct)))

            pref_mean, pref_lo, pref_hi = bootstrap_ci_paired(
                mg.tolist(), base_mg.tolist(), seed=seed
            )
            t_lo, t_hi = wilson_ci(int(target.sum()), n)
            b_lo, b_hi = wilson_ci(int(base_ok.sum()), n)
            f_lo, f_hi = wilson_ci(int(flip.sum()), n)

            rows.append(
                {
                    "layer": int(layer),
                    "relative_depth": round(layer / (n_layers - 1), 4),
                    "alpha": float(alpha),
                    "arm": arm,
                    "n_prompts": n,
                    "baseline_accuracy": round(float(base_ok.mean()), 6),
                    "baseline_acc_ci_lo": round(b_lo, 6),
                    "baseline_acc_ci_hi": round(b_hi, 6),
                    "target_answer_rate": round(float(target.mean()), 6),
                    "target_rate_ci_lo": round(t_lo, 6),
                    "target_rate_ci_hi": round(t_hi, 6),
                    "original_answer_rate": round(float(orig_ok.mean()), 6),
                    "preference_change": round(pref_mean, 6),
                    "pref_change_ci_lo": round(pref_lo, 6),
                    "pref_change_ci_hi": round(pref_hi, 6),
                    "flip_rate": round(float(flip.mean()), 6),
                    "flip_ci_lo": round(f_lo, 6),
                    "flip_ci_hi": round(f_hi, 6),
                    "flip_count": int(flip.sum()),
                    "n_correct_at_baseline": n_correct,
                    "flip_rate_of_correct": (
                        round(float(flip.sum()) / n_correct, 6) if n_correct else None
                    ),
                    "output_validity": round(float(validity), 6),
                    "n_ineligible": int(round((1.0 - validity) * n_rows_raw)),
                    "n_rows_saved": n_rows_raw,
                }
            )

    rows.sort(key=lambda r: (r["arm"], r["alpha"]))
    return rows