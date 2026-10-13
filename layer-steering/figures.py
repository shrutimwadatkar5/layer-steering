"""Generate the five required figures and the statistical summaries behind them.

A. Steering effectiveness versus layer depth
B. Positive versus negative steering effects
C. Steering-strength response curves
D. Baseline versus intervention performance
E. Confidence intervals

Every comparison is **paired by prompt**. Across-layer comparisons use the
Friedman omnibus followed by pairwise Wilcoxon with **Holm-Bonferroni**, and
the multiplicity correction is applied per strength so the reported adjusted
p-values are interpretable.

Selection discipline: the figures take a layer as a *reported* axis, not a
selected winner. ``layer_selection.json`` records that no layer was chosen using
the evaluation split -- the strongest layer was identified on the calibration
split and is carried here as a prediction to be checked, not as a result.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = [
    "holm_bonferroni",
    "layer_effect_test",
    "sign_asymmetry_test",
    "figure_a_depth",
    "figure_b_sign",
    "figure_c_response",
    "figure_d_baseline",
    "figure_e_ci",
    "write_layer_selection",
]

_ALPHA_KEY = "writes.push.do.add_scaled.alpha"


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Step-down adjusted p-values, monotone and capped at 1."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvalues[idx] * (m - rank)))
        adjusted[idx] = running
    return adjusted


def layer_effect_test(
    table: list[dict[str, Any]],
    raw_dir: Path | None = None,
    *,
    arm: str = "real",
    strength: float | None = None,
) -> dict[str, Any]:
    """Across-layer comparison at one strength, on per-prompt paired deltas.

    The per-cell ``preference_change`` in the table is a *mean*, so it cannot
    support a paired test on its own. This function therefore takes the
    per-prompt vectors from the raw run directories instead, which is why
    ``raw_dir`` is a required argument.
    """
    import numpy as np
    from scipy import stats

    if raw_dir is None:
        return {"error": "raw_dir required for a paired test"}

    from research.results_table import ALPHA_KEY, _load, _series

    raw_dir = Path(raw_dir)
    cells = [r for r in table if r["arm"] == arm]
    if strength is None:
        strength = max({r["alpha"] for r in cells}, key=abs)
    cells = [r for r in cells if r["alpha"] == strength]
    if len(cells) < 3:
        return {"error": "not enough layers", "n_layers": len(cells)}

    prefix = "steered" if arm == "real" else "control"
    per_layer: dict[int, tuple[list[str], np.ndarray]] = {}
    for r in cells:
        L = int(r["layer"])
        run = raw_dir / f"eval_L{L:02d}_{arm}"
        margin = _series(_load(run / f"{prefix}_margin.json"))
        base = _series(_load(run / "baseline_margin.json"))
        alpha = float(r["alpha"])
        zero = 0.0 if 0.0 in base else (min(base) if base else None)
        if alpha not in margin or zero is None or zero not in base:
            continue
        ids = sorted(set(margin[alpha]) & set(base[zero]))
        if len(ids) < 6:
            continue
        per_layer[L] = (
            ids,
            np.array([margin[alpha][i] - base[zero][i] for i in ids], dtype=float),
        )

    if len(per_layer) < 3:
        return {"error": "too few usable layers"}

    common = set.intersection(*[set(v[0]) for v in per_layer.values()])
    layers = sorted(per_layer)
    samples = [
        np.array([per_layer[L][1][per_layer[L][0].index(i)] for i in sorted(common)])
        for L in layers
    ]

    stat, pvalue = stats.friedmanchisquare(*samples)
    out: dict[str, Any] = {
        "arm": arm,
        "strength": strength,
        "n_layers": len(layers),
        "n_paired_prompts": len(common),
        "friedman_statistic": float(stat),
        "friedman_p": float(pvalue),
        "mean_delta_by_layer": {
            str(L): round(float(np.mean(samples[i])), 6) for i, L in enumerate(layers)
        },
    }

    # pairwise Wilcoxon on the same prompts, Holm-corrected across layers
    pairs = []
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            diff = samples[i] - samples[j]
            nz = diff[diff != 0]
            p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else 1.0
            pairs.append(
                {
                    "layer_a": layers[i],
                    "layer_b": layers[j],
                    "mean_diff": round(float(np.mean(diff)), 6),
                    "p_raw": p,
                }
            )
    for pair, adj in zip(pairs, holm_bonferroni([p["p_raw"] for p in pairs])):
        pair["p_holm"] = adj
    out["pairwise"] = pairs
    out["n_pairs_significant_after_holm"] = sum(1 for p in pairs if p["p_holm"] < 0.05)
    return out


def figure_a_depth(
    table: list[dict[str, Any]], out_dir: Path, *, arm: str = "real"
) -> dict[str, Any]:
    """A. Effectiveness (preference change) versus layer depth, per strength."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [r for r in table if r["arm"] == arm]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    strengths = sorted({r["alpha"] for r in cells})
    cmap = plt.get_cmap("coolwarm")
    for i, s in enumerate(strengths):
        pts = sorted(
            [(r["layer"], r["preference_change"]) for r in cells if r["alpha"] == s]
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = cmap(i / max(1, len(strengths) - 1))
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=color, label=f"alpha={s:+.1f}")
        # 95% CI band from the paired bootstrap
        los = [r["pref_change_ci_lo"] for r in sorted(cells, key=lambda r: r["layer"]) if r["alpha"] == s]
        his = [r["pref_change_ci_hi"] for r in sorted(cells, key=lambda r: r["layer"]) if r["alpha"] == s]
        ax.fill_between(xs, los, his, color=color, alpha=0.15, lw=0)

    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.set_xlabel("intervention layer")
    ax.set_ylabel("preference change  (logit margin, paired vs baseline)")
    ax.set_title(f"A. Steering effectiveness vs layer depth ({arm} direction)\n"
                 "shaded = 95% paired-bootstrap CI")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "A_effectiveness_vs_depth.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {"path": str(path), "arm": arm, "strengths": strengths}


def figure_b_sign(table: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """B. Positive versus negative steering, per layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real = [r for r in table if r["arm"] == "real"]
    pos_s = max({r["alpha"] for r in real if r["alpha"] > 0}, default=None)
    neg_s = min({r["alpha"] for r in real if r["alpha"] < 0}, default=None)
    if pos_s is None or neg_s is None:
        return {"error": "need both signs"}

    pos = {r["layer"]: r["preference_change"] for r in real if r["alpha"] == pos_s}
    neg = {r["layer"]: r["preference_change"] for r in real if r["alpha"] == neg_s}
    layers = sorted(set(pos) & set(neg))

    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38
    ax.bar([l - w / 2 for l in layers], [pos[l] for l in layers], w, label=f"+{pos_s:g}", color="#c44")
    ax.bar([l + w / 2 for l in layers], [neg[l] for l in layers], w, label=f"{neg_s:g}", color="#4c9")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xlabel("intervention layer")
    ax.set_ylabel("preference change")
    ax.set_title("B. Positive vs negative steering (matched |strength|)\n"
                 "equal magnitude, opposite sign")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = out_dir / "B_positive_vs_negative.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    asym = {l: abs(pos[l]) - abs(neg[l]) for l in layers}
    return {
        "path": str(path),
        "strength_pos": pos_s,
        "strength_neg": neg_s,
        "n_layers_positive_dominates": sum(1 for v in asym.values() if v > 0),
        "n_layers_negative_dominates": sum(1 for v in asym.values() if v < 0),
        "asymmetry_by_layer": {str(k): round(v, 6) for k, v in asym.items()},
    }


def figure_c_response(table: list[dict[str, Any]], out_dir: Path, *, layers: tuple[int, ...] = (4, 12, 16, 20, 24, 27)) -> dict[str, Any]:
    """C. Strength response curves for selected layers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real = [r for r in table if r["arm"] == "real"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    drawn = []
    for L in layers:
        pts = sorted(
            [(r["alpha"], r["preference_change"]) for r in real if r["layer"] == L]
        )
        if len(pts) < 3:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, label=f"L{L}")
        drawn.append(L)
    ax.axhline(0.0, color="black", lw=0.9, ls="--")
    ax.axvline(0.0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("normalized steering strength (alpha)")
    ax.set_ylabel("preference change")
    ax.set_title("C. Steering-strength response curves\n(monotone in strength => dose-response)")
    ax.legend(title="layer", fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "C_strength_response.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"path": str(path), "layers_drawn": drawn}


def figure_d_baseline(table: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """D. Baseline versus intervention performance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real = [r for r in table if r["arm"] == "real"]

    fig, ax = plt.subplots(figsize=(9, 5))
    layers = sorted({r["layer"] for r in real})
    base = [real[0]["baseline_accuracy"] if real else 0]
    zero = [
        next((r["original_answer_rate"] for r in real if r["layer"] == L and r["alpha"] == 0.0), None)
        for L in layers
    ]
    strong = [
        next(
            (
                r["original_answer_rate"]
                for r in real
                if r["layer"] == L and r["alpha"] == max(x["alpha"] for x in real if x["layer"] == L)
            ),
            None,
        )
        for L in layers
    ]
    ax.plot(layers, [base[0]] * len(layers), "k--", lw=1.4, label="baseline (unintervened)")
    ax.plot(layers, zero, "o-", ms=4, color="#888", label="zero-strength (alpha=0)")
    ax.plot(layers, strong, "s-", ms=4, color="#c44", label="strongest positive steering")
    ax.set_xlabel("intervention layer")
    ax.set_ylabel("original-answer accuracy")
    ax.set_title("D. Baseline vs intervention performance\n(zero-strength must coincide with baseline)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "D_baseline_vs_intervention.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"path": str(path)}


def figure_e_ci(table: list[dict[str, Any]], out_dir: Path, *, arm: str = "real") -> dict[str, Any]:
    """E. Effect sizes with 95% CIs, strongest dose, per layer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [r for r in table if r["arm"] == arm]
    if not cells:
        return {"error": "no rows"}
    strongest = max({r["alpha"] for r in cells}, key=abs)
    cells = sorted([r for r in cells if r["alpha"] == strongest], key=lambda r: r["layer"])

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [r["layer"] for r in cells]
    ys = [r["preference_change"] for r in cells]
    los = [r["pref_change_ci_lo"] for r in cells]
    his = [r["pref_change_ci_hi"] for r in cells]
    err_lo = [y - lo for y, lo in zip(ys, los, strict=True)]
    err_hi = [hi - y for y, hi in zip(ys, his, strict=True)]
    ax.errorbar(xs, ys, yerr=[err_lo, err_hi], fmt="o-", ms=4, capsize=2.5, lw=1.4)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xlabel("intervention layer")
    ax.set_ylabel("preference change (95% CI)")
    ax.set_title(f"E. Effect size with uncertainty, {arm} direction at alpha={strongest:+g}\n"
                 "intervals exclude 0 => effect distinguishable from no-op")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "E_effects_with_ci.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    n_excl = sum(1 for r in cells if r["pref_change_ci_lo"] > 0 or r["pref_change_ci_hi"] < 0)
    return {
        "path": str(path),
        "strength": strongest,
        "n_layers": len(cells),
        "n_layers_ci_excludes_zero": n_excl,
        "fraction_significant": round(n_excl / max(1, len(cells)), 4),
    }


def write_layer_selection(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")