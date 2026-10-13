"""Sign-asymmetry test: is positive steering distinguishable from negative?

At matched magnitude the two arms are compared per prompt. A Wilcoxon signed-rank
on the per-prompt difference, then a binomial sign test on how many layers
favour each arm, with a Holm correction across the per-layer comparisons.

An important caveat recorded in the output: the direction ``v_l`` is defined as
``mean(A) - mean(B)``, so ``+z`` pushes toward group A and ``-z`` toward group B.
Any asymmetry therefore reflects how differently the network propagates the two
halves of the contrast, not a symmetry the experiment assumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["sign_asymmetry_test"]


def sign_asymmetry_test(raw_dir: Path, *, strength: float = 8.0) -> dict[str, Any]:
    """Compare +z and -z at matched magnitude, per layer, paired by prompt."""
    import numpy as np
    from scipy import stats

    from research.results_table import _load, _series

    raw_dir = Path(raw_dir)
    per_layer: dict[int, dict[str, float]] = {}

    for run in sorted(raw_dir.glob("eval_L*_real")):
        try:
            layer = int(run.name.split("_L")[-1].split("_")[0])
        except ValueError:
            continue
        margin = _series(_load(run / "steered_margin.json"))
        if not margin:
            continue
        base = _series(_load(run / "baseline_margin.json"))
        # pin to the genuine zero; the baseline is duplicated at every alpha
        zero = 0.0 if 0.0 in base else (min(base) if base else None)
        ids = None
        deltas: dict[str, float] = {}
        for alpha in (strength, -strength):
            if alpha not in margin or zero is None or zero not in base:
                continue
            cell = sorted(set(margin[alpha]) & set(base[zero]))
            ids = cell if ids is None else sorted(set(ids) & set(cell))
            deltas["pos" if alpha > 0 else "neg"] = alpha
        if ids is None or len(ids) < 6 or len(deltas) < 2:
            continue
        if zero not in base:
            continue
        base_vec = {i: base[zero][i] for i in ids}
        pos = np.array([margin[deltas["pos"]][i] - base_vec[i] for i in ids])
        neg = np.array([margin[deltas["neg"]][i] - base_vec[i] for i in ids])
        diff = pos - neg
        nz = diff[diff != 0]
        p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else 1.0
        per_layer[layer] = {
            "mean_pos": float(pos.mean()),
            "mean_neg": float(neg.mean()),
            "abs_pos": abs(float(pos.mean())),
            "abs_neg": abs(float(neg.mean())),
            "p_value": p,
            "n_paired": len(ids),
        }

    if not per_layer:
        return {"error": "no layers with both arms"}

    layers = sorted(per_layer)
    raw_p = [per_layer[L]["p_value"] for L in layers]
    from research.figures import holm_bonferroni

    adjusted = holm_bonferroni(raw_p)
    for L, adj in zip(layers, adjusted, strict=True):
        per_layer[L]["p_holm"] = adj

    n_pos = sum(1 for L in layers if per_layer[L]["abs_pos"] > per_layer[L]["abs_neg"])
    n_neg = sum(1 for L in layers if per_layer[L]["abs_neg"] > per_layer[L]["abs_pos"])
    n_sig = sum(1 for L in layers if per_layer[L]["p_holm"] < 0.05)
    # sign test on which arm dominates, across layers
    k, m = max(n_pos, n_neg), n_pos + n_neg
    sign_p = float(stats.binomtest(k, m, 0.5).pvalue) if m else float("nan")

    return {
        "strength": strength,
        "n_layers": len(layers),
        "n_layers_positive_dominates": n_pos,
        "n_layers_negative_dominates": n_neg,
        "n_layers_significant_after_holm": n_sig,
        "sign_test_p": sign_p,
        "interpretation": (
            "positive and negative steering are distinguishable as a pair"
            if sign_p < 0.05
            else "no overall asymmetry across layers"
        ),
        "caveat": (
            "v_l = mean(A) - mean(B), so +z pushes toward A and -z toward B; "
            "asymmetry reflects asymmetric propagation of the two halves, not "
            "an assumed symmetry"
        ),
        "by_layer": {str(k2): {kk: round(vv, 8) if isinstance(vv, float) else vv
                               for kk, vv in v.items()} for k2, v in per_layer.items()},
    }