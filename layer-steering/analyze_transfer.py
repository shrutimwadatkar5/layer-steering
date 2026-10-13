"""Analysis for the cross-layer transfer grid: heatmap, classification, uncertainty.

Produces, for the ``cross_layer_transfer`` experiment only:

* a source-by-target heatmap of the behavioural-effect metric;
* the same-layer vs cross-layer comparison;
* directional asymmetry along the grid;
* paired-bootstrap CIs per cell;
* baseline and zero-strength controls;
* an explicit per-cell **classification** into the four outcomes the brief asks
  to be kept apart.

The classification is deliberately conservative, and it never says a direction is
"absent" from a layer. See :func:`classify_cell`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = [
    "ALPHA_KEY",
    "OUTCOMES",
    "collect_cells",
    "classify_cell",
    "heatmap_matrix",
    "summary",
    "figure_heatmap",
]

ALPHA_KEY = "writes.push.do.add_scaled.alpha"

#: The four outcomes the brief requires be kept distinct.
OUTCOMES = {
    "successful_transfer": (
        "CI excludes zero AND the effect is at least half the same-layer "
        "reference: the source direction moved behaviour at the target layer"
    ),
    "failed_transfer_possible_misalignment": (
        "CI includes zero while the cell's dose was calibrated and verified: "
        "consistent with the direction not being represented at the target in a "
        "usable basis, but NOT evidence the feature is absent there"
    ),
    "intervention_induced_degradation": (
        "original-answer accuracy fell below baseline while the target-answer "
        "rate did not rise: the intervention disrupted the computation without "
        "redirecting it"
    ),
    "inconclusive": "anything else, including a near-null with a wide interval",
}

#: A cross-layer cell is called a successful transfer only if it keeps at least
#: this fraction of the same-layer reference effect. Below it, the cell is
#: reported as partial rather than successful.
_SUCCESS_FRACTION = 0.5


def _series(rows: list[dict[str, Any]]) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for row in rows:
        if ALPHA_KEY not in row:
            continue
        out.setdefault(float(row[ALPHA_KEY]), {})[str(row["example_id"])] = float(
            row["value"]
        )
    return out


def _load(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else [data]


def _z_of(alpha: float, sigma: float | None) -> float | None:
    """Recover z from a raw alpha, given the cell's own sigma_ij.

    Each cell is calibrated independently, so raw alphas are **not** comparable
    across the grid; z is. Selection therefore happens in z-space, and the alpha
    is only reported for provenance.
    """
    if sigma in (None, 0):
        return None
    return float(alpha) / float(sigma)


def collect_cells(
    transfer_dir: Path, *, strength: float | None = None, z: float | None = None
) -> list[dict[str, Any]]:
    """One record per (source, target) cell at a chosen **z** (or raw alpha).

    ``z`` is the correct axis: the dose was calibrated per cell, so equal z means
    equal relative dose while equal alpha does not.
    """
    from research.results_table import bootstrap_ci_paired

    cells: list[dict[str, Any]] = []
    for run in sorted(Path(transfer_dir).glob("s*_t*")):
        meta_path = run / "transfer_config.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        source = int(meta["source_layer"])
        target = int(meta["target_layer"])
        sigma = meta.get("sigma_table", {}).get("sigma")

        margin = _series(_load(run / "transfer_margin.json"))
        base = _series(_load(run / "baseline_margin.json"))
        orig = _series(_load(run / "transfer_original_rate.json"))
        cf = _series(_load(run / "transfer_cf_rate.json"))
        base_acc = _series(_load(run / "baseline_accuracy.json"))
        if not margin or not base:
            continue

        zero = 0.0 if 0.0 in base else min(base)

        # choose this cell's alpha for the requested axis
        chosen: float | None = None
        if strength is not None:
            chosen = strength
        else:
            candidates = []
            for alpha in margin:
                zz = _z_of(alpha, sigma)
                if zz is not None and abs(alpha) > 0:
                    candidates.append((abs(zz), alpha, zz))
            if not candidates:
                continue
            if z is not None:
                # `sigma` is persisted rounded to 8 decimals, so a recovered z
                # can differ from the requested one in the last bits. Match with
                # a tolerance rather than exact equality, or every cell is
                # silently dropped.
                exact = [c for c in candidates if abs(c[2] - float(z)) < 1e-3]
                if not exact:
                    continue
                chosen = min(exact, key=lambda c: abs(c[2] - float(z)))[1]
            else:
                chosen = max(candidates)[1]
        if chosen is None or chosen not in margin:
            continue

        ids = sorted(
            set(margin[chosen])
            & set(base[zero])
            & set(orig.get(chosen, {}))
            & set(cf.get(chosen, {}))
            & set(base_acc.get(zero, {}))
        )
        if len(ids) < 6:
            continue

        mean, lo, hi = bootstrap_ci_paired(
            [margin[chosen][i] for i in ids], [base[zero][i] for i in ids]
        )
        baseline_acc = sum(base_acc[zero][i] for i in ids) / len(ids)
        orig_rate = sum(orig[chosen][i] for i in ids) / len(ids)
        cf_rate = sum(cf[chosen][i] for i in ids) / len(ids)

        zero_delta = None
        if 0.0 in margin:
            z_ids = sorted(set(margin[0.0]) & set(base[zero]))
            if z_ids:
                zero_delta = sum(
                    margin[0.0][i] - base[zero][i] for i in z_ids
                ) / len(z_ids)

        cells.append(
            {
                "source_layer": source,
                "target_layer": target,
                "same_layer": source == target,
                "alpha": round(float(chosen), 6),
                "z": (
                    round(_z_of(chosen, sigma), 4) if sigma else None
                ),
                "n_prompts": len(ids),
                "preference_change": round(mean, 6),
                "ci_lo": round(lo, 6),
                "ci_hi": round(hi, 6),
                "significant": bool(lo > 0 or hi < 0),
                "baseline_accuracy": round(baseline_acc, 6),
                "original_answer_rate": round(orig_rate, 6),
                "target_answer_rate": round(cf_rate, 6),
                "accuracy_change": round(orig_rate - baseline_acc, 6),
                "zero_strength_delta": (
                    round(zero_delta, 8) if zero_delta is not None else None
                ),
                "sigma_ij": sigma,
            }
        )
    cells.sort(key=lambda c: (c["source_layer"], c["target_layer"]))
    return cells


def classify_cell(cell: dict[str, Any], reference: dict[str, Any] | None) -> str:
    """Assign one of :data:`OUTCOMES` to a cell.

    Order matters: degradation is checked first, because a cell can be both
    "significant" and disruptive, and reporting it as a clean transfer would
    overstate the result.
    """
    acc_change = cell["accuracy_change"]
    cf_rate = cell["target_answer_rate"]

    if acc_change < -1e-9 and cf_rate <= 1e-9:
        return "intervention_induced_degradation"

    if not cell["significant"]:
        # a well-calibrated, verified no-op with no detectable effect
        return "failed_transfer_possible_misalignment"

    if reference is not None and reference["preference_change"] != 0:
        ratio = abs(cell["preference_change"]) / abs(reference["preference_change"])
        if ratio >= _SUCCESS_FRACTION:
            return "successful_transfer"

    return "inconclusive"


def heatmap_matrix(
    cells: list[dict[str, Any]], metric: str = "preference_change"
) -> dict[str, dict[str, float | None]]:
    """``{source_layer: {target_layer: value}}`` for the requested metric."""
    out: dict[str, dict[str, float | None]] = defaultdict(dict)
    for cell in cells:
        out[str(cell["source_layer"])][str(cell["target_layer"])] = cell[metric]
    return dict(out)


def summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Same-layer vs cross-layer, plus the per-cell classification."""
    diagonal = {c["source_layer"]: c for c in cells if c["same_layer"]}
    for cell in cells:
        cell["outcome"] = classify_cell(cell, diagonal.get(cell["source_layer"]))

    same = [c for c in cells if c["same_layer"]]
    cross = [c for c in cells if not c["same_layer"]]

    def _agg(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "mean_abs_effect": round(
                sum(abs(c["preference_change"]) for c in group) / len(group), 6
            ),
            "n_significant": sum(1 for c in group if c["significant"]),
            "outcomes": {
                k: sum(1 for c in group if c["outcome"] == k) for k in OUTCOMES
            },
        }

    # asymmetry along the grid: source i at target j vs the transpose
    by_key = {(c["source_layer"], c["target_layer"]): c for c in cells}
    asym = []
    for (i, j), cell in by_key.items():
        other = by_key.get((j, i))
        if other is None or i == j:
            continue
        asym.append(
            {
                "pair": f"{i}<->{j}",
                "forward": cell["preference_change"],
                "reverse": other["preference_change"],
                "forward_significant": cell["significant"],
                "reverse_significant": other["significant"],
            }
        )

    zero_devs = [
        abs(c["zero_strength_delta"])
        for c in cells
        if c["zero_strength_delta"] is not None
    ]

    return {
        "experiment": "cross_layer_transfer",
        "not_pooled_with_layerwise": True,
        "n_cells": len(cells),
        "same_layer": _agg(same),
        "cross_layer": _agg(cross),
        "baseline_accuracy": sorted({c["baseline_accuracy"] for c in cells}),
        "zero_strength_max_abs_delta": (
            round(max(zero_devs), 8) if zero_devs else None
        ),
        "outcomes": {k: sum(1 for c in cells if c["outcome"] == k) for k in OUTCOMES},
        "outcome_definitions": OUTCOMES,
        "directional_asymmetry": asym,
        "interpretation_caveats": [
            "A failed transfer does NOT show the feature is absent from the "
            "target layer; it is consistent with several alternatives "
            "(representation in a rotated basis, dose still too small, effect "
            "below the resolution of a 2-way argmax task).",
            "A successful transfer shows the direction has behavioural effect "
            "at the target layer. It does NOT show the target layer encodes a "
            "uniquely interpretable feature.",
        ],
    }


def figure_heatmap(
    cells: list[dict[str, Any]],
    out_dir: Path,
    *,
    metric: str = "preference_change",
    title: str | None = None,
) -> str:
    """Source-by-target heatmap with significance marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sources = sorted({c["source_layer"] for c in cells})
    targets = sorted({c["target_layer"] for c in cells})
    grid = np.full((len(sources), len(targets)), np.nan)
    sig = np.zeros_like(grid, dtype=bool)
    lookup = {(c["source_layer"], c["target_layer"]): c for c in cells}
    for a, i in enumerate(sources):
        for b, j in enumerate(targets):
            cell = lookup.get((i, j))
            if cell:
                grid[a, b] = cell[metric]
                sig[a, b] = cell["significant"]

    fig, ax = plt.subplots(figsize=(1.5 * len(targets) + 3, 1.4 * len(sources) + 3))
    limit = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")

    for a, i in enumerate(sources):
        for b, j in enumerate(targets):
            cell = lookup.get((i, j))
            if cell is None:
                continue
            marker = "*" if cell["same_layer"] else ""
            ax.text(
                b,
                a,
                f"{grid[a, b]:+.3f}{marker}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    ax.set_xticks(range(len(targets)), [f"t{j}" for j in targets])
    ax.set_yticks(range(len(sources)), [f"source {i}" for i in sources])
    ax.set_xlabel("target layer")
    ax.set_ylabel("source layer")
    ax.set_title(
        title
        or "Cross-layer transfer: source x target\n"
        "* = same-layer reference;  bold outline would mark CI excluding 0"
    )
    for a, b in zip(*np.where(sig)):
        ax.add_patch(
            plt.Rectangle(
                (b - 0.5, a - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.0
            )
        )
    fig.colorbar(im, ax=ax, label=metric)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transfer_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)