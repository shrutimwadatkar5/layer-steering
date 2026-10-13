"""Paired statistics across the calibrated steering sweep.

Every comparison is **paired by prompt**: the same calibration prompt is scored
under every strength, so the unit of analysis is the prompt and differences are
computed within-prompt.

Two questions, kept separate on purpose:

* **Within a layer** -- does increasing z move the logit margin away from the
  baseline? Wilcoxon signed-rank on the paired per-prompt differences against
  the ``z = 0`` cell, two-sided. Non-parametric because margins are bounded and
  the distributions are skewed at high z.
* **Across layers** -- at matched z, do layers differ? Friedman test over the
  four layers on the per-prompt margin, followed by pairwise Wilcoxon with
  **Holm-Bonferroni** across the six comparisons.

Choice of the reported layer is made on the *calibration* split. The evaluation
split is untouched, so nothing here is a held-out claim.

Usage::

    python -m research.scripts.analyze --tables-dir research/results/tables --out research/results/tables/statistics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import load_config  # noqa: E402

__all__ = ["paired_within_layer", "friedman_across_layers", "analyze"]

_ALPHA_KEY = "writes.push.do.add_scaled.alpha"


def _layers_from_config(config) -> tuple[int, ...]:
    """Layers come from the configuration, not a module constant.

    This was the reproducibility bug: the committed script swept a hardcoded
    4 layers while the headline result covered all 28, so re-running the repo
    could not regenerate the reported statistics.
    """
    return tuple(int(x) for x in config.layers)


def _run_prefix(config) -> str:
    return "dense" if len(config.layers) > 8 else "cal"


def _z_index(
    data: dict[int, dict[float, dict[str, float]]], layers: tuple[int, ...]
) -> dict[int, dict[float, float]]:
    """Map each layer's alpha grid onto the shared z grid.

    Reads ``dose_calibration.json`` so the mapping is the *measured* one rather
    than recomputed. Returns ``{layer: {z: alpha}}``; a layer missing from the
    table is skipped by the caller through the set intersection.
    """
    table = json.loads(
        (Path(data["_tables_dir"]) / "dose_calibration.json").read_text()
    )
    mapping: dict[int, dict[float, float]] = {}
    for entry in table["layers"]:
        layer = int(entry["layer"])
        if layer not in layers:
            continue
        mapping[layer] = {
            float(z): float(a) for z, a in zip(entry["z_grid"], entry["alpha_grid"])
        }
    return mapping


def _margin_by_alpha(run_dir: Path) -> dict[float, dict[str, float]]:
    """``{alpha: {example_id: margin}}`` from one layer's raw margin table."""
    path = Path(run_dir) / "logit_margin.json"
    out: dict[float, dict[str, float]] = {}
    for row in json.loads(path.read_text()):
        alpha = float(row[_ALPHA_KEY])
        out.setdefault(alpha, {})[str(row["example_id"])] = float(row["value"])
    return out


def paired_within_layer(run_dir: Path) -> dict[str, Any]:
    """Wilcoxon signed-rank of each strength against z=0, paired by prompt."""
    import numpy as np
    from scipy import stats

    by_alpha = _margin_by_alpha(run_dir)
    alphas = sorted(by_alpha)
    zero = by_alpha[alphas[0]]
    base_mean = float(np.mean(list(zero.values())))

    rows = []
    for alpha in alphas[1:]:
        current = by_alpha[alpha]
        ids = sorted(set(zero) & set(current))
        deltas = [zero[i] - current[i] for i in ids]
        nonzero = [d for d in deltas if d != 0.0]
        if len(nonzero) < 5:
            pvalue = 1.0
            stat = float("nan")
        else:
            result = stats.wilcoxon(nonzero)
            pvalue = float(result.pvalue)
            stat = float(result.statistic)
        rows.append(
            {
                "alpha": alpha,
                "n_paired": len(ids),
                "n_nonzero": len(nonzero),
                "margin_mean": float(np.mean([current[i] for i in ids])),
                "delta_mean": float(np.mean(deltas)),
                "wilcoxon_stat": stat,
                "p_value": pvalue,
            }
        )
    return {"baseline_margin_mean": base_mean, "comparisons": rows}


def friedman_across_layers(
    tables_dir: Path, *, layers: tuple[int, ...], prefix: str = "dense"
) -> dict[str, Any]:
    """Friedman across layers at each matched z, then pairwise Wilcoxon+Holm."""
    import numpy as np
    from scipy import stats

    tables_dir = Path(tables_dir)
    data: dict[int, Any] = {L: _margin_by_alpha(tables_dir.parent / "raw" / f"{prefix}_L{L:02d}")
            for L in layers}
    data["_tables_dir"] = str(tables_dir)

    # The per-layer alpha grids are DIFFERENT (that is the whole point of
    # calibration: alpha = z * sigma_within, and sigma differs by layer), so
    # comparing at a shared alpha would compare unequal doses -- exactly the
    # mistake the calibration was introduced to fix. Layers are therefore
    # matched on the shared **z** grid instead.
    z_by_layer = _z_index(data, layers)  # reads data["_tables_dir"], then drops it
    data.pop("_tables_dir", None)
    out: dict[str, Any] = {"friedman": [], "pairwise": []}

    for z in sorted(set.intersection(*[set(v) for v in z_by_layer.values()])):
        # align prompts across layers at this z
        common = set.intersection(
            *[set(data[L][z_by_layer[L][z]]) for L in layers]
        )
        if len(common) < 6:
            continue
        samples = []
        for L in layers:
            table = data[L][z_by_layer[L][z]]
            samples.append(
                np.array([table[i] for i in sorted(common)], dtype=float)
            )
        stat, pvalue = stats.friedmanchisquare(*samples)
        out["friedman"].append(
            {
                "z": z,
                "alpha_by_layer": {
                    str(L): z_by_layer[L][z] for L in layers
                },
                "n_paired": len(common),
                "statistic": float(stat),
                "p_value": float(pvalue),
                "margin_mean_by_layer": {
                    str(L): float(np.mean(samples[i])) for i, L in enumerate(layers)
                },
            }
        )

        # pairwise at this z, Holm-corrected
        pairs = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                a = samples[i]
                b = samples[j]
                diffs = (a - b)[a != b]
                if len(diffs) < 5:
                    p = 1.0
                else:
                    p = float(stats.wilcoxon(diffs).pvalue)
                pairs.append(
                    {
                        "layer_a": layers[i],
                        "layer_b": layers[j],
                        "mean_diff": float(np.mean(a) - np.mean(b)),
                        "p_raw": p,
                    }
                )
        order = sorted(range(len(pairs)), key=lambda k: pairs[k]["p_raw"])
        running = 0.0
        m = len(pairs)
        for rank, idx in enumerate(order):
            adjusted = min(1.0, pairs[idx]["p_raw"] * (m - rank))
            running = max(running, adjusted)
            pairs[idx]["p_holm"] = running
        out["pairwise"].extend([{**p, "z": z} for p in pairs])

    return out


def analyze(tables_dir: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    layers = _layers_from_config(config)
    prefix = _run_prefix(config)
    within = {
        str(L): paired_within_layer(
            Path(tables_dir).parent / "raw" / f"{prefix}_L{L:02d}"
        )
        for L in layers
    }
    across = friedman_across_layers(Path(tables_dir), layers=layers, prefix=prefix)
    return {
        "config": config_path.name,
        "split": "calibration",
        "n_prompts": None,
        "test_choice": (
            "Wilcoxon signed-rank on prompt-paired differences (within layer); "
            "Friedman across layers with pairwise Wilcoxon + Holm correction "
            "(across layers). Non-parametric because margins are bounded and "
            "skewed at high z."
        ),
        "multiple_comparisons": "Holm-Bonferroni within each alpha",
        "within_layer": within,
        "across_layers": across,
        "caveat": (
            "selection is on the calibration split; the evaluation split is "
            "untouched, so no held-out claim is made here"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired steering statistics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--tables-dir", type=Path, default=REPO_ROOT / "research" / "results" / "tables"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = analyze(args.tables_dir, args.config)
    out = args.out or Path(args.tables_dir) / "statistics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    print("WITHIN LAYER (vs z=0, paired by prompt)")
    for layer, block in result["within_layer"].items():
        for c in block["comparisons"]:
            print(
                f"  L{layer:>2} alpha={c['alpha']:>6.2f} "
                f"delta={c['delta_mean']:>6.3f} p={c['p_value']:.3g}"
            )
    print("\nACROSS LAYERS (Friedman)")
    for f in result["across_layers"]["friedman"]:
        print(f"  alpha={f['alpha']:>6.2f} chi2={f['statistic']:.2f} p={f['p_value']:.3g}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())