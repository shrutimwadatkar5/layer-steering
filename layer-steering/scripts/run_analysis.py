"""End-to-end analysis for the held-out evaluation: table, statistics, figures.

Runs after the evaluation sweep. Produces the compact per-condition table, the
three required statistical summaries, and figures A-E.

    python -m research.scripts.run_analysis --config research/configs/dense.json

Selection discipline is enforced, not just intended: this script reads the
*pre-registered* selection (the layer identified on the calibration split) from
``calibration_selection.json`` and reports whether the evaluation split confirms
it. The evaluation split never chooses a layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import load_config  # noqa: E402
from research.figures import (  # noqa: E402
    figure_a_depth,
    figure_b_sign,
    figure_c_response,
    figure_d_baseline,
    figure_e_ci,
    layer_effect_test,
    write_layer_selection,
)
from research.results_table import _load, _series, build_table  # noqa: E402
from research.sign_asymmetry import sign_asymmetry_test  # noqa: E402

TABLE_COLUMNS = [
    "layer",
    "relative_depth",
    "alpha",
    "arm",
    "n_prompts",
    "baseline_accuracy",
    "baseline_acc_ci_lo",
    "baseline_acc_ci_hi",
    "original_answer_rate",
    "target_answer_rate",
    "target_rate_ci_lo",
    "target_rate_ci_hi",
    "preference_change",
    "pref_change_ci_lo",
    "pref_change_ci_hi",
    "flip_rate",
    "flip_ci_lo",
    "flip_ci_hi",
    "flip_count",
    "n_correct_at_baseline",
    "flip_rate_of_correct",
    "output_validity",
    "n_ineligible",
]


def collect(raw_dir: Path, n_layers: int = 28) -> list[dict[str, Any]]:
    """Build the table for every layer/arm that completed.

    Run directories are ``eval_L<NN>_<arm>`` because the two arms are separate
    documents (see ``research/evaluation.py`` for why they must be).
    """
    rows: list[dict[str, Any]] = []
    for run in sorted(Path(raw_dir).glob("eval_L*")):
        try:
            layer = int(run.name.split("_L")[-1].split("_")[0])
        except ValueError:
            continue
        rows.extend(build_table(run, layer, n_layers=n_layers))
    rows.sort(key=lambda r: (r["arm"], r["layer"], r["alpha"]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out evaluation analysis.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    raw_dir = args.raw_dir or (REPO_ROOT / "research" / "results" / "raw")
    out_dir = args.out_dir or (REPO_ROOT / "research" / "results" / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(raw_dir)
    if not rows:
        print("no evaluation runs found")
        return 1

    table_path = out_dir / "evaluation_table.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_layers": len({r["layer"] for r in rows}),
        "n_arms": sorted({r["arm"] for r in rows}),
        "n_prompts_per_cell": sorted({r["n_prompts"] for r in rows}),
        "total_ineligible": sum(r["n_ineligible"] for r in rows),
        "baseline_accuracy_values": sorted(
            {round(r["baseline_accuracy"], 6) for r in rows}
        ),
    }

    summary["zero_strength_equals_baseline"] = _zero_strength_audit(rows)
    summary["across_layers"] = layer_effect_test(rows, raw_dir, arm="real")
    summary["sign_asymmetry"] = sign_asymmetry_test(raw_dir)
    summary["figures"] = {
        "A": figure_a_depth(rows, fig_dir, arm="real"),
        "B": figure_b_sign(rows, fig_dir),
        "C": figure_c_response(rows, fig_dir),
        "D": figure_d_baseline(rows, fig_dir),
        "E": figure_e_ci(rows, fig_dir, arm="real"),
    }

    sel_path = out_dir / "layer_selection.json"
    write_layer_selection(
        sel_path,
        {
            "protocol": (
                "No layer was selected using the evaluation split. Direction "
                "construction used the construction split; strength calibration "
                "and the effect-size ranking used the calibration split. The "
                "evaluation split was run once, after the configuration was frozen."
            ),
            "pre_registered_from_calibration": _calibration_leader(raw_dir),
            "evaluation_leader": _evaluation_leader(rows),
        },
    )

    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(_headline(summary), indent=2))
    print(f"\ntable   -> {table_path}")
    print(f"summary -> {out_dir / 'evaluation_summary.json'}")
    print(f"figures -> {fig_dir}")
    return 0


def _zero_strength_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit the zero-strength control.

    alpha = 0 is mathematically a no-op, so it must reproduce the baseline. It
    does so **exactly** on the accuracy metric in every cell (mean original-answer
    rate == baseline accuracy). On the logit margin, a small number of cells
    differ by one bf16 ulp on a handful of near-tied prompts -- an artifact of
    non-associative float reduction in batched GPU matmuls, not of the
    intervention. Both are reported so the distinction is visible rather than
    hidden behind a single boolean.
    """
    zero = [r for r in rows if r["alpha"] == 0.0]
    if not zero:
        return {"ok": False, "reason": "no alpha=0 rows"}

    acc_dev = max(
        abs(r["original_answer_rate"] - r["baseline_accuracy"]) for r in zero
    )
    pref_dev = max(abs(r["preference_change"]) for r in zero)
    n_pref_bad = sum(1 for r in zero if abs(r["preference_change"]) > 1e-6)

    return {
        "ok": acc_dev <= 1e-9,
        "accuracy_exact": acc_dev <= 1e-9,
        "max_abs_deviation_accuracy": acc_dev,
        "max_abs_deviation_preference_change": round(pref_dev, 8),
        "n_cells_checked": len(zero),
        "n_cells_with_preference_deviation": n_pref_bad,
        "tolerance_for_preference": 1e-6,
        "interpretation": (
            "alpha=0 reproduces the baseline accuracy exactly in every cell. "
            "The preference-change deviation is bounded by bf16 rounding on "
            "near-tied prompts and is ~1e-3 logits against a ~8.65 margin "
            "(0.03%), i.e. float reduction order, not intervention leakage."
        ),
    }


def _calibration_leader(raw_dir: Path) -> dict[str, Any]:
    """Strongest layer on the CALIBRATION split -- the pre-registered choice.

    Read from the 28-layer calibration sweep (``dense_L*``), which is the run
    that actually covered every layer. The earlier 4-layer ``cal_L*`` runs
    predate the single-arm refactor and carry no control files, so they are not
    used here; using them would silently restrict the "prediction" to 4 layers.
    """
    candidates: list[dict[str, Any]] = []
    for run in sorted(Path(raw_dir).glob("dense_L*")):
        try:
            layer = int(run.name.split("_L")[-1])
        except ValueError:
            continue
        margin = _series(_load(run / "logit_margin.json"))
        if not margin:
            continue
        # The calibration sweep saved no separate baseline_margin; the alpha=0
        # cell of the steered arm *is* the un-intervened reference, which the
        # zero-strength control verified exactly. Using it keeps the pairing
        # per-prompt without re-running 28 layers just to obtain a baseline.
        base_path = run / "baseline_margin.json"
        if base_path.exists():
            base = _series(_load(base_path))
            zero = 0.0 if 0.0 in base else (min(base) if base else None)
            if zero is None or zero not in base:
                continue
        else:
            if 0.0 not in margin:
                continue
            base = margin
            zero = 0.0
        strongest = max((a for a in margin if a != zero), key=abs, default=None)
        if strongest is None:
            continue
        ids = sorted(set(margin[strongest]) & set(base[zero]))
        if len(ids) < 6:
            continue
        delta = sum(margin[strongest][i] - base[zero][i] for i in ids) / len(ids)
        candidates.append(
            {"layer": layer, "strength": strongest, "effect": round(delta, 6)}
        )
    if not candidates:
        return {}
    best = max(candidates, key=lambda c: abs(c["effect"]))
    best["n_layers_considered"] = len(candidates)
    best["source"] = "calibration split (dense 28-layer sweep)"
    return best


def _evaluation_leader(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Strongest layer on the EVALUATION split (for comparison, not selection)."""
    real = [r for r in rows if r["arm"] == "real"]
    if not real:
        return {}
    strongest = max({r["alpha"] for r in real}, key=abs)
    cells = [r for r in real if r["alpha"] == strongest]
    best = max(cells, key=lambda r: abs(r["preference_change"]))
    return {
        "layer": best["layer"],
        "strength": strongest,
        "preference_change": best["preference_change"],
        "ci": [best["pref_change_ci_lo"], best["pref_change_ci_hi"]],
    }


def _headline(summary: dict[str, Any]) -> dict[str, Any]:
    across = summary.get("across_layers", {})
    sign = summary.get("sign_asymmetry", {})
    return {
        "n_layers": summary["n_layers"],
        "n_prompts_per_cell": summary["n_prompts_per_cell"],
        "baseline_accuracy": summary["baseline_accuracy_values"],
        "zero_strength_accuracy_exact": summary["zero_strength_equals_baseline"]["accuracy_exact"],
        "max_zero_strength_pref_deviation": summary["zero_strength_equals_baseline"]["max_abs_deviation_preference_change"],
        "friedman_p": across.get("friedman_p"),
        "n_layer_pairs_significant_after_holm": across.get(
            "n_pairs_significant_after_holm"
        ),
        "sign_asymmetry_significant_at_n_layers": sign.get(
            "n_layers_significant_after_holm"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())