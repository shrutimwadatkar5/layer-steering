"""Driver for the cross-layer transfer analysis.

    python -m research.scripts.run_transfer_analysis \\
        --config research/configs/dense.json \\
        --transfer-dir research/results/transfer

Writes, all under ``research/results/transfer_analysis/`` and all tagged
``experiment = "cross_layer_transfer"``:

* ``transfer_table.csv``      one row per (source, target) cell
* ``transfer_summary.json``   same-layer vs cross-layer, outcomes, asymmetry
* ``figures/transfer_heatmap.png``
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

from research.analyze_transfer import (  # noqa: E402
    OUTCOMES,
    collect_cells,
    figure_heatmap,
    heatmap_matrix,
    summary,
)
from research.config import load_config  # noqa: E402

COLUMNS = [
    "source_layer",
    "target_layer",
    "same_layer",
    "alpha",
    "z",
    "n_prompts",
    "baseline_accuracy",
    "original_answer_rate",
    "target_answer_rate",
    "accuracy_change",
    "preference_change",
    "ci_lo",
    "ci_hi",
    "significant",
    "zero_strength_delta",
    "sigma_ij",
    "outcome",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-layer transfer analysis.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--transfer-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="compare cells at this z (the calibrated axis). Default: each "
        "cell's largest-magnitude z.",
    )
    args = parser.parse_args(argv)

    load_config(args.config)
    transfer_dir = args.transfer_dir or (REPO_ROOT / "research" / "results" / "transfer")
    out_dir = args.out_dir or (
        REPO_ROOT / "research" / "results" / "transfer_analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = collect_cells(Path(transfer_dir), z=args.z)
    if not cells:
        print("no transfer runs found")
        return 1

    report = summary(cells)

    table_path = out_dir / "transfer_table.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cells)

    report["z"] = cells[0]["z"]
    report["heatmap_preference_change"] = heatmap_matrix(cells, "preference_change")
    report["heatmap_accuracy_change"] = heatmap_matrix(cells, "accuracy_change")
    report["figure"] = figure_heatmap(cells, out_dir / "figures")
    (out_dir / "transfer_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"cells: {report['n_cells']}  (same {report['same_layer']['n']}, "
          f"cross {report['cross_layer']['n']})")
    print(f"baseline accuracy: {report['baseline_accuracy']}")
    print(f"zero-strength max |delta|: {report['zero_strength_max_abs_delta']}")
    print(f"same-layer: mean|effect|={report['same_layer'].get('mean_abs_effect')} "
          f"significant={report['same_layer'].get('n_significant')}/"
          f"{report['same_layer'].get('n')}")
    print(f"cross-layer: mean|effect|={report['cross_layer'].get('mean_abs_effect')} "
          f"significant={report['cross_layer'].get('n_significant')}/"
          f"{report['cross_layer'].get('n')}")
    print("outcomes:")
    for key in OUTCOMES:
        print(f"   {key}: {report['outcomes'][key]}")
    print(f"\ntable  -> {table_path}")
    print(f"report -> {out_dir / 'transfer_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())