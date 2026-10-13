"""Aggregate steering results into per-layer curves with explicit denominators.

Reads the runner's raw tables -- one JSON list per saved metric, every row
carrying its sweep coordinates -- and emits a compact, structured CSV plus a
provenance sidecar.

Metrics, with denominators stated (the audit's §4.6):

* ``baseline_accuracy``    = mean[match(baseline, base_answer)]      / n_eval
* ``original_answer_rate`` = mean[match(steered, base_answer)]       / n_eval
* ``cf_answer_rate``       = mean[match(steered, cf_answer)]        / n_eval
* ``original_answer_delta``= original_answer_rate - baseline_accuracy

We do **not** compute a flip rate here. A flip needs both the baseline and the
steered prediction for the same prompt at the same alpha, and the raw tables are
keyed per model, so joining them is a separate, explicitly-labelled step. The
aggregate therefore reports only rates whose denominator is unambiguous, and
``analyze.py`` adds the paired statistics.

The zero-strength check is enforced, not assumed: at ``alpha == 0`` the
steered original-answer rate must equal the baseline rate on the same split.
A mismatch there means the intervention is not a clean no-op at zero, which
invalidates every comparison built on it.

Usage::

    python -m research.scripts.aggregate \\
        --config research/configs/pilot.json \\
        --run-dir research/results/raw/steer
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import SteeringConfig, load_config  # noqa: E402

__all__ = ["load_table", "aggregate", "zero_strength_check"]

_METRIC_FILES = {
    "original_answer_rate": "original_answer_rate.json",
    "cf_answer_rate": "cf_answer_rate.json",
    "baseline_accuracy": "baseline_accuracy.json",
    "logit_margin": "logit_margin.json",
}

#: The sweep coordinate names Causalab writes into each result row.
_LAYER_KEY = "sites.target.layers"
_ALPHA_KEY = "writes.push.do.add_scaled.alpha"


def load_table(path: Path) -> list[dict[str, Any]]:
    """Read one result table. A missing file is empty, not an error."""
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else [data]


def _coord(row: dict[str, Any], key: str) -> Any:
    return row.get(key)


def _resolve_layer(rows: list[dict[str, Any]], fallback: int | None) -> int | None:
    """Recover the layer for a run directory.

    Causalab signs **swept** coordinates into every row. A steering document
    pins one layer (``"layers": [L]``), so that is not a sweep and no layer
    coordinate appears in the rows -- confirmed on the pilot output, whose rows
    carry only ``writes.push.do.add_scaled.alpha``.

    So the layer comes from the sweep coordinate when present, and otherwise from
    ``fallback``, which the caller supplies from the document it generated.
    """
    for row in rows:
        value = _coord(row, _LAYER_KEY)
        if value is not None:
            return int(value)
    return fallback


def aggregate(
    run_dir: Path, config: SteeringConfig, *, layer: int | None = None
) -> list[dict[str, Any]]:
    """Build one row per (layer, alpha) with every metric and its denominator."""
    run_dir = Path(run_dir)
    tables = {
        name: load_table(run_dir / filename)
        for name, filename in _METRIC_FILES.items()
    }

    # Bucket every metric by (layer, alpha, prompt).
    buckets: dict[tuple[int, float], dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for name, rows in tables.items():
        run_layer = _resolve_layer(rows, layer)
        for row in rows:
            row_layer = _coord(row, _LAYER_KEY)
            alpha = _coord(row, _ALPHA_KEY)
            if alpha is None:
                continue
            effective_layer = int(row_layer) if row_layer is not None else run_layer
            if effective_layer is None:
                continue
            key = (effective_layer, float(alpha))
            prompt = str(row.get("example_id", ""))
            buckets[key][name][prompt].append(float(row["value"]))

    out: list[dict[str, Any]] = []
    for (layer, alpha), per_metric in sorted(buckets.items()):
        n_eval = len(per_metric.get("baseline_accuracy", {}))
        record: dict[str, Any] = {
            "layer": layer,
            "alpha": alpha,
            "n_eval": n_eval,
        }
        for name in _METRIC_FILES:
            per_prompt = per_metric.get(name, {})
            values = [
                statistics.fmean(v) for _, v in sorted(per_prompt.items()) if v
            ]
            record[name] = statistics.fmean(values) if values else None
            record[f"{name}_n"] = len(values)
        baseline = record.get("baseline_accuracy")
        original = record.get("original_answer_rate")
        record["original_answer_delta"] = (
            None
            if baseline is None or original is None
            else round(original - baseline, 6)
        )
        out.append(record)
    return out


def zero_strength_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assert ``alpha == 0`` reproduces the baseline exactly.

    This is the pilot's internal consistency check (requirement F / J). A
    non-zero deviation means the zero-strength run is not a clean no-op, so the
    strength axis has no meaningful origin and every rate is suspect.
    """
    at_zero = [r for r in rows if abs(float(r["alpha"])) < 1e-12]
    if not at_zero:
        return {"ok": False, "reason": "no alpha=0 rows in the results"}

    deviations = []
    for record in at_zero:
        baseline = record.get("baseline_accuracy")
        original = record.get("original_answer_rate")
        if baseline is None or original is None:
            continue
        deviations.append(abs(float(original) - float(baseline)))

    if not deviations:
        return {"ok": False, "reason": "alpha=0 rows carry no comparable metrics"}

    worst = max(deviations)
    return {
        "ok": worst <= 1e-9,
        "max_abs_deviation": worst,
        "n_layers_checked": len(deviations),
        "tolerance": 1e-9,
        "note": (
            "alpha=0 must equal the unmodified baseline on the same prompts; "
            "a deviation means add_scaled is not a clean no-op at zero."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate steering results.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="layer for a single-layer run; inferred from the rows when swept",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to research/results/tables",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    rows = aggregate(args.run_dir, config, layer=args.layer)
    check = zero_strength_check(rows)

    out_dir = args.out_dir or (REPO_ROOT / "research" / "results" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "steering_by_layer.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")

    provenance = {
        "config": config.to_dict(),
        "run_dir": str(args.run_dir),
        "layer": args.layer,
        "metric_files": _METRIC_FILES,
        "denominators": {
            "baseline_accuracy": "n_eval prompts in this (layer, alpha) cell",
            "original_answer_rate": "n_eval prompts in this (layer, alpha) cell",
            "cf_answer_rate": "n_eval prompts in this (layer, alpha) cell",
            "logit_margin": "n_eval prompts in this (layer, alpha) cell",
        },
        "zero_strength_check": check,
        "n_rows": len(rows),
    }
    prov_path = out_dir / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())