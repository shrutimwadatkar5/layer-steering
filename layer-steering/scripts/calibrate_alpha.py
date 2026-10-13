"""Compute the per-layer dose table from the construction harvest.

Reads the already-harvested construction activations and the already-built
directions, then measures how much the activations naturally vary *along* each
steering direction. Emits ``dose_calibration.json`` and, with ``--emit-alpha``,
the per-layer ``alpha`` grid to steer at.

No model forward passes: everything here is arithmetic on saved tensors.

Usage::

    python -m research.scripts.calibrate_alpha \\
        --config research/configs/pilot.json \\
        --harvest-dir research/results/raw/harvest_construction \\
        --directions-dir research/directions \\
        --out research/results/tables/dose_calibration.json
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

from research.calibration import (  # noqa: E402
    DEFAULT_Z_GRID,
    alpha_for_z,
    compute_dose_stats,
)
from research.config import SteeringConfig, load_config  # noqa: E402
from research.scripts.build_directions import (  # noqa: E402
    _find_layer_tensor,
    group_rows,
)

__all__ = ["build_dose_table"]


def build_dose_table(
    config: SteeringConfig,
    harvest_dir: Path,
    directions_dir: Path,
    z_grid: tuple[float, ...] = DEFAULT_Z_GRID,
) -> dict[str, Any]:
    """One :class:`DoseStats` per pilot layer, plus the alpha grid."""
    from causalab.io.step_io import read_tensor

    rows_path = Path(config.data_root) / f"{config.construction_dataset}.json"
    rows = json.loads(rows_path.read_text())
    group_a, group_b = group_rows(rows)

    bundle = Path(harvest_dir) / "acts.safetensors"
    entries: list[dict[str, Any]] = []

    for layer in config.layers:
        acts = _find_layer_tensor(bundle, layer)
        direction = read_tensor(
            Path(directions_dir) / f"v_L{layer:02d}.safetensors",
            slot="weight",
            what=f"direction for layer {layer}",
        )
        if acts.shape[0] != len(rows):
            raise ValueError(
                f"layer {layer}: harvest has {acts.shape[0]} rows, construction "
                f"table has {len(rows)}; they must align"
            )
        stats = compute_dose_stats(
            acts, direction, group_a, group_b, layer=int(layer)
        )
        stats["z_grid"] = list(z_grid)
        stats["alpha_grid"] = alpha_for_z(stats["sigma_within"], z_grid)
        entries.append(dict(stats))
        del acts

    return {
        "config": config.to_dict(),
        "method": (
            "alpha = z * sigma_within, where sigma_within is the pooled "
            "within-group standard deviation of the activation projected onto "
            "the layer's steering direction. z is therefore a within-group "
            "z-score and means the same dose at every layer."
        ),
        "supersedes": (
            "the first pilot swept a shared alpha on a unit-norm direction, "
            "which reached only ~0.05 sigma at layer 26 and produced a null"
        ),
        "layers": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-layer alpha calibration from artifacts."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--harvest-dir", required=True, type=Path)
    parser.add_argument("--directions-dir", required=True, type=Path)
    parser.add_argument(
        "--z-grid",
        default=",".join(str(z) for z in DEFAULT_Z_GRID),
        help="comma-separated z-scores; must include 0.0",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    z_grid = tuple(float(v) for v in args.z_grid.split(",") if v.strip())
    if 0.0 not in z_grid:
        parser.error("z grid must include 0.0 (the zero-strength control)")

    config = load_config(args.config)
    table = build_dose_table(
        config, args.harvest_dir, args.directions_dir, z_grid=z_grid
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n")

    print(f"{'L':>3} {'sigma':>9} {'sep':>9} {'d':>7}   alpha grid")
    for entry in table["layers"]:
        print(
            f"{entry['layer']:>3} {entry['sigma_within']:>9.3f} "
            f"{entry['raw_separation']:>9.3f} {entry['cohens_d']:>7.2f}   "
            + " ".join(f"{a:.2f}" for a in entry["alpha_grid"])
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())