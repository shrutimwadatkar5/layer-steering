"""Build the cross-layer transfer grid: per-cell dose calibration and documents.

Reads the already-harvested construction activations and the already-built
per-layer directions. No model forward passes -- the sigma table is arithmetic on
saved tensors, so calibrating a 16-cell grid is free.

Emits one protocol document per (source, target) cell, plus a manifest.

Usage::

    python -m research.scripts.prepare_transfer --config research/configs/dense.json \\
        --harvest-dir research/results/raw/harvest_dense
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

from research.calibration import project_onto_direction  # noqa: E402
from research.config import SteeringConfig, load_config  # noqa: E402
from research.scripts.build_directions import (  # noqa: E402
    _find_layer_tensor,
    group_rows,
)
from research.transfer import (  # noqa: E402
    EXPERIMENT_NAME,
    TransferConfig,
    alpha_grid_for,
    check_dimension_compatibility,
    transfer_document,
    transfer_sigma,
)

__all__ = ["build_sigma_table", "prepare"]


def build_sigma_table(
    tcfg: TransferConfig, harvest_bundle: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-cell ``sigma_ij`` plus the direction-alignment matrix.

    The alignment matrix (cosine between source directions) is reported alongside
    the dose table because it is the natural explanation for any transfer pattern:
    a pair whose directions are near-orthogonal has no aligned axis to push along,
    while a pair with high cosine (here L20/L26 at 0.45) shares structure.
    """
    import torch

    from causalab.io.step_io import read_tensor

    group_a, group_b = group_rows(rows)

    needed = sorted(set(tcfg.source_layers) | set(tcfg.target_layers))
    directions = {
        layer: read_tensor(
            Path(tcfg.directions_dir) / f"v_L{layer:02d}.safetensors",
            slot="weight",
            what=f"direction v_L{layer:02d}",
        )
        for layer in needed
    }
    check_dimension_compatibility(directions)

    alignment: dict[str, dict[str, float]] = {}
    for i in tcfg.source_layers:
        row = {}
        for j in tcfg.source_layers:
            cos = float(
                torch.nn.functional.cosine_similarity(
                    directions[i].double(), directions[j].double(), dim=0
                )
            )
            row[str(j)] = round(cos, 6)
        alignment[str(i)] = row

    sigma_table: dict[str, Any] = {}
    for source in tcfg.source_layers:
        for target in tcfg.target_layers:
            acts = _find_layer_tensor(harvest_bundle, target)
            sigma = transfer_sigma(
                acts, directions[source], group_a, group_b
            )
            key = f"{source}->{target}"
            sigma_table[key] = {
                "source_layer": source,
                "target_layer": target,
                "same_layer": source == target,
                "sigma": round(float(sigma), 8),
                "alpha_grid": alpha_grid_for(sigma, tcfg.z_grid),
            }
            del acts

    return {
        "groups": {"n_a": len(group_a), "n_b": len(group_b)},
        "direction_alignment_cosine": alignment,
        "sigma_table": sigma_table,
    }


def prepare(
    config: SteeringConfig, tcfg: TransferConfig, harvest_dir: Path, out_dir: Path
) -> dict[str, Path]:
    rows = json.loads(
        (Path(config.data_root) / f"{config.construction_dataset}.json").read_text()
    )
    bundle = Path(harvest_dir) / "acts.safetensors"
    table = build_sigma_table(tcfg, bundle, rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for (source, target) in tcfg.cells:
        key = f"{source}->{target}"
        alphas = table["sigma_table"][key]["alpha_grid"]
        doc = transfer_document(
            config,
            tcfg,
            source_layer=source,
            target_layer=target,
            alphas=alphas,
        )
        path = out_dir / f"transfer_s{source:02d}_t{target:02d}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        written[key] = path

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "transfer_config": tcfg.to_dict(),
        **table,
        "documents": {k: str(v) for k, v in written.items()},
    }
    manifest_path = out_dir / "transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    written["manifest"] = manifest_path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the transfer grid.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--harvest-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sources", default="4,12,20,26")
    parser.add_argument("--targets", default="4,12,20,26")
    parser.add_argument("--z-grid", default="-8,-4,0,4,8")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    tcfg = TransferConfig(
        source_layers=tuple(int(x) for x in args.sources.split(",")),
        target_layers=tuple(int(x) for x in args.targets.split(",")),
        z_grid=tuple(float(x) for x in args.z_grid.split(",")),
    )
    out_dir = args.out_dir or (REPO_ROOT / "research" / "protocols" / "transfer")
    written = prepare(config, tcfg, args.harvest_dir, out_dir)

    manifest = json.loads(Path(written["manifest"]).read_text())
    print("sigma table (target \\ source):")
    srcs = tcfg.source_layers
    print("        " + "".join(f"  src{s:<7}" for s in srcs))
    for target in tcfg.target_layers:
        row = "".join(
            f"  {manifest['sigma_table'][f'{s}->{target}']['sigma']:<9.4f}"
            for s in srcs
        )
        print(f"  t{target:<5}" + row)
    print(f"\n{len(written) - 1} cell documents -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())