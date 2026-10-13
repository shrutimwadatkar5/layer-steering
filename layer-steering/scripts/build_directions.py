"""Build steering directions from harvested activations, one per pilot layer.

Consumes the ``acts.safetensors`` bundles the harvest protocol saved and writes
one ``(hidden_size,)`` direction per layer, plus a random control matched to the
real direction's norm.

The bundle format is Causalab's: a ``.safetensors`` file with a ``weight``
slot, which is exactly what a protocol document's
``params: {"steer": {"file_path": ..., "entry": {"slot": "weight"}}}`` expects.
We write through ``causalab.io.step_io.write_tensor`` so the format cannot drift
from what the engine reads.

Group construction. The mean difference needs two contrasting groups. We split
the construction rows on the *observed answer letter*: group A is the rows whose
``base_answer`` is the first symbol in the prompt, group B the rest. The split
is deterministic given the rows and recorded in the provenance.

Usage::

    python -m research.scripts.build_directions \\
        --config research/configs/pilot.json \\
        --harvest-dir research/results/raw/harvest_construction
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import SteeringConfig, load_config  # noqa: E402
from research.steering import (  # noqa: E402
    mean_difference_direction,
    random_control_direction,
    validate_direction,
)

__all__ = ["group_rows", "build_directions"]


def group_rows(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """Split construction row indices into two contrast groups.

    Group A: rows whose base answer is the *second* option's symbol.
    Group B: rows whose base answer is the *first* option's symbol.

    The symbols themselves move between prompts (that is what the task does),
    so the contrast is carried by the answer *position*, which is the task's
    own ``TARGET_VARIABLE``. This keeps the two groups balanced by construction
    whenever the dataset is, and avoids letting a specific letter define one
    group.
    """
    a: list[int] = []
    b: list[int] = []
    for index, row in enumerate(rows):
        choices = (
            str(row.get("choices[0]", "")).strip(),
            str(row.get("choices[1]", "")).strip(),
        )
        answer_word = str(row.get("base_answer", "")).strip()
        # base_answer is the letter; recover which choice is correct by
        # matching the row's colour field when present.
        color = str(row.get("color", "")).strip()
        if not color:
            # fall back to the letter itself being the group key
            (a if answer_word else b).append(index)
            continue
        if choices[1] == color:
            a.append(index)
        else:
            b.append(index)
    return a, b


def build_directions(
    config: SteeringConfig,
    harvest_dir: Path,
    out_dir: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write one direction per layer plus a norm-matched random control."""
    import torch

    from causalab.io.step_io import write_tensor

    harvest_dir = Path(harvest_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if rows is None:
        # same resolution Causalab uses for a dataset ref (io/env.py:_file)
        rows_path = (
            Path(config.data_root) / f"{config.construction_dataset}.json"
        )
        rows = json.loads(rows_path.read_text())

    group_a, group_b = group_rows(rows)
    if not group_a or not group_b:
        raise ValueError(
            f"contrast groups are empty (A={len(group_a)}, B={len(group_b)}); "
            "the construction split does not support a mean difference"
        )

    records: list[dict[str, Any]] = []
    for layer in config.layers:
        acts = _find_layer_tensor(Path(harvest_dir) / "acts.safetensors", layer)
        if acts.ndim == 1:
            raise ValueError(
                f"layer {layer} harvest is already mean-reduced to {tuple(acts.shape)}; "
                "a direction needs per-row activations"
            )
        n_rows = acts.shape[0]
        if n_rows != len(rows):
            raise ValueError(
                f"layer {layer} harvest has {n_rows} rows but the construction "
                f"table has {len(rows)}; they must align"
            )
        pos = acts[group_a]
        neg = acts[group_b]
        direction, stats = mean_difference_direction(
            pos,
            neg,
            normalize=config.normalize_direction,
            name_positive=f"construction group A (layer {layer})",
            name_negative=f"construction group B (layer {layer})",
        )
        hidden_size = int(acts.shape[-1])
        validate_direction(direction, hidden_size=hidden_size)

        out_path = out_dir / f"v_L{layer:02d}.safetensors"
        write_tensor(out_path, direction, slot="weight")

        control = random_control_direction(
            hidden_size,
            seed=config.seed + 1000 + layer,
            target_norm=float(stats["norm"]),
        )
        control_path = out_dir / f"control_L{layer:02d}.safetensors"
        write_tensor(control_path, control, slot="weight")

        records.append(
            {
                "layer": int(layer),
                "hidden_size": hidden_size,
                "direction_file": str(out_path),
                "control_file": str(control_path),
                "n_group_a": len(group_a),
                "n_group_b": len(group_b),
                "stats": dict(stats),
            }
        )
        del acts

    manifest = {
        "config": config.to_dict(),
        "groups": {"n_a": len(group_a), "n_b": len(group_b)},
        "normalize_direction": config.normalize_direction,
        "directions": records,
    }
    manifest_path = out_dir / "directions_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _find_layer_tensor(harvest_path: Path, layer: int) -> Any:
    """Return the ``(rows, positions, hidden)`` activations for one layer.

    The runner writes **one** bundle holding every sweep point, with keys named
    by the sweep coordinate -- measured on the pilot harvest:
    ``act[target.layers=4] (189, 1, 1536)``, and likewise for 12, 20, 26.
    So the layer is selected by key, not by directory.
    """
    from causalab.io.tensor_files import load_file

    if not harvest_path.exists():
        raise FileNotFoundError(
            f"no harvest bundle at {harvest_path}; run the harvest protocol first"
        )
    bundle = load_file(harvest_path)
    keys = list(bundle.keys())

    # exact coordinate key, then the looser spellings the runner may use
    exact = f"act[target.layers={layer}]"
    candidates = [exact, f"act[sites.target.layers={layer}]"]
    for name in candidates:
        if name in bundle:
            return bundle[name]

    # fall back to a substring match, refusing an ambiguous bundle rather than
    # silently picking the first key
    matches = [k for k in keys if f"layers={layer}]" in k or f"layers={layer}," in k]
    if len(matches) == 1:
        return bundle[matches[0]]
    if len(matches) > 1:
        raise ValueError(
            f"layer {layer} is ambiguous in {harvest_path}: {sorted(matches)}"
        )
    raise KeyError(
        f"layer {layer} not present in {harvest_path}; bundle holds {sorted(keys)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build steering directions.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--harvest-dir", required=True, type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="defaults to research/directions"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = args.out_dir or (REPO_ROOT / "research" / "directions")
    manifest = build_directions(config, args.harvest_dir, out_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())