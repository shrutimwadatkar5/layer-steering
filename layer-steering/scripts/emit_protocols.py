"""Harvest block_output activations on the construction split, for every pilot layer.

This does **not** reimplement Causalab's intervention machinery. It emits a
protocol document and hands execution to the Causalab CLI, so the forward
passes, the padding-aware position resolution, and the sweep expansion are all
the engine's, not this script's.

Why a document rather than a Python forward hook:

* Causalab resolves a negative position index against each row's *first real*
  token (``token_index = index - first`` in
  ``causalab/protocol/positions/resolve.py``). We measured that padded
  positions carry a non-zero residual norm (~53 at block_output on
  Qwen2.5-1.5B), so a raw ``-1`` would address padding. Letting the engine
  resolve positions removes that entire class of bug.
* Sweeping ``sites.target.layers`` expands to N points that share the model
  load and dedupe common sub-values.

Writes two documents:

* ``harvest_construction.json`` -- reads block_output at every pilot layer on
  the construction split and saves one tensor per layer.
* ``harvest_calibration.json`` -- the same on calibration, for the zero-strength
  control and for measuring activation norms per layer.

Usage::

    python -m research.scripts.emit_protocols --config research/configs/pilot.json
    causalab run research/protocols/harvest_construction.json \\
        --engine auto --data-root research/data --out <run_dir>
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

from research.config import SplitNames, SteeringConfig, load_config  # noqa: E402
from research.steering import STEERING_DIRECTION_SHAPE_NOTE  # noqa: E402

__all__ = [
    "harvest_document",
    "steer_document",
    "write_documents",
]


def harvest_document(
    config: SteeringConfig, split: str, *, description_suffix: str = ""
) -> dict[str, Any]:
    """A protocol that saves ``block_output`` at every pilot layer.

    The layer axis is a sweep, so the engine expands one point per layer and
    each saved tensor carries its ``sites.target.layers`` coordinate.
    """
    dataset = config.dataset_for[split]
    return {
        "header": {
            "protocol_version": "4",
            "description": (
                "Harvest the residual stream at the answer slot on the "
                f"{split} split, one tensor per pilot layer. "
                f"{description_suffix}"
            ).strip(),
        },
        "model": {
            "key": config.model_key,
            "revision": config.model_revision,
            "dtype": config.dtype,
        },
        "data": {"base": {"dataset": dataset, "field": "input"}},
        "method": {
            "intervened_models": {"original": {"input": "base", "reads": ["act"]}},
            "positions": {"slot": dict(config.token_position)},
            "sites": {
                "target": {
                    "component": "block_output",
                    "layers": {"sweep": list(config.layers)},
                }
            },
            "reads": {"act": {"site": "target", "pos": "slot"}},
            "save": [
                {
                    "read": "act",
                    "model": "original",
                    "file_path": "acts.safetensors",
                }
            ],
        },
    }


def steer_document(
    config: SteeringConfig,
    *,
    split: str,
    layer: int,
    direction_file: str,
    alphas: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """A protocol that adds ``alpha * v`` at one layer, over a strength sweep.

    This is the intervention proper, and it uses Causalab's own ``add_scaled``:
    ``apply_delta`` returns ``alpha * op`` and the executor writes
    ``f = f0 + delta``. Nothing here replaces an activation -- the mechanism is
    additive by construction.

    Two models are run in the same document so the comparison is paired
    row-for-row on identical prompts:

    * ``baseline`` -- base text, no writes. The unmodified reference.
    * ``steered`` -- base text, ``add_scaled`` active.

    ``alpha = 0`` appears in the sweep by construction (the config enforces
    it), which is the zero-strength control: it must reproduce the baseline
    exactly, because ``f0 + 0 * v == f0``.
    """
    dataset = config.dataset_for[split]
    strength = list(alphas if alphas is not None else config.alphas)
    return {
        "header": {
            "protocol_version": "4",
            "description": (
                f"Additive steering at layer {layer} on the {split} split at "
                f"normalized strengths {strength}. The direction is a unit-norm "
                "mean difference built from the construction split only. "
                "alpha=0 is the zero-strength control and must equal the "
                f"baseline. {STEERING_DIRECTION_SHAPE_NOTE}"
            ),
        },
        "model": {
            "key": config.model_key,
            "revision": config.model_revision,
            "dtype": config.dtype,
        },
        "data": {"base": {"dataset": dataset, "field": "input"}},
        "method": {
            "intervened_models": {
                "baseline": {"input": "base", "reads": ["logits_base"]},
                "steered": {"input": "base", "reads": ["logits"], "writes": ["push"]},
            },
            "positions": {"slot": dict(config.token_position)},
            "sites": {
                "target": {"component": "block_output", "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "params": {
                "steer": {"file_path": direction_file, "entry": {"slot": "weight"}}
            },
            "reads": {
                "logits": {"site": "lm_head", "pos": -1},
                "logits_base": {"site": "lm_head", "pos": -1},
            },
            "writes": {
                "push": {
                    "site": "target",
                    "pos": "slot",
                    "do": {
                        "add_scaled": {
                            "op": "steer",
                            "alpha": {"sweep": strength},
                        }
                    },
                }
            },
            "save": [
                {
                    "read": "logits",
                    "model": "steered",
                    "aggregation": {"kind": "match", "expected": "base_answer"},
                    "file_path": "original_answer_rate.json",
                },
                {
                    "read": "logits",
                    "model": "steered",
                    "aggregation": {"kind": "match", "expected": "cf_answer"},
                    "file_path": "cf_answer_rate.json",
                },
                {
                    "read": "logits_base",
                    "model": "baseline",
                    "aggregation": {"kind": "match", "expected": "base_answer"},
                    "file_path": "baseline_accuracy.json",
                },
                {
                    "read": "logits",
                    "model": "steered",
                    "aggregation": {
                        "kind": "logit_diff",
                        "a": "base_answer",
                        "b": "cf_answer",
                    },
                    "file_path": "logit_margin.json",
                },
            ],
        },
    }


def write_documents(config: SteeringConfig, out_dir: Path) -> dict[str, Path]:
    """Emit the harvest documents and a manifest for the pilot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split in (SplitNames.CONSTRUCTION, SplitNames.CALIBRATION):
        doc = harvest_document(config, split)
        path = out_dir / f"harvest_{split}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        written[split] = path

    manifest = {
        "config": config.to_dict(),
        "documents": {k: str(v) for k, v in written.items()},
        "notes": {
            "hook_shape": STEERING_DIRECTION_SHAPE_NOTE,
            "why_a_document": (
                "Causalab resolves negative positions against each row's first "
                "real token, so a ragged batch cannot steer a padding slot."
            ),
            "mechanism": (
                "add_scaled returns alpha*op; the executor writes f0 + delta. "
                "This is additive, never a replacement."
            ),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    written["manifest"] = manifest_path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit pilot protocol documents.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to research/protocols",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = args.out_dir or (REPO_ROOT / "research" / "protocols")
    written = write_documents(config, out_dir)
    print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())