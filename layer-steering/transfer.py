"""Cross-layer transfer: a direction built at source layer i, applied at target layer j.

Separated from the layer-wise experiment by design
--------------------------------------------------
This module has its own configuration, its own output namespace, and its own
statistics. Results are **never** pooled with the same-layer experiment: same-layer
cells here exist only as *reference points within this grid*, and every table
carries an ``experiment`` field so the two can never be silently conflated.

Why a shared alpha would have been wrong
-----------------------------------------
Requirement 9 warns not to assume that equal-dimensional residual streams share
a coordinate system. Measured on the construction split, that is exactly the
situation:

* all 28 directions have width 1536, so Causalab accepts **any** cross-layer
  pairing without error -- dimensions are compatible;
* but they are nearly **orthogonal** across depth: cos(v_4, v_12) = 0.13,
  cos(v_4, v_20) = 0.01, cos(v_4, v_26) = 0.01, with the one exception
  cos(v_20, v_26) = 0.45;
* and the within-group spread at the target layer *along the source direction*
  is 2-20x smaller off the diagonal:

      target \\ src     4        12       20       26
        L4          2.83     0.47     0.16     0.17
        L12         0.55     1.99     0.21     0.40
        L20         0.53     0.83     2.76     1.51
        L26         1.15     1.21     2.36     6.19

So a direction that is "the same size" is not "the same size *along the axis
that matters*". Steering at a fixed raw alpha off-diagonal would push along an
axis where the activations barely vary, delivering a much smaller effective dose
than the diagonal -- and a null result would be uninterpretable.

**The strength convention therefore mirrors the layer-wise experiment and is
calibrated per (source, target) cell:**

    alpha_ij(z) = z * sigma_ij

where ``sigma_ij`` is the pooled within-group standard deviation of the target
layer's activations *projected onto the source direction*. ``z`` is then a
within-group z-score at the target along that specific axis, and z = 0 is an
exact no-op in every cell. This makes "same z" mean "same relative dose" across
the whole grid, which is the only way a transfer comparison can mean anything.

What a null result does and does not show
-----------------------------------------
A failed transfer is consistent with several explanations, and this module does
not pick between them:

* the feature is not represented at the target layer;
* it is represented but in a rotated basis (note the 0.45 cosine between L20 and
  L26, which share structure the other pairs do not);
* the dose was still too small even after calibration;
* the effect exists but is below the resolution of a 2-way argmax task.

Conversely, a successful transfer shows the direction has effect at the target
layer; it does **not** show the target layer represents "the same feature" in any
uniquely interpretable sense. Both cautions are carried in the output.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "TransferConfig",
    "EXPERIMENT_NAME",
    "check_dimension_compatibility",
    "transfer_sigma",
    "alpha_grid_for",
    "transfer_document",
    "write_transfer_config",
]

EXPERIMENT_NAME = "cross_layer_transfer"


@dataclass(frozen=True)
class TransferConfig:
    """Independently configurable transfer experiment.

    Kept separate from :class:`research.config.SteeringConfig` on purpose: this
    experiment must be runnable and tunable without touching the layer-wise one.
    """

    source_layers: tuple[int, ...] = (4, 12, 20, 26)
    target_layers: tuple[int, ...] = (4, 12, 20, 26)
    z_grid: tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0)
    include_diagonal: bool = True
    directions_dir: str = "research/directions_dense"
    n_layers: int = 28

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.source_layers:
            raise ValueError("source_layers must not be empty")
        if not self.target_layers:
            raise ValueError("target_layers must not be empty")
        if len(set(self.source_layers)) != len(self.source_layers):
            raise ValueError("source_layers contains duplicates")
        if len(set(self.target_layers)) != len(self.target_layers):
            raise ValueError("target_layers contains duplicates")
        for name, layers in (
            ("source_layers", self.source_layers),
            ("target_layers", self.target_layers),
        ):
            if list(layers) != sorted(layers):
                raise ValueError(f"{name} must be sorted ascending, got {layers}")
            if any(l < 0 for l in layers):
                raise ValueError(f"{name} must be non-negative, got {layers}")
            if any(l >= self.n_layers for l in layers):
                raise ValueError(
                    f"{name} contains a layer >= n_layers ({self.n_layers}): {layers}"
                )
        if 0.0 not in self.z_grid:
            raise ValueError(
                "z_grid must contain 0.0: the zero-strength control is required "
                "in every (source, target) cell"
            )
        if list(self.z_grid) != sorted(self.z_grid):
            raise ValueError(f"z_grid must be sorted, got {self.z_grid}")
        if not self.include_diagonal and not (
            set(self.source_layers) & set(self.target_layers)
        ):
            raise ValueError(
                "source and target layers are disjoint and include_diagonal is "
                "False, so there would be no reference points at all"
            )

    @property
    def cells(self) -> list[tuple[int, int]]:
        """Every (source, target) pair to run, in a deterministic order."""
        return [
            (i, j) for i in self.source_layers for j in self.target_layers
        ]

    @property
    def n_cells(self) -> int:
        return len(self.source_layers) * len(self.target_layers)

    @property
    def n_diagonal(self) -> int:
        return sum(
            1 for i, j in self.cells if i == j
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_layers"] = list(self.source_layers)
        data["target_layers"] = list(self.target_layers)
        data["z_grid"] = list(self.z_grid)
        data["experiment"] = EXPERIMENT_NAME
        return data


# --------------------------------------------------------------------------- #
# compatibility + calibration
# --------------------------------------------------------------------------- #
def check_dimension_compatibility(directions: dict[int, Any], *, expected: int | None = None) -> None:
    """Verify every direction shares one hidden width (requirement 8).

    Causalab would happily broadcast a width-1536 direction into a width-1536
    write at *any* layer, so a mismatch here is silently accepted rather than
    refused. Checking it explicitly is the only guard.

    Raises ``ValueError`` listing the offending layers.
    """
    widths = {layer: int(t.shape[-1]) for layer, t in directions.items()}
    distinct = set(widths.values())
    if len(distinct) > 1:
        detail = ", ".join(f"L{l}={w}" for l, w in sorted(widths.items()))
        raise ValueError(
            f"directions have inconsistent hidden widths: {detail} -- a "
            "cross-layer transfer needs one shared width"
        )
    width = distinct.pop()
    if expected is not None and width != expected:
        raise ValueError(
            f"direction width {width} does not match the model's hidden size "
            f"{expected}"
        )


def transfer_sigma(
    target_activations: Any, source_direction: Any, group_a: list[int], group_b: list[int]
) -> float:
    """Within-group sigma at the TARGET layer along the SOURCE direction.

    This is the dose scale that makes ``z`` comparable across the whole grid.
    See the module docstring for the measurement that motivates it.
    """
    import torch

    from research.calibration import project_onto_direction

    proj = project_onto_direction(target_activations, source_direction)
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(
            f"groups need >= 2 rows each, got A={len(group_a)}, B={len(group_b)}"
        )
    pa = proj[group_a].to(torch.float64)
    pb = proj[group_b].to(torch.float64)
    dof = (len(group_a) - 1) + (len(group_b) - 1)
    pooled = (
        (len(group_a) - 1) * float(pa.var(unbiased=True))
        + (len(group_b) - 1) * float(pb.var(unbiased=True))
    ) / dof
    return pooled ** 0.5


def alpha_grid_for(sigma: float, z_grid: tuple[float, ...]) -> list[float]:
    """Raw strengths for one (source, target) cell."""
    return [float(z) * float(sigma) for z in z_grid]


# --------------------------------------------------------------------------- #
# protocol document
# --------------------------------------------------------------------------- #
def transfer_document(
    base_cfg: Any,
    tcfg: TransferConfig,
    *,
    source_layer: int,
    target_layer: int,
    alphas: list[float],
    split: str = "calibration",
) -> dict[str, Any]:
    """One transfer cell as a Causalab document.

    The document carries ``EXPERIMENT_NAME`` in its description and uses the
    source direction at the target layer. One document per cell keeps the arms
    single-write (see ``research/evaluation.py`` for why two writes at one module
    silently collide) and makes each cell independently resumable.
    """
    if 0.0 not in alphas:
        raise ValueError("transfer cell must include alpha = 0.0")
    dataset = base_cfg.dataset_for[split]
    direction = f"{tcfg.directions_dir}/v_L{source_layer:02d}.safetensors"
    same_layer = source_layer == target_layer
    kind = "same-layer reference" if same_layer else "cross-layer transfer"

    return {
        "header": {
            "protocol_version": "4",
            "description": (
                f"[{EXPERIMENT_NAME}] {kind}: direction built at source layer "
                f"{source_layer}, applied at target layer {target_layer}, at "
                f"strengths {alphas}. alpha = z * sigma_ij where sigma_ij is the "
                f"target layer's within-group spread along the source direction, "
                f"so z is a within-group z-score at the target. Split: {split}. "
                "Scoring only, no generation."
            ),
        },
        "model": {
            "key": base_cfg.model_key,
            "revision": base_cfg.model_revision,
            "dtype": base_cfg.dtype,
        },
        "data": {"base": {"dataset": dataset, "field": "input"}},
        "method": {
            "intervened_models": {
                "baseline": {"input": "base", "reads": ["logits_base"]},
                "transfer": {
                    "input": "base",
                    "reads": ["logits"],
                    "writes": ["push"],
                },
            },
            "positions": {"slot": dict(base_cfg.token_position)},
            "sites": {
                "target": {"component": "block_output", "layers": [target_layer]},
                "lm_head": {"component": "lm_head"},
            },
            "params": {
                "steer": {"file_path": direction, "entry": {"slot": "weight"}},
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
                            "alpha": {"sweep": [float(a) for a in alphas]},
                        }
                    },
                }
            },
            "save": [
                {
                    "read": "logits",
                    "model": "transfer",
                    "aggregation": {"kind": "match", "expected": "base_answer"},
                    "file_path": "transfer_original_rate.json",
                },
                {
                    "read": "logits",
                    "model": "transfer",
                    "aggregation": {"kind": "match", "expected": "cf_answer"},
                    "file_path": "transfer_cf_rate.json",
                },
                {
                    "read": "logits",
                    "model": "transfer",
                    "aggregation": {
                        "kind": "logit_diff",
                        "a": "base_answer",
                        "b": "cf_answer",
                    },
                    "file_path": "transfer_margin.json",
                },
                {
                    "read": "logits_base",
                    "model": "baseline",
                    "aggregation": {"kind": "match", "expected": "base_answer"},
                    "file_path": "baseline_accuracy.json",
                },
                {
                    "read": "logits_base",
                    "model": "baseline",
                    "aggregation": {
                        "kind": "logit_diff",
                        "a": "base_answer",
                        "b": "cf_answer",
                    },
                    "file_path": "baseline_margin.json",
                },
            ],
        },
    }


def write_transfer_config(
    out_dir: Path, base_cfg: Any, tcfg: TransferConfig, sigma_table: dict, **extra
) -> Path:
    """Stamp the full transfer configuration into a results directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "transfer_config": tcfg.to_dict(),
        "engine_config": {
            "model_key": base_cfg.model_key,
            "revision": base_cfg.model_revision,
            "dtype": base_cfg.dtype,
            "token_position": dict(base_cfg.token_position),
            "component": "block_output",
        },
        "strength_convention": (
            "alpha = z * sigma_ij; sigma_ij is the pooled within-group sd of "
            "the TARGET layer's activations projected onto the SOURCE "
            "direction, computed on the construction split. z is therefore a "
            "within-group z-score at the target along that axis."
        ),
        "normalization": (
            "each source direction is unit-norm (divided by its own L2 norm), "
            "matching the layer-wise experiment"
        ),
        "sigma_table": sigma_table,
        "separability_note": (
            "results of this experiment are NOT pooled with the same-layer "
            "layer-wise experiment; diagonal cells here are reference points "
            "within this grid only"
        ),
    }
    payload.update(extra)
    path = out_dir / "transfer_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path