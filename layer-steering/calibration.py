"""Per-layer dose calibration for the steering strength.

Why this module exists
----------------------
The first pilot swept a shared ``alpha`` over a **unit-norm** direction and got
a null everywhere. The measured cause was dose, not direction quality: at
``alpha = 2`` the perturbation was 0.7%-5.5% of the local residual norm, and
because residual norms grow with depth the same ``alpha`` delivered 7.5x less
relative dose at layer 26 than at layer 4.

Normalizing by the activation norm fixes the *ratio* but not the *meaning*: the
quantity that actually matters is how far the perturbation moves the activation
**along the steering direction**, compared with how much the activation already
varies along that direction between prompts.

That quantity is directly measurable from the construction harvest:

    p_i = <h_i, v_l>                    projection of prompt i onto v_l
    mu_A, mu_B                          group means of p
    sigma_within                        pooled within-group standard deviation
    cohens_d = (mu_A - mu_B) / sigma_within

``sigma_within`` is the natural scale of the steering axis: how far apart two
*ordinary* prompts sit on it. So expressing strength as

    alpha = z * sigma_within

makes ``z`` a z-score, and ``z`` means the same thing at every layer by
construction. ``z = 0`` is the zero-strength control, ``z = 2`` moves the
activation two within-group standard deviations along ``v_l``.

This turns the layer-dependence question into a well-posed one: *at equal
effective dose, which layers respond?*

Functions here only read artifacts and compute statistics; they never run the
model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "DoseStats",
    "project_onto_direction",
    "compute_dose_stats",
    "load_dose_table",
    "alpha_for_z",
    "DEFAULT_Z_GRID",
]

#: z-score grid. Deliberately extends well past the old alpha=2 ceiling: the
#: old sweep reached only ~0.05 sigma at layer 26, so nothing near a threshold
#: was ever tested. Includes 0.0 as the mandatory zero-strength control.
DEFAULT_Z_GRID: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


class DoseStats(dict):
    """Per-layer dose statistics. A dict so it serializes directly."""

    @property
    def sigma_within(self) -> float:
        return float(self["sigma_within"])

    @property
    def alpha_per_z(self) -> float:
        return float(self["sigma_within"])


def project_onto_direction(activations: Any, direction: Any) -> Any:
    """Project rows of activations onto one direction.

    Args:
        activations: ``(..., d)`` -- leading dims flattened to rows.
        direction: ``(d,)``.

    Returns:
        ``(n_rows,)`` float64 tensor of projections.
    """
    import torch

    rows = activations.reshape(-1, activations.shape[-1]).to(torch.float64)
    v = direction.reshape(-1).to(torch.float64)
    if rows.shape[-1] != v.shape[0]:
        raise ValueError(
            f"cannot project {rows.shape[-1]}-dim activations onto a "
            f"{v.shape[0]}-dim direction"
        )
    return rows @ v


def compute_dose_stats(
    activations: Any,
    direction: Any,
    group_a: list[int],
    group_b: list[int],
    *,
    layer: int,
) -> DoseStats:
    """Within-group spread and effect size along ``direction`` at one layer.

    Args:
        activations: ``(rows, positions, d)`` or ``(rows, d)`` harvest for this
            layer, row-aligned with the construction table.
        direction: the ``(d,)`` direction for this layer.
        group_a / group_b: row indices of the two contrast groups.
        layer: recorded for provenance.

    Returns:
        ``DoseStats`` with the group means, the pooled within-group sigma, the
        raw separation, and Cohen's d along the steering axis.

    Raises:
        ValueError: if a group has fewer than two rows (a standard deviation
            from one sample is meaningless).
    """
    import torch

    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError(
            f"layer {layer}: groups need >= 2 rows each for a variance estimate, "
            f"got A={len(group_a)}, B={len(group_b)}"
        )

    proj = project_onto_direction(activations, direction)
    a = proj[group_a]
    b = proj[group_b]

    # Pooled within-group standard deviation (ddof=1 per group).
    var_a = float(torch.var(a, unbiased=True))
    var_b = float(torch.var(b, unbiased=True))
    dof = (len(group_a) - 1) + (len(group_b) - 1)
    pooled_var = ((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b) / dof
    sigma = pooled_var ** 0.5

    raw_separation = float(a.mean() - b.mean())
    cohens_d = raw_separation / sigma if sigma > 0 else float("nan")

    return DoseStats(
        {
            "layer": int(layer),
            "n_a": len(group_a),
            "n_b": len(group_b),
            "mean_projection_a": float(a.mean()),
            "mean_projection_b": float(b.mean()),
            "raw_separation": raw_separation,
            "sigma_within": sigma,
            "cohens_d": cohens_d,
            "alpha_per_z": sigma,
            "sigma_note": (
                "alpha = z * sigma_within makes z a within-group z-score along "
                "the steering axis, so the same z means the same relative dose "
                "at every layer"
            ),
        }
    )


def alpha_for_z(sigma_within: float, z_grid: tuple[float, ...]) -> list[float]:
    """Convert a z-score grid into raw ``alpha`` values for one layer."""
    return [float(z) * float(sigma_within) for z in z_grid]


def load_dose_table(path: Path) -> dict[int, DoseStats]:
    """Read ``dose_calibration.json`` keyed by layer."""
    data = json.loads(Path(path).read_text())
    return {int(entry["layer"]): DoseStats(entry) for entry in data["layers"]}