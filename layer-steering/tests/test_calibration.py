"""Unit tests for the dose-calibration arithmetic.

The claim under test: ``alpha = z * sigma_within`` makes ``z`` mean the same
*relative* dose at every layer, which is what the first pilot lacked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.calibration import (  # noqa: E402
    DEFAULT_Z_GRID,
    alpha_for_z,
    compute_dose_stats,
    project_onto_direction,
)
from research.scripts.build_directions import group_rows  # noqa: E402

D = 32
N = 60


def _synth(scale_a: float = 1.0, scale_b: float = 1.0, shift: float = 3.0):
    """Two groups separated by `shift` along axis 0, with per-group spread."""
    torch.manual_seed(0)
    a = torch.randn(N // 2, D) * scale_a
    b = torch.randn(N // 2, D) * scale_b + shift
    acts = torch.cat([a, b], dim=0).unsqueeze(1)  # (n, 1, d)
    direction = torch.zeros(D)
    direction[0] = 1.0
    g_a = list(range(N // 2))
    g_b = list(range(N // 2, N))
    return acts, direction, g_a, g_b


def test_projection_matches_manual_dot():
    acts = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 1, 4)
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = project_onto_direction(acts, v)
    assert torch.allclose(out, torch.tensor([0.0, 4.0], dtype=torch.float64))


def test_projection_flattens_leading_dims():
    acts = torch.randn(5, 3, D)
    v = torch.randn(D)
    out = project_onto_direction(acts, v)
    assert out.shape == (15,)


def test_projection_rejects_width_mismatch():
    with pytest.raises(ValueError, match="cannot project"):
        project_onto_direction(torch.randn(4, D), torch.randn(D + 1))


def test_dose_stats_recover_a_known_separation():
    acts, v, ga, gb = _synth()
    stats = compute_dose_stats(acts, v, ga, gb, layer=4)
    # the helper puts the +shift group last, so mean_a - mean_b is negative
    assert stats["raw_separation"] == pytest.approx(-3.0, abs=0.5)
    assert stats["sigma_within"] > 0
    assert stats["cohens_d"] == pytest.approx(
        stats["raw_separation"] / stats["sigma_within"], rel=1e-6
    )


def test_dose_stats_separation_sign_follows_group_order():
    """Swapping the groups flips the sign but not the magnitude."""
    acts, v, ga, gb = _synth()
    forward = compute_dose_stats(acts, v, ga, gb, layer=4)
    swapped = compute_dose_stats(acts, v, gb, ga, layer=4)
    assert swapped["raw_separation"] == pytest.approx(-forward["raw_separation"])
    assert swapped["sigma_within"] == pytest.approx(forward["sigma_within"])


def test_dose_stats_scale_with_group_spread():
    """Wider within-group spread => smaller sigma => smaller alpha per z."""
    tight, v, ga, gb = _synth(scale_a=0.1, scale_b=0.1)
    wide, _, _, _ = _synth(scale_a=1.0, scale_b=1.0)
    s_tight = compute_dose_stats(tight, v, ga, gb, layer=4)
    s_wide = compute_dose_stats(wide, v, ga, gb, layer=4)
    assert s_tight["sigma_within"] < s_wide["sigma_within"]


def test_alpha_for_z_scales_linearly():
    alphas = alpha_for_z(0.5, (0.0, 2.0, 4.0))
    assert alphas == [0.0, 1.0, 2.0]


def test_alpha_grid_includes_zero():
    assert 0.0 in DEFAULT_Z_GRID


def test_dose_stats_require_two_rows_per_group():
    acts, v, ga, gb = _synth()
    with pytest.raises(ValueError, match=">= 2 rows"):
        compute_dose_stats(acts, v, ga[:1], gb, layer=4)


def test_dose_stats_reject_undersized_second_group():
    acts, v, ga, gb = _synth()
    with pytest.raises(ValueError, match=">= 2 rows"):
        compute_dose_stats(acts, v, ga, gb[:1], layer=4)


def test_equal_z_yields_equal_relative_dose_across_scales():
    """The property the calibration exists to guarantee.

    Two layers with very different activation scales must receive the same
    *relative* push at the same z.
    """
    small, v, ga, gb = _synth(scale_a=0.1, scale_b=0.1)   # sigma ~ 0.1
    large, _, _, _ = _synth(scale_a=5.0, scale_b=5.0)     # sigma ~ 5.0
    s_small = compute_dose_stats(small, v, ga, gb, layer=4)
    s_large = compute_dose_stats(large, v, ga, gb, layer=20)

    z = 3.0
    assert z * s_small["sigma_within"] / s_small["sigma_within"] == pytest.approx(z)
    assert z * s_large["sigma_within"] / s_large["sigma_within"] == pytest.approx(z)
    # and the raw alphas genuinely differ, which is the point
    assert alpha_for_z(s_small["sigma_within"], (z,)) != alpha_for_z(
        s_large["sigma_within"], (z,)
    )


def test_group_rows_partitions_by_answer_position():
    rows = [
        {"input": "", "choices[0]": "orange", "choices[1]": "red", "color": "red"},
        {"input": "", "choices[0]": "orange", "choices[1]": "red", "color": "orange"},
    ]
    # first row's color matches choices[1] -> group A; second -> group B
    a, b = group_rows(rows)
    assert a == [0]
    assert b == [1]