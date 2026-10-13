"""Unit tests for the cross-layer transfer module.

Scope: source/target indexing, direction dimensions, dose calibration, and the
document shape. The engine's own intervention semantics are already covered by
upstream ``test_write_oracle.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import SteeringConfig  # noqa: E402
from research.transfer import (  # noqa: E402
    EXPERIMENT_NAME,
    TransferConfig,
    alpha_grid_for,
    check_dimension_compatibility,
    transfer_document,
    transfer_sigma,
)

D = 24


# --------------------------------------------------------------------------- #
# configuration / indexing
# --------------------------------------------------------------------------- #
def test_default_config_is_valid():
    cfg = TransferConfig()
    assert cfg.n_cells == 16
    assert cfg.n_diagonal == 4


def test_cells_are_the_full_cross_product_in_order():
    cfg = TransferConfig(source_layers=(2, 8), target_layers=(2, 8, 16))
    assert cfg.cells == [(2, 2), (2, 8), (2, 16), (8, 2), (8, 8), (8, 16)]


def test_cells_cover_every_pair_exactly_once():
    cfg = TransferConfig(source_layers=(1, 5, 9), target_layers=(1, 5))
    assert len(set(cfg.cells)) == len(cfg.cells) == 3 * 2


def test_rejects_duplicate_sources():
    with pytest.raises(ValueError, match="duplicates"):
        TransferConfig(source_layers=(4, 4))


def test_rejects_unsorted_layers():
    with pytest.raises(ValueError, match="sorted"):
        TransferConfig(target_layers=(20, 4))


def test_rejects_layer_beyond_model_depth():
    with pytest.raises(ValueError, match="n_layers"):
        TransferConfig(source_layers=(99,), n_layers=28)


def test_rejects_z_grid_without_zero():
    """Every cell needs its own no-op control."""
    with pytest.raises(ValueError, match="must contain 0.0"):
        TransferConfig(z_grid=(-4.0, 4.0))


def test_rejects_disjoint_grid_with_no_reference():
    with pytest.raises(ValueError, match="no reference points"):
        TransferConfig(
            source_layers=(4,), target_layers=(20,), include_diagonal=False
        )


def test_config_to_dict_tags_the_experiment():
    payload = TransferConfig().to_dict()
    assert payload["experiment"] == EXPERIMENT_NAME
    assert payload["source_layers"] == [4, 12, 20, 26]


# --------------------------------------------------------------------------- #
# dimension compatibility (requirement 8)
# --------------------------------------------------------------------------- #
def test_accepts_uniform_widths():
    check_dimension_compatibility({4: torch.randn(D), 12: torch.randn(D)})


def test_rejects_mismatched_widths():
    """Requirement 8. The engine would accept this silently; we must not."""
    with pytest.raises(ValueError, match="inconsistent hidden widths"):
        check_dimension_compatibility({4: torch.randn(D), 12: torch.randn(D + 8)})


def test_rejects_width_mismatch_against_model():
    with pytest.raises(ValueError, match="hidden size"):
        check_dimension_compatibility({4: torch.randn(D)}, expected=1536)


# --------------------------------------------------------------------------- #
# dose calibration
# --------------------------------------------------------------------------- #
def _grid(along: int, scale: float, shift_a: float = 0.0):
    torch.manual_seed(0)
    a = torch.randn(20, D) * scale
    b = torch.randn(20, D) * scale
    acts = torch.cat([a, b], 0)
    direction = torch.zeros(D)
    direction[along] = 1.0
    ga = list(range(20))
    gb = list(range(20, 40))
    return acts, direction, ga, gb, shift_a


def test_transfer_sigma_uses_target_layer_along_source_axis():
    acts, direction, ga, gb, _ = _grid(along=0, scale=1.0)
    sigma = transfer_sigma(acts, direction, ga, gb)
    assert sigma == pytest.approx(1.0, rel=0.1)


def test_transfer_sigma_is_zero_along_a_dead_axis():
    """Pushing along an axis the layer does not use delivers no dose.

    The activations are built with variance only on axis 0, so every other axis
    is exactly dead -- which is the extreme version of the off-diagonal case.
    """
    torch.manual_seed(0)
    a = torch.randn(20, D)
    b = torch.randn(20, D)
    a[:, 1:] = 0.0
    b[:, 1:] = 0.0
    acts = torch.cat([a, b], 0)
    dead = torch.zeros(D)
    dead[5] = 1.0
    live = torch.zeros(D)
    live[0] = 1.0
    ga = list(range(20))
    gb = list(range(20, 40))

    assert transfer_sigma(acts, dead, ga, gb) < 1e-9
    assert transfer_sigma(acts, live, ga, gb) > 0.5


def test_transfer_sigma_differs_by_direction_at_same_layer():
    """The motivating measurement: dose depends on the direction, not just the layer."""
    acts, _, ga, gb, _ = _grid(along=0, scale=1.0)
    acts2 = acts.clone()
    acts2[:, 7] = acts2[:, 7] * 10.0  # widen variance on axis 7
    strong = torch.zeros(D)
    strong[7] = 1.0
    weak = torch.zeros(D)
    weak[0] = 1.0
    assert transfer_sigma(acts2, strong, ga, gb) > transfer_sigma(
        acts2, weak, ga, gb
    )


def test_transfer_sigma_requires_two_rows_per_group():
    acts, direction, ga, gb, _ = _grid(along=0, scale=1.0)
    with pytest.raises(ValueError, match=">= 2 rows"):
        transfer_sigma(acts, direction, ga[:1], gb)


def test_alpha_grid_is_linear_in_z_and_includes_zero():
    alphas = alpha_grid_for(0.5, (-4.0, 0.0, 4.0))
    assert alphas == [-2.0, 0.0, 2.0]


# --------------------------------------------------------------------------- #
# document shape
# --------------------------------------------------------------------------- #
def _doc(source: int, target: int, **kw):
    base = SteeringConfig(model_key="org/name")
    tcfg = TransferConfig()
    return transfer_document(
        base,
        tcfg,
        source_layer=source,
        target_layer=target,
        alphas=kw.pop("alphas", [-8.0, -4.0, 0.0, 4.0, 8.0]),
        **kw,
    )


def test_document_applies_source_direction_at_target_layer():
    doc = _doc(4, 20)
    assert doc["method"]["sites"]["target"]["layers"] == [20]
    assert doc["method"]["params"]["steer"]["file_path"].endswith("v_L04.safetensors")


def test_document_uses_the_source_not_the_target_direction():
    doc = _doc(12, 26)
    assert "v_L12.safetensors" in json.dumps(doc)


def test_document_has_one_write_only():
    """Two writes at one module silently collide (see research/evaluation.py)."""
    doc = _doc(4, 20)
    assert len(doc["method"]["writes"]) == 1


def test_document_labels_same_layer_reference():
    doc = _doc(20, 20)
    assert "same-layer reference" in doc["header"]["description"]
    assert EXPERIMENT_NAME in doc["header"]["description"]


def test_document_labels_cross_layer_transfer():
    doc = _doc(4, 20)
    assert "cross-layer transfer" in doc["header"]["description"]


def test_document_includes_baseline_margin():
    doc = _doc(4, 20)
    files = {s["file_path"] for s in doc["method"]["save"]}
    assert "baseline_margin.json" in files
    assert "transfer_margin.json" in files


def test_document_sweep_has_both_signs_and_zero():
    doc = _doc(4, 20)
    sweep = doc["method"]["writes"]["push"]["do"]["add_scaled"]["alpha"]["sweep"]
    assert any(a < 0 for a in sweep)
    assert 0.0 in sweep
    assert any(a > 0 for a in sweep)


def test_document_rejects_grid_without_zero():
    base = SteeringConfig(model_key="org/name")
    with pytest.raises(ValueError, match="alpha = 0.0"):
        transfer_document(
            base, TransferConfig(), source_layer=4, target_layer=20, alphas=[1.0, 2.0]
        )


def test_document_selects_the_requested_split():
    doc = _doc(4, 20, split="calibration")
    assert doc["data"]["base"]["dataset"].endswith("calibration")


def test_document_is_json_serializable():
    json.dumps(_doc(4, 20))