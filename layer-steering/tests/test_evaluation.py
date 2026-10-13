"""Unit tests for the evaluation document shape and the statistics helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import SteeringConfig  # noqa: E402
from research.evaluation import (  # noqa: E402
    DECODE_CONFIG,
    evaluation_document,
    write_run_config,
)
from research.figures import holm_bonferroni  # noqa: E402


# --------------------------------------------------------------------------- #
# document shape
# --------------------------------------------------------------------------- #
def _doc(**kwargs):
    config = SteeringConfig(model_key="org/name")
    kwargs.setdefault("control_file", "research/directions_dense/control_L20.safetensors")
    kwargs.setdefault("arm", "real")
    return evaluation_document(config, layer=20, alphas=(-4.0, 0.0, 4.0), **kwargs)


def test_real_arm_has_baseline_and_steering():
    doc = _doc(arm="real")
    models = doc["method"]["intervened_models"]
    assert "baseline" in models           # baseline
    assert "steered" in models             # zero-strength + positive/negative


def test_control_arm_is_a_separate_document():
    """Regression guard for the write-collision bug.

    Putting both writes in one document with two sites at the same
    block_output made the control arm's alpha sweep silently not bind -- its
    margin was identical at every strength. Each arm therefore gets its own
    document with exactly one site and one write.
    """
    doc = _doc(arm="control")
    models = doc["method"]["intervened_models"]
    assert "baseline" in models
    assert "controlled" in models
    assert "steered" not in models
    assert len(doc["method"]["writes"]) == 1
    assert len(doc["method"]["sites"]) == 2  # target + lm_head


def test_each_arm_document_has_exactly_one_intervening_write():
    for arm in ("real", "control"):
        doc = _doc(arm=arm)
        writers = [
            name
            for name, m in doc["method"]["intervened_models"].items()
            if m.get("writes")
        ]
        assert len(writers) == 1, f"{arm} arm has {writers}"


def test_control_arm_loads_the_control_direction():
    doc = _doc(arm="control")
    params = doc["method"]["params"]
    assert "control" in params
    assert params["control"]["file_path"].endswith("control_L20.safetensors")
    assert doc["method"]["writes"]["push"]["do"]["add_scaled"]["op"] == "control"


def test_control_arm_requires_a_control_file():
    config = SteeringConfig(model_key="org/name")
    with pytest.raises(ValueError, match="requires control_file"):
        evaluation_document(
            config, layer=4, alphas=(0.0, 1.0), control_file=None, arm="control"
        )


def test_rejects_unknown_arm():
    config = SteeringConfig(model_key="org/name")
    with pytest.raises(ValueError, match="arm must be"):
        evaluation_document(
            config, layer=4, alphas=(0.0, 1.0), control_file="x", arm="bogus"
        )


def test_document_has_both_signs_and_zero():
    doc = _doc()
    sweep = doc["method"]["writes"]["push"]["do"]["add_scaled"]["alpha"]["sweep"]
    assert any(a < 0 for a in sweep)
    assert any(a > 0 for a in sweep)
    assert 0.0 in sweep


def test_zero_strength_is_in_the_sweep_not_a_separate_model():
    """alpha=0 must be a point on the strength axis, so it is paired."""
    doc = _doc()
    models = doc["method"]["intervened_models"]
    assert not any("zero" in name for name in models)
    assert 0.0 in doc["method"]["writes"]["push"]["do"]["add_scaled"]["alpha"]["sweep"]


def test_single_site_document():
    doc = _doc(arm="real")
    assert set(doc["method"]["sites"]) == {"target", "lm_head"}
    assert doc["method"]["writes"]["push"]["site"] == "target"


def test_document_uses_evaluation_split():
    doc = _doc()
    dataset = doc["data"]["base"]["dataset"]
    assert dataset.endswith("evaluation")


def test_document_records_no_generation():
    assert DECODE_CONFIG["generation"] is False
    assert DECODE_CONFIG["seeded_sampling"] is False
    assert "deterministic" in DECODE_CONFIG["sampling"]


def test_document_saves_baseline_margin_for_pairing():
    """A paired preference change needs the baseline margin per prompt."""
    for arm, prefix in (("real", "steered"), ("control", "control")):
        doc = _doc(arm=arm)
        files = {s["file_path"] for s in doc["method"]["save"]}
        assert "baseline_margin.json" in files
        assert f"{prefix}_margin.json" in files


def test_real_arm_needs_no_control_file():
    config = SteeringConfig(model_key="org/name")
    doc = evaluation_document(
        config, layer=20, alphas=(-4.0, 0.0, 4.0), control_file=None, arm="real"
    )
    assert "controlled" not in doc["method"]["intervened_models"]


def test_rejects_grid_without_zero():
    config = SteeringConfig(model_key="org/name")
    with pytest.raises(ValueError, match="alpha=0"):
        evaluation_document(config, layer=4, alphas=(-2.0, 2.0), control_file=None)


def test_document_is_json_serializable():
    json.dumps(_doc())


# --------------------------------------------------------------------------- #
# run config stamping
# --------------------------------------------------------------------------- #
def test_run_config_records_decoding_and_splits(tmp_path):
    config = SteeringConfig(model_key="org/name")
    path = write_run_config(tmp_path / "run", config, 20, [-4.0, 0.0, 4.0])
    payload = json.loads(path.read_text())
    assert payload["decode_config"]["generation"] is False
    assert payload["relative_depth"] == pytest.approx(20 / 27)
    assert "evaluation" in payload["split_config"]
    assert payload["engine_config"]["component"] == "block_output"
    assert payload["alphas"] == [-4.0, 0.0, 4.0]


# --------------------------------------------------------------------------- #
# multiplicity correction
# --------------------------------------------------------------------------- #
def test_holm_is_monotone_and_capped():
    ps = [0.001, 0.01, 0.02, 0.5]
    adj = holm_bonferroni(ps)
    assert all(0 <= p <= 1 for p in adj)
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]
    assert adj[0] == pytest.approx(0.004)


def test_holm_never_shrinks_a_p_value():
    ps = [0.01, 0.02, 0.03]
    adj = holm_bonferroni(ps)
    for p, a in zip(ps, adj):
        assert a >= p - 1e-12


def test_holm_single_comparison_is_unchanged():
    assert holm_bonferroni([0.03])[0] == pytest.approx(0.03)


def test_holm_handles_empty():
    assert holm_bonferroni([]) == []