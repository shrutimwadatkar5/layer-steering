"""Unit tests for transfer result aggregation and cell classification.

The classifier decides what a cell *means*, so its ordering and thresholds are
pinned here rather than left implicit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.analyze_transfer import (  # noqa: E402
    ALPHA_KEY,
    OUTCOMES,
    classify_cell,
    collect_cells,
    heatmap_matrix,
    summary,
)

HIDDEN = 8


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def _series_table(values: dict[float, dict[str, float]]) -> list[dict]:
    out = []
    for alpha, per_id in values.items():
        for eid, value in per_id.items():
            out.append(
                {
                    "example_id": eid,
                    "metric": "m",
                    "value": value,
                    ALPHA_KEY: alpha,
                    "eligible": True,
                    "unit": "fraction",
                }
            )
    return out


def _make_run(
    run: Path,
    *,
    source: int,
    target: int,
    effect: float,
    orig_rate: float = 1.0,
    cf_rate: float = 0.0,
    baseline_acc: float = 1.0,
    strength: float = 8.0,
    zero_delta: float = 0.0,
):
    """Write a complete transfer run directory."""
    run.mkdir(parents=True, exist_ok=True)
    ids = [str(i) for i in range(10)]
    base_margin = {i: 4.0 for i in ids}
    transfer_margin = {i: 4.0 + effect for i in ids}
    _write(
        run / "baseline_margin.json",
        _series_table({0.0: base_margin, strength: dict(base_margin)}),
    )
    _write(
        run / "transfer_margin.json",
        _series_table({0.0: {i: 4.0 + zero_delta for i in ids},
                       strength: transfer_margin}),
    )
    _write(
        run / "transfer_original_rate.json",
        _series_table({0.0: {i: 1.0 for i in ids},
                       strength: {i: orig_rate for i in ids}}),
    )
    _write(
        run / "transfer_cf_rate.json",
        _series_table({0.0: {i: 0.0 for i in ids},
                       strength: {i: cf_rate for i in ids}}),
    )
    _write(
        run / "baseline_accuracy.json",
        _series_table({0.0: {i: baseline_acc for i in ids},
                       strength: {i: baseline_acc for i in ids}}),
    )
    (run / "transfer_config.json").write_text(
        json.dumps(
            {
                "source_layer": source,
                "target_layer": target,
                "split": "calibration",
                "sigma_table": {"sigma": 1.5},
            }
        )
    )


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def test_collect_cells_indexes_source_and_target(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5)
    cells = collect_cells(tmp_path)
    assert len(cells) == 1
    assert cells[0]["source_layer"] == 4
    assert cells[0]["target_layer"] == 20
    assert cells[0]["same_layer"] is False


def test_collect_marks_the_diagonal(tmp_path):
    _make_run(tmp_path / "s20_t20", source=20, target=20, effect=-0.5)
    cells = collect_cells(tmp_path)
    assert cells[0]["same_layer"] is True


def test_collect_records_sigma_from_the_config(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.1)
    assert collect_cells(tmp_path)[0]["sigma_ij"] == 1.5


def test_collect_skips_run_without_config(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.1)
    (tmp_path / "s04_t20" / "transfer_config.json").unlink()
    assert collect_cells(tmp_path) == []


def test_collect_indexes_by_z_not_raw_alpha(tmp_path):
    """Cells are calibrated independently, so raw alphas differ at equal z.

    Two cells with different sigma_ij must both be selectable at z = +8 even
    though their raw alphas are unrelated.
    """
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5, strength=8.0)
    run2 = tmp_path / "s20_t04"
    # sigma 5.0 -> to reach the same z the raw alpha must be 5x larger
    _make_run(run2, source=20, target=4, effect=-0.2, strength=40.0)
    cfg = json.loads((run2 / "transfer_config.json").read_text())
    cfg["sigma_table"]["sigma"] = 5.0
    (run2 / "transfer_config.json").write_text(json.dumps(cfg))

    # cell 1: sigma 1.5, so z = 8 needs alpha = 12
    cfg1 = json.loads((tmp_path / "s04_t20" / "transfer_config.json").read_text())
    cfg1["sigma_table"]["sigma"] = 1.5
    (tmp_path / "s04_t20" / "transfer_config.json").write_text(json.dumps(cfg1))
    _rewrite_alpha(tmp_path / "s04_t20", 12.0)

    cells = collect_cells(tmp_path, z=8.0)
    assert len(cells) == 2, [c["alpha"] for c in cells]
    assert {round(c["z"], 3) for c in cells} == {8.0}
    # and the raw alphas are genuinely different
    assert len({c["alpha"] for c in cells}) == 2


def _rewrite_alpha(run: Path, alpha: float) -> None:
    """Replace the strongest-strength alpha inside a synthetic run."""
    for name in (
        "transfer_margin.json",
        "transfer_original_rate.json",
        "transfer_cf_rate.json",
        "baseline_margin.json",
        "baseline_accuracy.json",
    ):
        rows = json.loads((run / name).read_text())
        for row in rows:
            if row[ALPHA_KEY] == 8.0:
                row[ALPHA_KEY] = alpha
        (run / name).write_text(json.dumps(rows))


def test_collect_picks_the_strongest_z_by_default(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5, strength=8.0)
    assert collect_cells(tmp_path)[0]["z"] == pytest.approx(8.0 / 1.5, rel=1e-3)


def test_collect_records_zero_strength_delta(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5, zero_delta=0.25)
    assert collect_cells(tmp_path)[0]["zero_strength_delta"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def _cell(**kw):
    base = {
        "preference_change": -0.5,
        "ci_lo": -0.6,
        "ci_hi": -0.4,
        "significant": True,
        "accuracy_change": 0.0,
        "target_answer_rate": 0.0,
        "same_layer": False,
    }
    base.update(kw)
    return base


def test_successful_transfer_needs_half_the_reference():
    ref = _cell(preference_change=-1.0)
    assert (
        classify_cell(_cell(preference_change=-0.8), ref) == "successful_transfer"
    )
    assert (
        classify_cell(_cell(preference_change=-0.4), ref) == "inconclusive"
    )


def test_significant_but_weak_is_inconclusive_not_success():
    ref = _cell(preference_change=-1.0)
    assert classify_cell(_cell(preference_change=-0.2), ref) == "inconclusive"


def test_degradation_outranks_success():
    """A cell can be significant AND disruptive; report the disruption."""
    ref = _cell(preference_change=-1.0)
    cell = _cell(preference_change=-0.9, accuracy_change=-0.4, target_answer_rate=0.0)
    assert classify_cell(cell, ref) == "intervention_induced_degradation"


def test_degradation_not_reported_when_target_rate_rises():
    ref = _cell(preference_change=-1.0)
    cell = _cell(
        preference_change=-0.9, accuracy_change=-0.4, target_answer_rate=0.2
    )
    assert classify_cell(cell, ref) != "intervention_induced_degradation"


def test_non_significant_is_failed_transfer():
    cell = _cell(ci_lo=-0.1, ci_hi=0.1, significant=False)
    assert classify_cell(cell, None) == "failed_transfer_possible_misalignment"


def test_every_outcome_is_a_defined_key():
    assert set(OUTCOMES) == {
        "successful_transfer",
        "failed_transfer_possible_misalignment",
        "intervention_induced_degradation",
        "inconclusive",
    }


def test_failed_transfer_language_is_cautious():
    text = OUTCOMES["failed_transfer_possible_misalignment"]
    assert "NOT evidence" in text
    assert "absent" not in text.lower().replace("not evidence the feature is absent", "")


# --------------------------------------------------------------------------- #
# summary + heatmap
# --------------------------------------------------------------------------- #
def test_summary_separates_same_and_cross_layer(tmp_path):
    _make_run(tmp_path / "s20_t20", source=20, target=20, effect=-0.5)
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5)
    cells = collect_cells(tmp_path)
    report = summary(cells)
    assert report["same_layer"]["n"] == 1
    assert report["cross_layer"]["n"] == 1
    assert report["not_pooled_with_layerwise"] is True


def test_heatmap_matrix_is_source_by_target(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5)
    _make_run(tmp_path / "s20_t04", source=20, target=4, effect=-0.1)
    matrix = heatmap_matrix(collect_cells(tmp_path), "preference_change")
    assert matrix["4"]["20"] == pytest.approx(-0.5, abs=1e-3)
    assert matrix["20"]["4"] == pytest.approx(-0.1, abs=1e-3)


def test_summary_reports_zero_strength_deviation(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5, zero_delta=0.0)
    report = summary(collect_cells(tmp_path))
    assert report["zero_strength_max_abs_delta"] == pytest.approx(0.0, abs=1e-6)


def test_summary_carries_interpretation_caveats(tmp_path):
    _make_run(tmp_path / "s04_t20", source=4, target=20, effect=-0.5)
    report = summary(collect_cells(tmp_path))
    joined = " ".join(report["interpretation_caveats"])
    assert "does NOT" in joined or "NOT show" in joined