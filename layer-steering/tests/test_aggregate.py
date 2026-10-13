"""Unit tests for result aggregation, zero-strength checking, and leakage guards.

These cover the compact-result pipeline (requirement H) and the leakage
assertions (the brief's data-leakage requirement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.scripts.aggregate import (  # noqa: E402
    aggregate,
    load_table,
    zero_strength_check,
)
from research.config import SplitNames  # noqa: E402
from research.scripts.build_dataset import (  # noqa: E402
    assert_no_leakage,
    assert_symbol_balance,
)
from research.config import SteeringConfig  # noqa: E402


def _write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def _steer_row(example_id: str, layer: int, alpha: float, value: float) -> dict:
    return {
        "example_id": example_id,
        "metric": "original_answer_rate",
        "value": value,
        "sites.target.layers": layer,
        "writes.push.do.add_scaled.alpha": alpha,
        "unit": "fraction",
        "eligible": True,
    }


# --------------------------------------------------------------------------- #
# table loading
# --------------------------------------------------------------------------- #
def test_load_table_missing_file_is_empty_not_error(tmp_path):
    assert load_table(tmp_path / "nope.json") == []


def test_load_table_reads_a_list(tmp_path):
    path = tmp_path / "t.json"
    _write_table(path, [{"value": 1.0}])
    assert load_table(path) == [{"value": 1.0}]


def test_load_table_wraps_a_single_object(tmp_path):
    path = tmp_path / "t.json"
    _write_table(path, {"value": 0.5})
    assert load_table(path) == [{"value": 0.5}]


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _synthetic_run_dir(tmp_path: Path) -> Path:
    """A two-layer, three-alpha run with a well-formed zero-strength cell."""
    layers = [4, 12]
    alphas = [-1.0, 0.0, 1.0]
    ids = ["0", "1", "2", "3"]

    original, cf, baseline, margin = [], [], [], []
    for layer in layers:
        for alpha in alphas:
            for i, eid in enumerate(ids):
                value = 1.0 if alpha == 0.0 else (1.0 if alpha > 0 else 0.0)
                original.append(_steer_row(eid, layer, alpha, value))
                cf.append(_steer_row(eid, layer, alpha, 0.0))
                baseline.append(
                    {
                        "example_id": eid,
                        "metric": "baseline_accuracy",
                        "value": 1.0,
                        "sites.target.layers": layer,
                        "writes.push.do.add_scaled.alpha": alpha,
                        "unit": "fraction",
                    }
                )
                margin.append(
                    {
                        "example_id": eid,
                        "metric": "logit_margin",
                        "value": 2.0 - alpha,
                        "sites.target.layers": layer,
                        "writes.push.do.add_scaled.alpha": alpha,
                        "unit": "logit",
                    }
                )
    _write_table(tmp_path / "original_answer_rate.json", original)
    _write_table(tmp_path / "cf_answer_rate.json", cf)
    _write_table(tmp_path / "baseline_accuracy.json", baseline)
    _write_table(tmp_path / "logit_margin.json", margin)
    return tmp_path


def test_aggregate_emits_one_row_per_layer_alpha_cell(tmp_path):
    rows = aggregate(_synthetic_run_dir(tmp_path), SteeringConfig())
    assert len(rows) == 6  # 2 layers x 3 alphas
    keys = {(r["layer"], r["alpha"]) for r in rows}
    assert keys == {(l, a) for l in (4, 12) for a in (-1.0, 0.0, 1.0)}


def test_aggregate_records_denominator_n_eval(tmp_path):
    rows = aggregate(_synthetic_run_dir(tmp_path), SteeringConfig())
    assert all(r["n_eval"] == 4 for r in rows)


def test_aggregate_computes_original_answer_delta(tmp_path):
    rows = aggregate(_synthetic_run_dir(tmp_path), SteeringConfig())
    by_key = {(r["layer"], r["alpha"]): r for r in rows}
    # at alpha=0 steered == baseline, so delta is 0
    assert by_key[(4, 0.0)]["original_answer_delta"] == 0.0
    # at alpha=+1 steered=1.0, baseline=1.0 -> delta 0
    assert by_key[(4, 1.0)]["original_answer_delta"] == 0.0
    # at alpha=-1 steered=0.0, baseline=1.0 -> delta -1
    assert by_key[(4, -1.0)]["original_answer_delta"] == -1.0


def test_aggregate_ignores_rows_without_coordinates(tmp_path):
    run_dir = tmp_path
    _write_table(
        run_dir / "original_answer_rate.json",
        [
            _steer_row("0", 4, 0.0, 1.0),
            {"example_id": "9", "value": 1.0},  # no sweep coords
        ],
    )
    _write_table(
        run_dir / "baseline_accuracy.json",
        [
            {
                "example_id": "0",
                "metric": "baseline_accuracy",
                "value": 1.0,
                "sites.target.layers": 4,
                "writes.push.do.add_scaled.alpha": 0.0,
                "unit": "fraction",
            }
        ],
    )
    rows = aggregate(run_dir, SteeringConfig())
    # the coordinate-less row is dropped, leaving exactly one (layer, alpha) cell
    assert len(rows) == 1
    assert rows[0]["layer"] == 4
    assert rows[0]["original_answer_rate_n"] == 1
    assert rows[0]["n_eval"] == 1


# --------------------------------------------------------------------------- #
# zero-strength check
# --------------------------------------------------------------------------- #
def test_zero_strength_check_passes_when_clean(tmp_path):
    rows = aggregate(_synthetic_run_dir(tmp_path), SteeringConfig())
    check = zero_strength_check(rows)
    assert check["ok"] is True
    assert check["max_abs_deviation"] == 0.0
    assert check["n_layers_checked"] == 2


def test_zero_strength_check_fails_on_deviation():
    rows = [
        {
            "layer": 4,
            "alpha": 0.0,
            "baseline_accuracy": 1.0,
            "original_answer_rate": 0.5,
        }
    ]
    check = zero_strength_check(rows)
    assert check["ok"] is False
    assert check["max_abs_deviation"] == 0.5


def test_zero_strength_check_reports_missing_zero():
    rows = [{"layer": 4, "alpha": 1.0, "baseline_accuracy": 1.0, "original_answer_rate": 1.0}]
    check = zero_strength_check(rows)
    assert check["ok"] is False
    assert "no alpha=0" in check["reason"]


# --------------------------------------------------------------------------- #
# leakage guards
# --------------------------------------------------------------------------- #
def _example(prompt: str, obj: str) -> dict:
    return {"input": {"raw_input": prompt, "object": obj}}


def test_no_leakage_passes_on_disjoint_splits():
    grouped = {
        "construction": [_example("p1", "cup"), _example("p2", "car")],
        "calibration": [_example("p3", "hat")],
        "evaluation": [_example("p4", "bag")],
    }
    report = assert_no_leakage(grouped)
    assert report["n_unique_prompts"] == 4
    assert report["rows_per_split"] == {
        "construction": 2,
        "calibration": 1,
        "evaluation": 1,
    }


def test_no_leakage_detects_shared_prompt():
    grouped = {
        "construction": [_example("p1", "cup")],
        "evaluation": [_example("p1", "bag")],
    }
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(grouped)


def test_no_leakage_detects_shared_group_value():
    """Two distinct prompts carrying the same group value still leak.

    This is the check that rejected ``group_key="input"`` for MCQA: prompts
    differ, but the colour word they share would be visible to a direction
    built on one split and tested on the other.
    """
    grouped = {
        "construction": [{"input": {"raw_input": "p1", "color": "red"}}],
        "evaluation": [{"input": {"raw_input": "p2", "color": "red"}}],
    }
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(grouped, group_key="color")


def test_no_leakage_ignores_group_value_when_grouping_by_input():
    grouped = {
        "construction": [{"input": {"raw_input": "p1", "color": "red"}}],
        "evaluation": [{"input": {"raw_input": "p2", "color": "red"}}],
    }
    report = assert_no_leakage(grouped, group_key="input")
    assert report["n_unique_groups"] == 0
    assert report["n_unique_prompts"] == 2


def test_example_helper_supports_group_value():
    row = {"input": {"raw_input": "p1", "object": "cup", "color": "red"}}
    grouped = {"construction": [row], "evaluation": [dict(row)]}
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(grouped, group_key="object")


def test_symbol_balance_reports_inventory():
    grouped = {
        "construction": [
            {"input": {"raw_input": "a", "symbols[0]": "A", "symbols[1]": "B"}},
            {"input": {"raw_input": "b", "symbols[0]": "C", "symbols[1]": "D"}},
        ]
    }
    report = assert_symbol_balance(grouped)
    assert report["n_construction_rows"] == 2
    assert report["symbol_counts"] == {"A": 1, "B": 1, "C": 1, "D": 1}
    assert report["n_distinct_symbols"] == 4


def test_symbol_balance_requires_construction_rows():
    """A construction split too small to form a difference must fail loudly."""
    with pytest.raises(AssertionError, match="needs at least 2"):
        assert_symbol_balance({"construction": [{"input": {"raw_input": "a"}}]})