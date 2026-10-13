"""Unit tests for the results table: denominators, flip rate, and CIs.

These matter because a wrong denominator makes every downstream number
misleading, and the flip rate in particular is easy to get wrong: it is *not*
the target-answer rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.results_table import (  # noqa: E402
    ALPHA_KEY,
    bootstrap_ci_paired,
    build_table,
    wilson_ci,
)


# --------------------------------------------------------------------------- #
# interval helpers
# --------------------------------------------------------------------------- #
def test_wilson_ci_brackets_the_point_estimate():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_wilson_ci_at_zero_is_nonzero_width():
    """Wald collapses at 0; Wilson must not, or a null result looks certain."""
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_ci_at_full_is_below_one():
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0
    assert lo < 1.0


def test_wilson_ci_handles_empty():
    lo, hi = wilson_ci(0, 0)
    assert lo != lo and hi != hi  # both NaN


def test_bootstrap_ci_paired_is_deterministic():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.0, 1.0, 2.0, 3.0, 4.0]
    first = bootstrap_ci_paired(a, b, seed=7)
    second = bootstrap_ci_paired(a, b, seed=7)
    assert first == second


def test_bootstrap_ci_paired_is_zero_for_constant_shift():
    a = [2.0, 3.0, 4.0, 5.0, 6.0]
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, lo, hi = bootstrap_ci_paired(a, b)
    assert mean == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_paired_brackets_mean():
    a = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 2.0, 7.0]
    b = [0.0, 4.0, 1.0, 7.0, 2.0, 8.0, 1.0, 6.0]
    mean, lo, hi = bootstrap_ci_paired(a, b)
    assert lo <= mean <= hi


def test_bootstrap_pairs_are_smaller_than_unpaired_would_be():
    """A constant per-prompt offset has zero variance when paired."""
    a = [10.0, 20.0, 30.0, 40.0, 50.0]
    b = [9.0, 19.0, 29.0, 39.0, 49.0]
    _, lo, hi = bootstrap_ci_paired(a, b)
    assert (hi - lo) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# table construction
# --------------------------------------------------------------------------- #
def _write(path: Path, series: dict[float, dict[str, float]], metric: str) -> None:
    rows = []
    for alpha, per_id in series.items():
        for eid, value in per_id.items():
            rows.append(
                {
                    "example_id": eid,
                    "metric": metric,
                    "value": value,
                    ALPHA_KEY: alpha,
                    "unit": "fraction",
                    "eligible": True,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def _synthetic(tmp_path: Path) -> Path:
    """Four prompts, one positive alpha, with a known flip configuration."""
    ids = ["0", "1", "2", "3"]
    alpha = 4.0
    zero = 0.0

    # baseline correct on 0,1,2 and wrong on 3
    _write(
        tmp_path / "baseline_accuracy.json",
        {zero: {"0": 1.0, "1": 1.0, "2": 1.0, "3": 0.0}},
        "baseline_accuracy",
    )
    _write(
        tmp_path / "baseline_margin.json",
        {zero: {"0": 5.0, "1": 5.0, "2": 5.0, "3": -5.0}},
        "logit_diff",
    )
    # steered: keeps original on 0,1; flips 2 to cf; 3 unchanged
    _write(
        tmp_path / "steered_original_rate.json",
        {zero: dict(zip(ids, [1.0, 1.0, 0.0, 0.0])),
         alpha: {i: 1.0 for i in ids}},
        "match",
    )
    _write(
        tmp_path / "steered_cf_rate.json",
        {zero: {i: 0.0 for i in ids}, alpha: {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0}},
        "match",
    )
    _write(
        tmp_path / "steered_margin.json",
        {zero: {"0": 5.0, "1": 5.0, "2": 5.0, "3": -5.0},
         alpha: {"0": 5.0, "1": 5.0, "2": 0.0, "3": -5.0}},
        "logit_diff",
    )
    return tmp_path


def test_build_table_reports_required_columns(tmp_path):
    _synthetic(tmp_path)
    rows = build_table(tmp_path, layer=12)
    assert rows
    required = {
        "layer",
        "relative_depth",
        "alpha",
        "n_prompts",
        "baseline_accuracy",
        "target_answer_rate",
        "preference_change",
        "flip_rate",
        "output_validity",
        "pref_change_ci_lo",
        "pref_change_ci_hi",
    }
    assert required <= set(rows[0])


def test_flip_rate_is_not_target_rate(tmp_path):
    """The distinction the whole table depends on."""
    _synthetic(tmp_path)
    rows = {r["alpha"]: r for r in build_table(tmp_path, layer=12)}
    at_alpha = rows[4.0]
    # prompt 2 flipped; 4 prompts total
    assert at_alpha["flip_rate"] == pytest.approx(0.25)
    assert at_alpha["flip_count"] == 1
    # target rate is also 0.25 here, but computed from a different definition
    assert at_alpha["target_answer_rate"] == pytest.approx(0.25)
    # of the 3 prompts correct at baseline, 1 flipped
    assert at_alpha["n_correct_at_baseline"] == 3
    assert at_alpha["flip_rate_of_correct"] == pytest.approx(1 / 3)


def test_flip_rate_zero_when_baseline_all_wrong(tmp_path):
    run = tmp_path
    ids = ["0", "1", "2"]
    zero = 0.0
    alpha = 2.0
    _write(run / "baseline_accuracy.json", {zero: {i: 0.0 for i in ids}}, "m")
    _write(run / "baseline_margin.json", {zero: {i: -1.0 for i in ids}}, "m")
    _write(
        run / "steered_original_rate.json",
        {zero: {i: 0.0 for i in ids}, alpha: {i: 0.0 for i in ids}},
        "m",
    )
    # steering pushes every prompt to the counterfactual
    _write(run / "steered_cf_rate.json", {alpha: {i: 1.0 for i in ids}}, "m")
    _write(
        run / "steered_margin.json",
        {zero: {i: -1.0 for i in ids}, alpha: {i: -3.0 for i in ids}},
        "m",
    )
    rows = {r["alpha"]: r for r in build_table(run, layer=4)}
    assert rows[alpha]["target_answer_rate"] == pytest.approx(1.0)
    # but nothing "flipped" because nothing was right to begin with
    assert rows[alpha]["flip_rate"] == pytest.approx(0.0)
    assert rows[alpha]["flip_rate_of_correct"] is None


def test_preference_change_is_paired_delta(tmp_path):
    _synthetic(tmp_path)
    rows = {r["alpha"]: r for r in build_table(tmp_path, layer=12)}
    # margins: baseline [5,5,5,-5], steered@4 [5,5,0,-5] -> mean delta -5/4
    assert rows[4.0]["preference_change"] == pytest.approx(-1.25)


def test_relative_depth_is_fraction_of_depth(tmp_path):
    _synthetic(tmp_path)
    rows = build_table(tmp_path, layer=27, n_layers=28)
    assert rows[0]["relative_depth"] == pytest.approx(1.0)
    rows = build_table(tmp_path, layer=0, n_layers=28)
    assert rows[0]["relative_depth"] == pytest.approx(0.0)


def test_output_validity_reflects_ineligible_rows(tmp_path):
    run = _synthetic(tmp_path)
    path = run / "steered_original_rate.json"
    rows = json.loads(path.read_text())
    for r in rows:
        if r[ALPHA_KEY] == 4.0 and r["example_id"] == "0":
            r["eligible"] = False
    path.write_text(json.dumps(rows))
    table = {r["alpha"]: r for r in build_table(run, layer=12)}
    assert table[4.0]["output_validity"] < 1.0
    assert table[4.0]["n_ineligible"] >= 1


def test_missing_control_yields_real_arm_only(tmp_path):
    _synthetic(tmp_path)
    rows = build_table(tmp_path, layer=12)
    assert {r["arm"] for r in rows} == {"real"}


def test_build_table_empty_on_missing_run(tmp_path):
    assert build_table(tmp_path / "nope", layer=4) == []

# --------------------------------------------------------------------------- #
# regression: the baseline is duplicated at every swept alpha
# --------------------------------------------------------------------------- #
def test_zero_alpha_is_not_misbounded(tmp_path):
    """Regression guard for a real bug.

    The baseline table repeats identically at every swept alpha, so its alpha
    coordinate is a label. Keying the reference cell on ``min()`` picked the most
    *negative* alpha, which mislabelled the arm's own alpha=0 row and made the
    zero-strength control look violated by up to 0.53 logits.

    Here the negative-alpha cells deliberately carry a very different value, so
    a min()-based lookup produces a visibly wrong zero row.
    """
    ids = ["0", "1", "2", "3"]
    grid = (-8.0, -4.0, 0.0, 4.0, 8.0)

    # baseline: identical at every alpha
    _write(
        tmp_path / "baseline_accuracy.json",
        {a: {i: 1.0 for i in ids} for a in grid},
        "m",
    )
    _write(
        tmp_path / "baseline_margin.json",
        {a: {i: 4.0 for i in ids} for a in grid},
        "m",
    )
    # arm: alpha=0 matches baseline exactly; negative alphas are very different
    steered_orig = {a: {i: 1.0 for i in ids} for a in grid}
    steered_cf = {a: {i: 0.0 for i in ids} for a in grid}
    steered_marg = {a: {i: 4.0 for i in ids} for a in grid}
    for a in (-8.0, -4.0):
        steered_orig[a] = {i: 0.0 for i in ids}
        steered_marg[a] = {i: -9.0 for i in ids}
    _write(tmp_path / "steered_original_rate.json", steered_orig, "m")
    _write(tmp_path / "steered_cf_rate.json", steered_cf, "m")
    _write(tmp_path / "steered_margin.json", steered_marg, "m")

    rows = {r["alpha"]: r for r in build_table(tmp_path, layer=8)}
    # the alpha=0 row must be an exact no-op against the baseline
    assert rows[0.0]["preference_change"] == pytest.approx(0.0)
    assert rows[0.0]["original_answer_rate"] == pytest.approx(1.0)
    # and it must not have inherited the -8.0 cell's behaviour
    assert rows[-8.0]["preference_change"] == pytest.approx(-13.0)
