"""Build the construction / calibration / evaluation splits for the steering pilot.

Uses Causalab's own ``generate_split_dataset``, which the audit identified as
the right primitive: it partitions **unique inputs** into disjoint groups, forms
counterfactual pairs *within* each split, and makes the split a property of each
row so disjointness is checkable from the bytes.

Verified API facts (measured, not inferred)
--------------------------------------------
* ``generate_split_dataset`` returns a ``SplitDataset(examples, splits, audit)``
  where each ``examples`` entry is a **plain dict** with ``"input"`` and
  ``"counterfactual_inputs"`` keys, each a ``CausalTrace``. ``CausalTrace``
  raises ``KeyError`` on unknown names and must be indexed by name, never
  iterated (``sorted(trace)`` yields integer indices).
* Its ``generator=`` argument must name a ``generate_dataset(model, n, seed)``
  function. MCQA's bare generators (``different_symbol`` etc.) take no
  arguments and raise ``TypeError`` here; ``generate_dataset`` is the loader
  convention that matches.
* MCQA's input space is 6,760,000 combinations, so ``_input_pool`` takes the
  *generator* branch (its ``max_inputs`` cap) rather than enumeration.
* ``enumerate_inputs()`` is broken for MCQA -- ``choices.index(color)`` calls
  ``.index`` on a ``FamilyDom``, which is neither tuple nor iterable. We
  therefore always pass an explicit ``max_inputs`` and never rely on
  enumeration. This is an upstream bug, not something this script introduces;
  see the report.

On the confound the brief warns about: the MCQA generator resamples symbols per
pair, so a mean-difference direction over these pairs can encode symbol
frequency rather than the semantic contrast. ``assert_symbol_balance`` reports
the inventory so that risk is visible rather than assumed away.

Usage::

    python -m research.scripts.build_dataset --config research/configs/pilot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import SplitNames, SteeringConfig, load_config  # noqa: E402

__all__ = [
    "build_splits",
    "assert_no_leakage",
    "assert_symbol_balance",
    "write_splits",
]

#: Symbols per split is not balanced by construction (the MCQA generator draws
#: a fresh alphabet slice per pair), so this is a *report*, not an assertion.
#: The assertion that matters is prompt/group disjointness.


def _base_row(example: dict[str, Any]) -> Any:
    return example["input"]


def _trace_get(trace: Any, name: str, default: Any = "") -> Any:
    """Read one variable from a ``CausalTrace``, or ``default`` if absent.

    ``CausalTrace.get(variable)`` takes no default and raises ``KeyError`` on an
    unknown name (verified in ``causalab/causal/model.py:612``), so a
    dict-style ``.get(name, default)`` is not available.
    """
    try:
        return trace[name]
    except (KeyError, IndexError):
        return default


def build_splits(
    config: SteeringConfig,
    *,
    pool_size: int = 400,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Generate one group-disjoint table and split it three ways.

    Args:
        config: Validated pilot configuration.
        pool_size: Input pool size, passed as ``max_inputs`` to force the
            generator branch of ``_input_pool`` (enumeration is broken for
            MCQA; see module docstring).

    Returns:
        ``(grouped, audit)`` where ``grouped`` maps split name to a list of
        example dicts, and ``audit`` is Causalab's own provenance record.
    """
    from causalab.tasks.loader import load_task
    from causalab.tasks.splits import generate_split_dataset

    task = load_task("MCQA")
    split_dataset = generate_split_dataset(
        task,
        seed=config.seed,
        fractions=config.split_fractions_map,
        group_key=config.group_key,
        require_label_change=config.require_label_change,
        target_variable="answer",
        generator="generate_dataset",
        max_inputs=pool_size,
    )

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in config.split_names}
    for example, split in zip(
        split_dataset.examples, split_dataset.splits, strict=True
    ):
        grouped[split].append(example)
    return grouped, split_dataset.audit


def assert_no_leakage(
    grouped: dict[str, list[dict[str, Any]]], *, group_key: str = "color"
) -> dict[str, Any]:
    """Verify split disjointness from the rows themselves.

    Raises ``AssertionError`` if a base prompt appears in more than one split, or
    if the configured group variable spans splits. The second check is the one
    that matters: grouping by prompt alone is not a real holdout when prompts
    share content words, so whichever variable the split is keyed on must not
    cross a split boundary.
    """
    prompt_to_split: dict[str, str] = {}
    group_to_split: dict[str, str] = {}
    collisions: list[str] = []

    for split, examples in grouped.items():
        for example in examples:
            row = _base_row(example)
            prompt = str(row["raw_input"])
            if prompt in prompt_to_split and prompt_to_split[prompt] != split:
                collisions.append(f"prompt {prompt!r} in {prompt_to_split[prompt]}")
            prompt_to_split[prompt] = split

            if group_key == "input":
                continue
            group_value = str(_trace_get(row, group_key, default=""))
            if not group_value:
                continue
            if (
                group_value in group_to_split
                and group_to_split[group_value] != split
            ):
                collisions.append(
                    f"{group_key}={group_value!r} in {group_to_split[group_value]}"
                )
            group_to_split[group_value] = split

    if collisions:
        raise AssertionError(
            f"leakage between splits (grouped by {group_key!r}): "
            + "; ".join(sorted(set(collisions))[:5])
        )
    return {
        "group_key": group_key,
        "n_unique_prompts": len(prompt_to_split),
        "n_unique_groups": len(group_to_split),
        "rows_per_split": {k: len(v) for k, v in grouped.items()},
    }


def assert_symbol_balance(
    grouped: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Report the symbol inventory used to build the construction direction.

    Not an assertion: MCQA draws a fresh symbol pair per pair, so the inventory
    is deliberately varied. It is recorded because a direction built from an
    unbalanced symbol distribution risks encoding letter frequency -- the
    confound the design must not fall into.
    """
    construction = grouped.get(SplitNames.CONSTRUCTION, [])
    if len(construction) < 2:
        raise AssertionError(
            f"construction split has {len(construction)} rows; a mean difference "
            "needs at least 2, and in practice far more to average out prompt "
            "idiosyncrasies"
        )
    symbols: Counter[str] = Counter()
    for example in construction:
        row = _base_row(example)
        for key in ("symbols[0]", "symbols[1]"):
            value = _trace_get(row, key, default=None)
            if value is not None:
                symbols[str(value)] += 1
    return {
        "n_construction_rows": len(construction),
        "symbol_counts": dict(sorted(symbols.items())),
        "n_distinct_symbols": len(symbols),
    }


def write_splits(
    grouped: dict[str, list[dict[str, Any]]], config: SteeringConfig
) -> dict[str, Path]:
    """Write each split as the JSON table Causalab protocol documents read.

    The table is a list of rows; each row carries ``input``,
    ``counterfactual_inputs[0]``, ``base_answer``, ``cf_answer``, and ``split``.
    That is the schema the shipped demo protocols consume
    (``demos/onboarding_tutorial/protocols/mcqa_*.json``), so no new dataset
    loader is needed downstream.
    """
    written: dict[str, Path] = {}
    for split, examples in grouped.items():
        dataset_name = config.dataset_for[split]
        # Causalab resolves a dataset ref to ``{data_root}/{ref}.json``
        # (causalab/io/env.py:_file), so the nested name is a real directory.
        out_path = Path(config.data_root) / f"{dataset_name}.json"
        rows = [_serialize_row(example, split) for example in examples]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2) + "\n")
        written[split] = out_path
    return written


def _serialize_row(example: dict[str, Any], split: str) -> dict[str, Any]:
    """One table row, matching the demo dataset schema."""
    base = example["input"]
    cf = example["counterfactual_inputs"][0]
    row: dict[str, Any] = {
        "input": str(base["raw_input"]),
        "counterfactual_inputs[0]": str(cf["raw_input"]),
        "base_answer": str(base["raw_output"]),
        "cf_answer": str(cf["raw_output"]),
        "base_answer_forms": [str(base["raw_output"]), str(base["raw_output"]).strip()],
        "cf_answer_forms": [str(cf["raw_output"]), str(cf["raw_output"]).strip()],
        "split": split,
    }
    for key in ("object", "color", "choices[0]", "choices[1]", "symbols[0]", "symbols[1]"):
        value = _trace_get(base, key, default=None)
        if value is None:
            continue
        row[key] = str(value)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build steering pilot splits.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pool-size", type=int, default=400)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    grouped, audit = build_splits(config, pool_size=args.pool_size)
    leakage_report = assert_no_leakage(grouped, group_key=config.group_key)
    symbol_report = assert_symbol_balance(grouped)
    written = write_splits(grouped, config)

    report = {
        "config": config.to_dict(),
        "causalab_audit": audit,
        "leakage": leakage_report,
        "symbols": symbol_report,
        "files": {k: str(v) for k, v in written.items()},
    }
    report_path = Path(config.data_root) / "build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())