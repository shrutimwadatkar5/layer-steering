"""Held-out evaluation protocol: real direction and control direction together.

This is the first and only time the evaluation split is used. Everything
upstream — direction construction, strength calibration, layer selection — ran
on the construction and calibration splits, so the configuration is frozen here
and the evaluation split stays a genuine held-out set.

Why the control direction is in the same document
-------------------------------------------------
A large steering effect is only attributable to the *direction* if a
same-norm random direction does something different. The control is therefore
matched to the real direction's norm (see ``build_directions.py``), which is
what makes the comparison fair: any effect both share is attributable to
perturbing the residual stream by that much, not to what the direction encodes.

Both arms share the layer, the token position, the strength grid, the prompts,
and the model, and are swept over the *same* alphas, so the comparison is paired
by construction.

Decoding configuration (requirement 9)
--------------------------------------
This experiment scores ``match`` and ``logit_diff`` against a read at
``lm_head``, position ``-1``. No tokens are generated, so there is no sampling:
every value is an exact argmax comparison and therefore deterministic given the
weights. ``decode_config`` below is recorded into every results directory by
``write_run_config`` so a reader never has to infer it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["evaluation_document", "write_run_config", "DECODE_CONFIG"]

#: Recorded verbatim into every evaluation run directory. There is no sampling
#: and no generation: the metrics compare logits at one position.
DECODE_CONFIG: dict[str, Any] = {
    "task": "multiple_choice_2way",
    "generation": False,
    "decoding": "none (scoring only)",
    "sampling": "deterministic — no tokens generated",
    "read_position": -1,
    "read_component": "lm_head",
    "metrics": ["match", "logit_diff"],
    "greedy": True,
    "seeded_sampling": False,
    "note": (
        "match and logit_diff are exact comparisons of logits at position -1, "
        "so results are a deterministic function of the weights and the prompt. "
        "A repeat run reproduced logits to 0.0."
    ),
}


def evaluation_document(
    config: Any,
    *,
    layer: int,
    alphas: tuple[float, ...],
    control_file: str | None = None,
    real_file: str | None = None,
    arm: str = "real",
) -> dict[str, Any]:
    """One held-out document for ONE arm.

    The two arms are **separate documents by necessity, not by preference**.

    A first attempt put the real and control writes in one document with two
    sites (``target`` and ``target_control``) both resolving to
    ``block_output`` at the same layer. That validated and ran, but the control
    arm's alpha sweep silently never bound: its margin was *identical* at every
    strength (8.6353 for alpha = -8 .. +8) while the real arm varied correctly.
    Two writes targeting one module collide on the hook, and the first site's
    write wins for both.

    So each arm gets its own document, with a single site and a single write.
    The cost is one extra model load per layer; the benefit is that both arms
    receive their own swept alpha, which is the entire point of a control.

    Args:
        config: a validated :class:`research.config.SteeringConfig`.
        layer: the intervention layer (single, not swept — one document each,
            so a failure is isolated and the run is resumable).
        alphas: calibrated raw strengths; must contain ``0.0``.
        control_file: path to the norm-matched random control direction.
        real_file: path to the real direction; defaults to
            ``research/directions_dense/v_L{layer}.safetensors``.
        arm: ``"real"`` or ``"control"``.

    Returns:
        A Causalab protocol document for that arm.
    """
    if arm not in ("real", "control"):
        raise ValueError(f"arm must be 'real' or 'control', got {arm!r}")
    if arm == "control" and control_file is None:
        raise ValueError("arm='control' requires control_file")
    dataset = config.dataset_for["evaluation"]
    strength = [float(a) for a in alphas]
    if 0.0 not in strength:
        raise ValueError("evaluation grid must include alpha=0.0")

    if arm == "real":
        direction = real_file or f"research/directions_dense/v_L{layer:02d}.safetensors"
        arm_name, param_name, model_name = "steered", "steer", "steered"
    else:
        direction = control_file
        arm_name, param_name, model_name = "control", "control", "controlled"

    return {
        "header": {
            "protocol_version": "4",
            "description": (
                f"HELD-OUT evaluation, arm={arm}, layer {layer}: baseline plus "
                f"steering at strengths {strength}. Evaluated once, on the "
                f"evaluation split only. Scoring only, no generation. "
                f"{DECODE_CONFIG['note']}"
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
                model_name: {
                    "input": "base",
                    "reads": ["logits"],
                    "writes": ["push"],
                },
            },
            "positions": {"slot": dict(config.token_position)},
            "sites": {
                "target": {"component": "block_output", "layers": [layer]},
                "lm_head": {"component": "lm_head"},
            },
            "params": {
                param_name: {
                    "file_path": direction,
                    "entry": {"slot": "weight"},
                }
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
                            "op": param_name,
                            "alpha": {"sweep": strength},
                        }
                    },
                }
            },
            "save": [
                {
                    "read": "logits",
                    "model": model_name,
                    "aggregation": {"kind": "match", "expected": "base_answer"},
                    "file_path": f"{arm_name}_original_rate.json",
                },
                {
                    "read": "logits",
                    "model": model_name,
                    "aggregation": {"kind": "match", "expected": "cf_answer"},
                    "file_path": f"{arm_name}_cf_rate.json",
                },
                {
                    "read": "logits",
                    "model": model_name,
                    "aggregation": {
                        "kind": "logit_diff",
                        "a": "base_answer",
                        "b": "cf_answer",
                    },
                    "file_path": f"{arm_name}_margin.json",
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


def write_run_config(out_dir: Path, config: Any, layer: int, alphas, **extra) -> Path:
    """Stamp the full configuration into a results directory (requirement 9/10)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer": int(layer),
        "relative_depth": int(layer) / 27.0,
        "n_layers": 28,
        "alphas": [float(a) for a in alphas],
        "decode_config": DECODE_CONFIG,
        "engine_config": {
            "model_key": config.model_key,
            "revision": config.model_revision,
            "dtype": config.dtype,
            "token_position": dict(config.token_position),
            "component": "block_output",
            "seed": config.seed,
        },
        "split_config": {
            "construction": config.construction_dataset,
            "calibration": config.calibration_dataset,
            "evaluation": config.evaluation_dataset,
            "group_key": config.group_key,
            "note": (
                "direction built on construction only; strengths calibrated on "
                "calibration only; these results are the evaluation split, run once"
            ),
        },
    }
    payload.update(extra)
    path = out_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path