"""Pilot configuration for the layer-dependent activation-steering experiment.

Single source of truth for every knob the pilot exposes: model identifier,
dataset paths, seed, split assignment, layer selection, token position,
direction normalization, intervention strength, and output directory.

Every field is validated in :meth:`SteeringConfig.validate`, which is called by
``__post_init__`` so an invalid configuration can never reach a runner.

Design constraints taken from the repository audit:

* The pilot is deliberately **not** a full layer sweep. ``PILOT_LAYERS`` is a
  small hand-picked set spanning early / middle / late depth, and the config
  refuses a layer list that looks like a full sweep unless explicitly allowed.
* Direction normalization is **on by default**. Residual-stream norms grow
  substantially with depth, so an un-normalized direction makes raw ``alpha``
  incomparable across layers. Normalizing makes ``alpha`` mean "fraction of
  direction norm added".
* The zero-strength control is always included in ``ALPHAS``; dropping it is an
  error, because alpha=0 is the invariance check the whole design rests on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = [
    "PILOT_LAYERS",
    "DEFAULT_ALPHAS",
    "SplitNames",
    "SteeringConfig",
    "load_config",
]

#: Early / middle / late depths for a 28-layer model. Deliberately small:
#: this is a pilot, and the audit's staged plan puts the full sweep later.
PILOT_LAYERS: tuple[int, ...] = (4, 12, 20, 26)

#: Normalized steering strengths. Symmetric around zero so the negative arm is
#: genuinely comparable to the positive one (hypothesis H2). ``0.0`` is the
#: zero-strength control and must always be present.
DEFAULT_ALPHAS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


class SplitNames:
    """Split names carried as a *property of each row* (see the audit).

    Causalab's ``tasks/splits.py`` deliberately makes a split a row attribute so
    disjointness is checkable from the bytes rather than asserted in prose.
    """

    CONSTRUCTION = "construction"
    CALIBRATION = "calibration"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class SteeringConfig:
    """Immutable pilot configuration.

    Attributes:
        model_key: Hugging Face model identifier. Must be in Causalab's model
            registry or ``--register-from-hf`` must be passed.
        data_root: Directory Causalab resolves dataset names against.
        construction_dataset / calibration_dataset / evaluation_dataset: Dataset
            names, relative to ``data_root``. All three must come from the same
            generated table so the audit's leakage checks hold.
        seed: Master seed. Fixes split assignment, pair formation, and the
            random control direction.
        layers: Layers to steer at. Each is validated against the model's
            layer count at runtime, not here, because the config is
            model-agnostic.
        token_position: The position spec, in Causalab's document vocabulary.
            ``{"index": -1}`` is the final prompt token. Causalab resolves a
            negative index against each row's *first real* token, so this is
            padding-correct even in a ragged batch (verified: see README).
        normalize_direction: Divide the mean-difference direction by its L2
            norm. Strongly recommended; see module docstring.
        alphas: Normalized intervention strengths. Must contain 0.0.
        output_dir: Where raw and tabular results are written.
        require_label_change: Drop construction/calibration pairs whose
            counterfactual does not change the answer label.
        group_key: Split grouping strategy. ``"input"`` holds out by prompt;
            an input variable name holds out by that variable's value. The
            pilot defaults to ``"color"``: holding out colour words forces the
            steering direction to generalize across unseen content words rather
            than memorizing them, which is what hypothesis H3 asks. MCQA has
            only 10 colours, so this yields 10 groups split 5/2/3 -- coarse,
            but a real holdout. ``"input"`` (group by prompt) is too weak here:
            with 10 objects, prompts about the same object span all three
            splits, which the leakage assertion correctly rejects.
        allow_full_sweep: Escape hatch permitting a layer list spanning the
            whole model. Off by default per the audit's staged plan.
    """

    model_key: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "main"
    dtype: str = "bf16"

    data_root: Path = Path("research/data")
    construction_dataset: str = "mcqa_steering/construction"
    calibration_dataset: str = "mcqa_steering/calibration"
    evaluation_dataset: str = "mcqa_steering/evaluation"

    seed: int = 42

    layers: tuple[int, ...] = PILOT_LAYERS
    token_position: dict[str, Any] = field(default_factory=lambda: {"index": -1})

    normalize_direction: bool = True
    alphas: tuple[float, ...] = DEFAULT_ALPHAS

    output_dir: Path = Path("research/results")
    artifacts_dir: Path | None = None

    require_label_change: bool = True
    group_key: str = "color"
    split_fractions: tuple[float, ...] = (0.5, 0.2, 0.3)

    allow_full_sweep: bool = False

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Raise ``ValueError`` on any invalid field. Called on construction."""
        if not self.model_key or "/" not in self.model_key:
            raise ValueError(
                f"model_key must be a Hugging Face id like 'org/name', got "
                f"{self.model_key!r}"
            )
        if self.dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError(
                f"dtype must be one of fp32/bf16/fp16, got {self.dtype!r}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        if not self.layers:
            raise ValueError("layers must not be empty")
        if len(set(self.layers)) != len(self.layers):
            dupes = sorted({l for l in self.layers if list(self.layers).count(l) > 1})
            raise ValueError(f"layers contains duplicates: {dupes}")
        if any(l < 0 for l in self.layers):
            raise ValueError(f"layers must be non-negative, got {self.layers}")
        if list(self.layers) != sorted(self.layers):
            raise ValueError(
                f"layers must be sorted ascending for a readable sweep axis, got "
                f"{self.layers}"
            )
        if not self.allow_full_sweep:
            span = max(self.layers) - min(self.layers) + 1
            if len(self.layers) > 8 or span > 24:
                raise ValueError(
                    f"layers={self.layers} looks like a full sweep ({len(self.layers)} "
                    f"layers spanning {span}). The audit's staged plan requires a "
                    "small pilot first. Pass allow_full_sweep=True to override."
                )

        if not self.alphas:
            raise ValueError("alphas must not be empty")
        if 0.0 not in self.alphas:
            raise ValueError(
                "alphas must contain 0.0: the zero-strength control is the "
                "invariance check the whole design rests on"
            )
        if any(not isinstance(a, (int, float)) for a in self.alphas):
            raise ValueError(f"alphas must all be numeric, got {self.alphas}")
        if list(self.alphas) != sorted(self.alphas):
            raise ValueError(
                f"alphas must be sorted ascending, got {self.alphas}"
            )

        pos = self.token_position
        if not isinstance(pos, dict) or len(pos) != 1:
            raise ValueError(
                f"token_position must be a single-key Causalab position spec, got "
                f"{pos!r}"
            )
        key = next(iter(pos))
        allowed = {"index", "variable", "column", "span", "generated"}
        if key not in allowed:
            raise ValueError(
                f"token_position key {key!r} is not one of {sorted(allowed)}"
            )

        if len(self.split_fractions) != 3:
            raise ValueError(
                "split_fractions must have three entries "
                f"(construction, calibration, evaluation), got {self.split_fractions}"
            )
        if any(f <= 0 for f in self.split_fractions):
            raise ValueError(
                f"every split fraction must be positive, got {self.split_fractions}"
            )
        total = sum(self.split_fractions)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"split_fractions must sum to 1.0, got {total} from "
                f"{self.split_fractions}"
            )

    # ------------------------------------------------------------------ #
    # derived views
    # ------------------------------------------------------------------ #
    @property
    def split_names(self) -> tuple[str, ...]:
        return (
            SplitNames.CONSTRUCTION,
            SplitNames.CALIBRATION,
            SplitNames.EVALUATION,
        )

    @property
    def split_fractions_map(self) -> dict[str, float]:
        return dict(zip(self.split_names, self.split_fractions, strict=True))

    @property
    def dataset_for(self) -> dict[str, str]:
        return {
            SplitNames.CONSTRUCTION: self.construction_dataset,
            SplitNames.CALIBRATION: self.calibration_dataset,
            SplitNames.EVALUATION: self.evaluation_dataset,
        }

    def with_layers(self, layers: tuple[int, ...]) -> "SteeringConfig":
        """Return a copy steering at ``layers``. Validated like any config."""
        return replace(self, layers=tuple(layers))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view, for provenance stamping."""
        data = asdict(self)
        data["layers"] = list(self.layers)
        data["alphas"] = list(self.alphas)
        data["split_fractions"] = list(self.split_fractions)
        for key in ("data_root", "output_dir"):
            data[key] = str(data[key])
        if data["artifacts_dir"] is not None:
            data["artifacts_dir"] = str(data["artifacts_dir"])
        return data

    def save(self, path: Path) -> None:
        """Write the config next to its results, for requirement G."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def load_config(path: str | Path) -> SteeringConfig:
    """Load a JSON config written by :meth:`SteeringConfig.save`."""
    data = json.loads(Path(path).read_text())
    for key in ("layers", "alphas", "split_fractions"):
        if key in data and data[key] is not None:
            data[key] = tuple(data[key])
    for key in ("data_root", "output_dir", "artifacts_dir"):
        if key in data and data[key] is not None:
            data[key] = Path(data[key])
    return SteeringConfig(**data)