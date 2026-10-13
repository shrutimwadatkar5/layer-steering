"""Steering direction construction and additive-intervention arithmetic.

This module is deliberately thin. Causalab already implements the intervention
itself: ``add_scaled`` in ``causalab/neural/shared/mechanisms.py`` returns
``alpha * op`` and the executor writes ``f = f0 + delta``
(``causalab/neural/shared/executor/writes.py``). This module therefore does
**not** reimplement the forward hook. It provides the pieces that are not
protocol documents:

* :func:`mean_difference_direction` -- the contrastive direction
  ``v = mean(h | A) - mean(h | B)``, reusing the float64-accumulation decision
  already made by ``causalab/analysis/harvest_difference.py``.
* :func:`apply_additive` -- the reference arithmetic, used by tests to pin the
  semantics the protocol must reproduce.
* :func:`random_control_direction` -- a norm-matched random control.
* :func:`validate_direction` -- shape/device/finiteness checks.

The norm convention matters. Residual-stream norms grow with depth (measured on
Qwen2.5-1.5B: mean per-token L2 norm at ``block_output`` ranges from ~630 at
layer 0's example to ~3893 for a short row), so an un-normalized direction makes
raw ``alpha`` incomparable across layers. Normalizing by default makes ``alpha``
mean "fraction of direction norm added" rather than "absolute activation
offset".
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "STEERING_DIRECTION_SHAPE_NOTE",
    "mean_difference_direction",
    "apply_additive",
    "random_control_direction",
    "validate_direction",
    "DirectionStats",
]

#: Recorded so a result file can be read without re-running anything.
#:
#: The intervention hook is Causalab's ``block_output`` component, which
#: ``causalab/protocol/registry/families.py:587`` defines as ``Tap("block")``
#: -- the output of the whole transformer block, not an interior tensor.
#: Empirically measured on Qwen/Qwen2.5-1.5B-Instruct (28 layers,
#: ``hidden_size=1536``), a forward hook on ``model.model.layers[L]`` sees
#: ``(batch, seq_len, 1536)`` in bfloat16 on the model device.
#:
#: ``seq_len`` is the *padded* width: in a ragged batch the padded positions
#: carry a non-zero residual norm (~53 observed), so a raw ``-1`` index would
#: address padding rather than the answer slot. Causalab resolves positions
#: against each row's first real token (``token_index = index - first`` in
#: ``causalab/protocol/positions/resolve.py``), which is why the pilot uses a
#: position spec rather than a hard-coded index.
STEERING_DIRECTION_SHAPE_NOTE = (
    "block_output hook returns (batch, padded_seq_len, hidden_size); "
    "hidden_size=1536 for Qwen2.5-1.5B-Instruct. The steering direction is "
    "(hidden_size,) and broadcasts across the position and batch axes."
)


class DirectionStats(dict):
    """Provenance recorded alongside every direction.

    A ``dict`` subclass so it serializes with ``json.dump`` directly.
    """

    @property
    def norm(self) -> float:
        return float(self["norm"])


def _as_rows(tensor: Any, name: str) -> Any:
    """Flatten leading dims to ``(n, d)``.

    Matches ``harvest_difference._rows``: a ``(d,)`` mean-reduced harvest is one
    row, and ``(n, positions, d)`` contributes ``n * positions`` rows.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}; expected rows of a hidden "
            "dimension"
        )
    return tensor.reshape(-1, tensor.shape[-1])


def mean_difference_direction(
    positive: Any,
    negative: Any,
    *,
    normalize: bool = True,
    name_positive: str = "construction group A",
    name_negative: str = "construction group B",
) -> tuple[Any, DirectionStats]:
    """Contrastive direction ``mean(A) - mean(B)``.

    Args:
        positive: Activations for group A. Any shape ending in the hidden
            dimension; leading dims are flattened.
        negative: Activations for group B, same hidden width.
        normalize: Divide by the L2 norm. Default ``True`` -- see module
            docstring for why raw alpha is otherwise not layer-comparable.
        name_positive / name_negative: Labels for error messages.

    Returns:
        ``(direction, stats)`` where ``direction`` is a 1-D float32 tensor of
        width ``d`` and ``stats`` records counts, norms, and the mean norms.

    Raises:
        ValueError: On empty inputs, mismatched hidden widths, or -- when
            ``normalize`` is set -- a zero-norm difference. A zero direction
            would steer nothing while appearing to have steered, so it is an
            error rather than a silent pass-through.
    """
    import torch

    pos = _as_rows(positive, name_positive).to(torch.float64)
    neg = _as_rows(negative, name_negative).to(torch.float64)

    if pos.shape[-1] != neg.shape[-1]:
        raise ValueError(
            f"{name_positive} has width {pos.shape[-1]} but {name_negative} has "
            f"width {neg.shape[-1]} -- a difference needs one hidden dimension"
        )
    if pos.shape[0] == 0:
        raise ValueError(f"{name_positive} has no rows")
    if neg.shape[0] == 0:
        raise ValueError(f"{name_negative} has no rows")

    # float64 accumulation, for the same reason upstream does it: a bf16 mean
    # over many rows loses the low bits the difference is made of.
    delta = pos.mean(dim=0) - neg.mean(dim=0)
    raw_norm = float(torch.linalg.vector_norm(delta))
    mean_norm_pos = float(torch.linalg.vector_norm(pos, dim=-1).mean())
    mean_norm_neg = float(torch.linalg.vector_norm(neg, dim=-1).mean())

    if normalize and raw_norm == 0.0:
        raise ValueError(
            "the two groups have identical means, so there is no direction to "
            "normalize -- a zero direction would steer nothing while looking "
            "like it steered"
        )

    unit = delta / raw_norm if normalize else delta
    direction = unit.to(torch.float32)

    stats = DirectionStats(
        {
            "n_positive_rows": int(pos.shape[0]),
            "n_negative_rows": int(neg.shape[0]),
            "hidden_size": int(pos.shape[-1]),
            "raw_norm": raw_norm,
            "norm": float(torch.linalg.vector_norm(direction)),
            "mean_norm_positive": mean_norm_pos,
            "mean_norm_negative": mean_norm_neg,
            "norm_ratio": (
                raw_norm / mean_norm_pos if mean_norm_pos > 0 else float("nan")
            ),
            "normalized": bool(normalize),
        }
    )
    return direction, stats


def apply_additive(activation: Any, direction: Any, alpha: float) -> Any:
    """Reference additive intervention ``h + alpha * v``.

    Mirrors what ``add_scaled`` does in the executor, kept here so the tests
    pin the arithmetic independently of a GPU run.

    Args:
        activation: ``(..., d)`` tensor at the intervention site.
        direction: ``(d,)`` tensor, broadcasting over leading dims.
        alpha: Normalized strength. ``0.0`` returns ``activation`` unchanged.

    Returns:
        A tensor of the same shape and dtype as ``activation``.
    """
    import torch

    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be numeric, got {type(alpha)!r}")
    a = float(alpha)
    if not torch.isfinite(torch.tensor(a)):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    if activation.shape[-1] != direction.shape[-1]:
        raise ValueError(
            f"cannot add a direction of width {direction.shape[-1]} to an "
            f"activation of width {activation.shape[-1]}"
        )
    # Explicit short-circuit: at alpha=0 the result must be the *same tensor*,
    # not merely an equal one, so a zero-strength run provably leaves the
    # model untouched.
    if a == 0.0:
        return activation
    delta = direction.to(device=activation.device, dtype=activation.dtype)
    return activation + a * delta


def random_control_direction(
    hidden_size: int,
    *,
    seed: int,
    target_norm: float = 1.0,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """A seeded random control direction with a prescribed norm.

    The control isolates "did a *meaningful* direction do the work" from "did
    *any* perturbation of this size do the work". Matching the real direction's
    norm is what makes the comparison fair.
    """
    import torch

    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if target_norm <= 0:
        raise ValueError(f"target_norm must be positive, got {target_norm}")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    norm = float(torch.linalg.vector_norm(raw))
    if norm == 0.0:
        raise ValueError(
            f"seed {seed} produced a zero random vector; choose another seed"
        )
    scaled = raw / norm * float(target_norm)
    if device is not None or dtype is not None:
        scaled = scaled.to(device=device, dtype=dtype)
    return scaled


def validate_direction(direction: Any, *, hidden_size: int, device: Any = None) -> None:
    """Check a direction before it is written or applied.

    Raises ``ValueError`` on a wrong width, a non-finite entry, or a device
    mismatch. Called by the tests and by the direction writer.
    """
    import torch

    if not isinstance(direction, torch.Tensor):
        raise TypeError(f"direction must be a torch.Tensor, got {type(direction)!r}")
    if direction.ndim != 1:
        raise ValueError(
            f"direction must be 1-D (hidden_size,), got shape {tuple(direction.shape)}"
        )
    if direction.shape[0] != hidden_size:
        raise ValueError(
            f"direction has width {direction.shape[0]}, expected hidden_size "
            f"{hidden_size}"
        )
    if not torch.isfinite(direction).all():
        raise ValueError("direction contains non-finite values (nan or inf)")
    norm = float(torch.linalg.vector_norm(direction))
    if norm == 0.0:
        raise ValueError("direction has zero norm and would steer nothing")
    if device is not None and direction.device != torch.device(device):
        raise ValueError(
            f"direction is on {direction.device} but expected {torch.device(device)}"
        )