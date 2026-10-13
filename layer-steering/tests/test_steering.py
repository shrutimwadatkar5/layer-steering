"""Unit tests for the steering pilot's own arithmetic and configuration.

Scope note. These tests cover **this research code**, not Causalab's engine.
Causalab's intervention semantics are already pinned upstream by
``tests/neural/engines/pytorch_hooks/test_write_oracle.py``, which contains
``test_add_scaled_matches_oracle_steer``. What these tests add is:

* the direction construction and its guards,
* the reference additive arithmetic, including the alpha=0 identity,
* layer selection and configuration validation,
* determinism of the seeded control direction.

Run with::

    uv run pytest research/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.config import (  # noqa: E402
    DEFAULT_ALPHAS,
    PILOT_LAYERS,
    SplitNames,
    SteeringConfig,
    load_config,
)
from research.steering import (  # noqa: E402
    apply_additive,
    mean_difference_direction,
    random_control_direction,
    validate_direction,
)

HIDDEN = 16


# --------------------------------------------------------------------------- #
# direction construction
# --------------------------------------------------------------------------- #
def test_mean_difference_is_mean_a_minus_mean_b():
    """The direction must be exactly mean(A) - mean(B)."""
    torch.manual_seed(0)
    a = torch.randn(32, HIDDEN)
    b = torch.randn(32, HIDDEN) + 1.5

    raw, stats = mean_difference_direction(a, b, normalize=False)
    expected = a.double().mean(0) - b.double().mean(0)

    assert raw.shape == (HIDDEN,)
    assert torch.allclose(raw.double(), expected, atol=1e-6)
    assert stats["normalized"] is False
    assert stats["n_positive_rows"] == 32
    assert stats["n_negative_rows"] == 32


def test_normalized_direction_has_unit_norm():
    a = torch.randn(16, HIDDEN) * 3.0
    b = torch.randn(16, HIDDEN) * 3.0

    direction, stats = mean_difference_direction(a, b, normalize=True)

    assert torch.allclose(
        torch.linalg.vector_norm(direction), torch.tensor(1.0), atol=1e-5
    )
    assert abs(stats["norm"] - 1.0) < 1e-5


def test_zero_mean_difference_is_rejected_not_normalized():
    """A zero direction must raise, not silently pass through.

    Normalizing a zero vector would produce NaNs that look like a result.
    """
    same = torch.randn(8, HIDDEN)
    with pytest.raises(ValueError, match="identical means"):
        mean_difference_direction(same, same.clone(), normalize=True)


def test_mismatched_hidden_width_fails_clearly():
    with pytest.raises(ValueError, match="needs one hidden dimension"):
        mean_difference_direction(torch.randn(4, HIDDEN), torch.randn(4, HIDDEN + 1))


def test_empty_group_fails_clearly():
    with pytest.raises(ValueError, match="no rows"):
        mean_difference_direction(torch.zeros(0, HIDDEN), torch.randn(4, HIDDEN))


def test_leading_dims_are_flattened():
    """(n, positions, d) and (n, d) must mean the same thing."""
    a = torch.randn(4, 3, HIDDEN)
    b = torch.randn(4, 3, HIDDEN)

    flat_a = a.reshape(-1, HIDDEN)
    flat_b = b.reshape(-1, HIDDEN)

    d1, _ = mean_difference_direction(a, b, normalize=False)
    d2, _ = mean_difference_direction(flat_a, flat_b, normalize=False)

    assert torch.allclose(d1, d2, atol=1e-6)


def test_direction_shape_is_one_dimensional():
    """The direction must be (d,) -- it broadcasts over batch and position."""
    direction, _ = mean_difference_direction(
        torch.randn(5, HIDDEN), torch.randn(5, HIDDEN), normalize=True
    )
    assert direction.ndim == 1
    assert direction.shape == (HIDDEN,)


# --------------------------------------------------------------------------- #
# intervention arithmetic
# --------------------------------------------------------------------------- #
def test_alpha_zero_reproduces_unmodified_activation():
    """Requirement J: alpha=0 must return the activation unchanged.

    Identity, not just equality: the same tensor object is returned, so no
    arithmetic happens at all on the no-op path.
    """
    activation = torch.randn(4, 7, HIDDEN)
    direction = torch.randn(HIDDEN)

    out = apply_additive(activation, direction, 0.0)

    assert out is activation
    assert torch.equal(out, activation)


def test_additive_matches_hand_computed_arithmetic():
    activation = torch.randn(3, 5, HIDDEN)
    direction = torch.randn(HIDDEN)
    alpha = 1.7

    out = apply_additive(activation, direction, alpha)

    expected = activation + 1.7 * direction
    assert torch.allclose(out, expected, atol=1e-6)


def test_additive_broadcasts_over_batch_and_position():
    """A (d,) direction applies to every position and every row."""
    activation = torch.randn(2, 9, HIDDEN)
    direction = torch.randn(HIDDEN)

    out = apply_additive(activation, direction, 1.0)

    assert out.shape == activation.shape
    # every position got the same delta
    delta = out - activation
    for b in range(activation.shape[0]):
        for p in range(activation.shape[1]):
            assert torch.allclose(delta[b, p], direction, atol=1e-6)


def test_additive_is_addition_not_replacement():
    """Guard against confusing adding a direction with overwriting.

    The result must retain the original activation's structure; a replacement
    would discard it.
    """
    activation = torch.randn(1, 1, HIDDEN) + 7.0
    direction = torch.randn(HIDDEN) * 0.01

    out = apply_additive(activation, direction, 1.0)

    assert not torch.allclose(out, direction.expand_as(out))
    assert torch.allclose(out - direction, activation, atol=1e-6)


def test_negative_alpha_moves_the_other_way():
    activation = torch.zeros(1, 1, HIDDEN)
    direction = torch.ones(HIDDEN)

    assert torch.allclose(
        apply_additive(activation, direction, -1.0), -direction.expand(1, 1, HIDDEN)
    )


def test_dimension_mismatch_fails_clearly():
    activation = torch.randn(2, HIDDEN)
    direction = torch.randn(HIDDEN + 1)
    with pytest.raises(ValueError, match="cannot add a direction"):
        apply_additive(activation, direction, 1.0)


def test_non_finite_alpha_fails_clearly():
    with pytest.raises(ValueError, match="finite"):
        apply_additive(torch.randn(1, HIDDEN), torch.randn(HIDDEN), float("nan"))


# --------------------------------------------------------------------------- #
# direction validation
# --------------------------------------------------------------------------- #
def test_validate_direction_accepts_well_formed():
    validate_direction(torch.randn(HIDDEN), hidden_size=HIDDEN)


@pytest.mark.parametrize(
    "bad, match",
    [
        (torch.randn(2, HIDDEN), r"must be 1-D"),
        (torch.randn(HIDDEN + 3), r"expected hidden_size"),
        (torch.zeros(HIDDEN), r"zero norm"),
    ],
)
def test_validate_direction_rejects_bad_tensors(bad, match):
    with pytest.raises(ValueError, match=match):
        validate_direction(bad, hidden_size=HIDDEN)


def test_validate_direction_rejects_non_finite():
    bad = torch.randn(HIDDEN)
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_direction(bad, hidden_size=HIDDEN)


# --------------------------------------------------------------------------- #
# random control + determinism
# --------------------------------------------------------------------------- #
def test_random_control_is_deterministic_for_a_seed():
    a = random_control_direction(HIDDEN, seed=7, target_norm=1.0)
    b = random_control_direction(HIDDEN, seed=7, target_norm=1.0)
    assert torch.equal(a, b)


def test_random_control_differs_across_seeds():
    a = random_control_direction(HIDDEN, seed=1)
    b = random_control_direction(HIDDEN, seed=2)
    assert not torch.equal(a, b)


def test_random_control_has_requested_norm():
    direction = random_control_direction(HIDDEN, seed=3, target_norm=2.5)
    assert abs(float(torch.linalg.vector_norm(direction)) - 2.5) < 1e-5


def test_random_control_rejects_zero_target_norm():
    with pytest.raises(ValueError, match="target_norm"):
        random_control_direction(HIDDEN, seed=1, target_norm=0.0)


def test_random_control_shape_and_dtype():
    direction = random_control_direction(HIDDEN, seed=5, target_norm=1.0)
    assert direction.shape == (HIDDEN,)
    assert direction.dtype == torch.float32


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_default_config_is_valid():
    config = SteeringConfig()
    assert config.layers == PILOT_LAYERS
    assert 0.0 in config.alphas
    assert len(config.split_fractions) == 3


def test_config_roundtrips_through_json(tmp_path):
    config = SteeringConfig(seed=7, layers=(2, 10, 18))
    path = tmp_path / "cfg.json"
    config.save(path)
    loaded = load_config(path)
    assert loaded.seed == 7
    assert loaded.layers == (2, 10, 18)
    assert loaded.to_dict() == config.to_dict()


def test_config_rejects_empty_layers():
    with pytest.raises(ValueError, match="must not be empty"):
        SteeringConfig(layers=())


def test_config_rejects_duplicate_layers():
    with pytest.raises(ValueError, match="duplicates"):
        SteeringConfig(layers=(4, 4))


def test_config_rejects_unsorted_layers():
    with pytest.raises(ValueError, match="sorted"):
        SteeringConfig(layers=(20, 4))


def test_config_rejects_alphas_without_zero():
    """The zero-strength control is mandatory -- it is the invariance check."""
    with pytest.raises(ValueError, match="must contain 0.0"):
        SteeringConfig(alphas=(1.0, 2.0))


def test_config_rejects_unsorted_alphas():
    with pytest.raises(ValueError, match="sorted"):
        SteeringConfig(alphas=(1.0, 0.0))


def test_config_rejects_full_sweep_by_default():
    """The audit's staged plan requires a pilot first."""
    with pytest.raises(ValueError, match="looks like a full sweep"):
        SteeringConfig(layers=tuple(range(0, 28)))


def test_config_allows_full_sweep_when_explicitly_permitted():
    config = SteeringConfig(layers=tuple(range(0, 28)), allow_full_sweep=True)
    assert len(config.layers) == 28


def test_config_rejects_bad_token_position():
    with pytest.raises(ValueError, match="single-key"):
        SteeringConfig(token_position={"index": -1, "scope": "x"})


def test_config_rejects_unknown_position_key():
    with pytest.raises(ValueError, match="not one of"):
        SteeringConfig(token_position={"banana": -1})


def test_config_rejects_fractions_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        SteeringConfig(split_fractions=(0.5, 0.2, 0.2))


def test_config_rejects_nonpositive_fraction():
    with pytest.raises(ValueError, match="positive"):
        SteeringConfig(split_fractions=(0.5, 0.0, 0.5))


def test_config_rejects_bad_model_key():
    with pytest.raises(ValueError, match="Hugging Face id"):
        SteeringConfig(model_key="gpt2")


def test_config_rejects_bad_dtype():
    with pytest.raises(ValueError, match="dtype"):
        SteeringConfig(dtype="float16-ish")


def test_split_names_and_dataset_mapping():
    config = SteeringConfig()
    assert config.split_names == (
        SplitNames.CONSTRUCTION,
        SplitNames.CALIBRATION,
        SplitNames.EVALUATION,
    )
    mapping = config.dataset_for
    assert set(mapping) == set(config.split_names)
    assert len(set(mapping.values())) == 3


def test_default_alphas_include_both_signs_and_zero():
    """H2 needs distinguishable positive and negative strengths."""
    assert any(a < 0 for a in DEFAULT_ALPHAS)
    assert any(a > 0 for a in DEFAULT_ALPHAS)
    assert 0.0 in DEFAULT_ALPHAS


def test_with_layers_revalidates():
    config = SteeringConfig()
    narrowed = config.with_layers((2, 6))
    assert narrowed.layers == (2, 6)
    with pytest.raises(ValueError):
        config.with_layers((0, 28))