"""Invariant tests on synthetic data. No feature CSVs needed.

Run with  python -m pytest tests/  or  python tests/test_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import RULES  # noqa: E402
from src.data import _clean  # noqa: E402
from src.models import assert_rule_parity, correct_knn_proba  # noqa: E402
from src.rules import apply_noise, canonical_rule, compute_public_stats, rule_scales  # noqa: E402

RNG = np.random.default_rng(7)
D = 62


def _synthetic(n: int = 400):
    """Synthetic cohort with the real shape, d=62 and about 7% prevalence."""
    X = RNG.normal(size=(n, D)) * RNG.uniform(0.1, 100.0, size=D)
    y = (RNG.uniform(size=n) < 0.073).astype(int)
    y[:5] = 1  # guarantee both classes at any n
    public_stats = compute_public_stats(RNG.normal(size=(n, D)) * RNG.uniform(0.1, 100.0, size=D))
    return X, y, public_stats


def test_rule_parity():
    """Every (model, rule) cell trains on a perturbed matrix via sklearn."""
    X, y, ps = _synthetic()
    assert_rule_parity(X, y, prevalence=float(y.mean()), public_stats=ps)


def test_budget_sums_to_d_eps():
    """Implied per-feature budgets sum to d*eps for every rule."""
    _, _, ps = _synthetic()
    eps = 1.0
    for rule in RULES:
        scales = rule_scales(rule, ps["ranges"], ps["variances"], eps)
        # invert scales = ranges / (eps_per + guard). uniform has no guard
        guard = 0.0 if rule == "uniform" else 1e-10
        eps_per = ps["ranges"] / scales - guard
        assert np.isclose(eps_per.sum(), eps * D, rtol=1e-6), (
            f"{rule}: implied budget {eps_per.sum():.6f} != {eps * D:.6f}"
        )


def test_bounded_inverse_share_ratio():
    """bounded_inverse_feature: no feature gets more than 4x another's budget."""
    _, _, ps = _synthetic()
    eps = 1.0
    scales = rule_scales("bounded_inverse_feature", ps["ranges"], ps["variances"], eps)
    eps_per = ps["ranges"] / scales - 1e-10
    ratio = eps_per.max() / eps_per.min()
    assert ratio <= 4.0 * (1 + 1e-9), f"budget ratio {ratio:.4g} exceeds the 4:1 bound"
    assert np.isclose(eps_per.mean(), eps, rtol=1e-6), "mean share drifted from eps"


def test_legacy_alias_is_same_rule():
    """'sample_specific' and 'bounded_inverse_feature' are byte-identical."""
    X, _, ps = _synthetic()
    a = apply_noise(X, "sample_specific", 1.0, seed=42, public_stats=ps)
    b = apply_noise(X, "bounded_inverse_feature", 1.0, seed=42, public_stats=ps)
    assert np.array_equal(a, b), "legacy alias diverged from the canonical rule"
    assert canonical_rule("sample_specific") == "bounded_inverse_feature"


def test_unknown_rule_fails_loud():
    """A typo'd rule name raises immediately with the valid list."""
    X, _, ps = _synthetic()
    try:
        apply_noise(X, "adaptve_feature", 1.0, public_stats=ps)
    except ValueError as e:
        assert "Valid" in str(e)
    else:
        raise AssertionError("unknown rule silently accepted")


def test_baseline_is_untouched_copy():
    """'baseline' returns an identical, independent copy."""
    X, _, _ = _synthetic()
    out = apply_noise(X, "baseline", 1.0)
    assert np.array_equal(out, X)
    out[0, 0] += 1.0
    assert out[0, 0] != X[0, 0], "baseline returned a view, not a copy"


def test_noise_is_seed_deterministic():
    """Same seed same noise, different seed different noise."""
    X, _, ps = _synthetic()
    a = apply_noise(X, "adaptive_feature", 1.0, seed=42, public_stats=ps)
    b = apply_noise(X, "adaptive_feature", 1.0, seed=42, public_stats=ps)
    c = apply_noise(X, "adaptive_feature", 1.0, seed=43, public_stats=ps)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_public_box_clip():
    """With public stats, outputs start from values clipped into the public box."""
    X, _, ps = _synthetic()
    X_wild = X.copy()
    X_wild[0, :] = 1e9  # far outside any public range
    # huge eps means negligible noise, so the clip dominates
    out = apply_noise(X_wild, "uniform", 1e12, seed=0, public_stats=ps)
    hi = ps["mean"] + ps["ranges"] / 2.0
    assert np.all(out[0] <= hi + 1e-3), "public-box clip did not bound the outlier row"


def test_clean_imputes_with_column_means():
    """NaN and inf impute to the column mean of the finite entries."""
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.inf, 8.0]])
    out = _clean(X)
    assert np.isfinite(out).all()
    assert out[0, 1] == 6.0  # mean of {4, 8}
    assert out[2, 0] == 2.0  # mean of {1, 3}


def test_knn_prior_correction_shape_and_bounds():
    """KNN prior correction returns a valid binary probability matrix."""
    proba = np.column_stack([np.linspace(0.1, 0.9, 9), 1 - np.linspace(0.1, 0.9, 9)])
    out = correct_knn_proba(proba, train_prior=0.073)
    assert out.shape == proba.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out > 0).all() and (out < 1).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
