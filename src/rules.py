"""The four budget-allocation rules and the shared Laplace noise path.

Every rule spends the same record-level budget d*epsilon and differs only in
how it splits that across features. All calibration comes from public stats,
never from the data being noised. Variance stands in for feature relevance
here. None of these rules is an LRP relevance mechanism.

bounded_inverse_feature used to be called sample_specific. RULE_ALIASES keeps
the old CSVs loading.
"""
from __future__ import annotations

import warnings

import numpy as np

from .constants import RULE_ALIASES, RULES


def canonical_rule(name: str) -> str:
    """Map a rule name (or legacy alias) to its canonical form."""
    name = RULE_ALIASES.get(name, name)
    if name not in RULES and name != "baseline":
        raise ValueError(
            f"Unknown rule {name!r}. Valid: {('baseline',) + RULES}; "
            f"legacy aliases: {sorted(RULE_ALIASES)}"
        )
    return name


def get_safe_feature_ranges(X: np.ndarray) -> np.ndarray:
    """Per-feature range with zero ranges replaced by 1e-10 (noise no-ops)."""
    X = np.asarray(X, dtype=np.float64)
    ranges = X.max(axis=0) - X.min(axis=0)
    return np.where(ranges == 0, 1e-10, ranges)


def _sanitize(vec: np.ndarray) -> np.ndarray:
    """Coerce non-finite or non-positive entries to 1e-10."""
    vec = np.asarray(vec, dtype=np.float64)
    return np.where(np.isfinite(vec) & (vec > 0), vec, 1e-10)


def rule_scales(
    rule: str,
    ranges: np.ndarray,
    variances: np.ndarray | None,
    epsilon: float,
) -> np.ndarray:
    """Per-feature Laplace scales for one rule at one budget."""
    rule = canonical_rule(rule)
    ranges = _sanitize(ranges)
    d = len(ranges)

    if rule == "uniform":
        return ranges / epsilon

    if variances is None:
        raise ValueError(f"{rule} needs per-feature variances; got None")
    variances = _sanitize(variances)
    importance = variances / (np.sum(variances) + 1e-10)

    if rule == "adaptive_feature":
        # inverse share: high-variance features get less budget, so more noise
        inv = np.clip(1 - importance, 0.01, 0.99)
        eps_per = epsilon * d * inv / (np.sum(inv) + 1e-10)
        eps_per = np.clip(eps_per, 0.01 * epsilon, epsilon * d)
        # the clip can't bind at d=62 but could at a smaller d
        assert np.isclose(eps_per.sum(), epsilon * d, rtol=1e-9), (
            f"budget sums to {eps_per.sum():.6f}, expected {epsilon * d:.6f}"
        )
    elif rule == "adaptive_feature_reversed":
        # the one line that differs from adaptive_feature: proportional, not inverse
        fwd = np.clip(importance, 0.01, 0.99)
        eps_per = epsilon * d * fwd / (np.sum(fwd) + 1e-10)
        eps_per = np.clip(eps_per, 0.01 * epsilon, epsilon * d)
        # this clip can bind when one feature dominates, so renormalize
        eps_per = eps_per / (eps_per.sum() + 1e-10) * (epsilon * d)
        assert np.isclose(eps_per.sum(), epsilon * d, rtol=1e-9)
    else:  # bounded_inverse_feature
        # same direction as adaptive_feature, shares bounded to [0.5, 2]x even
        inv = 1.0 / (importance + 1e-10)
        shares = inv / (np.sum(inv) + 1e-10)
        shares = np.clip(shares, 0.5 / d, 2.0 / d)
        shares = shares / (np.sum(shares) + 1e-10)
        eps_per = epsilon * d * shares

    return ranges / (eps_per + 1e-10)


def apply_noise(
    X: np.ndarray,
    rule: str,
    epsilon: float,
    seed: int = 42,
    public_stats: dict | None = None,
) -> np.ndarray:
    """Return a noised copy of X under the given rule ('baseline' = unchanged copy)."""
    rule = canonical_rule(rule)
    X = np.asarray(X, dtype=np.float64).copy()
    if rule == "baseline":
        return X
    X = np.where(np.isfinite(X), X, 0.0)

    ps = public_stats or {}
    ranges = _sanitize(ps["ranges"]) if ps.get("ranges") is not None \
        else get_safe_feature_ranges(X)
    variances = None
    if rule != "uniform":
        variances = ps["variances"] if ps.get("variances") is not None \
            else np.var(X, axis=0)

    # clip to the public box so sensitivity = range actually holds. the
    # public stats have no min/max, so the box is mean +/- range/2. every
    # rule clips the same X to the same box, so this can't favor any rule
    if ps.get("mean") is not None:
        mean = np.asarray(ps["mean"], dtype=np.float64)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        X = np.clip(X, mean - ranges / 2.0, mean + ranges / 2.0)

    scales = rule_scales(rule, ranges, variances, epsilon)
    rng = np.random.default_rng(seed)
    out = X + rng.laplace(0, scales, size=X.shape)
    return np.nan_to_num(out, nan=0.0, posinf=1e10, neginf=-1e10)


def compute_public_stats(X_public: np.ndarray) -> dict:
    """Per-feature ranges/variances/mean/min/max from a public donor matrix."""
    X_public = np.asarray(X_public, dtype=np.float64).copy()
    X_public = np.where(np.isfinite(X_public), X_public, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN columns warn; sanitized below
        col_max = np.nanmax(X_public, axis=0)
        col_min = np.nanmin(X_public, axis=0)
        col_mean = np.nanmean(X_public, axis=0)
        col_var = np.nanvar(X_public, axis=0)

    ranges = col_max - col_min
    # all-NaN columns (e.g. Chapman's height/weight) become noise no-ops
    ranges = np.where(np.isfinite(ranges) & (ranges > 0), ranges, 1e-10)
    col_var = np.where(np.isfinite(col_var) & (col_var > 0), col_var, 1e-10)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)

    # explicit min/max for callers that want bounds
    col_min = np.where(np.isfinite(col_min), col_min, col_mean - ranges / 2.0)
    col_max = np.where(np.isfinite(col_max), col_max, col_mean + ranges / 2.0)

    return {
        "ranges": ranges,
        "variances": col_var,
        "mean": col_mean,
        "min": col_min,
        "max": col_max,
    }
