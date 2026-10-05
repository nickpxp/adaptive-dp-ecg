"""Feature-CSV loading, public-donor stats, and patient-disjoint splits.

Splits go by strat_fold, which is patient-disjoint. Folds 1-8 train, 9
validates, 10 tests. For MIA the target trains on 1-4 and 5-8 is the shadow
and non-member pool. Chapman folds were assigned the same way with
StratifiedGroupKFold.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    DATA_FILES,
    DATASETS,
    DESCRIPTOR_PREVALENCE,
    FOLD_COL,
    MIA_SHADOW_FOLDS,
    MIA_TARGET_TRAIN_FOLDS,
    N_FEATURES,
    NON_FEATURE_COLS,
    PREVALENCE_TOLERANCE,
    PUBLIC_SOURCE_MAP,
    TARGET_COL,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VAL_FOLDS,
)
from .rules import compute_public_stats


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns, everything that isn't an id, metadata or the label."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def compute_prevalence(y: np.ndarray) -> float:
    """Positive-class fraction as a Python float."""
    return float(np.asarray(y).mean())


def verify_prevalence(dataset: str, y: np.ndarray) -> float:
    """Descriptor prevalence for a dataset, with a warning if the labels disagree."""
    expected = DESCRIPTOR_PREVALENCE[dataset]
    observed = compute_prevalence(y)
    if abs(observed - expected) > PREVALENCE_TOLERANCE:
        warnings.warn(
            f"{dataset}: label prevalence {observed:.4f} deviates from the "
            f"descriptor value {expected:.4f} by more than "
            f"{PREVALENCE_TOLERANCE}. Check the label extraction and fold "
            f"assignment before trusting threshold-dependent metrics.",
            stacklevel=2,
        )
    return expected


def _clean(X: np.ndarray) -> np.ndarray:
    """Impute NaN/inf with column means. All-NaN columns become zero."""
    X = np.where(np.isinf(X), np.nan, X.astype(np.float64))
    with warnings.catch_warnings():
        # all-NaN columns (Chapman has no height/weight) warn here and
        # become zero below, which is what we want
        warnings.simplefilter("ignore", RuntimeWarning)
        col_means = np.nan_to_num(np.nanmean(X, axis=0))
    X = np.where(np.isnan(X), col_means, X)  # broadcast beats a per-column loop
    return np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)


def split_by_folds(
    df: pd.DataFrame,
    pools: dict[str, tuple[int, ...]],
    feature_cols: list[str],
    target_col: str = TARGET_COL,
    fold_col: str = FOLD_COL,
) -> dict:
    """Cleaned X/y arrays for each named fold pool, plus the feature list."""
    out: dict = {"feature_cols": feature_cols}
    for name, folds in pools.items():
        mask = df[fold_col].isin(folds)
        out[f"X_{name}"] = _clean(df.loc[mask, feature_cols].values)
        out[f"y_{name}"] = df.loc[mask, target_col].values.astype(int)
    return out


def canonical_split(df: pd.DataFrame, feature_cols: list[str], **kwargs) -> dict:
    """Patient-disjoint train (1-8) / val (9) / test (10) split."""
    pools = {"train": TRAIN_FOLDS, "val": VAL_FOLDS, "test": TEST_FOLDS}
    return split_by_folds(df, pools, feature_cols, **kwargs)


def canonical_split_for_mia(df: pd.DataFrame, feature_cols: list[str], **kwargs) -> dict:
    """MIA split: target_train (1-4) / shadow_pool (5-8) / val (9) / test (10)."""
    pools = {
        "target_train": MIA_TARGET_TRAIN_FOLDS,
        "shadow_pool": MIA_SHADOW_FOLDS,
        "val": VAL_FOLDS,
        "test": TEST_FOLDS,
    }
    return split_by_folds(df, pools, feature_cols, **kwargs)


def _read_features(name: str, data_dir: str | Path) -> pd.DataFrame:
    """Read one dataset's feature CSV with the required-column guards."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; valid: {DATASETS}")
    path = Path(data_dir) / DATA_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the feature CSVs per README 'Data' "
            f"and place them in {Path(data_dir).resolve()}."
        )
    df = pd.read_csv(path)
    # no fold column, no patient-disjoint split. don't fake one
    if FOLD_COL not in df.columns:
        raise ValueError(
            f"{name} has no {FOLD_COL!r} column; patient-disjoint split not possible"
        )
    # a wrong column set once got through unnoticed, so check the count
    n_feat = len(feature_columns(df))
    if n_feat != N_FEATURES:
        raise ValueError(
            f"{name} yields {n_feat} feature columns, expected {N_FEATURES}. "
            f"The CSV must come from notebooks/0*_preprocessing.ipynb unchanged; "
            f"check NON_FEATURE_COLS if the schema was extended on purpose."
        )
    return df.dropna(subset=[TARGET_COL, FOLD_COL]).reset_index(drop=True)


def load_dataset(name: str, data_dir: str | Path = "data", mia: bool = False) -> dict:
    """Load one dataset and return its canonical or MIA splits."""
    df = _read_features(name, data_dir)
    cols = feature_columns(df)
    split = canonical_split_for_mia if mia else canonical_split
    splits = split(df, feature_cols=cols)
    splits["dataset_name"] = name
    return splits


def load_dataframe(name: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Cleaned DataFrame for callers that need the raw columns."""
    return _read_features(name, data_dir)


def load_public_stats(target_dataset: str, data_dir: str | Path = "data") -> dict:
    """Calibration stats from the cross-dataset public donor."""
    public_name = PUBLIC_SOURCE_MAP[target_dataset]
    df = _read_features(public_name, data_dir)
    cols = feature_columns(df)
    # inf counts as missing. median-impute, then zero any all-NaN columns
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
    df[cols] = df[cols].fillna(df[cols].median())
    df[cols] = df[cols].fillna(0.0)
    return compute_public_stats(df[cols].values.astype(np.float64))
