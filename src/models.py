"""Classifier factories, thresholds, and the single DP training path.

Noise goes on the feature matrix only. Labels are treated as public, so the
class weights and priors below read y freely and spend no budget. Every rule
reaches the model through the same input-perturbation path; assert_rule_parity
exists to keep it that way.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from .constants import (
    DEFAULT_KNN_NEIGHBORS,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_VAR_SMOOTHING,
    MODEL_TYPES,
    RULES,
)
from .rules import apply_noise


# ---------------------------------------------------------------------------
# Classifier factories
# ---------------------------------------------------------------------------
def make_random_forest(
    n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Random forest with balanced_subsample class weights."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def make_gaussian_nb(
    prevalence: float,
    var_smoothing: float = DEFAULT_VAR_SMOOTHING,
) -> GaussianNB:
    """GaussianNB with the public prevalence as prior."""
    prevalence = float(np.clip(prevalence, 1e-6, 1 - 1e-6))
    return GaussianNB(
        priors=np.asarray([1.0 - prevalence, prevalence]),
        var_smoothing=var_smoothing,
    )


def make_knn(n_neighbors: int = DEFAULT_KNN_NEIGHBORS) -> KNeighborsClassifier:
    """Distance-weighted KNN. Prior correction happens at scoring time."""
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)


def make_logreg(random_state: int | None = None) -> LogisticRegression:
    """Logistic regression with balanced class weights."""
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)


def make_classifier(model_type: str, seed: int, prevalence: float):
    """Fresh non-DP classifier of the requested type."""
    if model_type == "knn":
        return make_knn()
    if model_type == "naive_bayes":
        return make_gaussian_nb(prevalence)
    if model_type == "random_forest":
        return make_random_forest(random_state=seed)
    if model_type == "logreg":
        return make_logreg(random_state=seed)
    raise ValueError(f"Unknown model type {model_type!r}; valid: {MODEL_TYPES}")


def correct_knn_proba(
    proba: np.ndarray,
    train_prior: float,
    target_prior: float = 0.5,
) -> np.ndarray:
    """Rescale binary predict_proba from one class prior to another."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"Expected binary proba matrix, got {proba.shape}.")
    p1 = np.clip(proba[:, 1], 1e-12, 1 - 1e-12)
    train_prior = float(np.clip(train_prior, 1e-6, 1 - 1e-6))
    target_prior = float(np.clip(target_prior, 1e-6, 1 - 1e-6))
    odds = (p1 / (1.0 - p1)) * (
        (target_prior / (1.0 - target_prior)) / (train_prior / (1.0 - train_prior))
    )
    p1_new = odds / (1.0 + odds)
    return np.column_stack([1.0 - p1_new, p1_new])


def tune_var_smoothing(
    x_train: np.ndarray,
    y_train: np.ndarray,
    prevalence: float,
    grid: np.ndarray | None = None,
    n_splits: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
) -> tuple[float, dict]:
    """Grid-search var_smoothing on clean data. Returns (best, diagnostics)."""
    grid = np.logspace(0, -9, 25) if grid is None else np.asarray(grid)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores_mean = np.zeros_like(grid, dtype=np.float64)
    scores_std = np.zeros_like(grid, dtype=np.float64)

    for idx, vs in enumerate(grid):
        fold_scores = []
        for train_idx, val_idx in skf.split(x_train, y_train):
            clf = make_gaussian_nb(prevalence, var_smoothing=float(vs))
            clf.fit(x_train[train_idx], y_train[train_idx])
            p = clf.predict_proba(x_train[val_idx])[:, 1]
            if scoring == "average_precision":
                fold_scores.append(average_precision_score(y_train[val_idx], p))
            elif scoring == "roc_auc":
                fold_scores.append(roc_auc_score(y_train[val_idx], p))
            else:
                raise ValueError(f"Unknown scoring: {scoring!r}")
        scores_mean[idx] = float(np.mean(fold_scores))
        scores_std[idx] = float(np.std(fold_scores, ddof=1))

    best = int(np.argmax(scores_mean))
    diagnostics = {
        "grid": grid.tolist(),
        "scores_mean": scores_mean.tolist(),
        "scores_std": scores_std.tolist(),
        "best_var_smoothing": float(grid[best]),
        "best_score": float(scores_mean[best]),
        "best_score_std": float(scores_std[best]),
        "scoring": scoring,
        "n_splits": n_splits,
    }
    return float(grid[best]), diagnostics


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def threshold_prevalence_matched(prevalence: float) -> float:
    """Threshold at the public prevalence."""
    return float(np.clip(prevalence, 1e-6, 1 - 1e-6))


def threshold_f1_optimal(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold that maximizes F1 on the validation fold."""
    precision, recall, thr = precision_recall_curve(y_val, p_val)
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if len(thr) == 0 or len(f1) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1))])


def threshold_youden(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Threshold at max sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(tpr - fpr))])


# ---------------------------------------------------------------------------
# The one DP training path
# ---------------------------------------------------------------------------
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    rule: str,
    epsilon: float,
    seed: int,
    prevalence: float,
    public_stats: dict | None,
):
    """Noise X under the rule and fit. Returns (model, X_used)."""
    if rule == "baseline":
        model = make_classifier(model_type, seed, prevalence)
        model.fit(X_train, y_train)
        return model, X_train

    # missing public_stats would silently calibrate on the private data
    if public_stats is None:
        raise RuntimeError("public_stats is None for a DP arm; pass load_public_stats(dataset)")
    X_noisy = apply_noise(X_train, rule, epsilon, seed, public_stats=public_stats)
    model = make_classifier(model_type, seed, prevalence)
    model.fit(X_noisy, y_train)
    return model, X_noisy


def assert_rule_parity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prevalence: float,
    public_stats: dict,
    epsilon: float = 1.0,
    seed: int = 42,
) -> None:
    """Every (model, rule) cell must use input perturbation at the same d*eps."""
    problems = []
    for mt in MODEL_TYPES:
        for rule in RULES:
            model, X_used = train_model(
                X_train, y_train, mt, rule, epsilon, seed, prevalence, public_stats
            )
            if np.allclose(X_used, X_train):
                problems.append(f"{mt}/{rule}: trained on the clean matrix")
            mod = type(model).__module__
            if "sklearn" not in mod:
                problems.append(f"{mt}/{rule}: {type(model).__name__} from {mod}, not sklearn")
    for rule in RULES:
        Xn = apply_noise(X_train, rule, epsilon, seed, public_stats=public_stats)
        if Xn.shape != X_train.shape:
            problems.append(f"{rule}: apply_noise changed the matrix shape")
    if problems:
        raise AssertionError("RULE PARITY VIOLATED:\n  " + "\n  ".join(problems))
    d = X_train.shape[1]
    print(f"rule parity OK: {len(MODEL_TYPES)}x{len(RULES)} cells at d*eps = {d * epsilon}")
