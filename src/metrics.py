"""Utility metric bundle, MIA metrics, and the stats every runner shares."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

from .constants import DEFAULT_FPR_TARGETS, EPSILON_VALUES, MODEL_TYPES, RULES


@dataclass
class ClassificationReport:
    """Metric bundle for imbalanced binary classification."""

    prevalence: float
    threshold: float
    auroc: float
    auc_pr: float
    mcc: float
    f1_at_tau: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    brier: float
    accuracy_at_tau: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_classifier(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    threshold: float,
) -> ClassificationReport:
    """Metric bundle at a fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    if np.any((p_pred < 0.0) | (p_pred > 1.0)):
        raise ValueError("p_pred must be probabilities in [0, 1].")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only {0, 1}.")

    y_hat = (p_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    # single-class y_true breaks AUROC/AP, report NaN instead of crashing
    if len(np.unique(y_true)) < 2:
        auroc, auc_pr = float("nan"), float("nan")
    else:
        auroc = float(roc_auc_score(y_true, p_pred))
        auc_pr = float(average_precision_score(y_true, p_pred))

    return ClassificationReport(
        prevalence=float(y_true.mean()),
        threshold=float(threshold),
        auroc=auroc,
        auc_pr=auc_pr,
        mcc=float(matthews_corrcoef(y_true, y_hat)),
        f1_at_tau=float(f1_score(y_true, y_hat, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_hat)),
        sensitivity=float(tp / max(tp + fn, 1)),
        specificity=float(tn / max(tn + fp, 1)),
        ppv=float(tp / max(tp + fp, 1)),
        npv=float(tn / max(tn + fn, 1)),
        brier=float(brier_score_loss(y_true, p_pred)),
        accuracy_at_tau=float((tp + tn) / max(tp + tn + fp + fn, 1)),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


# ---------------------------------------------------------------------------
# MIA score metrics
# ---------------------------------------------------------------------------
def _scores_and_labels(member_scores, nonmember_scores):
    """Stack member and non-member scores into (scores, labels), None if degenerate."""
    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)
    if len(member_scores) == 0 or len(nonmember_scores) == 0:
        return None
    labels = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    scores = np.concatenate([member_scores, nonmember_scores])
    if len(np.unique(scores)) < 2:
        return None
    return scores, labels


def mia_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Attack AUC over membership-confidence scores."""
    pair = _scores_and_labels(member_scores, nonmember_scores)
    if pair is None:
        return float("nan")
    scores, labels = pair
    return float(roc_auc_score(labels, scores))


def mia_tpr_at_fpr(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    fpr_targets: tuple[float, ...] = DEFAULT_FPR_TARGETS,
) -> dict:
    """TPR at fixed low FPR levels."""
    pair = _scores_and_labels(member_scores, nonmember_scores)
    if pair is None:
        return {float(t): float("nan") for t in fpr_targets}
    scores, labels = pair
    fpr, tpr, _ = roc_curve(labels, scores)
    return {float(t): float(np.interp(t, fpr, tpr)) for t in fpr_targets}


# ---------------------------------------------------------------------------
# Statistics shared by every runner
# ---------------------------------------------------------------------------
def compute_95_ci(data: np.ndarray) -> tuple[float, float]:
    """95% t-interval for the mean, (nan, nan) below two observations."""
    if len(data) < 2:
        return (np.nan, np.nan)
    return tuple(stats.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=stats.sem(data)))


def ci95_halfwidth(x: np.ndarray) -> float:
    """95% t-interval half-width, NaN-safe."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled SD, NaN below two observations per group."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def pairwise_rule_tests(
    results_df: pd.DataFrame,
    metric: str,
    rules: tuple[str, ...] = RULES,
    model_types: tuple[str, ...] = MODEL_TYPES,
    epsilons: tuple[float, ...] = EPSILON_VALUES,
    rule_col: str = "mechanism",
) -> pd.DataFrame:
    """Welch's t + Cohen's d for every rule pair at each (model, epsilon)."""
    records = []
    n_comparisons = len(list(combinations(rules, 2)))
    for model_type in model_types:
        md = results_df[results_df["model_type"] == model_type]
        for epsilon in epsilons:
            groups = {
                r: md[(md[rule_col] == r) & (md["epsilon"] == epsilon)][metric].values
                for r in rules
            }
            for r1, r2 in combinations(rules, 2):
                a, b = groups[r1], groups[r2]
                if len(a) < 2 or len(b) < 2:
                    continue
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
                p_bonf = min(1.0, p_val * n_comparisons)
                records.append({
                    "model_type": model_type,
                    "epsilon": epsilon,
                    "comparison": f"{r1}_vs_{r2}",
                    "metric": metric,
                    "mean_diff": np.mean(a) - np.mean(b),
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "p_value_bonferroni": p_bonf,
                    "cohens_d": cohens_d(a, b),
                    "significant_005": p_bonf < 0.05,
                    "n1": len(a),
                    "n2": len(b),
                })
    return pd.DataFrame(records)
