"""The attacks: shadow-model MIA, LiRA, reconstruction, attribute inference.

The adversary holds clean records and queries the target on clean inputs.
Querying on noised inputs would detect the noise rather than membership.
Attribute inference is the confidence rule of CSMIA on its own, so its
advantages run conservative. Continuous attributes use 10 quantile bins.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import KBinsDiscretizer

from .constants import (
    MIA_SAMPLE_SIZE,
    N_ATTACK_TARGETS,
    N_BINS_CONTINUOUS,
    N_LIRA_SHADOWS,
    N_RECONSTRUCTION_CANDIDATES,
    N_SHADOW_MODELS,
)
from .models import train_model


# ---------------------------------------------------------------------------
# Shared confidence helpers
# ---------------------------------------------------------------------------
def max_confidence(model, X: np.ndarray) -> np.ndarray:
    """Max class probability per record (predicted labels if no proba)."""
    if hasattr(model, "predict_proba"):
        return np.max(model.predict_proba(X), axis=1)
    return model.predict(X).astype(float)


def true_class_confidence(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True-class probability per record, the signal LiRA scores on."""
    p = model.predict_proba(X)
    return p[np.arange(len(y)), y.astype(int)]


# ---------------------------------------------------------------------------
# Shadow-model MIA
# ---------------------------------------------------------------------------
def mia_attack_shadow(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    X_shadow_pool: np.ndarray, y_shadow_pool: np.ndarray,
    rule: str, epsilon: float, model_type: str, seed: int, prevalence: float,
    public_stats: dict | None,
    n_shadow_models: int = N_SHADOW_MODELS,
    sample_size: int = MIA_SAMPLE_SIZE,
) -> dict:
    """Shadow-model MIA. Returns accuracy, advantage and per-sample scores."""
    rng = np.random.default_rng(seed)
    attack_features, attack_labels = [], []
    n_shadow = len(X_shadow_pool)

    for i in range(n_shadow_models):
        shadow_seed = seed + i * 100
        indices = np.random.default_rng(shadow_seed).permutation(n_shadow)
        split = n_shadow // 2
        X_in, y_in = X_shadow_pool[indices[:split]], y_shadow_pool[indices[:split]]
        X_out, y_out = X_shadow_pool[indices[split:]], y_shadow_pool[indices[split:]]

        shadow_model, _ = train_model(
            X_in, y_in, model_type, rule, epsilon, shadow_seed, prevalence, public_stats
        )
        # training saw noise, queries are clean
        for X_q, y_q, is_member in ((X_in, y_in, 1.0), (X_out, y_out, 0.0)):
            conf = max_confidence(shadow_model, X_q)
            correct = (shadow_model.predict(X_q) == y_q).astype(float)
            attack_features.append(np.column_stack([conf, correct]))
            attack_labels.append(np.full(len(conf), is_member))

    attack_model = RandomForestClassifier(n_estimators=100, random_state=seed)
    attack_model.fit(np.vstack(attack_features), np.concatenate(attack_labels))

    n_attack = min(sample_size, len(X_train), len(X_test))
    member_idx = rng.choice(len(X_train), n_attack, replace=False)
    nonmember_idx = rng.choice(len(X_test), n_attack, replace=False)

    def _attack_features(X_q, y_q):
        conf = max_confidence(model, X_q)
        correct = (model.predict(X_q) == y_q).astype(float)
        return np.column_stack([conf, correct])

    Xm = _attack_features(X_train[member_idx], y_train[member_idx])
    Xn = _attack_features(X_test[nonmember_idx], y_test[nonmember_idx])

    pred_m, pred_n = attack_model.predict(Xm), attack_model.predict(Xn)
    tpr = float(np.mean(pred_m == 1))
    fpr = float(np.mean(pred_n == 1))
    return {
        "mia_accuracy": (np.sum(pred_m == 1) + np.sum(pred_n == 0)) / (len(pred_m) + len(pred_n)),
        "mia_advantage": tpr - fpr,
        "tpr": tpr,
        "fpr": fpr,
        "attack_type": "shadow",
        "member_scores": attack_model.predict_proba(Xm)[:, 1],
        "nonmember_scores": attack_model.predict_proba(Xn)[:, 1],
    }


# ---------------------------------------------------------------------------
# LiRA
# ---------------------------------------------------------------------------
def lira_scaled_logit(p_true: np.ndarray, clip: float = 1e-6) -> np.ndarray:
    """Scaled logit of the true-class probability."""
    p = np.clip(np.asarray(p_true, float), clip, 1 - clip)
    return np.log(p) - np.log(1 - p)


def lira_collect(pool_X, pool_y, fit_shadow, confidence, n_shadow, in_frac, seed):
    """Train shadows on random halves. Returns per-shadow logits and in-masks."""
    n = len(pool_X)
    n_in = max(1, int(round(in_frac * n)))
    rng = np.random.default_rng(seed)
    phi = np.empty((n_shadow, n))
    in_mask = np.zeros((n_shadow, n), bool)
    for s in range(n_shadow):
        idx = rng.choice(n, size=n_in, replace=False)
        in_mask[s, idx] = True
        m = fit_shadow(pool_X[idx], pool_y[idx], int(rng.integers(1 << 31)))
        phi[s] = lira_scaled_logit(confidence(m, pool_X, pool_y))
    return phi, in_mask


def lira_score(phi, in_mask, target_phi, labels, fix_variance: bool = True):
    """Per-example online and offline LiRA scores from shadow logits."""
    out_mask = ~in_mask
    n_in, n_out = in_mask.sum(0), out_mask.sum(0)
    mu_in = np.where(in_mask, phi, 0).sum(0) / np.maximum(n_in, 1)
    mu_out = np.where(out_mask, phi, 0).sum(0) / np.maximum(n_out, 1)

    def _var(mask, mu, nn):
        d = np.where(mask, phi - mu[None, :], 0)
        return (d ** 2).sum(0) / np.maximum(nn - 1, 1)

    v_in, v_out = _var(in_mask, mu_in, n_in), _var(out_mask, mu_out, n_out)
    if fix_variance:
        pooled = np.concatenate([v_in[n_in > 1], v_out[n_out > 1]])
        gv = float(np.median(pooled)) if pooled.size else 1.0
        sg_in = sg_out = np.full(len(labels), np.sqrt(max(gv, 1e-8)))
    else:
        sg_in = np.sqrt(np.maximum(v_in, 1e-8))
        sg_out = np.sqrt(np.maximum(v_out, 1e-8))

    online = norm.logpdf(target_phi, loc=mu_in, scale=sg_in) \
        - norm.logpdf(target_phi, loc=mu_out, scale=sg_out)
    offline = (target_phi - mu_out) / sg_out
    return online, offline


def lira_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Attack AUC, NaN when labels are degenerate."""
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def lira_tpr_at_fpr(scores, labels, fprs=(1e-3, 1e-2, 1e-1)) -> dict:
    """TPR at the requested FPR levels."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return {f: float(np.interp(f, fpr, tpr)) for f in fprs}


def mia_attack_lira(
    model,
    X_mem: np.ndarray, y_mem: np.ndarray,
    X_non: np.ndarray, y_non: np.ndarray,
    rule: str, epsilon: float, model_type: str, seed: int, prevalence: float,
    public_stats: dict | None,
    n_shadow: int = N_LIRA_SHADOWS,
    sample_size: int = MIA_SAMPLE_SIZE,
) -> dict:
    """Per-example LiRA with shadows on the same DP path as the target."""
    rng = np.random.default_rng(seed)
    n_m, n_n = min(sample_size, len(X_mem)), min(sample_size, len(X_non))
    mi = rng.choice(len(X_mem), n_m, replace=False)
    ni = rng.choice(len(X_non), n_n, replace=False)
    pool_X = np.vstack([X_mem[mi], X_non[ni]])
    pool_y = np.concatenate([y_mem[mi], y_non[ni]])
    labels = np.concatenate([np.ones(n_m), np.zeros(n_n)]).astype(int)

    def fit_shadow(Xin, yin, s):
        m, _ = train_model(Xin, yin, model_type, rule, epsilon, s, prevalence, public_stats)
        return m

    phi, in_mask = lira_collect(
        pool_X, pool_y, fit_shadow, true_class_confidence, n_shadow, 0.5, seed
    )
    target_phi = lira_scaled_logit(true_class_confidence(model, pool_X, pool_y))
    online, offline = lira_score(phi, in_mask, target_phi, labels, fix_variance=True)
    t = lira_tpr_at_fpr(online, labels)
    return {
        "lira_auc": lira_auc(online, labels),
        "lira_auc_offline": lira_auc(offline, labels),
        "lira_tpr_at_fpr_001": t[1e-3],
        "lira_tpr_at_fpr_01": t[1e-2],
        "lira_tpr_at_fpr_1": t[1e-1],
        "lira_n_shadow": n_shadow,
    }


# ---------------------------------------------------------------------------
# Reconstruction (candidate search)
# ---------------------------------------------------------------------------
def reconstruction_attack(
    model,
    X_train_original: np.ndarray,
    n_targets: int = N_ATTACK_TARGETS,
    n_candidates: int = N_RECONSTRUCTION_CANDIDATES,
    seed: int = 42,
) -> tuple[float, float]:
    """Candidate-search reconstruction against clean records. Returns MAE mean and std."""
    rng = np.random.default_rng(seed)
    n_targets = min(n_targets, len(X_train_original))
    target_indices = rng.choice(len(X_train_original), n_targets, replace=False)

    # candidate bounds from the clean matrix, noisy bounds would inflate MAE
    feature_mins = X_train_original.min(axis=0)
    feature_maxs = X_train_original.max(axis=0)

    def _proba(X):
        try:
            return model.predict_proba(X)
        except AttributeError:
            preds = model.predict(X)
            return np.column_stack([1 - preds, preds])

    mae_list = []
    for idx in target_indices:
        target_pred = _proba(X_train_original[idx:idx + 1])[0]
        candidates = rng.uniform(
            feature_mins, feature_maxs, (n_candidates, X_train_original.shape[1])
        )
        distances = np.sum((_proba(candidates) - target_pred) ** 2, axis=1)
        best = candidates[np.argmin(distances)]
        mae_list.append(np.mean(np.abs(X_train_original[idx] - best)))

    return float(np.mean(mae_list)), float(np.std(mae_list, ddof=1))


# ---------------------------------------------------------------------------
# Attribute inference
# ---------------------------------------------------------------------------
def attack_candidates(df_train, df_test, attribute: str, attribute_type: str):
    """Candidate values and ground-truth bin index per test record."""
    if attribute_type == "binary":
        candidates = np.array(sorted(df_train[attribute].unique()), dtype=float)
        true_bins = np.array([
            int(np.argmin(np.abs(candidates - v))) for v in df_test[attribute].values
        ])
        return candidates, true_bins, None

    if attribute_type == "continuous":
        # bins fit on train only, test values clipped into range
        discretizer = KBinsDiscretizer(
            n_bins=N_BINS_CONTINUOUS, encode="ordinal", strategy="quantile"
        )
        discretizer.fit(df_train[[attribute]].values)
        edges = discretizer.bin_edges_[0]
        candidates = 0.5 * (edges[:-1] + edges[1:])
        true_bins = discretizer.transform(df_test[[attribute]].values).astype(int).flatten()
        true_bins = np.clip(true_bins, 0, N_BINS_CONTINUOUS - 1)
        return candidates, true_bins, edges

    raise ValueError(f"Unknown attribute_type: {attribute_type!r}")


def confidence_rank_attack(
    model,
    X_target: np.ndarray,
    y_target: np.ndarray,
    sensitive_idx: int,
    candidates: np.ndarray,
    knn_prevalence: float | None = None,
) -> np.ndarray:
    """Pick the candidate value that maximizes true-class confidence per target."""
    from .models import correct_knn_proba  # local import avoids a cycle

    n_targets, n_cands = X_target.shape[0], len(candidates)
    X_query = np.repeat(X_target, n_cands, axis=0)
    X_query[:, sensitive_idx] = np.tile(candidates, n_targets)

    proba = model.predict_proba(X_query)
    if knn_prevalence is not None:
        proba = correct_knn_proba(proba, knn_prevalence)

    y_tiled = np.repeat(y_target, n_cands)
    conf = proba[np.arange(len(proba)), y_tiled]
    return np.argmax(conf.reshape(n_targets, n_cands), axis=1)


def attribute_attack_metrics(
    predicted_bins: np.ndarray,
    true_bins: np.ndarray,
    attribute_type: str,
) -> dict:
    """Top-1 accuracy, majority baseline, advantage, and within-1 for continuous."""
    counts = np.bincount(true_bins)
    majority_baseline = float((true_bins == int(np.argmax(counts))).mean())
    top1 = float((predicted_bins == true_bins).mean())
    out = {
        "top1_acc": top1,
        "majority_baseline": majority_baseline,
        "advantage_over_baseline": top1 - majority_baseline,
    }
    if attribute_type == "continuous":
        out["within1_acc"] = float((np.abs(predicted_bins - true_bins) <= 1).mean())
    return out
