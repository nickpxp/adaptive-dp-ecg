"""Why the rules differ in utility but not privacy.

Four checks, all calibrated on the public donor through src.rules.rule_scales:
  1. total injected noise per rule at matched eps
  2. post-noise SNR on signal-bearing features
  3. variance vs classifier reliance (Spearman)
  4. GNB variance inflation and RF importance stability (--fit, trains models)

Writes CSVs under results/rule_analysis/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import _path  # noqa: F401
from src.constants import DATASETS, DESCRIPTOR_PREVALENCE, NOISE_SEEDS, TRAIN_FOLDS
from src.data import (
    feature_columns,
    load_dataframe,
    load_public_stats,
)
from src.rules import apply_noise, rule_scales

EPS_GRID = (0.1, 0.5, 1.0, 2.0, 5.0)
EPS_HEADLINE = 1.0  # headline operating point
TOP_SIGNAL_FRAC = 0.25


def training_matrix(ds: str, data_dir: str):
    """Training-fold X, y, and column names for one dataset."""
    df = load_dataframe(ds, data_dir)
    cols = feature_columns(df)
    sub = df.loc[df["strat_fold"].isin(TRAIN_FOLDS)]
    X = np.where(np.isfinite(sub[cols].to_numpy(dtype=np.float64)), 
                 sub[cols].to_numpy(dtype=np.float64), 0.0)
    y = sub["af_detected"].to_numpy(dtype=int)
    return X, y, cols


def total_noise_variance(scales: np.ndarray) -> float:
    """Per-record injected noise variance: Laplace(b) has variance 2*b^2."""
    return float(np.sum(2.0 * np.asarray(scales, dtype=np.float64) ** 2))


# ---------------------------------------------------------------------------
# Check 1: total injected noise at matched nominal eps
# ---------------------------------------------------------------------------
def check_noise_totals(data_dir: str, out_dir: Path) -> None:
    """Total injected noise per rule, relative to Uniform."""
    rows = []
    for ds in DATASETS:
        ps = load_public_stats(ds, data_dir)
        for eps in EPS_GRID:
            u = total_noise_variance(rule_scales("uniform", ps["ranges"], None, eps))
            a = total_noise_variance(
                rule_scales("adaptive_feature", ps["ranges"], ps["variances"], eps)
            )
            rows.append({
                "dataset": ds, "eps": eps,
                "uniform_noise_var": u, "adaptive_noise_var": a,
                "ratio_adaptive_over_uniform": a / u if u > 0 else np.nan,
            })
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "noise_totals.csv", index=False)
    print("\nTotal injected noise variance per record (lower = less privacy):")
    print(res.to_string(index=False))
    mean_ratio = res["ratio_adaptive_over_uniform"].mean()
    # ratio < 1 means less total noise, not a free utility win
    if mean_ratio < 0.9:
        verdict = "AdaptiveFeature injects materially LESS noise: effective-privacy artifact"
    elif mean_ratio > 1.1:
        verdict = "AdaptiveFeature injects MORE noise: gain is real, not a budget artifact"
    else:
        verdict = "roughly equal: gain is allocation-driven, not a budget artifact"
    print(f"mean ratio {mean_ratio:.3f} -> {verdict}")


# ---------------------------------------------------------------------------
# Check 2: post-noise SNR redistribution
# ---------------------------------------------------------------------------
def check_snr(data_dir: str, out_dir: Path) -> None:
    """Post-noise SNR on signal-bearing features, AdaptiveFeature vs Uniform."""
    rows = []
    print(f"\nPer-feature SNR after noise at eps={EPS_HEADLINE} "
          f"(signal = |class mean diff|, noise std = scale*sqrt(2))")
    for ds in DATASETS:
        X, y, _ = training_matrix(ds, data_dir)
        signal = np.abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
        ps = load_public_stats(ds, data_dir)

        su = rule_scales("uniform", ps["ranges"], None, EPS_HEADLINE)
        sa = rule_scales("adaptive_feature", ps["ranges"], ps["variances"], EPS_HEADLINE)
        snr_u = signal / (su * np.sqrt(2) + 1e-12)
        snr_a = signal / (sa * np.sqrt(2) + 1e-12)

        order = np.argsort(signal)[::-1]
        k = max(1, int(TOP_SIGNAL_FRAC * len(signal)))
        top = order[:k]
        rows.append({
            "dataset": ds,
            "snr_ratio_top": float(np.median(snr_a[top] / (snr_u[top] + 1e-12))),
            "snr_ratio_all": float(np.median(snr_a / (snr_u + 1e-12))),
            "frac_top_better": float(np.mean(snr_a[top] > snr_u[top])),
            "k": k,
            "d": len(signal),
        })
        print(f"  {ds}: top-{k} SNR ratio={rows[-1]['snr_ratio_top']:.3f}  "
              f"all={rows[-1]['snr_ratio_all']:.3f}  "
              f"top better={rows[-1]['frac_top_better']:.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "snr_redistribution.csv", index=False)
    mean_top, mean_all = res["snr_ratio_top"].mean(), res["snr_ratio_all"].mean()
    if mean_top > 1.05 and mean_top > mean_all:
        print("CONFIRMED: extra noise lands on low-signal features, signal features keep better SNR.")
    elif mean_top > 1.05:
        print("PARTIAL: signal-feature SNR higher under adaptive, but not "
              "clearly above the all-feature ratio.")
    else:
        print("NOT confirmed: adaptive does not preserve higher SNR on "
              "signal features; report the measured ratios honestly.")


# ---------------------------------------------------------------------------
# Check 3: variance vs classifier reliance
# ---------------------------------------------------------------------------
def nb_mean_separation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|class mean difference| / pooled SD per feature."""
    x0, x1 = X[y == 0], X[y == 1]
    pooled = np.sqrt((x0.var(axis=0, ddof=1) + x1.var(axis=0, ddof=1)) / 2.0)
    pooled = np.where(pooled > 0, pooled, 1e-10)
    return np.abs(x1.mean(axis=0) - x0.mean(axis=0)) / pooled


def check_alignment(data_dir: str, out_dir: Path) -> None:
    """Spearman of per-feature variance against RF, NB and LR reliance."""
    summaries = []
    for ds in DATASETS:
        X, y, cols = training_matrix(ds, data_dir)
        print(f"\n=== {ds} === features={X.shape[1]}, "
              f"positives={int(y.sum())}/{len(y)} ({y.mean() * 100:.2f}%)")
        variance = X.var(axis=0)
        rf_imp = RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ).fit(X, y).feature_importances_
        nb_sep = nb_mean_separation(X, y)
        lr_coef = np.abs(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            .fit(StandardScaler().fit_transform(X), y).coef_.ravel()
        )

        pd.DataFrame({
            "feature": cols, "variance": variance, "rf_importance": rf_imp,
            "nb_mean_separation": nb_sep, "lr_abs_coef_std": lr_coef,
        }).sort_values("variance").to_csv(
            out_dir / f"variance_importance_{ds}.csv", index=False
        )

        out = {}
        for label, imp, key in (("RF importance", rf_imp, "rf"),
                                ("NB mean-separation", nb_sep, "nb"),
                                ("LR |coef| (std)", lr_coef, "lr")):
            rho, p = spearmanr(variance, imp)
            out[key] = rho
            if rho < -0.15 and p < 0.05:
                reading = "ALIGNED (low-var = relied-on)"
            elif rho > 0.15 and p < 0.05:
                reading = "ANTI (high-var = relied-on)"
            else:
                reading = "no clear alignment"
            print(f"  {label:<22}rho={rho:>7.3f}  p={p:.2e}  {reading}")
        out["dataset"] = ds
        out["aligned"] = (out["rf"] < -0.15) and (out["nb"] < -0.15)
        summaries.append(out)

    res = pd.DataFrame(summaries)
    res.to_csv(out_dir / "variance_alignment_summary.csv", index=False)
    n = int(res["aligned"].sum())
    print(f"\nNB/RF low-variance alignment holds in {n}/{len(res)} datasets.")


# ---------------------------------------------------------------------------
# Check 4: fitted-model diagnostics (GNB variance inflation, RF stability)
# ---------------------------------------------------------------------------
def check_fitted_models(data_dir: str, out_dir: Path, eps: float = EPS_HEADLINE) -> None:
    """Fitted-model diagnostics over 30 runs at matched eps.

    GNB: does noise inflate the per-class variances equally across classes?
    RF: does the importance ranking hold between Uniform and AdaptiveFeature?
    """
    from sklearn.naive_bayes import GaussianNB

    for ds in DATASETS:
        X, y, _ = training_matrix(ds, data_dir)
        ps = load_public_stats(ds, data_dir)
        pub_var = np.asarray(ps["variances"], float)
        prevalence = DESCRIPTOR_PREVALENCE[ds]  # public, as in every runner

        def _noised(rule, seed):
            return apply_noise(X, rule, eps, seed, public_stats=ps)

        def fit_gnb(rule, seed):
            m = GaussianNB(priors=np.asarray([1.0 - prevalence, prevalence]),
                           var_smoothing=1e-3)
            m.fit(_noised(rule, seed), y)
            return m

        def fit_rf(rule, seed):
            m = RandomForestClassifier(n_estimators=300,
                                       class_weight="balanced_subsample",
                                       random_state=seed, n_jobs=-1)
            m.fit(_noised(rule, seed), y)
            return m

        # (1) GNB per-class variance inflation vs the no-DP baseline
        gnb_rules = ("baseline", "uniform", "adaptive_feature", "adaptive_feature_reversed")
        var_by_rule = {r: [] for r in gnb_rules}
        for seed in NOISE_SEEDS:
            for rule in gnb_rules:
                var_by_rule[rule].append(np.asarray(fit_gnb(rule, seed).var_))
        base = np.mean(var_by_rule["baseline"], axis=0)  # [2, n_features]

        gnb_rows = []
        for rule in gnb_rules[1:]:
            infl = np.mean(var_by_rule[rule], axis=0) - base
            rho, p = spearmanr(infl[0], infl[1])
            ratio = (np.mean(infl[1]) / np.mean(infl[0])
                     if np.mean(infl[0]) != 0 else np.nan)
            rho_v, p_v = spearmanr(pub_var, infl.mean(0))
            gnb_rows.append({
                "dataset": ds, "mechanism": rule,
                "mean_inflation_class0": float(np.mean(infl[0])),
                "mean_inflation_class1": float(np.mean(infl[1])),
                "class1_over_class0_ratio": float(ratio),
                "across_class_inflation_spearman": float(rho),
                "across_class_inflation_p": float(p),
                "inflation_vs_pubvar_spearman": float(rho_v),
                "inflation_vs_pubvar_p": float(p_v),
            })
            print(f"[GNB {ds} {rule:26s}] c0={np.mean(infl[0]):.3g} "
                  f"c1={np.mean(infl[1]):.3g} ratio={ratio:.3f} rho={rho:.3f}")
        pd.DataFrame(gnb_rows).to_csv(out_dir / f"gnb_inflation_{ds}.csv", index=False)

        # (2) RF split-importance stability, Uniform vs AdaptiveFeature
        imp = {"uniform": [], "adaptive_feature": []}
        for seed in NOISE_SEEDS:
            for rule in imp:
                imp[rule].append(np.asarray(fit_rf(rule, seed).feature_importances_))
        imp_u = np.mean(imp["uniform"], axis=0)
        imp_a = np.mean(imp["adaptive_feature"], axis=0)
        rho_imp, p_imp = spearmanr(imp_u, imp_a)
        rf_rows = []
        for k in (5, 10):
            top_u = set(np.argsort(imp_u)[-k:])
            top_a = set(np.argsort(imp_a)[-k:])
            rf_rows.append({
                "dataset": ds, "k": k,
                "topk_overlap": len(top_u & top_a) / k,
                "importance_rank_spearman": float(rho_imp),
                "importance_rank_p": float(p_imp),
            })
        pd.DataFrame(rf_rows).to_csv(out_dir / f"rf_stability_{ds}.csv", index=False)
        print(f"[RF {ds}] rank Spearman={rho_imp:.3f}  "
              f"top5={rf_rows[0]['topk_overlap']:.2f}  "
              f"top10={rf_rows[1]['topk_overlap']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--fit", action="store_true",
                        help="also run the fitted-model diagnostics (slow)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / "rule_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    check_noise_totals(args.data_dir, out_dir)
    check_snr(args.data_dir, out_dir)
    check_alignment(args.data_dir, out_dir)
    if args.fit:
        check_fitted_models(args.data_dir, out_dir)
    print(f"\nrule analysis -> {out_dir}")


if __name__ == "__main__":
    main()
