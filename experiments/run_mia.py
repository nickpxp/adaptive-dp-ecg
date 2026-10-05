"""Membership inference sweep, shadow-model and LiRA, plus the per-class split.

Writes under results/mia_<dataset>/:
  results_mia_<dataset>.csv     raw per-run records
  summary_mia_<dataset>.csv     mean/std/CI per (model, rule, epsilon)
  stats_mia_<dataset>.csv       Welch t + Cohen's d over rule pairs
  perclass_mia_<dataset>.csv    minority/majority LiRA split (--perclass)
  perclass_generalization_<dataset>.csv
  lira_scores_<dataset>.npz     raw scores for the log-log ROC figure
  *.png                         quick diagnostic plots
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to disk; never open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve
from joblib import Parallel, delayed

import _path  # noqa: F401  (repo root on sys.path, warnings, per-dataset isolation)
from src.attacks import (
    lira_collect,
    lira_scaled_logit,
    lira_score,
    mia_attack_lira,
    mia_attack_shadow,
    true_class_confidence,
)
from src.constants import (
    DATASETS,
    EPSILON_VALUES,
    MIA_SAMPLE_SIZE,
    MODEL_TYPES,
    N_JOBS,
    N_LIRA_SHADOWS,
    NOISE_SEEDS,
    RULE_COLORS,
    RULE_LABELS,
    RULES,
)
from src.data import load_dataset, load_public_stats, verify_prevalence
from src.metrics import compute_95_ci, evaluate_classifier, mia_auc, mia_tpr_at_fpr, pairwise_rule_tests
from src.models import correct_knn_proba, threshold_prevalence_matched, train_model

SUMMARY_EXTRA_COLS = (
    "mia_auc", "mia_tpr_at_fpr_001", "mia_tpr_at_fpr_01", "mia_tpr_at_fpr_1",
    # LiRA columns, so the summary stands on its own
    "lira_auc", "lira_auc_offline",
    "lira_tpr_at_fpr_001", "lira_tpr_at_fpr_01", "lira_tpr_at_fpr_1",
    "util_auroc", "util_auc_pr", "util_mcc", "util_f1_at_tau",
    "util_balanced_accuracy", "util_brier", "util_sensitivity", "util_specificity",
)


def run_single_experiment(data, model_type, rule, epsilon, seed, prevalence, public_stats):
    """One (model, rule, epsilon, seed) cell. Train, score utility, run both attacks."""
    start = time.time()
    X_tr, y_tr = data["X_target_train"], data["y_target_train"]
    # non-members come from the shadow pool, folds 5-8
    X_non, y_non = data["X_shadow_pool"], data["y_shadow_pool"]
    X_te, y_te = data["X_test"], data["y_test"]

    model, _ = train_model(X_tr, y_tr, model_type, rule, epsilon, seed, prevalence, public_stats)

    # utility on the clean test fold
    model_accuracy = accuracy_score(y_te, model.predict(X_te))
    try:
        proba = model.predict_proba(X_te)
        if model_type == "knn":
            proba = correct_knn_proba(proba, prevalence)
        tau = threshold_prevalence_matched(prevalence)
        util = evaluate_classifier(y_te, proba[:, 1], threshold=tau).to_dict()
    except (AttributeError, ValueError, IndexError):
        util = {}

    # both attacks query the target on clean records
    shadow = mia_attack_shadow(
        model, X_tr, y_tr, X_non, y_non, X_non, y_non,
        rule, epsilon, model_type, seed, prevalence, public_stats,
    )
    attack_auc = mia_auc(shadow["member_scores"], shadow["nonmember_scores"])
    tpr_at_fpr = mia_tpr_at_fpr(shadow["member_scores"], shadow["nonmember_scores"])

    try:
        lira = mia_attack_lira(
            model, X_tr, y_tr, X_non, y_non,
            rule, epsilon, model_type, seed, prevalence, public_stats,
        )
    except Exception:
        lira = {k: float("nan") for k in (
            "lira_auc", "lira_auc_offline", "lira_tpr_at_fpr_001",
            "lira_tpr_at_fpr_01", "lira_tpr_at_fpr_1")}
        lira["lira_n_shadow"] = N_LIRA_SHADOWS

    record = {
        "dataset": data["dataset_name"],
        "model_type": model_type,
        "mechanism": rule,
        "epsilon": epsilon,
        "seed": seed,
        "model_accuracy": model_accuracy,
        "mia_accuracy": shadow["mia_accuracy"],
        "mia_advantage": shadow["mia_advantage"],
        "tpr": shadow["tpr"],
        "fpr": shadow["fpr"],
        "attack_type": shadow["attack_type"],
        "mia_auc": attack_auc,
        "mia_tpr_at_fpr_001": tpr_at_fpr.get(0.001, float("nan")),
        "mia_tpr_at_fpr_01": tpr_at_fpr.get(0.01, float("nan")),
        "mia_tpr_at_fpr_1": tpr_at_fpr.get(0.1, float("nan")),
        **lira,
        "runtime_seconds": time.time() - start,
    }
    record.update({f"util_{k}": v for k, v in util.items()})
    return record


def run_all_experiments(data, public_stats) -> pd.DataFrame:
    """Full (model x rule x epsilon x seed) sweep for one dataset."""
    ds = data["dataset_name"]
    # public descriptor value, labels are only checked against it
    prevalence = verify_prevalence(ds, data["y_target_train"])
    print(f"\n[{ds}] descriptor prevalence={prevalence:.4f}", flush=True)

    all_results = []
    for model_type in MODEL_TYPES:
        print(f"\n[{ds}] MODEL: {model_type.upper()}")
        cells = [("baseline", 0.0)] + [(r, e) for r in RULES for e in EPSILON_VALUES]
        for rule, epsilon in cells:
            results = Parallel(n_jobs=N_JOBS, verbose=0, max_nbytes=None)(
                delayed(run_single_experiment)(
                    data, model_type, rule, epsilon, seed, prevalence, public_stats
                ) for seed in NOISE_SEEDS
            )
            all_results.extend(results)
            print(f"  {rule:26s} eps={epsilon}: "
                  f"MIA={np.mean([r['mia_accuracy'] for r in results]):.4f}  "
                  f"AUROC={np.mean([r.get('util_auroc', np.nan) for r in results]):.4f}",
                  flush=True)

    df = pd.DataFrame(all_results)
    df["prevalence"] = prevalence
    return df


def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std and CI per cell from the raw sweep."""
    records = []
    for model_type in MODEL_TYPES:
        md = results_df[results_df["model_type"] == model_type]
        for rule in ("baseline",) + RULES:
            for epsilon in ((0.0,) if rule == "baseline" else EPSILON_VALUES):
                sub = md[(md["mechanism"] == rule) & (md["epsilon"] == epsilon)]
                if len(sub) == 0:
                    continue
                mia_ci = compute_95_ci(sub["mia_accuracy"].values)
                adv_ci = compute_95_ci(sub["mia_advantage"].values)
                row = {
                    "model_type": model_type,
                    "mechanism": rule,
                    "epsilon": epsilon,
                    "mia_accuracy_mean": sub["mia_accuracy"].mean(),
                    "mia_accuracy_std": sub["mia_accuracy"].std(ddof=1),
                    "mia_accuracy_ci_lower": mia_ci[0],
                    "mia_accuracy_ci_upper": mia_ci[1],
                    "mia_advantage_mean": sub["mia_advantage"].mean(),
                    "mia_advantage_std": sub["mia_advantage"].std(ddof=1),
                    "mia_advantage_ci_lower": adv_ci[0],
                    "mia_advantage_ci_upper": adv_ci[1],
                    "tpr_mean": sub["tpr"].mean(),
                    "fpr_mean": sub["fpr"].mean(),
                    "model_accuracy_mean": sub["model_accuracy"].mean(),
                    "model_accuracy_std": sub["model_accuracy"].std(ddof=1),
                    "n_runs": len(sub),
                }
                for col in SUMMARY_EXTRA_COLS:
                    if col in sub.columns:
                        row[f"{col}_mean"] = float(np.nanmean(sub[col].values))
                        row[f"{col}_std"] = float(np.nanstd(sub[col].values, ddof=1))
                records.append(row)
    return pd.DataFrame(records)


def run_perclass(data, public_stats, out_dir: Path, epsilon: float = 1.0):
    """Per-class LiRA, minority AF vs majority."""
    ds = data["dataset_name"]
    X_mem, y_mem = data["X_target_train"], data["y_target_train"]
    X_non, y_non = data["X_shadow_pool"], data["y_shadow_pool"]
    X_tst, y_tst = data["X_test"], data["y_test"]
    prevalence = verify_prevalence(ds, y_mem)
    fpr_target = 1e-3

    def _subgroup_tpr(scores, labels):
        if len(np.unique(labels)) < 2 or len(labels) < 10:
            return np.nan
        fpr, tpr, _ = roc_curve(labels, scores)
        return float(np.interp(fpr_target, fpr, tpr))

    rows, gen_rows, roc_dump = [], [], {}
    for model_type in MODEL_TYPES:
        for rule in ("baseline",) + RULES:
            per = {"min_acc": [], "maj_acc": [], "min_tpr": [], "maj_tpr": [], "ovr_tpr": []}
            for run, seed in enumerate(NOISE_SEEDS):
                model, _ = train_model(
                    X_mem, y_mem, model_type, rule, epsilon, seed, prevalence, public_stats
                )
                rng = np.random.default_rng(seed)
                n_m = min(MIA_SAMPLE_SIZE, len(X_mem))
                n_n = min(MIA_SAMPLE_SIZE, len(X_non))
                mi = rng.choice(len(X_mem), n_m, replace=False)
                ni = rng.choice(len(X_non), n_n, replace=False)
                pool_X = np.vstack([X_mem[mi], X_non[ni]])
                pool_y = np.concatenate([y_mem[mi], y_non[ni]])
                labels = np.concatenate([np.ones(n_m), np.zeros(n_n)]).astype(int)
                classes = pool_y.astype(int)

                # shadows through the same train path as the target
                def fit_shadow(Xin, yin, s):
                    m, _ = train_model(
                        Xin, yin, model_type, rule, epsilon, s, prevalence, public_stats
                    )
                    return m

                phi, in_mask = lira_collect(
                    pool_X, pool_y, fit_shadow, true_class_confidence,
                    N_LIRA_SHADOWS, 0.5, seed,
                )
                target_phi = lira_scaled_logit(true_class_confidence(model, pool_X, pool_y))
                online, _ = lira_score(phi, in_mask, target_phi, labels, fix_variance=True)
                per["ovr_tpr"].append(_subgroup_tpr(online, labels))

                # per-class generalization gap, aggregates only
                for c in (0, 1):
                    trs, tes = y_mem.astype(int) == c, y_tst.astype(int) == c
                    if trs.sum() < 10 or tes.sum() < 10:
                        continue
                    gen_rows.append({
                        "model_type": model_type, "mechanism": rule,
                        "epsilon": epsilon, "run": run, "true_class": c,
                        "n_train": int(trs.sum()), "n_test": int(tes.sum()),
                        "train_acc": float((model.predict(X_mem[trs]) == c).mean()),
                        "test_acc": float((model.predict(X_tst[tes]) == c).mean()),
                        "train_conf": float(true_class_confidence(model, X_mem[trs], y_mem[trs]).mean()),
                        "test_conf": float(true_class_confidence(model, X_tst[tes], y_tst[tes]).mean()),
                    })
                # c=1 is the minority AF class
                for c, ka, kt in ((1, "min_acc", "min_tpr"), (0, "maj_acc", "maj_tpr")):
                    sel = classes == c
                    if sel.sum() < 10 or len(np.unique(labels[sel])) < 2:
                        continue
                    sc, lb = online[sel], labels[sel]
                    thr = np.median(sc)  # subgroup MIA accuracy at its own median
                    per[ka].append(float(np.mean((sc >= thr).astype(int) == lb)))
                    per[kt].append(_subgroup_tpr(sc, lb))
                if run == 0:  # keep one run's raw scores for the ROC figure
                    roc_dump[f"{model_type}|{rule}|scores"] = online
                    roc_dump[f"{model_type}|{rule}|labels"] = labels
                    roc_dump[f"{model_type}|{rule}|classes"] = classes

            def _ms(v):
                a = np.asarray(v, float)
                a = a[~np.isnan(a)]
                return (np.nan, np.nan) if a.size == 0 else (float(a.mean()), float(a.std()))

            min_acc, min_sd = _ms(per["min_acc"])
            maj_acc, maj_sd = _ms(per["maj_acc"])
            min_tpr, min_tpr_sd = _ms(per["min_tpr"])
            maj_tpr, maj_tpr_sd = _ms(per["maj_tpr"])
            ovr_tpr, _ = _ms(per["ovr_tpr"])
            rows.append({
                "dataset": ds, "model_type": model_type, "mechanism": rule,
                "epsilon": epsilon, "overall_tpr_0p1": ovr_tpr,
                "minority_AFpos_mia_acc": min_acc, "minority_AFpos_mia_acc_sd": min_sd,
                "majority_AFneg_mia_acc": maj_acc, "majority_AFneg_mia_acc_sd": maj_sd,
                "minority_AFpos_tpr_0p1": min_tpr, "minority_AFpos_tpr_0p1_sd": min_tpr_sd,
                "majority_AFneg_tpr_0p1": maj_tpr, "majority_AFneg_tpr_0p1_sd": maj_tpr_sd,
            })
            print(f"{model_type:14s} {rule:26s} minAcc={min_acc:.3f}+/-{min_sd:.3f} "
                  f"majAcc={maj_acc:.3f}+/-{maj_sd:.3f}")

    pd.DataFrame(rows).to_csv(out_dir / f"perclass_mia_{ds}.csv", index=False)
    pd.DataFrame(gen_rows).to_csv(out_dir / f"perclass_generalization_{ds}.csv", index=False)
    np.savez_compressed(out_dir / f"lira_scores_{ds}.npz", **roc_dump)
    print(f"wrote per-class tables and lira_scores_{ds}.npz")


def quick_plots(summary_df: pd.DataFrame, ds: str, out_dir: Path):
    """Quick diagnostic plots. The paper figures are in run_utility_figures."""
    for metric, ylabel, fname, ylim in (
        ("mia_accuracy", "MIA Accuracy", f"mia_accuracy_{ds}.png", (0.35, 0.75)),
        ("mia_advantage", "MIA Advantage (TPR - FPR)", f"mia_advantage_{ds}.png", None),
    ):
        fig, axes = plt.subplots(1, len(MODEL_TYPES), figsize=(5 * len(MODEL_TYPES), 5))
        for ax, model_type in zip(axes, MODEL_TYPES):
            md = summary_df[summary_df["model_type"] == model_type]
            ref = 0.5 if metric == "mia_accuracy" else 0.0
            ax.axhline(ref, color="gray", linestyle="--", alpha=0.6)
            base = md[md["mechanism"] == "baseline"]
            if metric == "mia_accuracy" and len(base):
                ax.axhline(base[f"{metric}_mean"].values[0], color="red",
                           linestyle=":", alpha=0.7, label="No DP")
            for rule in RULES:
                rd = md[md["mechanism"] == rule]
                if len(rd) == 0:
                    continue
                ax.errorbar(rd["epsilon"], rd[f"{metric}_mean"], yerr=rd[f"{metric}_std"],
                            label=RULE_LABELS[rule], marker="o", capsize=3,
                            color=RULE_COLORS[rule])
            ax.set_xlabel("per-feature budget eps")
            ax.set_ylabel(ylabel)
            ax.set_title(model_type.upper())
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--perclass", action="store_true",
                        help="also run the per-class LiRA tables (slow)")
    parser.add_argument("--perclass-only", action="store_true",
                        help="run ONLY the per-class tables, skipping a sweep already on disk")
    parser.add_argument("--summary-only", action="store_true",
                        help="rebuild summary and stats from the raw CSV, no refitting")
    parser.add_argument(_path.SINGLE_FLAG, dest="single", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # one process per dataset
    if len(args.datasets) > 1 and not args.single and not args.summary_only:
        raise SystemExit(_path.run_per_dataset(__file__, args.datasets))

    if args.summary_only:
        # re-aggregate from the raw sweep, no refit needed
        for ds in args.datasets:
            out_dir = Path(args.out_dir) / f"mia_{ds}"
            raw = out_dir / f"results_mia_{ds}.csv"
            if not raw.exists():
                raise FileNotFoundError(f"{raw} not found; run the sweep first")
            results_df = pd.read_csv(raw)
            compute_summary(results_df).to_csv(out_dir / f"summary_mia_{ds}.csv", index=False)
            pairwise_rule_tests(results_df, metric="mia_accuracy").to_csv(
                out_dir / f"stats_mia_{ds}.csv", index=False)
            print(f"--- {ds} summary rebuilt -> {out_dir} ---")
        return

    for ds in args.datasets:
        out_dir = Path(args.out_dir) / f"mia_{ds}"
        out_dir.mkdir(parents=True, exist_ok=True)
        public_stats = load_public_stats(ds, args.data_dir)
        data = load_dataset(ds, args.data_dir, mia=True)

        if args.perclass_only:
            # needs the sweep CSVs already on disk
            sweep = out_dir / f"results_mia_{ds}.csv"
            if not sweep.exists():
                raise FileNotFoundError(
                    f"{sweep} not found; --perclass-only assumes the sweep already "
                    f"ran. Drop the flag to run both."
                )
            run_perclass(data, public_stats, out_dir)
            print(f"--- {ds} per-class complete -> {out_dir} ---")
            continue

        results_df = run_all_experiments(data, public_stats)
        results_df.to_csv(out_dir / f"results_mia_{ds}.csv", index=False)

        summary_df = compute_summary(results_df)
        summary_df.to_csv(out_dir / f"summary_mia_{ds}.csv", index=False)

        stats_df = pairwise_rule_tests(results_df, metric="mia_accuracy")
        stats_df.to_csv(out_dir / f"stats_mia_{ds}.csv", index=False)

        quick_plots(summary_df, ds, out_dir)
        if args.perclass:
            run_perclass(data, public_stats, out_dir)
        print(f"--- {ds} complete -> {out_dir} ---")


if __name__ == "__main__":
    main()
