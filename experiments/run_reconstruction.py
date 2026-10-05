"""Reconstruction sweep.

Writes under results/reconstruction_<dataset>/:
  results_reconstruction_<dataset>.csv   raw per-run records
  metrics_reconstruction_<dataset>.csv   aggregates + protection gain/utility loss
  stats_reconstruction_<dataset>.csv     Welch t + Cohen's d over rule pairs
  *.png                                  diagnostic plots
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import Parallel, delayed

import _path  # noqa: F401
from src.attacks import reconstruction_attack
from src.constants import (
    DATASETS,
    EPSILON_VALUES,
    MODEL_TYPES,
    N_JOBS,
    NOISE_SEEDS,
    RULE_COLORS,
    RULE_LABELS,
    RULES,
)
from src.data import load_dataset, load_public_stats, verify_prevalence
from src.metrics import compute_95_ci, evaluate_classifier, pairwise_rule_tests
from src.models import correct_knn_proba, threshold_prevalence_matched, train_model

UTIL_COLS = (
    "util_auroc", "util_auc_pr", "util_mcc", "util_f1_at_tau",
    "util_balanced_accuracy", "util_brier", "util_sensitivity", "util_specificity",
)


def run_single_experiment(data, model_type, rule, epsilon, seed, prevalence, public_stats):
    """One cell. Train under the rule, score utility, run the attack."""
    start = time.time()
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    model, _ = train_model(X_tr, y_tr, model_type, rule, epsilon, seed, prevalence, public_stats)

    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="weighted")
    try:
        proba = model.predict_proba(X_te)
        if model_type == "knn":
            proba = correct_knn_proba(proba, prevalence)
        y_proba = proba[:, 1]
        auc = roc_auc_score(y_te, y_proba)
        tau = threshold_prevalence_matched(prevalence)
        report = evaluate_classifier(y_te, y_proba, threshold=tau).to_dict()
    except (AttributeError, ValueError, IndexError):
        auc, report = 0.5, {}

    # the attack targets the original clean records
    mae_mean, mae_std = reconstruction_attack(model, X_tr, seed=seed)

    record = {
        "model_type": model_type,
        "mechanism": rule,
        "epsilon": epsilon,
        "seed": seed,
        "accuracy": accuracy,
        "f1_score": f1,
        "auc": auc,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "runtime_seconds": time.time() - start,
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    record.update({f"util_{k}": v for k, v in report.items()})
    return record


def run_all_experiments(data, public_stats) -> pd.DataFrame:
    """Full (model x rule x epsilon x seed) sweep for one dataset."""
    ds = data["dataset_name"]
    prevalence = verify_prevalence(ds, data["y_train"])
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
                  f"MAE={np.mean([r['mae_mean'] for r in results]):.4f}  "
                  f"Acc={np.mean([r['accuracy'] for r in results]):.4f}", flush=True)

    df = pd.DataFrame(all_results)
    df["dataset"] = ds
    df["prevalence"] = prevalence
    return df


def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MAE and accuracy with CIs, plus protection gain and utility loss."""
    records = []
    for model_type in MODEL_TYPES:
        md = results_df[results_df["model_type"] == model_type]
        baseline = md[md["mechanism"] == "baseline"]
        if len(baseline) == 0:
            continue
        baseline_mae = baseline["mae_mean"].mean()
        baseline_acc = baseline["accuracy"].mean()

        for rule in RULES:
            for epsilon in EPSILON_VALUES:
                sub = md[(md["mechanism"] == rule) & (md["epsilon"] == epsilon)]
                if len(sub) == 0:
                    continue
                mae_vals, acc_vals = sub["mae_mean"].values, sub["accuracy"].values
                mae_ci, acc_ci = compute_95_ci(mae_vals), compute_95_ci(acc_vals)
                row = {
                    "model_type": model_type,
                    "mechanism": rule,
                    "epsilon": epsilon,
                    "mae_mean": np.mean(mae_vals),
                    "mae_std": np.std(mae_vals, ddof=1),
                    "mae_ci_lower": mae_ci[0],
                    "mae_ci_upper": mae_ci[1],
                    "accuracy_mean": np.mean(acc_vals),
                    "accuracy_std": np.std(acc_vals, ddof=1),
                    "accuracy_ci_lower": acc_ci[0],
                    "accuracy_ci_upper": acc_ci[1],
                    "protection_gain_pct": (np.mean(mae_vals) - baseline_mae)
                        / (baseline_mae + 1e-10) * 100,
                    "utility_loss_pct": (baseline_acc - np.mean(acc_vals))
                        / (baseline_acc + 1e-10) * 100,
                    "baseline_mae": baseline_mae,
                    "baseline_accuracy": baseline_acc,
                    "n_runs": len(mae_vals),
                }
                for col in UTIL_COLS:
                    if col in sub.columns:
                        row[f"{col}_mean"] = float(np.nanmean(sub[col].values))
                        row[f"{col}_std"] = float(np.nanstd(sub[col].values, ddof=1))
                records.append(row)
    return pd.DataFrame(records)


def quick_plots(results_df, metrics_df, ds: str, out_dir: Path):
    """Diagnostic MAE, protection-gain, and trade-off panels."""
    def _panels(plot_one, fname, suptitle):
        fig, axes = plt.subplots(1, len(MODEL_TYPES), figsize=(5 * len(MODEL_TYPES), 5))
        for ax, model_type in zip(axes, MODEL_TYPES):
            plot_one(ax, model_type)
            ax.set_title(model_type.upper())
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _mae(ax, model_type):
        md = results_df[results_df["model_type"] == model_type]
        base = md[md["mechanism"] == "baseline"]
        if len(base):
            bl = base["mae_mean"].mean()
            ax.axhline(bl, color="red", linestyle="--", alpha=0.7, label=f"Baseline ({bl:.3f})")
        for rule in RULES:
            grouped = md[md["mechanism"] == rule].groupby("epsilon")["mae_mean"].agg(["mean", "std"])
            if len(grouped):
                ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                            label=RULE_LABELS[rule], marker="o", capsize=3,
                            color=RULE_COLORS[rule])
        ax.set_xlabel("per-feature budget eps")
        ax.set_ylabel("Reconstruction MAE")

    def _gain(ax, model_type):
        md = metrics_df[metrics_df["model_type"] == model_type]
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        for rule in RULES:
            rd = md[md["mechanism"] == rule]
            if len(rd):
                ax.plot(rd["epsilon"], rd["protection_gain_pct"],
                        label=RULE_LABELS[rule], marker="o", color=RULE_COLORS[rule])
        ax.set_xlabel("per-feature budget eps")
        ax.set_ylabel("Protection Gain (%)")

    def _tradeoff(ax, model_type):
        md = metrics_df[metrics_df["model_type"] == model_type]
        for rule in RULES:
            rd = md[md["mechanism"] == rule]
            if len(rd):
                ax.scatter(rd["accuracy_mean"], rd["mae_mean"],
                           label=RULE_LABELS[rule], s=50, color=RULE_COLORS[rule])
                ax.plot(rd["accuracy_mean"], rd["mae_mean"], alpha=0.5,
                        color=RULE_COLORS[rule])
        ax.set_xlabel("Model Accuracy")
        ax.set_ylabel("Reconstruction MAE")

    _panels(_mae, f"reconstruction_mae_{ds}.png", f"Reconstruction MAE - {ds.upper()}")
    _panels(_gain, f"protection_gain_{ds}.png", f"Protection Gain - {ds.upper()}")
    _panels(_tradeoff, f"privacy_utility_{ds}.png", f"Privacy-Utility Trade-off - {ds.upper()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(_path.SINGLE_FLAG, dest="single", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if len(args.datasets) > 1 and not args.single:
        raise SystemExit(_path.run_per_dataset(__file__, args.datasets))

    for ds in args.datasets:
        out_dir = Path(args.out_dir) / f"reconstruction_{ds}"
        out_dir.mkdir(parents=True, exist_ok=True)
        public_stats = load_public_stats(ds, args.data_dir)
        data = load_dataset(ds, args.data_dir)

        results_df = run_all_experiments(data, public_stats)
        results_df.to_csv(out_dir / f"results_reconstruction_{ds}.csv", index=False)

        metrics_df = compute_metrics(results_df)
        metrics_df.to_csv(out_dir / f"metrics_reconstruction_{ds}.csv", index=False)

        stats_df = pairwise_rule_tests(results_df, metric="mae_mean")
        stats_df.to_csv(out_dir / f"stats_reconstruction_{ds}.csv", index=False)

        quick_plots(results_df, metrics_df, ds, out_dir)
        print(f"--- {ds} complete -> {out_dir} ---")


if __name__ == "__main__":
    main()
