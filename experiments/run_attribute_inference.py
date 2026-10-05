"""Attribute inference sweep.

Writes under results/attribute_<dataset>/:
  results_attribute_<dataset>.csv   raw per-run attack records
  metrics_attribute_<dataset>.csv   per-run utility records
  summary_attribute_<dataset>.csv   grouped means/CIs per cell

Synthetic data is left out on purpose. With engineered correlations, attribute
inference measures population-level inference, not memorization.
"""
from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import _path  # noqa: F401
from src.attacks import attack_candidates, attribute_attack_metrics, confidence_rank_attack
from src.constants import (
    ATTRIBUTE_TYPES,
    DATASETS,
    EPSILON_VALUES,
    FOLD_COL,
    MODEL_TYPES,
    N_TARGET_RECORDS,
    NOISE_SEEDS,
    SENSITIVE_ATTRIBUTES,
    TEST_FOLDS,
    TRAIN_FOLDS,
)
from src.data import canonical_split, feature_columns, load_dataframe, load_public_stats, verify_prevalence
from src.metrics import ci95_halfwidth, evaluate_classifier
from src.models import correct_knn_proba, threshold_prevalence_matched, train_model


def prepare_dataset(ds: str, data_dir: str):
    """Load, impute, split and precompute attack candidates for one dataset."""
    df = load_dataframe(ds, data_dir)
    missing = [a for a in SENSITIVE_ATTRIBUTES if a not in df.columns]
    if missing:
        raise ValueError(
            f"{ds} is missing sensitive attributes {missing}; the attack "
            f"needs them as feature columns — check the extraction step"
        )
    df = df.dropna(subset=list(SENSITIVE_ATTRIBUTES)).reset_index(drop=True)

    cols = feature_columns(df)
    df[cols] = df[cols].fillna(df[cols].median())
    splits = canonical_split(df, feature_cols=cols)
    df_train = df[df[FOLD_COL].isin(TRAIN_FOLDS)].reset_index(drop=True)
    df_test = df[df[FOLD_COL].isin(TEST_FOLDS)].reset_index(drop=True)

    setup = {}
    for attr in SENSITIVE_ATTRIBUTES:
        candidates, true_bins, _ = attack_candidates(
            df_train, df_test, attr, ATTRIBUTE_TYPES[attr]
        )
        setup[attr] = {
            "candidates": candidates,
            "true_bins_test": true_bins,
            "feature_idx": cols.index(attr),
        }
    return splits, setup


def run_dataset(ds: str, data_dir: str, out_dir: Path):
    """Full (rule x classifier x epsilon x seed) attribute-inference grid."""
    public_stats = load_public_stats(ds, data_dir)
    splits, setup = prepare_dataset(ds, data_dir)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    prevalence = verify_prevalence(ds, y_train)  # descriptor value, public

    # fixed target subset, deterministic across cells
    rng = np.random.default_rng(2026)
    if N_TARGET_RECORDS < len(X_test):
        idx = rng.choice(len(X_test), size=N_TARGET_RECORDS, replace=False)
    else:
        idx = np.arange(len(X_test))
    X_target, y_target = X_test[idx], y_test[idx]
    target_true_bins = {a: setup[a]["true_bins_test"][idx] for a in SENSITIVE_ATTRIBUTES}

    from src.constants import RULES
    cells = list(product(RULES, MODEL_TYPES, EPSILON_VALUES, NOISE_SEEDS))
    print(f"[{ds}] {len(cells)} cells, {len(idx)} target records, prevalence={prevalence:.4f}",
          flush=True)

    # long serial loop, print progress so a hung job is visible
    start = time.time()
    done = 0
    last_block = None

    results_rows, metrics_rows = [], []
    for rule, model_type, eps, seed in cells:
        if (rule, model_type) != last_block:
            last_block = (rule, model_type)
            if done:
                rate = done / (time.time() - start)
                eta = (len(cells) - done) / rate / 60
                print(f"  {done:5d}/{len(cells)} cells  {eta:.0f} min left  "
                      f"-> {rule}/{model_type}", flush=True)
            else:
                print(f"  starting {rule}/{model_type}", flush=True)
        done += 1
        try:
            model, _ = train_model(
                X_train, y_train, model_type, rule, eps, seed, prevalence, public_stats
            )
        except Exception as e:
            print(f"Training failed: {rule}/{model_type}/eps={eps}/seed={seed}: {e}")
            continue

        try:
            proba = model.predict_proba(X_test)
            if model_type == "knn":
                proba = correct_knn_proba(proba, prevalence)
            report = evaluate_classifier(
                y_test, proba[:, 1], threshold=threshold_prevalence_matched(prevalence)
            )
            metrics_rows.append({
                "dataset": ds, "mechanism": rule, "model_type": model_type,
                "epsilon": eps, "seed": seed,
                "auroc": report.auroc, "auc_pr": report.auc_pr, "mcc": report.mcc,
                "f1_at_tau": report.f1_at_tau, "brier": report.brier,
            })
        except Exception as e:
            print(f"Metrics failed for {rule}/{model_type}/eps={eps}/seed={seed}: {e}")

        for attr in SENSITIVE_ATTRIBUTES:
            try:
                predicted = confidence_rank_attack(
                    model, X_target, y_target,
                    setup[attr]["feature_idx"], setup[attr]["candidates"],
                    knn_prevalence=(prevalence if model_type == "knn" else None),
                )
                m = attribute_attack_metrics(
                    predicted, target_true_bins[attr], ATTRIBUTE_TYPES[attr]
                )
            except Exception as e:
                print(f"Attack failed: {rule}/{model_type}/eps={eps}/seed={seed}/{attr}: {e}")
                continue
            results_rows.append({
                "dataset": ds, "mechanism": rule, "model_type": model_type,
                "epsilon": eps, "seed": seed, "attribute": attr,
                "attribute_type": ATTRIBUTE_TYPES[attr],
                "n_candidates": len(setup[attr]["candidates"]),
                "n_targets": len(predicted),
                "top1_acc": m["top1_acc"],
                "within1_acc": m.get("within1_acc", np.nan),
                "majority_baseline": m["majority_baseline"],
                "advantage_over_baseline": m["advantage_over_baseline"],
            })

    results_df = pd.DataFrame(results_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    results_df.to_csv(out_dir / f"results_attribute_{ds}.csv", index=False)
    metrics_df.to_csv(out_dir / f"metrics_attribute_{ds}.csv", index=False)

    summary = (
        results_df.groupby(["mechanism", "model_type", "epsilon", "attribute"])
        .agg(
            top1_mean=("top1_acc", "mean"),
            top1_std=("top1_acc", "std"),
            top1_ci95=("top1_acc", ci95_halfwidth),
            within1_mean=("within1_acc", "mean"),
            within1_ci95=("within1_acc", ci95_halfwidth),
            baseline=("majority_baseline", "mean"),
            advantage_mean=("advantage_over_baseline", "mean"),
            advantage_ci95=("advantage_over_baseline", ci95_halfwidth),
            n_runs=("top1_acc", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"summary_attribute_{ds}.csv", index=False)
    sanity_checks(results_df)
    print(f"--- {ds} complete: {len(results_df)} attack rows -> {out_dir} ---")


def sanity_checks(results_df: pd.DataFrame) -> None:
    """Invariant checks on the finished grid. Hard failures raise."""
    avg_advantage = results_df.groupby("attribute")["advantage_over_baseline"].mean()
    print("Mean advantage over majority baseline, by attribute:")
    print(avg_advantage.round(4).to_string())
    if (avg_advantage < -0.02).any():
        # can happen at very low eps where noise is huge, so only flag it
        print("Note: an attribute underperforms the majority baseline on average; "
              "plausible at high noise, worth checking if it persists at high eps.")

    for attr in SENSITIVE_ATTRIBUTES:
        if ATTRIBUTE_TYPES[attr] != "continuous":
            continue
        sub = results_df[results_df["attribute"] == attr]
        viol = int((sub["within1_acc"] < sub["top1_acc"]).sum())
        assert viol == 0, f"{viol} runs have within1 < top1 for {attr}"

    sex = results_df[results_df["attribute"] == "sex"]
    if len(sex):
        assert sex["top1_acc"].between(0.0, 1.0).all(), "sex attack accuracy outside [0,1]"


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
        out_dir = Path(args.out_dir) / f"attribute_{ds}"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dataset(ds, args.data_dir, out_dir)


if __name__ == "__main__":
    main()
