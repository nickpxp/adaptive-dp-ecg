"""Reconstruction floor and the paper figures.

Reads the CSVs and score dumps the runners wrote and writes, under
results/figures/:
  naive_reconstruction_baseline.csv    marginal-sampling floor per dataset
  attack_vs_naive_floor.csv            attack vs floor, by classifier
  figure2_lira_roc.png                 LiRA log-log ROC, KNN, three datasets
  figure3_utility_ptbxl.png            utility AUROC vs budget, four classifiers
  figureA1_mae_vs_utility.png          reconstruction MAE vs utility AUROC
  figureA2_utility_chapman_native.png  as figure 3, Chapman-Native
  figureA3_utility_chapman_matched.png as figure 3, Chapman-Matched

Each rule gets a color, marker and line style so the panels survive greyscale.
Panel letters go in the title, legend sits below.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to disk; never open a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import _path  # noqa: F401
from src.constants import (
    DATA_FILES,
    DATASET_LABELS,
    DATASETS,
    EPSILON_VALUES,
    MODEL_LABELS,
    MODEL_TYPES,
    N_FEATURES,
    NOISE_SEEDS,
    N_TARGET_RECORDS,
    PUBLIC_SOURCE_MAP,
    RULE_ALIASES,
    RULE_COLORS,
    RULE_LABELS,
    RULE_MARKERS,
    RULE_STYLES,
    RULES,
    FIG_WIDTH_IN,
    FIG_DPI,
    MODEL_PANEL_TITLES,
)
from src.data import feature_columns

# features the reconstruction attack predicts. the floor has to use the same
# set or MAE isn't comparable. None = all shared features
TARGET_FEATURES: list[str] | None = None

FIG2_CLASSIFIER = "knn"
# 500 non-members resolve FPR no finer than 1/500
FIG2_RESOLUTION_LIMIT = 1.0 / 500.0
FIG_EPSILON = 1.0   # the budget the ROC figure is drawn at
PANEL_LETTERS = "abcdefgh"


def _canon(series: pd.Series) -> pd.Series:
    """Normalize legacy rule names in old sweep CSVs to the canonical ones."""
    return series.replace(RULE_ALIASES)


# ---------------------------------------------------------------------------
# Naive marginal-sampling floor
# ---------------------------------------------------------------------------
def usable_columns(public_csv: Path, protected_csv: Path, cols: list[str]):
    """Shared columns with data on both sides. Dropped names are returned."""
    cols = sorted(cols)
    pub = pd.read_csv(public_csv)[cols]
    prot = pd.read_csv(protected_csv)[cols]
    pub_ok, prot_ok = pub.notna().any(), prot.notna().any()
    usable = [c for c in cols if pub_ok[c] and prot_ok[c]]
    dropped = {
        "all-NaN in donor": [c for c in cols if not pub_ok[c]],
        "all-NaN in target": [c for c in cols if pub_ok[c] and not prot_ok[c]],
    }
    return usable, {k: v for k, v in dropped.items() if v}


def naive_reconstruction_baseline(
    public_csv: Path,
    protected_csv: Path,
    cols: list[str],
    n_targets: int = N_TARGET_RECORDS,
):
    """Marginal-sampling floor, each feature drawn from the donor's marginal."""
    cols, dropped = usable_columns(public_csv, protected_csv, cols)
    if not cols:
        raise ValueError(
            "no usable columns: every shared column is empty on one side. "
            "Check that these are the right feature files."
        )
    pub = pd.read_csv(public_csv)[cols].to_numpy(dtype=float)
    prot = pd.read_csv(protected_csv)[cols].to_numpy(dtype=float)
    pub[~np.isfinite(pub)] = np.nan
    prot[~np.isfinite(prot)] = np.nan
    pub_mean = np.nanmean(pub, axis=0)
    pub_col = [pub[np.isfinite(pub[:, j]), j] for j in range(pub.shape[1])]

    rows = []
    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(prot.shape[0], size=min(n_targets, prot.shape[0]), replace=False)
        target = prot[idx]
        draw = np.column_stack([
            rng.choice(pub_col[j], size=target.shape[0], replace=True)
            for j in range(target.shape[1])
        ])
        rows.append({
            "seed": seed,
            "mae_marginal_draw": np.nanmean(np.abs(target - draw)),
            "mae_marginal_mean": np.nanmean(np.abs(target - pub_mean[None, :])),
            "n_targets": target.shape[0],
            "n_features": len(cols),
        })
    return pd.DataFrame(rows), dropped


def compute_floor(data_dir: Path, out_dir: Path) -> pd.DataFrame | None:
    """Floor per dataset over the shared feature schema (or TARGET_FEATURES)."""
    paths = {ds: data_dir / DATA_FILES[ds] for ds in DATASETS}
    present = {ds: p for ds, p in paths.items() if p.exists()}
    if not present:
        print("SKIPPED floor: no feature CSVs found in", data_dir)
        return None

    schemas = {ds: set(feature_columns(pd.read_csv(p, nrows=200)))
               for ds, p in present.items()}
    shared = sorted(set.intersection(*schemas.values()))
    print(f"shared feature columns: {len(shared)}")
    if len(shared) < 50:
        print("WARNING: fewer than 50 shared columns; these files probably "
              "do not share a feature schema — check you picked the right ones.")
    if TARGET_FEATURES:
        missing = [f for f in TARGET_FEATURES if f not in shared]
        if missing:
            print(f"TARGET_FEATURES not present in the data: {missing}")
        shared = [f for f in TARGET_FEATURES if f in shared]
        print(f"floor restricted to the {len(shared)} attacked features")
    else:
        print("TARGET_FEATURES is None: the floor covers ALL shared features. "
              "If the attack predicts only a subset, set TARGET_FEATURES first.")

    results = []
    for ds, prot_p in present.items():
        pub_p = present.get(PUBLIC_SOURCE_MAP[ds])
        if pub_p is None:
            print(f"  {ds}: skipped (donor file missing)")
            continue
        per_seed, dropped = naive_reconstruction_baseline(pub_p, prot_p, shared)
        for reason, names in dropped.items():
            print(f"  {ds}: dropped {len(names)} columns, {reason}: {names[:6]}")
        per_seed.insert(0, "dataset", DATASET_LABELS[ds])
        results.append(per_seed)

    if not results:
        return None
    floor = pd.concat(results, ignore_index=True)
    floor.to_csv(out_dir / "naive_reconstruction_baseline.csv", index=False)
    n = floor.groupby("dataset").n_features.first()
    if (n != N_FEATURES).any():
        print(f"NOTE: not all datasets ran on {N_FEATURES} features")
    return floor


def attack_vs_floor(floor: pd.DataFrame, results_dir: Path, out_dir: Path) -> None:
    """Attack MAE vs the floor, by classifier."""
    from src.constants import MODEL_LABELS, MODEL_TYPES
    floors = floor.groupby("dataset").mae_marginal_mean.mean()
    rows = []
    for ds in DATASETS:
        label = DATASET_LABELS[ds]
        f = results_dir / f"reconstruction_{ds}" / f"metrics_reconstruction_{ds}.csv"
        if not f.exists() or label not in floors.index:
            continue
        m = pd.read_csv(f)
        m["mechanism"] = _canon(m["mechanism"])
        for mt in MODEL_TYPES:
            sub = m[m.model_type == mt]
            if not len(sub):
                continue
            base = float(sub.baseline_mae.iloc[0])
            dp = float(sub[sub.epsilon == 1.0].mae_mean.mean())
            fl = float(floors[label])
            rows.append({
                "dataset": label, "classifier": MODEL_LABELS[mt],
                "nonprivate_mae": round(base, 2), "dp_mae_eps1": round(dp, 2),
                "naive_floor": round(fl, 2),
                "nonprivate_x_floor": round(base / fl, 2),
                "dp_x_floor": round(dp / fl, 2),
            })
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "attack_vs_naive_floor.csv", index=False)
    print(comp.to_string(index=False))
    if len(comp):
        # the headline floor multiples
        print(f"strongest non-private result: {comp.nonprivate_mae.min():.2f} "
              f"= {comp.nonprivate_x_floor.min():.1f}x the floor")
        print(f"DP cells at eps=1.0: {comp.dp_mae_eps1.min():.0f} to "
              f"{comp.dp_mae_eps1.max():.0f}, {comp.dp_x_floor.min():.0f} to "
              f"{comp.dp_x_floor.max():.0f}x the floor")
        above = bool((comp.dp_x_floor > 1).all() and (comp.nonprivate_x_floor > 1).all())
        print(f"every cell above the floor: {above}")
    if not TARGET_FEATURES:
        print("WARNING: floor computed over all shared features. Set "
              "TARGET_FEATURES to the attacked subset before reading this table.")


# ---------------------------------------------------------------------------
# Manuscript figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 8})


def _style(rule: str) -> dict:
    """Color, marker and line style for one rule."""
    return {"color": RULE_COLORS[rule], "marker": RULE_MARKERS[rule] or None,
            "linestyle": RULE_STYLES[rule], "label": RULE_LABELS[rule],
            "markerfacecolor": "none", "markersize": 4, "linewidth": 1.3}


def _title(ax, i: int, text: str) -> None:
    """Panel title with its letter, left-aligned."""
    ax.set_title(f"{PANEL_LETTERS[i]}. {text}", loc="left")


def _legend_below(fig, handles, labels, ncol: int, bottom: float) -> None:
    """One shared legend under the panels."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, bottom, 1, 1))


def _save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / name, dpi=FIG_DPI)
    plt.close(fig)
    print("wrote", name)


def figure2_lira_roc(results_dir: Path, out_dir: Path) -> None:
    """Log-log LiRA ROC for KNN, no-DP and each rule at eps = 1.0, one panel per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 2.6))
    for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        f = results_dir / f"mia_{ds}" / f"lira_scores_{ds}.npz"
        if not f.exists():
            _title(ax, i, f"{DATASET_LABELS[ds]} (scores missing)")
            continue
        z = np.load(f, allow_pickle=True)
        n_non = 0
        for rule in ("baseline",) + RULES:
            keys = [rule] + [o for o, n in RULE_ALIASES.items() if n == rule]
            key = next((k for k in keys if f"{FIG2_CLASSIFIER}|{k}|scores" in z.files), None)
            if key is None:
                continue
            sc = np.asarray(z[f"{FIG2_CLASSIFIER}|{key}|scores"], float)
            y = np.asarray(z[f"{FIG2_CLASSIFIER}|{key}|labels"], int)
            ok = np.isfinite(sc)
            if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
                continue
            n_non = max(n_non, int((y[ok] == 0).sum()))
            fpr, tpr, _ = roc_curve(y[ok], sc[ok])
            st = _style(rule)
            st.update(marker=None, label="no DP" if rule == "baseline" else st["label"],
                      linestyle="-" if rule == "baseline" else st["linestyle"],
                      linewidth=1.6 if rule == "baseline" else 1.3)
            ax.plot(np.maximum(fpr, 1e-3), np.maximum(tpr, 1e-3), **st)
        ax.plot([1e-3, 1], [1e-3, 1], color="0.5", ls="--", lw=0.8)
        # the finest false-positive rate the non-members can resolve
        ax.axvline(1.0 / n_non if n_non else FIG2_RESOLUTION_LIMIT, color=RULE_COLORS["bounded_inverse_feature"],
                   ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.5e-3, 1)
        ax.set_ylim(1.5e-3, 1)
        ax.set_xlabel("false-positive rate")
        ax.grid(alpha=0.3, which="both", lw=0.4)
        _title(ax, i, DATASET_LABELS[ds])
    axes[0].set_ylabel("true-positive rate")
    h, l = axes[0].get_legend_handles_labels()
    _legend_below(fig, h, l, ncol=3, bottom=0.14)
    _save(fig, out_dir, "figure2_lira_roc.png")


def figure_utility(ds: str, results_dir: Path, out_dir: Path, fname: str) -> None:
    """Utility AUROC against record-level budget, 2x2 panels by classifier."""
    f = results_dir / f"mia_{ds}" / f"summary_mia_{ds}.csv"
    if not f.exists():
        print(f"SKIPPED {fname}: {f} missing")
        return
    s = pd.read_csv(f)
    s["mechanism"] = _canon(s["mechanism"])
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 5.07))
    for i, (ax, mt) in enumerate(zip(axes.ravel(), MODEL_TYPES)):
        md = s[s.model_type == mt]
        for rule in RULES:
            g = md[md.mechanism == rule].groupby("epsilon").util_auroc_mean.mean().sort_index()
            if not g.empty:
                ax.plot(g.index * N_FEATURES, g.values, **_style(rule))
        base = md[md.mechanism == "baseline"].util_auroc_mean.mean()
        if np.isfinite(base):
            ax.axhline(base, color="black", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("record-level budget dε")
        ax.set_ylabel("utility AUROC")
        ax.grid(alpha=0.3, lw=0.4)
        _title(ax, i, MODEL_PANEL_TITLES[mt])
    h, l = axes[0, 0].get_legend_handles_labels()
    _legend_below(fig, h, l, ncol=2, bottom=0.09)
    _save(fig, out_dir, fname)


def figure_a1_mae_vs_utility(floor: pd.DataFrame | None, results_dir: Path, out_dir: Path) -> None:
    """Reconstruction MAE against utility AUROC, three stacked panels by dataset."""
    if floor is None:
        saved = out_dir / "naive_reconstruction_baseline.csv"
        if saved.exists():
            floor = pd.read_csv(saved)
            print(f"floor line read from {saved.name}")
    if floor is None:
        raise FileNotFoundError(
            "no floor available: pass --data-dir so it can be computed, or keep "
            "naive_reconstruction_baseline.csv in the figures directory. The "
            "caption states the floor, so the figure must not be drawn without it."
        )
    floors = floor.groupby("dataset").mae_marginal_mean.mean()
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH_IN, 7.93))
    label_eps = (EPSILON_VALUES[0], 1.0, EPSILON_VALUES[-1])  # first, headline, last
    for i, (ax, ds) in enumerate(zip(axes, DATASETS)):
        label = DATASET_LABELS[ds]
        fm = results_dir / f"reconstruction_{ds}" / f"metrics_reconstruction_{ds}.csv"
        fu = results_dir / f"mia_{ds}" / f"summary_mia_{ds}.csv"
        if not (fm.exists() and fu.exists()):
            _title(ax, i, f"{label} (inputs missing)")
            continue
        m = pd.read_csv(fm)
        u = pd.read_csv(fu)
        m["mechanism"] = _canon(m["mechanism"])
        u["mechanism"] = _canon(u["mechanism"])
        mae = m.groupby(["mechanism", "epsilon"]).mae_mean.mean()
        auroc = u.groupby(["mechanism", "epsilon"]).util_auroc_mean.mean()
        for rule in RULES:
            if rule not in mae.index.get_level_values(0):
                continue
            g = pd.DataFrame({"mae": mae.loc[rule], "auroc": auroc.loc[rule]}).sort_index()
            ax.plot(g.auroc, g.mae, **_style(rule))
            if rule == "adaptive_feature_reversed":
                for eps in label_eps:
                    if eps in g.index:
                        ax.annotate(f"dε = {eps * N_FEATURES:.0f}", (g.loc[eps, "auroc"], g.loc[eps, "mae"]),
                                    fontsize=7, color="0.35", xytext=(4, -12), textcoords="offset points")
        if label in floors.index:
            fl = float(floors[label])
            ax.axhline(fl, color=RULE_COLORS["bounded_inverse_feature"], ls="--", lw=1.2, zorder=0)
            ax.annotate(f"marginal-sampling floor {fl:.2f}", xy=(0.01, fl), xycoords=("axes fraction", "data"),
                        fontsize=7, color=RULE_COLORS["bounded_inverse_feature"], va="bottom")
        ax.set_ylim(0, None)
        ax.set_xlabel("utility AUROC (mean over classifiers)")
        ax.set_ylabel("reconstruction MAE")
        ax.grid(alpha=0.3, lw=0.4)
        _title(ax, i, label)
    h, l = axes[0].get_legend_handles_labels()
    _legend_below(fig, h, l, ncol=2, bottom=0.06)
    _save(fig, out_dir, "figureA1_mae_vs_utility.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    floor = compute_floor(data_dir, out_dir)
    if floor is not None:
        attack_vs_floor(floor, results_dir, out_dir)

    figure2_lira_roc(results_dir, out_dir)
    for ds, fname in zip(DATASETS, ("figure3_utility_ptbxl.png",
                                    "figureA2_utility_chapman_native.png",
                                    "figureA3_utility_chapman_matched.png")):
        figure_utility(ds, results_dir, out_dir, fname)
    figure_a1_mae_vs_utility(floor, results_dir, out_dir)
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
