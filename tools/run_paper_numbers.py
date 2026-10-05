"""Every number the paper reports, regenerated from the results CSVs.

Reads results/ and writes, under results/paper_numbers/:
  table4_mia_accuracy.csv ... table8_fitted_parameters.csv   main-text tables
  tableA1_reconstruction_mae.csv, tableA2_attribute_advantage.csv   appendix
  numbers.json                                              every prose statistic

No model is fitted. Rule comparisons are paired t-tests against Uniform over
the 30 shared seeds, Bonferroni-corrected within cell (0.05/7) and family-wide
(0.05/252), with pooled Cohen's d. The 0.1% LiRA point is interpolated, 1% is
observed.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.constants import (  # noqa: E402
    DATASET_LABELS,
    DATASETS,
    EPSILON_VALUES,
    MODEL_LABELS,
    MODEL_TYPES,
    RULE_ALIASES,
    RULES,
)

OTHER_RULES = tuple(r for r in RULES if r != "uniform")
PARAM_FITTERS = ("naive_bayes", "random_forest", "logreg")
ALPHA_CELL = 0.05 / len(EPSILON_VALUES)
ALPHA_FAMILY = 0.05 / 252
TABLE_EPS = 1.0


def _read(path: Path) -> pd.DataFrame:
    """Read a results CSV with legacy rule names mapped to canonical ones."""
    df = pd.read_csv(path)
    if "mechanism" in df:
        df["mechanism"] = df["mechanism"].replace(RULE_ALIASES)
    return df


def pooled_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d with pooled SD, 0 when both groups are constant."""
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def paired(df: pd.DataFrame, metric: str, rule: str, base: str = "uniform") -> dict:
    """Paired comparison of one rule against a base rule over shared seeds."""
    a = df[df.mechanism == rule].set_index("seed")[metric]
    b = df[df.mechanism == base].set_index("seed")[metric]
    a, b = a.align(b, join="inner")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # identical groups warn; p is 1 there
        p = stats.ttest_rel(a, b).pvalue if (a - b).std() > 0 else 1.0
    return {"diff": float(a.mean() - b.mean()), "p": float(p), "d": pooled_d(a, b)}


class Numbers:
    """Loads one results directory and derives every reported quantity."""

    def __init__(self, root: Path):
        self.root = root
        self.mia = {d: _read(root / f"mia_{d}" / f"results_mia_{d}.csv") for d in DATASETS}
        self.mia_sum = {d: _read(root / f"mia_{d}" / f"summary_mia_{d}.csv") for d in DATASETS}
        self.perclass = {d: _read(root / f"mia_{d}" / f"perclass_mia_{d}.csv") for d in DATASETS}
        self.gen = {d: _read(root / f"mia_{d}" / f"perclass_generalization_{d}.csv") for d in DATASETS}
        self.rec = {d: _read(root / f"reconstruction_{d}" / f"metrics_reconstruction_{d}.csv") for d in DATASETS}
        self.attr = {d: _read(root / f"attribute_{d}" / f"results_attribute_{d}.csv") for d in DATASETS}
        self.attr_sum = {d: _read(root / f"attribute_{d}" / f"summary_attribute_{d}.csv") for d in DATASETS}
        self.floor = pd.read_csv(root / "figures" / "naive_reconstruction_baseline.csv")
        self.ra = root / "rule_analysis"

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    def table4(self) -> pd.DataFrame:
        """MIA accuracy (%) at baseline and each rule at eps = 1.0."""
        rows = []
        for d in DATASETS:
            s = self.mia_sum[d]
            for m in MODEL_TYPES:
                x = s[s.model_type == m]
                row = {"dataset": DATASET_LABELS[d], "classifier": MODEL_LABELS[m],
                       "baseline": 100 * x[x.mechanism == "baseline"].mia_accuracy_mean.iloc[0]}
                for r in RULES:
                    row[r] = 100 * x[(x.mechanism == r) & (x.epsilon == TABLE_EPS)].mia_accuracy_mean.iloc[0]
                rows.append(row)
        return pd.DataFrame(rows).round(1)

    def table5(self) -> pd.DataFrame:
        """LiRA AUC and TPR at 0.1% and 1% FPR, baseline and four-rule mean at eps = 1.0."""
        rows = []
        for d in DATASETS:
            r = self.mia[d]
            for m in MODEL_TYPES:
                x = r[r.model_type == m]
                b = x[x.mechanism == "baseline"]
                dp = x[(x.mechanism != "baseline") & (x.epsilon == TABLE_EPS)].groupby("mechanism")
                rows.append({"dataset": DATASET_LABELS[d], "classifier": MODEL_LABELS[m],
                             "base_auc": b.lira_auc.mean(), "dp_auc": dp.lira_auc.mean().mean(),
                             "base_tpr": b.lira_tpr_at_fpr_001.mean(),
                             "dp_tpr": dp.lira_tpr_at_fpr_001.mean().mean(),
                             "base_tpr_1pct": b.lira_tpr_at_fpr_01.mean(),
                             "dp_tpr_1pct": dp.lira_tpr_at_fpr_01.mean().mean()})
        return pd.DataFrame(rows).round(3)

    def table6(self) -> pd.DataFrame:
        """Per-class MIA accuracy (%) for KNN and RF, baseline and rule mean at eps = 1.0."""
        rows = []
        for d in DATASETS:
            p = self.perclass[d]
            for m in ("knn", "random_forest"):
                x = p[p.model_type == m]
                b = x[x.mechanism == "baseline"].iloc[0]
                dp = x[x.mechanism != "baseline"]
                rows.append({"dataset": DATASET_LABELS[d], "classifier": MODEL_LABELS[m],
                             "base_min": 100 * b.minority_AFpos_mia_acc,
                             "base_maj": 100 * b.majority_AFneg_mia_acc,
                             "dp_min": 100 * dp.minority_AFpos_mia_acc.mean(),
                             "dp_maj": 100 * dp.majority_AFneg_mia_acc.mean()})
        return pd.DataFrame(rows).round(1)

    def table7(self) -> pd.DataFrame:
        """Utility AUROC at baseline and eps = 1.0, with the two deltas."""
        rows = []
        for d in DATASETS:
            s = self.mia_sum[d]
            for m in MODEL_TYPES:
                x = s[s.model_type == m]
                at = lambda r: x[(x.mechanism == r) & (x.epsilon == TABLE_EPS)].util_auroc_mean.iloc[0]  # noqa: E731
                u, a, v = at("uniform"), at("adaptive_feature"), at("adaptive_feature_reversed")
                rows.append({"dataset": DATASET_LABELS[d], "classifier": MODEL_LABELS[m],
                             "baseline": x[x.mechanism == "baseline"].util_auroc_mean.iloc[0],
                             "uniform": u, "adaptive": a, "reversed": v,
                             "rev_minus_adapt": v - a, "adapt_minus_uniform": a - u})
        return pd.DataFrame(rows).round(3)

    def table8(self) -> pd.DataFrame:
        """GNB fitted-variance inflation and RF importance stability at eps = 1.0."""
        rows = []
        for d in DATASETS:
            g = _read(self.ra / f"gnb_inflation_{d}.csv")
            a = g[g.mechanism == "adaptive_feature"].iloc[0]
            u = g[g.mechanism == "uniform"].iloc[0]
            rf = _read(self.ra / f"rf_stability_{d}.csv")
            rows.append({
                "dataset": DATASET_LABELS[d],
                "inflation_ratio_c1_c0": round(a.class1_over_class0_ratio, 3),
                "cross_class_rho": round(a.across_class_inflation_spearman, 2),
                "inflation_variance_rho": round(a.inflation_vs_pubvar_spearman, 2),
                "adapt_over_uniform": round((a.mean_inflation_class0 + a.mean_inflation_class1)
                                            / (u.mean_inflation_class0 + u.mean_inflation_class1), 2),
                "rf_importance_rho": round(rf.importance_rank_spearman.iloc[0], 2),
                "top5_overlap": round(rf[rf.k == 5].topk_overlap.iloc[0], 2),
                "top10_overlap": round(rf[rf.k == 10].topk_overlap.iloc[0], 2),
            })
        return pd.DataFrame(rows)

    def table_a1(self) -> pd.DataFrame:
        """Reconstruction MAE at baseline and each rule at eps = 1.0."""
        rows = []
        for d in DATASETS:
            mt = self.rec[d]
            for m in MODEL_TYPES:
                x = mt[mt.model_type == m]
                row = {"dataset": DATASET_LABELS[d], "classifier": MODEL_LABELS[m],
                       "baseline": x.baseline_mae.iloc[0]}
                for r in RULES:
                    row[r] = x[(x.mechanism == r) & (x.epsilon == TABLE_EPS)].mae_mean.iloc[0]
                rows.append(row)
        return pd.DataFrame(rows).round(2)

    def _attr_eps(self, eps: float = TABLE_EPS) -> pd.DataFrame:
        """Attribute summary rows at one budget, advantage in percentage points."""
        frames = []
        for d in DATASETS:
            s = self.attr_sum[d]
            s = s[s.epsilon == eps].copy()
            s["dataset"] = d
            frames.append(s)
        out = pd.concat(frames, ignore_index=True)
        out["adv_pp"] = 100 * out.advantage_mean
        return out

    def table_a2(self) -> pd.DataFrame:
        """Attribute advantage range [min, max] across the four rules at eps = 1.0."""
        a = self._attr_eps()
        rows = []
        for d in DATASETS:
            for at in ("sex", "age", "heart_rate"):
                row = {"dataset": DATASET_LABELS[d], "attribute": at}
                for m in MODEL_TYPES:
                    v = a[(a.dataset == d) & (a.attribute == at) & (a.model_type == m)].adv_pp
                    row[MODEL_LABELS[m]] = f"[{v.min():+.1f}, {v.max():+.1f}]"
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Prose statistics
    # ------------------------------------------------------------------
    def mia_tests(self) -> pd.DataFrame:
        """All 252 paired rule-versus-Uniform tests on MIA accuracy."""
        rows = []
        for d in DATASETS:
            r = self.mia[d]
            for m in MODEL_TYPES:
                for e in EPSILON_VALUES:
                    x = r[(r.model_type == m) & (r.epsilon == e)]
                    for rule in OTHER_RULES:
                        rows.append({"dataset": d, "classifier": m, "epsilon": e, "rule": rule,
                                     **paired(x, "mia_accuracy", rule)})
        return pd.DataFrame(rows)

    def section_5_1(self) -> dict:
        s = pd.concat([self.mia_sum[d].assign(dataset=d) for d in DATASETS])
        base = s[s.mechanism == "baseline"]
        dp = s[s.mechanism != "baseline"]
        t = self.mia_tests()
        cell, fam = t[t.p < ALPHA_CELL], t[t.p < ALPHA_FAMILY]
        spread = (base.groupby("dataset").mia_accuracy_mean.max()
                  - base.groupby("dataset").mia_accuracy_mean.min()).max()

        lira = pd.concat([self.mia[d].assign(dataset=d) for d in DATASETS])
        lb = lira[(lira.mechanism == "baseline") & (lira.model_type == "knn")].groupby("dataset")
        knn_auc, knn_tpr = lb.lira_auc.mean(), lb.lira_tpr_at_fpr_001.mean()
        knn_dp_ptbxl = lira[(lira.dataset == "ptbxl") & (lira.model_type == "knn")
                            & (lira.mechanism != "baseline") & (lira.epsilon == TABLE_EPS)
                            ].groupby("mechanism").lira_tpr_at_fpr_001.mean().mean()

        t6 = self.table6()
        gaps = t6.base_min - t6.base_maj
        pc = pd.concat([self.perclass[d] for d in DATASETS])
        pc = pc[(pc.mechanism != "baseline") & pc.model_type.isin(["knn", "random_forest"])]
        pc_dev = max((100 * (pc[c] - 0.5)).abs().max()
                     for c in ("minority_AFpos_mia_acc", "majority_AFneg_mia_acc"))
        return {
            "baseline_spread_pp": round(100 * spread, 1),
            "knn_baseline_pct": [round(100 * base[base.model_type == "knn"].mia_accuracy_mean.min(), 1),
                                 round(100 * base[base.model_type == "knn"].mia_accuracy_mean.max(), 1)],
            "rf_baseline_pct": [round(100 * base[base.model_type == "random_forest"].mia_accuracy_mean.min(), 1),
                                round(100 * base[base.model_type == "random_forest"].mia_accuracy_mean.max(), 1)],
            "eps1_max_dev_pp": round(100 * (dp[dp.epsilon == TABLE_EPS].mia_accuracy_mean - 0.5).abs().max(), 1),
            "grid_max_dev_pp": round(100 * (dp.mia_accuracy_mean - 0.5).abs().max(), 1),
            "n_tests": len(t),
            "max_abs_d": round(t.d.abs().max(), 2),
            "max_abs_diff_pp": round(100 * t["diff"].abs().max(), 1),
            "within_cell_sig": len(cell),
            "within_cell_favor_uniform": int((cell["diff"] > 0).sum()),
            "within_cell_favor_rule": int((cell["diff"] < 0).sum()),
            "family_sig": len(fam),
            "family_hits": fam.round(5).to_dict("records"),
            "knn_lira_auc": [round(knn_auc.min(), 2), round(knn_auc.max(), 2)],
            "knn_tpr_by_dataset": {d: round(knn_tpr[d], 3) for d in DATASETS},
            "knn_tpr_times_chance": [round(knn_tpr.min() / 0.001), round(knn_tpr.max() / 0.001)],
            "knn_dp_tpr_ptbxl": round(knn_dp_ptbxl, 3),
            "knn_minority_base_pct": t6[t6.classifier == "KNN"].base_min.tolist(),
            "rf_minority_base_pct": t6[t6.classifier == "RF"].base_min.tolist(),
            "majority_base_pct": [t6.base_maj.min(), t6.base_maj.max()],
            "minority_gap_pp": [round(gaps.min()), round(gaps.max())],
            "perclass_dp_max_dev_pp": round(pc_dev, 1),
        }

    def section_5_2(self) -> dict:
        f = self.floor.groupby("dataset").mae_marginal_mean.agg(["mean", "std"])
        ta1 = self.table_a1()
        e1 = ta1[list(RULES)].to_numpy()
        floors = {DATASET_LABELS[d]: f.loc[DATASET_LABELS[d], "mean"] for d in DATASETS}
        mult = [ta1.loc[i, r] / floors[ta1.loc[i, "dataset"]] for i in ta1.index for r in RULES]
        nb = ta1[(ta1.dataset == "PTB-XL") & (ta1.classifier == "NB")].baseline.iloc[0]

        overlap_fail, reversed_sep = [], []
        three = ("uniform", "adaptive_feature", "bounded_inverse_feature")
        for d in DATASETS:
            e = self.rec[d][self.rec[d].epsilon == TABLE_EPS]
            for m in MODEL_TYPES:
                x = e[e.model_type == m].set_index("mechanism")
                for i, a in enumerate(three):
                    for b in three[i + 1:]:
                        if (x.loc[a, "mae_ci_lower"] > x.loc[b, "mae_ci_upper"]
                                or x.loc[b, "mae_ci_lower"] > x.loc[a, "mae_ci_upper"]):
                            overlap_fail.append([d, m, a, b])
                rv, u = x.loc["adaptive_feature_reversed"], x.loc["uniform"]
                if rv.mae_ci_upper < u.mae_ci_lower:
                    reversed_sep.append([d, m, "lower"])
                elif rv.mae_ci_lower > u.mae_ci_upper:
                    reversed_sep.append([d, m, "higher"])
        return {
            "floor": {k: round(v, 2) for k, v in floors.items()},
            "floor_max_sd": round(f["std"].max(), 2),
            "floor_n_features": sorted(self.floor.n_features.unique().tolist()),
            "three_rule_ci_overlap_failures": overlap_fail,
            "reversed_separates": reversed_sep,
            "nb_ptbxl_baseline": round(nb, 2),
            "nb_ptbxl_times_floor": round(nb / floors["PTB-XL"], 1),
            "eps1_mae_range": [round(e1.min()), round(e1.max())],
            "eps1_times_floor": [round(min(mult), 1), round(max(mult), 1)],
        }

    def attr_tests(self) -> pd.DataFrame:
        """Paired rule-versus-Uniform tests on attribute advantage over the whole grid."""
        rows = []
        for d in DATASETS:
            r = self.attr[d]
            for (at, m, e), g in r.groupby(["attribute", "model_type", "epsilon"]):
                for rule in OTHER_RULES:
                    rows.append({"dataset": d, "attribute": at, "classifier": m, "epsilon": e,
                                 "rule": rule, **paired(g, "advantage_over_baseline", rule)})
        return pd.DataFrame(rows)

    def section_5_3(self) -> dict:
        a = self._attr_eps()
        pos, neg = a.adv_pp.max(), a.adv_pp.min()
        t = self.attr_tests()
        fam_alpha = 0.05 / len(t)
        e1 = t[(t.epsilon == TABLE_EPS) & (t.rule != "adaptive_feature_reversed")]
        rev = t[t.rule == "adaptive_feature_reversed"]
        adapt_hits = t[(t.rule == "adaptive_feature") & (t.p < fam_alpha)]
        worst = a.loc[a.adv_pp.idxmin()]
        return {
            "max_abs_adv_pp": round(a.adv_pp.abs().max(), 1),
            "max_pos_adv_pp": round(pos, 1),
            "max_pos_cells": a[a.adv_pp >= pos - 0.05][["dataset", "attribute", "model_type"]]
                             .drop_duplicates().values.tolist(),
            "most_negative_pp": round(neg, 1),
            "most_negative_cell": [worst.dataset, worst.attribute, worst.model_type, worst.mechanism],
            "spread_by_classifier_pp": a.groupby("model_type").adv_pp
                                        .agg(lambda v: round(v.max() - v.min(), 1)).to_dict(),
            "eps1_three_rules_max_abs_d": round(e1.d.abs().max(), 2),
            "eps1_three_rules_within_cell_sig": int((e1.p < ALPHA_CELL).sum()),
            "family_alpha": fam_alpha,
            "reversed_grid_max_drop_pp": round(-100 * rev["diff"].min(), 1),
            "reversed_grid_max_abs_d": round(rev.d.abs().max(), 1),
            "reversed_grid_family_sig": int((rev.p < fam_alpha).sum()),
            "adaptive_grid_family_hits": adapt_hits.assign(diff_pp=100 * adapt_hits["diff"])
                                         [["dataset", "attribute", "classifier", "epsilon", "diff_pp", "d"]]
                                         .round(2).to_dict("records"),
        }

    def section_5_4(self) -> dict:
        rows, rev = [], []
        for d in DATASETS:
            e = self.mia[d][self.mia[d].epsilon == TABLE_EPS]
            for m in PARAM_FITTERS:
                x = e[e.model_type == m]
                for rule in ("adaptive_feature", "bounded_inverse_feature"):
                    rows.append({"rule": rule, **paired(x, "util_auroc", rule)})
                rev.append(paired(x, "util_auroc", "adaptive_feature_reversed"))
        cmp_, rev = pd.DataFrame(rows), pd.DataFrame(rev)
        fam = cmp_[cmp_.p < ALPHA_FAMILY]

        t7 = self.table7()
        pf = t7[t7.classifier.isin([MODEL_LABELS[m] for m in PARAM_FITTERS])]

        knn = pd.concat([self.mia[d].assign(dataset=d) for d in DATASETS])
        knn = knn[knn.model_type == "knn"]
        means = knn.groupby(["dataset", "mechanism", "epsilon"]).util_auroc.mean()
        three = ("uniform", "adaptive_feature", "bounded_inverse_feature")
        collapse = [means[(d, r, e)] for d in DATASETS for r in three for e in (0.1, 0.5, 1.0)]
        knn_d = []
        for d in DATASETS:
            for e in (0.1, 0.5, 1.0):
                x = knn[(knn.dataset == d) & (knn.epsilon == e)]
                for i, a in enumerate(three):
                    for b in three[i + 1:]:
                        knn_d.append(abs(paired(x, "util_auroc", a, b)["d"]))
        top = sum(means[d].drop("baseline", errors="ignore").xs(e, level="epsilon").idxmax()
                  == "adaptive_feature_reversed" for d in DATASETS for e in EPSILON_VALUES)
        lifts, lift_p, knn_rev_d = [], [], []
        for d in DATASETS:
            x = knn[(knn.dataset == d) & (knn.epsilon == TABLE_EPS)]
            best = means[d].xs(TABLE_EPS, level="epsilon").drop(
                ["baseline", "adaptive_feature_reversed"], errors="ignore").idxmax()
            c = paired(x, "util_auroc", "adaptive_feature_reversed", best)
            lifts.append(c["diff"]); lift_p.append(c["p"]); knn_rev_d.append(abs(c["d"]))
        return {
            "uniform_param_fitters_auroc": [round(pf.uniform.min(), 3), round(pf.uniform.max(), 3)],
            "uniform_max_gap_to_baseline": round((pf.baseline - pf.uniform).max(), 3),
            "adapt_minus_uniform_max_abs": round(cmp_[cmp_.rule == "adaptive_feature"]["diff"].abs().max(), 3),
            "bounded_minus_uniform_max_abs": round(cmp_[cmp_.rule == "bounded_inverse_feature"]["diff"].abs().max(), 3),
            "of_18_family_sig": len(fam),
            "of_18_largest": round(fam["diff"].min(), 3) if len(fam) else None,
            "any_favors_adaptive_rule": bool((fam["diff"] > 0).any()),
            "reversed_cost": [round(-rev["diff"].max(), 2), round(-rev["diff"].min(), 2)],
            "reversed_abs_d": [round(rev.d.abs().min(), 1), round(rev.d.abs().max())],
            "reversed_all_family_sig": bool((rev.p < ALPHA_FAMILY).all()),
            "knn_collapse_range": [round(min(collapse), 2), round(max(collapse), 2)],
            "knn_three_rules_max_abs_d": round(max(knn_d), 2),
            "knn_reversed_eps1": [round(min(means[(d, "adaptive_feature_reversed", 1.0)] for d in DATASETS), 2),
                                  round(max(means[(d, "adaptive_feature_reversed", 1.0)] for d in DATASETS), 2)],
            "knn_reversed_eps5": [round(min(means[(d, "adaptive_feature_reversed", 5.0)] for d in DATASETS), 2),
                                  round(max(means[(d, "adaptive_feature_reversed", 5.0)] for d in DATASETS), 2)],
            "knn_reversed_top_cells": f"{top} of {len(DATASETS) * len(EPSILON_VALUES)}",
            "knn_lift_exact": [round(v, 4) for v in lifts],
            "knn_lift_max_p": float(f"{max(lift_p):.1e}"),
            "knn_reversed_abs_d_eps1": [round(min(knn_rev_d), 2), round(max(knn_rev_d), 2)],
        }

    def section_5_5(self) -> dict:
        rows = []
        for d in DATASETS:
            g, p = self.gen[d], self.perclass[d]
            for rule in ("baseline",) + RULES:
                for m in MODEL_TYPES:
                    acc = g[(g.mechanism == rule) & (g.model_type == m)].groupby("true_class")[
                        ["train_acc", "test_acc"]].mean()
                    pr = p[(p.mechanism == rule) & (p.model_type == m)].iloc[0]
                    for c, col in ((1, "minority_AFpos_mia_acc"), (0, "majority_AFneg_mia_acc")):
                        if c in acc.index:
                            rows.append({"dataset": d, "rule": rule, "classifier": m, "cls": c,
                                         "gap": acc.loc[c, "train_acc"] - acc.loc[c, "test_acc"],
                                         "adv": pr[col] - 0.5})
        g = pd.DataFrame(rows)
        b = g[g.rule == "baseline"]
        dp = g[g.rule != "baseline"].groupby(["dataset", "classifier", "cls"])[["gap", "adv"]].mean()
        rho, p = stats.spearmanr(b.gap, b.adv)
        rho_dp, p_dp = stats.spearmanr(dp.gap, dp.adv)
        top = b.nlargest(4, "adv")
        noise = pd.read_csv(self.ra / "noise_totals.csv")
        snr = pd.read_csv(self.ra / "snr_redistribution.csv")
        align = pd.read_csv(self.ra / "variance_alignment_summary.csv")
        return {
            "baseline_cells": len(b),
            "gap_adv_rho": round(rho, 3), "gap_adv_p": float(f"{p:.0e}"),
            "gap_adv_rho_minority": round(stats.spearmanr(b[b.cls == 1].gap, b[b.cls == 1].adv)[0], 3),
            "gap_adv_rho_majority": round(stats.spearmanr(b[b.cls == 0].gap, b[b.cls == 0].adv)[0], 3),
            "top_advantage_cells_all_minority": bool((top.cls == 1).all()),
            "dp_rho": round(rho_dp, 2), "dp_p": round(p_dp, 2),
            "adaptive_noise_ratio_eps1": noise[noise.eps == 1.0].set_index("dataset")
                                          .ratio_adaptive_over_uniform.round(3).to_dict(),
            "snr_ratio_top": snr.set_index("dataset").snr_ratio_top.round(3).to_dict(),
            "snr_ratio_all": snr.set_index("dataset").snr_ratio_all.round(3).to_dict(),
            "variance_reliance_rho_rf_nb": [round(min(align.rf.min(), align.nb.min()), 2),
                                            round(max(align.rf.max(), align.nb.max()), 2)],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    root = Path(args.results_dir)
    out = root / "paper_numbers"
    out.mkdir(parents=True, exist_ok=True)
    n = Numbers(root)

    for name, fn in (("table4_mia_accuracy", n.table4), ("table5_lira", n.table5),
                     ("table6_perclass_mia", n.table6), ("table7_utility_auroc", n.table7),
                     ("table8_fitted_parameters", n.table8), ("tableA1_reconstruction_mae", n.table_a1),
                     ("tableA2_attribute_advantage", n.table_a2)):
        fn().to_csv(out / f"{name}.csv", index=False)

    numbers = {"section_5_1": n.section_5_1(), "section_5_2": n.section_5_2(),
               "section_5_3": n.section_5_3(), "section_5_4": n.section_5_4(),
               "section_5_5": n.section_5_5()}
    with open(out / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=2, default=float)
    print(f"tables and numbers.json -> {out}")


if __name__ == "__main__":
    main()
