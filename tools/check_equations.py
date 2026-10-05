"""Check that the paper's allocation equations match src.rules.

Transcribes each rule's budget formula from the paper, runs it on the public
stats, and compares against rule_scales(). No sweep, no training, no
protected data. Also reports injected noise and worst relative perturbation
per rule, relative to Uniform.

Usage:
    python tools/check_equations.py
    python tools/check_equations.py --calibration 1.38 1.48 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.constants import DATASETS, N_FEATURES, PUBLIC_SOURCE_MAP, RULE_LABELS, RULES  # noqa: E402
from src.data import feature_columns, load_dataframe, load_public_stats  # noqa: E402
from src.rules import rule_scales  # noqa: E402

CODE_MATCH_RTOL = 1e-4  # transcribed vs rule_scales; the 1e-10 guards leave ~1e-6 residue


# ---------------------------------------------------------------------------
# the four allocation equations, transcribed from the paper. v = public variance share
# ---------------------------------------------------------------------------
def shares(rule: str, v: np.ndarray, d: int, iterate_clip: bool = False) -> np.ndarray:
    """Budget share per feature under one rule, transcribed from the paper."""
    if rule == "uniform":
        return np.full(d, 1.0 / d)
    if rule == "adaptive_feature":
        w = np.clip(1.0 - v, 0.01, 0.99)
        return w / w.sum()
    if rule == "adaptive_feature_reversed":
        w = np.clip(v, 0.01, 0.99)
        eps_j = np.clip(d * w / w.sum(), 0.01, d)  # in units of eps
        return eps_j / eps_j.sum()
    if rule == "bounded_inverse_feature":
        inv = 1.0 / (v + 1e-10)  # the line-3 guard; binds when a donor column is empty
        s = inv / inv.sum()
        lo, hi = 0.5 / d, 2.0 / d
        s = np.clip(s, lo, hi)
        s = s / s.sum()
        if iterate_clip:
            # clip-renormalize to a fixed point, to see how far one pass drifts
            for _ in range(200):
                s2 = np.clip(s, lo, hi)
                s2 = s2 / s2.sum()
                if np.allclose(s2, s, rtol=0, atol=1e-15):
                    break
                s = s2
        return s
    raise ValueError(rule)


def injected_variance(delta: np.ndarray, s: np.ndarray, d: int) -> float:
    """Total injected noise variance, sum_j 2 (Delta_j / eps_j)^2 at eps = 1."""
    return float(np.sum(2.0 * (delta / (d * s)) ** 2))


def worst_relative_perturbation(delta, s, d, var) -> float:
    """Worst relative perturbation, max_j Var[eta_j] / var_j."""
    return float(np.max(2.0 * (delta / (d * s)) ** 2 / var))


# ---------------------------------------------------------------------------
def analyze(ds: str, data_dir: str, out_dir: Path) -> dict:
    """All derived quantities for one protected dataset."""
    ps = load_public_stats(ds, data_dir)
    delta = np.asarray(ps["ranges"], float)
    var = np.asarray(ps["variances"], float)
    d = len(delta)
    names = feature_columns(load_dataframe(PUBLIC_SOURCE_MAP[ds], data_dir))
    if d != N_FEATURES:
        print(f"  ! {ds}: d = {d}, not {N_FEATURES}", file=sys.stderr)

    pd.DataFrame({"feature": names, "variance": var, "range": delta}) \
        .to_csv(out_dir / f"public_stats_{ds}.csv", index=False)

    v = var / var.sum()
    s = {r: shares(r, v, d) for r in RULES}
    for r, sh in s.items():  # every allocation sums to d*eps
        assert abs(sh.sum() - 1.0) < 1e-12, f"{r}: shares sum to {sh.sum()}"

    total = {r: injected_variance(delta, s[r], d) for r in RULES}
    rel = {r: worst_relative_perturbation(delta, s[r], d, var) for r in RULES}
    s_iter = shares("bounded_inverse_feature", v, d, iterate_clip=True)

    # the actual test: transcribed scales against what the code draws
    code_gap = {}
    for r in RULES:
        transcribed = delta / (d * s[r])
        code = rule_scales(r, delta, var, 1.0)
        code_gap[r] = float(np.max(np.abs(transcribed - code) / code))

    return {
        "dataset": ds,
        "d": d,
        "ratio_total": {r: total[r] / total["uniform"] for r in RULES},
        "ratio_relmax": {r: rel[r] / rel["uniform"] for r in RULES},
        "ratio_total_bounded_iterated": injected_variance(delta, s_iter, d) / total["uniform"],
        "adaptive_eps_max": float(d * s["adaptive_feature"].max()),
        "bounded_eps_min": float(d * s["bounded_inverse_feature"].min()),
        "bounded_eps_max": float(d * s["bounded_inverse_feature"].max()),
        "variance_share_max": float(v.max()),
        "code_gap": code_gap,
        "ranges": delta,
        "variances": var,
    }


def report(rows: list[dict], calibration: tuple[float, float] | None) -> int:
    """Print the tables and verdicts. Returns a shell exit code."""
    labels = [RULE_LABELS[r] for r in RULES if r != "uniform"]
    print("\nEquations vs code: max relative gap, transcribed equation vs rule_scales")
    print("-" * 72)
    print(f"  {'dataset':<18}" + "".join(f"{RULE_LABELS[k]:>24}" for k in RULES))
    for r in rows:
        print(f"  {r['dataset']:<18}" + "".join(
            f"{r['code_gap'][k]:>24.1e}" for k in RULES))
    failed = [k for k in RULES
              if max(r["code_gap"][k] for r in rows) > CODE_MATCH_RTOL]
    code_ok = not failed
    print(f"  tolerance {CODE_MATCH_RTOL:.0e}: "
          + ("PASS for every rule" if code_ok else
             "FAIL for " + ", ".join(RULE_LABELS[k] for k in failed)))
    if not code_ok:
        print("  For a failing rule the printed equation and the code disagree; "
              "the usual cause is a share clip the code applies that the "
              "equation does not state. Reconcile before citing the rows below.")

    print("\nTotal injected noise variance per record, relative to Uniform")
    print("-" * 72)
    print(f"  {'dataset':<18}" + "".join(f"{l:>24}" for l in labels))
    for r in rows:
        print(f"  {r['dataset']:<18}" + "".join(
            f"{r['ratio_total'][k]:>24.3f}" for k in RULES if k != "uniform"))

    print("\nWorst relative perturbation max_j Var[eta_j]/var_j, relative to Uniform")
    print("-" * 72)
    print(f"  {'dataset':<18}" + "".join(f"{l:>24}" for l in labels))
    for r in rows:
        print(f"  {r['dataset']:<18}" + "".join(
            f"{r['ratio_relmax'][k]:>24.4g}" for k in RULES if k != "uniform"))

    # both Chapman cohorts use the PTB-XL donor, so their stats must match
    by = {r["dataset"]: r for r in rows}
    if "chapman_native" in by and "chapman_matched" in by:
        same = (np.allclose(by["chapman_native"]["ranges"], by["chapman_matched"]["ranges"])
                and np.allclose(by["chapman_native"]["variances"], by["chapman_matched"]["variances"]))
        print(f"\nshared-donor check (Chapman-Native == Chapman-Matched): "
              f"{'PASS' if same else 'FAIL — donor mapping is wrong'}")

    print("\nDiagnostics")
    bound = 0.99 * N_FEATURES / (0.99 * N_FEATURES - 1)  # max budget any feature can get under adaptive_feature
    for r in rows:
        print(f"  {r['dataset']}")
        print(f"    AdaptiveFeature max eps_j/eps      {r['adaptive_eps_max']:.4f}  "
              f"(bound {bound:.4f}: "
              f"{'ok' if r['adaptive_eps_max'] <= bound + 1e-9 else 'EXCEEDED'})")
        print(f"    BoundedInverse eps_j/eps           {r['bounded_eps_min']:.4f} to "
              f"{r['bounded_eps_max']:.4f}  (printed bounds 0.5 to 2.0)")
        print(f"    BoundedInverse ratio, iterated clip {r['ratio_total_bounded_iterated']:.3f}  "
              f"vs single-pass {r['ratio_total']['bounded_inverse_feature']:.3f}")
        print(f"    largest variance share             {r['variance_share_max']:.4f}")

    if calibration:
        lo, hi = calibration
        af = [r["ratio_total"]["adaptive_feature"] for r in rows]
        ok = all(lo <= x <= hi for x in af)
        print(f"\nCalibration: derived AdaptiveFeature ratio {min(af):.3f} to {max(af):.3f}; "
              f"measured band [{lo}, {hi}]: {'PASS' if ok else 'FAIL'}")
        if not ok:
            code_ok = False
    return 0 if code_ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/equation_check")
    parser.add_argument("--calibration", nargs=2, type=float, metavar=("LO", "HI"),
                        help="measured band for adaptive_feature's noise ratio")
    parser.add_argument("--json", help="also write the derived quantities as JSON")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [analyze(ds, args.data_dir, out_dir) for ds in DATASETS]
    code = report(rows, tuple(args.calibration) if args.calibration else None)
    if args.json:
        slim = [{k: v for k, v in r.items() if k not in ("ranges", "variances")} for r in rows]
        with open(args.json, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"wrote {args.json}")
    return code


if __name__ == "__main__":
    sys.exit(main())
