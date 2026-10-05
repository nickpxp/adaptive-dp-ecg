# adaptive-dp-ecg

Reproduction code for *A fixed privacy budget does not fix the utility cost
of noise allocation* (Pierrelouis and Liu). It covers the full attack-evaluation
sweep behind the paper's results, namely membership inference (shadow-model and
LiRA), reconstruction, attribute inference, and the rule-analysis diagnostics.

The paper compares four **budget-allocation rules** for input-perturbation
Laplace noise on ECG feature vectors, all spending the same record-level
budget d·ε and differing only in how it is split across the d = 62 features:

| Rule | Allocation |
|---|---|
| `uniform` | equal budget per feature |
| `adaptive_feature` | inverse to variance share (high-variance → more noise). A variance-weighted adaptation, not a relevance mechanism |
| `adaptive_feature_reversed` | proportional to variance share (sign-flipped control) |
| `bounded_inverse_feature` | inverse-variance shares bounded to [0.5, 2]× even split |

`bounded_inverse_feature` was called `sample_specific` in earlier sweeps; the
alias is still accepted everywhere (`src/constants.py: RULE_ALIASES`) so old
result CSVs and score dumps load unchanged.

## Layout

```
notebooks/      raw PhysioNet downloads -> the three feature CSVs (run first)
src/            shared library (constants, rules, data, models, metrics, attacks)
experiments/    one runner per results section
tools/          equations-vs-code audit; needs data, trains nothing
tests/          invariant tests that run on synthetic data, no CSVs needed
results/        per-seed result CSVs, LiRA score dumps and figures from the canonical run
data/           feature CSVs go here (not distributed; see Data)
```

## Data

The feature CSVs are derived from PTB-XL and Chapman-Shaoxing, which cannot be
redistributed here. Both are freely available from PhysioNet:

* PTB-XL: https://physionet.org/content/ptb-xl/
* Chapman-Shaoxing: https://physionet.org/content/ecg-arrhythmia/

### Data preparation

The three notebooks in `notebooks/` turn the raw downloads into the feature
CSVs the runners consume; run them in order (`pip install -r
requirements-preprocessing.txt` first):

1. `01_ptbxl_preprocessing.ipynb`: PTB-XL at 100 Hz, AF = AFIB ∪ AFLT by
   key presence in `scp_codes`, `strat_fold` taken from `ptbxl_database.csv`.
   Asserts the published record count and 7.27% prevalence.
2. `02_chapman_native_preprocessing.ipynb`: Chapman-Shaoxing with the
   identical 62-feature schema; folds assigned by StratifiedGroupKFold on
   patient id (seeded).
3. `03_chapman_matched_preprocessing.ipynb`: subsamples Chapman-Native to
   PTB-XL's size and prevalence (seed 2026) and re-stratifies folds, so
   prevalence and sample size are held constant in the cross-cohort comparison.

Each writes `data/<cohort>_features.csv` with 67 columns: `ecg_id`,
`patient_id`, `strat_fold`, `validated_by_human`, 62 features, `af_detected`.
Demographics (`age`, `sex`) and `heart_rate` are features, which attribute
inference relies on. `src/data.py` refuses a CSV that does not yield exactly
62 feature columns, because a wrong column set once reached a reported
number without anything downstream noticing.

## Running

```bash
pip install -r requirements.txt
python tests/test_parity.py                      # invariants, no data needed

cd experiments
python run_mia.py --perclass                     # Section 5.1, Table 6
python run_reconstruction.py                     # Section 5.2
python run_attribute_inference.py                # Section 5.3
python run_rule_analysis.py --fit                # Section 5.5
python run_utility_figures.py                    # floor + Figures 2, 3, A1-A3
python ../tools/run_paper_numbers.py             # Tables 4-8, A1, A2 + every prose number
python ../tools/draw_figure1.py                  # Figure 1 (study design)
```

Every runner accepts `--datasets`, `--data-dir`, and `--out-dir`. When more
than one dataset is requested, each runs in its own Python process, so a
failure in one cannot stall the others; the parent prints a summary and the
exact command to re-run whatever failed. `python run_mia.py --perclass-only`
adds the per-class tables to a sweep already on disk, and `--summary-only`
rebuilds the summary CSVs from the raw results without refitting.

DP covers the feature matrix only. Labels are treated as public inputs, so
no label privacy is claimed. The decision threshold, Naive Bayes prior, and KNN correction target are the
published cohort prevalence from each dataset's descriptor
(`DESCRIPTOR_PREVALENCE` in `src/constants.py`), never a statistic of the
protected labels. The labels are checked against it and a warning is raised
if they drift by more than `PREVALENCE_TOLERANCE`. The full
sweep is 30 seeds × 7 ε values × 4 rules × 4 classifiers per attack per
dataset; LiRA (64 shadows per cell) dominates the runtime. Lower
`N_LIRA_SHADOWS` in `src/constants.py` to 32 for a fast pass.

## Verification

1. `tests/test_parity.py` runs without data. Every (classifier, rule) cell
   trains through the same input-perturbation path at the same d·ε, budgets
   sum to d·ε for every rule, the `sample_specific` alias matches
   `bounded_inverse_feature` exactly, and seeds are deterministic.
2. `tools/check_equations.py` transcribes the allocation equations from the
   paper and checks them against `rule_scales` on the real public statistics.
   It also derives each rule's injected noise and worst relative perturbation
   from public statistics alone, so a reader can recompute those numbers
   without a sweep.
3. `tools/run_paper_numbers.py` rebuilds every table and every statistic in
   the Results from the results CSVs, without fitting a model. If a number in
   the paper isn't in its output, it has no code behind it.
4. The refactor from the original notebooks was accepted by re-running one
   full 30-seed sweep per attack and comparing the output CSVs against the
   canonical results.

## Notes

* Noise calibration always uses the cross-dataset public statistics
  (`PUBLIC_SOURCE_MAP`), never the protected data. `train_model` refuses to
  run a DP arm without them.
* Attacks query the target on clean records. Anything else detects noise, not
  membership.
* diffprivlib and ART appear in no reported result and are not dependencies.
* Synthetic data is left out of attribute inference on purpose. With
  engineered correlations it measures population-level inference, not
  memorization.

## Citation

If you use this code, please cite the paper and the archived release.
Citation metadata is in `CITATION.cff`, and GitHub shows a "Cite this
repository" button from it. The Zenodo DOI for the archived release is
listed below once minted.

DOI: to be added on release

## License

MIT. See `LICENSE`.
