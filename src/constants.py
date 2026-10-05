"""Every constant the sweep depends on. Change values here, nowhere else."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = ("ptbxl", "chapman_native", "chapman_matched")

DATA_FILES = {
    "ptbxl": "ptbxl_features.csv",
    "chapman_native": "chapman_native_features.csv",
    "chapman_matched": "chapman_matched_features.csv",
}

# public donor for noise calibration. only its ranges, variances and mean
# are used, and the protected data is noised once
PUBLIC_SOURCE_MAP = {
    "ptbxl": "chapman_native",
    "chapman_native": "ptbxl",
    "chapman_matched": "ptbxl",
}

DATASET_LABELS = {
    "ptbxl": "PTB-XL",
    "chapman_native": "Chapman-Native",
    "chapman_matched": "Chapman-Matched",
}

# published prevalence from each dataset's descriptor. used as the decision
# threshold, the NB prior and the KNN correction target. public, so it costs
# no budget, and never computed from the protected labels
DESCRIPTOR_PREVALENCE = {
    "ptbxl": 0.0728,
    "chapman_native": 0.2106,
    "chapman_matched": 0.0728,
}
PREVALENCE_TOLERANCE = 0.02  # computed fold prevalence may drift this far

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
TARGET_COL = "af_detected"
FOLD_COL = "strat_fold"
# ids, metadata, label. everything else in a feature CSV is a feature
NON_FEATURE_COLS = frozenset({
    "ecg_id", "patient_id", "strat_fold", "validated_by_human",
    "timestamp", "af_detected",
})
AF_POSITIVE_CODES = frozenset({"AFIB", "AFLT"})
N_FEATURES = 62  # the record-level budget is N_FEATURES * epsilon

# patient-disjoint folds
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLDS = (9,)
TEST_FOLDS = (10,)
# MIA needs four pools: members, shadow and non-member pool, val, test
MIA_TARGET_TRAIN_FOLDS = (1, 2, 3, 4)
MIA_SHADOW_FOLDS = (5, 6, 7, 8)

# ---------------------------------------------------------------------------
# Allocation rules
# ---------------------------------------------------------------------------
RULES = (
    "uniform",
    "adaptive_feature",
    "adaptive_feature_reversed",
    "bounded_inverse_feature",
)
# old sweep CSVs use the old name, accept it everywhere
RULE_ALIASES = {"sample_specific": "bounded_inverse_feature"}
RULE_LABELS = {
    "baseline": "Baseline",
    "uniform": "Uniform",
    "adaptive_feature": "AdaptiveFeature",
    "adaptive_feature_reversed": "AdaptiveFeatureReversed",
    "bounded_inverse_feature": "BoundedInverseFeature",
}
RULE_COLORS = {
    "baseline": "#000000",
    "uniform": "#1f77b4",
    "adaptive_feature": "#ff7f0e",
    "adaptive_feature_reversed": "#2ca02c",
    "bounded_inverse_feature": "#d62728",
}
# color, marker and line style together, so the panels survive greyscale
RULE_MARKERS = {
    "baseline": "",
    "uniform": "o",
    "adaptive_feature": "s",
    "adaptive_feature_reversed": "^",
    "bounded_inverse_feature": "D",
}
RULE_STYLES = {
    "baseline": ":",
    "uniform": "-",
    "adaptive_feature": "--",
    "adaptive_feature_reversed": "-.",
    "bounded_inverse_feature": ":",
}
# figure spec: full column width, panel letters, legend below
FIG_WIDTH_IN = 6.5
FIG_DPI = 600
MODEL_PANEL_TITLES = {
    "knn": "KNN", "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest", "logreg": "Logistic Regression",
}

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
EPSILON_VALUES = (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
DELTA = 1e-5
N_RUNS = 30
NOISE_SEEDS = tuple(range(42, 42 + N_RUNS))
TRAIN_TEST_SEED = 42
MODEL_TYPES = ("knn", "naive_bayes", "random_forest", "logreg")
MODEL_LABELS = {"knn": "KNN", "naive_bayes": "NB", "random_forest": "RF", "logreg": "LR"}
N_JOBS = -1

# ---------------------------------------------------------------------------
# Classifier defaults
# ---------------------------------------------------------------------------
DEFAULT_VAR_SMOOTHING = 1e-3
DEFAULT_KNN_NEIGHBORS = 25
DEFAULT_RF_N_ESTIMATORS = 300

# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
DEFAULT_FPR_TARGETS = (0.001, 0.01, 0.1)
N_SHADOW_MODELS = 3
N_LIRA_SHADOWS = 64  # dominates MIA runtime, 32 for a fast pass
MIA_SAMPLE_SIZE = 500
N_ATTACK_TARGETS = 30
N_RECONSTRUCTION_CANDIDATES = 1000
SENSITIVE_ATTRIBUTES = ("sex", "age", "heart_rate")
ATTRIBUTE_TYPES = {"sex": "binary", "age": "continuous", "heart_rate": "continuous"}
N_BINS_CONTINUOUS = 10  # quantile bins for age and heart rate
N_TARGET_RECORDS = 500
