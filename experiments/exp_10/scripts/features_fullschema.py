"""exp_10: full-schema feature frame -- Frame.md's 37-variable tabular schema (minus 2 exact
duplicates found during EDA, see DESIGN.md Section 2) + MRI-PCA(2), encoded with Frame.md's own
stated preprocessing convention (MinMax scaling + one-hot categoricals + explicit missing-flag
columns), not this project's build_preprocessor (median-impute) convention.

Column count verified programmatically (not by hand -- DESIGN.md's original count was wrong twice
before this): 37 raw source fields -> 48 encoded columns. See DESIGN.md Section 3 / this
experiment's IMPLEMENTATION.md finding #5.

One-hot encoding does NOT use sklearn.OneHotEncoder/ColumnTransformer (see IMPLEMENTATION.md
finding #3) -- every category is a small, fixed, known-in-advance set, so plain
`(series == "Category").astype(int)` columns are built directly into the returned frame, computed
once (deterministic, target-independent), exactly like this project's existing
cli_bx_missing/mri_missing flags. This means the frame's final 48 columns exist from the moment
select_exp10_feature_frame() returns -- no expansion happens later, so
X_frame.columns.get_loc(c)-style lookups (used throughout run_signals/run_reveal for occlusion
column indexing) work exactly like exp_3/exp_8/exp_9's frames, just wider.

Only 5 of the 48 columns carry genuine missingness needing a per-fold-fit median impute
(cli_pirads, path_hist_bx_isup, path_hist_bx_gl_prim, path_hist_bx_gl_sec, cli_fh_binary) --
handled by fit_transform_fullschema(), fit on train-fold rows only, mirroring how
build_preprocessor's SimpleImputer(median) is fit per fold everywhere else in this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Column groups (see DESIGN.md Section 2/3 for the EDA behind each choice)
# ---------------------------------------------------------------------------

# Complete (no missing) continuous fields, passed through as-is other than the
# vit_smoking_pack_years fill below. mri_pca_0/mri_pca_1 are appended by the caller-supplied
# mri_pca frame (already complete -- mri_pca_features() imputes missing-MRI cases to the PCA-space
# origin, plus its own separate mri_missing flag).
PASSTHROUGH_CONTINUOUS_COLS = [
    "cli_age", "cli_psa", "cli_psap", "cli_psav", "cli_psad", "cli_vol", "cli_months", "cli_cspca",
    "cli_comorbidity_count", "cli_allergies_count", "cli_ipss_score",
    "vit_weight_kg", "vit_height_cm", "vit_bmi", "vit_bp_systolic", "vit_bp_diastolic",
    "vit_heart_rate_bpm", "vit_smoking_pack_years",
    "psa_tr_count", "psa_tr_first_val", "psa_tr_min", "psa_tr_mean", "psa_tr_delta", "psa_tr_slope",
    "lab_creatinine_mg_dl", "lab_free_psa_ng_ml", "lab_free_total_ratio",
]  # 27 raw + mri_pca_0/mri_pca_1 joined in = 29

# psa_tr_last_val and psa_tr_max are EXCLUDED here -- exact duplicates of cli_psa (r=1.0, verified
# row-by-row on all 91 cases during DESIGN.md's EDA, Section 2a), not just correlated.

# Continuous fields with genuine missingness -- left as real NaN here, median-imputed per fold by
# fit_transform_fullschema() (never globally, to avoid leakage).
IMPUTE_NEEDED_COLS = [
    "cli_pirads", "path_hist_bx_isup", "path_hist_bx_gl_prim", "path_hist_bx_gl_sec", "cli_fh_binary",
]  # 5

# One-hot categories, per DESIGN.md Section 2/3 -- fixed, known in advance (not inferred from
# whatever happens to appear in a given CV fold).
DRE_CATEGORIES = ["Normal", "Nodus", "Abnormal", "Not done", "Suspicious"]
BX_CATEGORIES = ["Positive", "Negative"]  # missing rows get (0, 0), covered by cli_bx_missing
SMOKING_CATEGORIES = ["Never", "Ex-smoker", "Current"]

ONEHOT_COLS = (
    [f"cli_dre_{c}" for c in DRE_CATEGORIES]
    + [f"cli_bx_{c}" for c in BX_CATEGORIES]
    + [f"vit_smoking_status_{c}" for c in SMOKING_CATEGORIES]
)  # 10

MISSING_FLAG_COLS = ["cli_pirads_missing", "cli_bx_missing", "cli_fh_missing", "mri_missing"]  # 4

# Full 48-column final order: passthrough(29) + impute-needed(5) + flags(4) + one-hot(10)
FULL_SCHEMA_COLUMNS = (
    PASSTHROUGH_CONTINUOUS_COLS + ["mri_pca_0", "mri_pca_1"] + IMPUTE_NEEDED_COLS + MISSING_FLAG_COLS + ONEHOT_COLS
)


def select_exp10_feature_frame(inp: pd.DataFrame, mri_pca: pd.DataFrame) -> pd.DataFrame:
    """Build exp_10's 48-column full-schema frame.

    `mri_pca` is precomputed by the caller via
    `chimera_task1.train_decision.mri_pca_features(inp, n_components=2)` (or
    `experiments/exp_3/scripts/holdout_eval.py`'s `mri_pca_train_only` for held-out evaluation),
    indexed the same as `inp`, providing `mri_pca_0`, `mri_pca_1`, `mri_missing`.
    """
    frame = inp[PASSTHROUGH_CONTINUOUS_COLS].copy()
    # never-smokers have zero pack-years by construction, not an unknown quantity (DESIGN.md 2d) --
    # filled here, globally, since it's a fixed domain constant, not a statistic that could leak.
    frame["vit_smoking_pack_years"] = frame["vit_smoking_pack_years"].fillna(0.0)

    frame = frame.join(mri_pca[["mri_pca_0", "mri_pca_1"]])

    # Left as real NaN -- median-imputed per CV fold by fit_transform_fullschema(), not here.
    for col in IMPUTE_NEEDED_COLS:
        frame[col] = inp[col].values

    # Missing-flags: deterministic, target-independent, computed once (same discipline as this
    # project's existing cli_bx_missing/mri_missing flags).
    frame["cli_pirads_missing"] = inp["cli_pirads"].isna().astype("int64")
    bx_missing = inp["cli_bx"].isna()
    frame["cli_bx_missing"] = bx_missing.astype("int64")  # shared by isup/gl_prim/gl_sec, DESIGN.md 2e
    frame["cli_fh_missing"] = inp["cli_fh_binary"].isna().astype("int64")
    frame["mri_missing"] = mri_pca["mri_missing"].values

    # One-hot, built directly (no sklearn OneHotEncoder -- see module docstring / IMPLEMENTATION.md
    # finding #3). NaN never matches any category comparison, so missing cli_bx rows correctly get
    # (0, 0) across cli_bx_Positive/cli_bx_Negative with no separate handling needed.
    for cat in DRE_CATEGORIES:
        frame[f"cli_dre_{cat}"] = (inp["cli_dre"] == cat).astype("int64")
    for cat in BX_CATEGORIES:
        frame[f"cli_bx_{cat}"] = (inp["cli_bx"] == cat).astype("int64")
    for cat in SMOKING_CATEGORIES:
        frame[f"vit_smoking_status_{cat}"] = (inp["vit_smoking_status"] == cat).astype("int64")

    frame = frame[FULL_SCHEMA_COLUMNS]  # enforce the documented, stable column order
    return frame


# ---------------------------------------------------------------------------
# Per-factor restricted groups (weights_kdm_* occlusion loop, importance_comparison_fullschema.py)
# ---------------------------------------------------------------------------

_RESTRICTED_FEATURE_GROUP_FULLSCHEMA: dict[str, list[str]] = {
    "age": ["cli_age"],
    "psa": ["cli_psa"],  # psap/psav/psa_tr_* have no weight key of their own, same as exp_3-exp_9
    "dre": [f"cli_dre_{c}" for c in DRE_CATEGORIES],
    "bx": [f"cli_bx_{c}" for c in BX_CATEGORIES] + ["cli_bx_missing"],
    "pirads": ["cli_pirads", "cli_pirads_missing"],
    "psad": ["cli_psad"],
    "vol": ["cli_vol"],
    "cspca": ["cli_cspca"],
    "comorbidity": ["cli_comorbidity_count"],  # Frame.md has no comorb_* grouped flags, unlike exp_6-9
}


def restricted_feature_group_fullschema(factor: str) -> list[str]:
    """Column list for one factor's restricted-scope group, on exp_10's full-schema frame.

    `chimera_task1.features.restricted_feature_group()` hardcodes column names from the 19/23-col
    frames' engineered encoding (cli_dre_ordinal, cli_bx_positive, ...) that don't exist here --
    this is a new, exp_10-specific mapping, not an edit to that function (IMPLEMENTATION.md
    finding #2). `fh` is intentionally absent (same IN_SCOPE_FACTORS exclusion as every experiment
    since exp_2 -- unrelated to cli_fh_binary/cli_fh_missing still being raw predictive features).
    """
    return _RESTRICTED_FEATURE_GROUP_FULLSCHEMA[factor]


# ---------------------------------------------------------------------------
# Per-fold preprocessing: median-impute the 5 NaN-bearing columns (train-fold only), MinMax-scale
# everything (train-fold only) -- mirrors build_preprocessor + external StandardScaler's two-stage
# shape used by every other script in this project, per IMPLEMENTATION.md finding #1.
# ---------------------------------------------------------------------------

def fit_transform_fullschema(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Impute (median, train-fold only) + MinMax-scale (train-fold only), applied to both -- no
    leakage from test into the fit. Column order = X_train_raw.columns throughout (no expansion
    happens here, unlike a sklearn ColumnTransformer + OneHotEncoder pipeline would produce), so
    X_frame.columns.get_loc(c) lookups computed against the pre-transform frame stay valid against
    the transformed matrix's column positions.
    """
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    for col in IMPUTE_NEEDED_COLS:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    X_train_arr = X_train[FULL_SCHEMA_COLUMNS].values.astype(np.float64)
    X_test_arr = X_test[FULL_SCHEMA_COLUMNS].values.astype(np.float64)

    scaler = MinMaxScaler().fit(X_train_arr)
    return scaler.transform(X_train_arr), scaler.transform(X_test_arr)
