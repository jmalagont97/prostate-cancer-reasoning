"""Feature preparation for Task 1 (MRI-only diagnostic decision).

Column groups and encoding choices here mirror Section 5 of
``visualize_datasets.ipynb`` ("Input variable inventory — what's usable for
modeling"). This module is the single place that decides how a raw
``inputs.csv`` row (or, at inference time, the equivalent fields assembled
from ``structured-prompt.json`` + MCP-tool-revealed sections) turns into a
model-ready feature vector — see the module docstring warning in the build
plan: training and inference must go through the same code path here.

``TASK1_VARIABLE_TO_FEATURE`` maps the schema's 10 short ``variable_weights``
keys (``psa, age, dre, comorbidity, bx, pirads, psad, vol, cspca, fh``) to
the underlying engineered feature column(s) each one corresponds to — used
by the confidence/weight models (step 4) to know which feature(s) "ground"
each factor.
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Column groups (from the notebook's Section 5 inventory)
# ---------------------------------------------------------------------------

IDENTIFIER_COLS = ["case_id"]

# Dropped outright: free text (needs a separate NLP path, mostly restates a
# structured column elsewhere) and the near-constant pathology column.
# Exception: txt_comorbidities is still listed here (never fed as raw NLP text),
# but it is the source for the derived COMORBIDITY_GROUPS flags below rather
# than being discarded outright.
FREE_TEXT_COLS = [
    "txt_comorbidities",
    "txt_chief_complaint",
    "txt_history",
    "txt_physical_examination",
    "txt_prompt_summary_notes",
    "txt_full_prompt_narrative",
    "txt_radiology_report",
    "txt_previous_notes",
    "txt_psa_trend_summary",
    "txt_laboratory_results_summary",
    "txt_consolidated_ehr_narrative",
]
NEAR_CONSTANT_COLS = ["path_hist_bx_gl_tert"]

ORDINAL_COLS = [
    "cli_pirads",
    "cli_comorbidity_count",
    "cli_months",
    "path_hist_bx_isup",
    "path_hist_bx_gl_prim",
    "path_hist_bx_gl_sec",
]

BINARY_COLS = ["cli_fh_binary", "cli_allergies_count"]

# Text-valued but low-cardinality -> one-hot, with an explicit "missing"
# category rather than imputing (missingness can itself be informative,
# e.g. txt_allergies is 92% missing = "none recorded").
CATEGORICAL_COLS = [
    "cli_bx",
    "cli_dre",
    "vit_smoking_status",
    "txt_family_history_narrative",
    "txt_allergies",
]

CONTINUOUS_COLS = [
    "cli_age",
    "cli_psa",
    "cli_psap",
    "cli_psav",
    "cli_psad",
    "cli_vol",
    "cli_cspca",
    "cli_ipss_score",
    "lab_hemoglobin_g_dl",
    "lab_creatinine_mg_dl",
    "lab_free_psa_ng_ml",
    "lab_free_total_ratio",
    "psa_tr_first_val",
    "psa_tr_last_val",
    "psa_tr_min",
    "psa_tr_max",
    "psa_tr_mean",
    "psa_tr_delta",
    "psa_tr_slope",
    "vit_smoking_pack_years",
    "vit_weight_kg",
    "vit_height_cm",
    "vit_bmi",
    "vit_bp_systolic",
    "vit_bp_diastolic",
    "vit_heart_rate_bpm",
]

# Add an explicit missingness indicator for columns missing on a large
# fraction of cases, per the notebook's Section 5 recommendation.
HIGH_MISSING_COLS = ["lab_hemoglobin_g_dl", "vit_smoking_pack_years"]

MRI_EMB_PREFIX = "mri_emb_"

# ---------------------------------------------------------------------------
# Comorbidities: txt_comorbidities is a comma-joined free-text list (e.g.
# "Hypertension, Chronic kidney disease"). Only 10 distinct conditions occur
# anywhere in the 195-case training set, each with decent support (7-44
# cases), so rather than one flag per unique string (or dropping it as free
# text like the other txt_* columns), it's grouped into a handful of
# clinically-motivated categories for *this* decision (general comorbidity
# burden vs. renal vs. bleeding-risk-relevant vs. respiratory vs. BPH, which
# is not a frailty comorbidity at all but directly inflates PSA/volume).
# Osteoarthritis gets no flag of its own -- least plausibly relevant to a
# biopsy decision, still captured via cli_comorbidity_count.
# `comorb_other_unmatched` is a robustness catch-all: set when a case lists a
# condition that isn't one of the 10 known ones, e.g. new wording or a new
# condition in the validation/test sets that this training-set-derived
# grouping has never seen.
# ---------------------------------------------------------------------------

COMORBIDITY_GROUPS: dict[str, list[str]] = {
    "comorb_cardiometabolic": [
        "Hypertension",
        "Hypercholesterolaemia",
        "Coronary artery disease",
        "Type 2 diabetes",
        "Obesity",
    ],
    "comorb_renal": ["Chronic kidney disease"],
    "comorb_bleeding_risk": ["Atrial fibrillation"],
    "comorb_respiratory": ["COPD"],
    "comorb_bph": ["Benign prostatic hyperplasia"],
}
# Conditions known well enough not to trip comorb_other_unmatched, including
# Osteoarthritis even though it has no flag of its own.
_KNOWN_CONDITIONS = [c for conditions in COMORBIDITY_GROUPS.values() for c in conditions] + ["Osteoarthritis"]
_KNOWN_CONDITIONS_LOWER = {c.lower() for c in _KNOWN_CONDITIONS}

COMORBIDITY_FLAG_COLS = [*COMORBIDITY_GROUPS.keys(), "comorb_other_unmatched"]


def _split_conditions(raw: str) -> list[str]:
    """Split a comma-joined comorbidity string into condition names, dropping parenthetical detail
    (e.g. "Obesity (BMI >30)" -> "Obesity")."""
    parts = []
    for entry in raw.split(","):
        entry = re.sub(r"\s*\(.*?\)", "", entry).strip()
        if entry:
            parts.append(entry)
    return parts


def comorbidity_flags(txt_comorbidities: pd.Series) -> pd.DataFrame:
    """One 0/1 flag per clinically-grouped comorbidity category, plus `comorb_other_unmatched`.

    See the COMORBIDITY_GROUPS comment above for the grouping rationale.
    """
    parsed = txt_comorbidities.fillna("").apply(_split_conditions)

    out = {}
    for group, conditions in COMORBIDITY_GROUPS.items():
        cond_set = {c.lower() for c in conditions}
        out[group] = parsed.apply(lambda conds, cs=cond_set: int(any(c.lower() in cs for c in conds)))

    out["comorb_other_unmatched"] = parsed.apply(
        lambda conds: int(any(c.lower() not in _KNOWN_CONDITIONS_LOWER for c in conds))
    )
    return pd.DataFrame(out, index=txt_comorbidities.index)


# ---------------------------------------------------------------------------
# Official-schema feature path (exp_2): restricted to the 11 documented
# Task-1 input variables (psa, psap, psav, psad, vol, age, pirads, dre,
# cspca, bx, pmhx) rather than exp_1's 47-column engineered frame, which
# included vit_*/lab_*/psa_tr_*/path_hist_* columns outside the documented
# structured-prompt.json schema and may not exist at real inference time.
# `ct` is excluded (confirmed absent from this training data: only 2/195
# cases have any T-stage mention at all, buried in free text). `fh` stays on
# its own separate tool-revealed path (TASK1_VARIABLE_TO_FEATURE below), not
# part of this 11-variable table.
# ---------------------------------------------------------------------------

OFFICIAL_CONTINUOUS_COLS = [
    "cli_psa",
    "cli_psap",
    "cli_psav",
    "cli_psad",
    "cli_vol",
    "cli_age",
    "cli_cspca",
]

# dre's real categories (Normal 136, Nodus 49, Not done 5, Abnormal 4,
# Suspicious 1) include "Not done", which is missingness wearing a category
# label, not a severity level -- pulled into its own flag rather than an
# ordinal rung.
_DRE_ORDINAL_MAP = {"Normal": 0, "Abnormal": 1, "Nodus": 2, "Suspicious": 3}


def encode_dre_ordinal(dre: pd.Series) -> pd.DataFrame:
    """Ordinal-encode `dre` by clinical severity, with `Not done` pulled out as its own flag."""
    not_done = (dre == "Not done").astype("int64")
    ordinal = dre.map(_DRE_ORDINAL_MAP)  # "Not done" and NaN both map to NaN, imputed downstream
    return pd.DataFrame({"cli_dre_ordinal": ordinal, "cli_dre_not_done": not_done}, index=dre.index)


def encode_bx_binary(bx: pd.Series) -> pd.DataFrame:
    """Binary-encode `bx` (prior biopsy status): bx_positive + bx_missing; "Negative" is (0, 0)."""
    positive = (bx == "Positive").astype("int64")
    missing = bx.isna().astype("int64")
    return pd.DataFrame({"cli_bx_positive": positive, "cli_bx_missing": missing}, index=bx.index)


def select_official_feature_frame(inp: pd.DataFrame, *, comorbidity_treatment: str = "flags") -> pd.DataFrame:
    """Build a feature frame restricted to the 11 officially documented Task-1 input variables.

    `comorbidity_treatment`: "flags" (6 grouped `comorb_*` binary flags, from `comorbidity_flags`)
    or "count" (the single `cli_comorbidity_count` ordinal) -- the exp_2 comorbidity-treatment
    ablation axis.
    """
    if comorbidity_treatment not in ("flags", "count"):
        raise ValueError(f"comorbidity_treatment must be 'flags' or 'count', got {comorbidity_treatment!r}")

    frame = inp[[*OFFICIAL_CONTINUOUS_COLS, "cli_pirads"]].copy()
    frame = frame.join(encode_dre_ordinal(inp["cli_dre"]))
    frame = frame.join(encode_bx_binary(inp["cli_bx"]))
    if comorbidity_treatment == "flags":
        frame = frame.join(comorbidity_flags(inp["txt_comorbidities"]))
    else:
        frame["cli_comorbidity_count"] = inp["cli_comorbidity_count"]
    return frame


# exp_3: 19-column frame = exp_2's official-flags frame, minus the two most
# redundant PSA-family columns (psap r=0.99, psav r=0.95 with psa -- psad is
# kept since it's ~psa/vol, not purely a psa duplicate), plus a 2-component
# MRI-embedding PCA (84.25% cumulative variance) + missing-MRI flag. See
# experiments/exp_3/DESIGN.md for the full rationale.
EXP3_DROPPED_PSA_COLS = ["cli_psap", "cli_psav"]


def select_exp3_feature_frame(inp: pd.DataFrame, mri_pca: pd.DataFrame) -> pd.DataFrame:
    """Build exp_3's 19-column feature frame: official-flags frame minus psap/psav, plus MRI-PCA.

    `mri_pca` is precomputed by the caller via
    `chimera_task1.train_decision.mri_pca_features(inp, n_components=2)` (already generic over
    `n_components`, reused as-is) and passed in indexed the same as `inp`.
    """
    frame = select_official_feature_frame(inp, comorbidity_treatment="flags")
    frame = frame.drop(columns=EXP3_DROPPED_PSA_COLS)
    frame = frame.join(mri_pca)
    return frame


def select_exp4_feature_frame(inp: pd.DataFrame) -> pd.DataFrame:
    """exp_4: exp_3's 19-column frame minus the 3 MRI-PCA columns -- 16 columns, clinical +
    comorbidity flags only. Isolates whether exp_3's MRI-PCA addition was actually helping."""
    frame = select_official_feature_frame(inp, comorbidity_treatment="flags")
    frame = frame.drop(columns=EXP3_DROPPED_PSA_COLS)
    return frame


# Per-factor restricted feature groups for the weights_restricted_* conditions
# (exp_2). Only the 8 non-comorbidity in-scope factors -- "comorbidity" is
# handled by restricted_feature_group() below since its group depends on the
# comorbidity_treatment in use, and "fh" is excluded entirely (separate
# tool-revealed path, see TASK1_VARIABLE_TO_FEATURE below).
TASK1_VARIABLE_TO_FEATURE_GROUP: dict[str, list[str]] = {
    "psa": ["cli_psa"],  # psap/psav have no weight key of their own; kept out of every restricted group
    "age": ["cli_age"],
    "dre": ["cli_dre_ordinal", "cli_dre_not_done"],
    "bx": ["cli_bx_positive", "cli_bx_missing"],
    "pirads": ["cli_pirads"],
    "psad": ["cli_psad"],
    "vol": ["cli_vol"],
    "cspca": ["cli_cspca"],
}


def restricted_feature_group(factor: str, comorbidity_treatment: str = "flags") -> list[str]:
    """Column list for one factor's restricted-scope group (weights_restricted_* conditions).

    `comorbidity_treatment` only affects the "comorbidity" factor's group; ignored otherwise.
    """
    if factor == "comorbidity":
        return list(COMORBIDITY_FLAG_COLS) if comorbidity_treatment == "flags" else ["cli_comorbidity_count"]
    return TASK1_VARIABLE_TO_FEATURE_GROUP[factor]


# ---------------------------------------------------------------------------
# Schema variable -> feature column mapping (for the step-4 weight models)
# ---------------------------------------------------------------------------

TASK1_VARIABLE_TO_FEATURE: dict[str, str] = {
    "psa": "cli_psa",
    "age": "cli_age",
    "dre": "cli_dre",
    "comorbidity": "cli_comorbidity_count",
    "bx": "cli_bx",
    "pirads": "cli_pirads",
    "psad": "cli_psad",
    "vol": "cli_vol",
    "cspca": "cli_cspca",
    "fh": "cli_fh_binary",
}

# Which revealed section (schema's TOOL_REVEAL_INFO / ground truth's
# reveal_sequence strings) each schema variable can be "grounded" by, beyond
# always being visible in structured-prompt.json. Only 'fh' requires a tool
# call per TASK1_VARIABLES in the baseline's schema.py; the others are
# already in-context. Kept here for the step-4/step-5 reveal-policy logic.
VARIABLE_REQUIRES_REVEAL: dict[str, str | None] = {
    "psa": None,
    "age": None,
    "dre": None,
    "comorbidity": None,
    "bx": None,
    "pirads": None,
    "psad": None,
    "vol": None,
    "cspca": None,
    "fh": "family_history",
}


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with one ``<col>_missing`` flag per high-missing column."""
    out = df.copy()
    for col in HIGH_MISSING_COLS:
        out[f"{col}_missing"] = out[col].isna().astype("int64")
    return out


def select_feature_frame(inp: pd.DataFrame, *, include_mri_pca: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the model-ready feature frame from a raw ``inputs.csv``-shaped DataFrame.

    Drops identifiers/free-text/near-constant columns, adds missingness
    indicators and the derived comorbidity-group flags (from
    ``txt_comorbidities``, the one free-text column not simply discarded),
    and optionally appends pre-computed MRI-embedding PCA components (pass
    the DataFrame returned by a fitted PCA, indexed the same as ``inp``).
    """
    keep_cols = ORDINAL_COLS + BINARY_COLS + CATEGORICAL_COLS + CONTINUOUS_COLS
    missing_in_input = [c for c in keep_cols if c not in inp.columns]
    if missing_in_input:
        raise KeyError(f"Expected columns not found in input frame: {missing_in_input}")

    frame = add_missingness_indicators(inp[keep_cols])
    frame = frame.join(comorbidity_flags(inp["txt_comorbidities"]))
    if include_mri_pca is not None:
        frame = frame.join(include_mri_pca)
    return frame


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for ``feature_frame`` (impute + one-hot categoricals).

    Numeric columns (ordinal/binary/continuous/missingness flags/MRI-PCA)
    are median-imputed and passed through; categorical columns get an
    explicit "missing" category then one-hot encoding.
    """
    categorical_present = [c for c in CATEGORICAL_COLS if c in feature_frame.columns]
    numeric_present = [c for c in feature_frame.columns if c not in categorical_present]

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_present),
            ("cat", categorical_pipe, categorical_present),
        ]
    )
