"""Task3 model registry: every entry exposes the same .fit(df)/.predict_months(df) interface so
swapping models is a one-line change (task3_variant_tester.py / models_bridge_task3.py just pick a
different key from MODEL_FACTORIES). Mirrors task1's subtask_config.py pattern, simplified because
task3 has one sub-task (a single predicted `months_to_recurrence`), not three.

scikit-survival (Random Survival Forest / Gradient-Boosted Cox, the plan's T3.4) was evaluated for
install and is NOT available in this environment -- it needs a C++ compiler toolchain
("Microsoft Visual C++ 14.0 or greater") that isn't present here, confirmed by a failed pip install
(building its `ecos` dependency from source). The 6 models below are all from `lifelines` (already a
project dependency, no new install): Cox at 3 regularization strengths, plus the 3 standard AFT
families (Weibull/LogNormal/LogLogistic -- T3.5 from the plan) which predict an actual time directly,
a better structural match to the `months_to_recurrence` output than Cox's hazard ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task3_embeddings import EmbeddingSource  # noqa: E402

# exp_3's winner (structured_cleaned_plus_pathology, minus active_treatment_flag -- see
# experiments/exp_3/reports/summary.md Issue #2 for why that column was dropped).
NUMERIC_FEATURES = [
    "age", "psa", "dre_abnormal", "dre_bilateral", "dre_ct_stage_ordinal",
    "isup_grade_group", "pt_stage_ordinal", "epe_present", "margins_positive",
    "svi_present", "lvi_present", "ln_assessed", "ln_positive",
]
CATEGORICAL_FEATURES: list[str] = []

# Safety cap for predicted months: AFT/Cox expectation/median can return inf (or absurdly large
# values) for a very-low-risk case whose survival curve never crosses the relevant threshold
# within the observed horizon. Capped at ~3x the max observed follow-up (110 months, see
# notebooks/task3_eda.ipynb §2) rather than left as inf, which would break downstream arithmetic
# (MAE, the schema's plain-float requirement) even though C-index itself only needs a finite,
# correctly-ordered value.
MAX_MONTHS_CAP = 330.0

# Event (BCR status) prediction: horizon + S(t) threshold, swept under LOOCV against real
# ground-truth labels (experiments/exp_7/) -- 60 months (5-year BCR, the standard clinical
# benchmark) with threshold 0.7 gave the best balanced accuracy (0.779) among all combinations
# tried, well clear of a trivial "always 0" baseline's 0.500.
EVENT_HORIZON_MONTHS = 60.0
EVENT_THRESHOLD = 0.7


def build_preprocessor():
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES)])


class _CoxAdapter:
    """Wraps CoxPHFitter's partial-hazard model behind predict_months(); uses predict_expectation()
    (the integral of the survival curve -- well-behaved here, confirmed 0/75 inf/capped values on
    this data). predict_median() was tried first and rejected: it requires the survival curve to
    cross 50%, which it never does within the observed follow-up window for 62/75 cases (all
    silently collapsing to the same capped value), destroying most of the model's actual ranking
    resolution -- this was caught by comparing this adapter's LOOCV C-index (0.65) against exp_3's
    equivalent raw-partial-hazard-ranking result (0.77) and finding the gap traced entirely to that
    capping, not a real difference in the underlying model."""

    def __init__(self, penalizer: float):
        self.penalizer = penalizer
        self.pre = build_preprocessor()
        self.model = None

    def fit(self, df: pd.DataFrame):
        X = np.asarray(self.pre.fit_transform(df))
        cox_df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        cox_df["months_to_recurrence"] = df["months_to_recurrence"].to_numpy()
        cox_df["event"] = df["event"].to_numpy()
        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.model.fit(cox_df, duration_col="months_to_recurrence", event_col="event")
        return self

    def predict_months(self, df: pd.DataFrame) -> np.ndarray:
        X = np.asarray(self.pre.transform(df))
        cox_df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        # predict_expectation/predict_median squeeze to a bare scalar for a single-row frame (no
        # .to_numpy()) -- np.atleast_1d(np.asarray(...)) handles both the scalar and Series cases.
        pred = np.atleast_1d(np.asarray(self.model.predict_expectation(cox_df), dtype=float)).flatten()
        pred = np.where(np.isfinite(pred), pred, MAX_MONTHS_CAP)
        return np.clip(pred, 0.0, MAX_MONTHS_CAP)


class _AFTAdapter:
    """Wraps a lifelines AFT fitter (Weibull/LogNormal/LogLogistic) -- these predict an actual
    time directly (predict_expectation()), which is what the output schema wants, rather than a
    relative hazard that needs converting."""

    def __init__(self, fitter_cls):
        self.fitter_cls = fitter_cls
        self.pre = build_preprocessor()
        self.model = None

    def fit(self, df: pd.DataFrame):
        X = np.asarray(self.pre.fit_transform(df))
        aft_df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        aft_df["months_to_recurrence"] = df["months_to_recurrence"].to_numpy()
        aft_df["event"] = df["event"].to_numpy()
        self.model = self.fitter_cls(penalizer=0.1)
        self.model.fit(aft_df, duration_col="months_to_recurrence", event_col="event")
        return self

    def predict_months(self, df: pd.DataFrame) -> np.ndarray:
        X = np.asarray(self.pre.transform(df))
        aft_df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        pred = np.atleast_1d(np.asarray(self.model.predict_expectation(aft_df), dtype=float)).flatten()
        pred = np.where(np.isfinite(pred), pred, MAX_MONTHS_CAP)
        return np.clip(pred, 0.0, MAX_MONTHS_CAP)


class _CoxEmbeddingAdapter:
    """Cox model on NUMERIC_FEATURES + PCA-reduced pooled embedding(s) from one or more
    EmbeddingSource(s). Stateful across fit()/predict_months(): the PCA (and the imputer, and the
    train-fold mean fallback for sources with real per-case missingness) are fit ONCE in fit() and
    reused in predict_months() -- never refit on data being predicted, which would leak. Promoted
    from experiments/exp_5/scripts/run_stage_a.py's exploration code (same logic, hardened into a
    stateful class instead of a per-fold closure) after exp_5 confirmed prostatectomy_slides,
    PCA(k=5), is a genuine, bootstrap-significant win (see task3_embeddings.py's docstring)."""

    def __init__(self, penalizer: float, embedding_source_names: list[str], k: int,
                 percentile: float | None = None):
        self.penalizer = penalizer
        self.embedding_source_names = embedding_source_names
        self.k = k
        # percentile=None -> predict_expectation() (mean of the survival curve). A systematic-
        # miscalibration check found this mean is inflated far above reality by this cohort's heavy
        # censoring (75% never recur): +40mo mean bias, 42mo MAE on the 19 known-truth (event=1)
        # cases, up to 11x overestimate on cases that recurred within a year. predict_percentile(p)
        # -- the time by which (1-p) of the population would have recurred -- swept p in
        # {0.5,...,0.95}: p=0.90 both improves ranking_score (0.8575->0.8765, borderline-significant,
        # bootstrap 95% CI [-0.002,+0.046]) AND collapses the bias (+40.1mo -> -9.7mo, MAE 42.2->10.0mo)
        # and coverage (61/75 capped at the old median default -> 7/75 at p=0.90). Validated only for
        # this specific (penalizer, embedding) combination -- not extended to other MODEL_FACTORIES
        # entries without separately testing them.
        self.percentile = percentile
        self.sources = {name: EmbeddingSource(name, "single" if name == "mri" else "bag")
                         for name in embedding_source_names}
        self.pcas: dict[str, PCA] = {}
        self.fallbacks: dict[str, np.ndarray] = {}
        self.has_missingness: dict[str, bool] = {}
        self.imputer = SimpleImputer(strategy="median")
        self.model = None
        self.feature_cols: list[str] = []

    def _embedding_columns(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for name, source in self.sources.items():
            case_ids = df["case_id"].tolist()
            pooled = {cid: source.pooled_vector(cid) for cid in case_ids}
            if fitting:
                self.has_missingness[name] = any(v is None for v in pooled.values())
                valid = [v for v in pooled.values() if v is not None]
                self.fallbacks[name] = np.mean(valid, axis=0) if valid else np.zeros(source.raw_dim())
                matrix = np.array([pooled[c] if pooled[c] is not None else self.fallbacks[name]
                                    for c in case_ids])
                n_components = min(self.k, matrix.shape[0] - 1, matrix.shape[1])
                self.pcas[name] = PCA(n_components=n_components, random_state=0).fit(matrix)

            fallback = self.fallbacks[name]
            matrix = np.array([pooled[c] if pooled[c] is not None else fallback for c in case_ids])
            reduced = self.pcas[name].transform(matrix)
            for j in range(reduced.shape[1]):
                out[f"{name}_{j}"] = reduced[:, j]
            if self.has_missingness[name]:
                out[f"has_{name}"] = [1.0 if pooled[c] is not None else 0.0 for c in case_ids]
        return out

    def fit(self, df: pd.DataFrame):
        emb_cols = self._embedding_columns(df, fitting=True)
        self.feature_cols = NUMERIC_FEATURES + list(emb_cols.columns)
        X = self.imputer.fit_transform(pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True),
                                                    emb_cols.reset_index(drop=True)], axis=1))
        cox_df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        cox_df["months_to_recurrence"] = df["months_to_recurrence"].to_numpy()
        cox_df["event"] = df["event"].to_numpy()
        self.model = CoxPHFitter(penalizer=self.penalizer)
        self.model.fit(cox_df, duration_col="months_to_recurrence", event_col="event")
        return self

    def _build_cox_df(self, df: pd.DataFrame) -> pd.DataFrame:
        emb_cols = self._embedding_columns(df, fitting=False)
        X = self.imputer.transform(pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True),
                                               emb_cols.reset_index(drop=True)], axis=1))
        return pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])

    def predict_months(self, df: pd.DataFrame) -> np.ndarray:
        cox_df = self._build_cox_df(df)
        if self.percentile is None:
            raw = self.model.predict_expectation(cox_df)
        else:
            raw = self.model.predict_percentile(cox_df, p=self.percentile)
        pred = np.atleast_1d(np.asarray(raw, dtype=float)).flatten()
        pred = np.where(np.isfinite(pred), pred, MAX_MONTHS_CAP)
        return np.clip(pred, 0.0, MAX_MONTHS_CAP)

    def predict_event(self, df: pd.DataFrame, horizon: float = EVENT_HORIZON_MONTHS,
                       threshold: float = EVENT_THRESHOLD) -> np.ndarray:
        """Binary BCR-status prediction (0/1) required by the actual challenge output contract
        (confirmed against the org's own webpage and the reference agent's real write path in
        run.py -- Task3Output's pydantic schema, used only for an internal LLM form-fill step,
        omits it and is NOT the true submission contract). Derived from the same fitted survival
        curve predict_months() already uses: event=1 if the cumulative incidence by `horizon`
        exceeds `1-threshold` (i.e. S(horizon|x) < threshold). horizon=60 months (the standard
        5-year BCR benchmark) and threshold=0.7 were swept, not assumed -- see
        experiments/exp_7/ for the full LOOCV grid against real ground-truth event labels
        (balanced accuracy 0.779 vs. a trivial "always 0" baseline's 0.500)."""
        cox_df = self._build_cox_df(df)
        sf = self.model.predict_survival_function(cox_df, times=[horizon])
        s_at_h = np.asarray(sf.values, dtype=float).flatten()
        return (s_at_h < threshold).astype(int)


MODEL_FACTORIES = {
    "cox_p001": lambda: _CoxAdapter(penalizer=0.01),   # lighter regularization
    "cox_p01": lambda: _CoxAdapter(penalizer=0.1),     # exp_3's original default
    "cox_p03": lambda: _CoxAdapter(penalizer=0.3),
    "cox_p05": lambda: _CoxAdapter(penalizer=0.5),
    "cox_p07": lambda: _CoxAdapter(penalizer=0.7),
    "cox_p10": lambda: _CoxAdapter(penalizer=1.0),
    "cox_p20": lambda: _CoxAdapter(penalizer=2.0),
    "weibull_aft": lambda: _AFTAdapter(WeibullAFTFitter),
    "lognormal_aft": lambda: _AFTAdapter(LogNormalAFTFitter),
    "loglogistic_aft": lambda: _AFTAdapter(LogLogisticAFTFitter),
    # exp_5 winners -- Cox(penalizer=0.3) + PCA-reduced prostatectomy-slide embedding(s).
    "cox_p03_prostatectomy_pca2": lambda: _CoxEmbeddingAdapter(0.3, ["prostatectomy_slides"], 2),
    "cox_p03_prostatectomy_pca5": lambda: _CoxEmbeddingAdapter(0.3, ["prostatectomy_slides"], 5),
    "cox_p03_prostatectomy_pca10": lambda: _CoxEmbeddingAdapter(0.3, ["prostatectomy_slides"], 10),
    "cox_p03_all_embeddings_pca5": lambda: _CoxEmbeddingAdapter(
        0.3, ["mri", "biopsy_slides", "prostatectomy_slides"], 5),
    # Calibration-check follow-up: same winning (penalizer, embedding) combo, but predicting the
    # 90th-percentile survival time instead of the mean -- see _CoxEmbeddingAdapter's percentile
    # docstring for why.
    "cox_p03_prostatectomy_pca5_p90": lambda: _CoxEmbeddingAdapter(
        0.3, ["prostatectomy_slides"], 5, percentile=0.90),
}

# Promoted default -- only changed if a swap beats this under LOOCV concordance_index (the exact
# evaluate.py function), per this project's promotion-gated discipline. Set by task3_variant_tester.py's
# sweep; update by hand here once a new winner is confirmed (mirrors task1's subtask_config.py).
#
# History:
#   exp_4 (2026-08-24): penalizer swept 0.01-2.0 on the tabular+text feature set, a clean unimodal
#   curve peaking at 0.3 (ranking_score=0.7655) -- beat the original 0.1 default (0.7487) but that
#   +0.017 gain did NOT survive a bootstrap significance check (95% CI crossed zero).
#   exp_5 (2026-08-25): adding PCA-reduced prostatectomy-slide embeddings (k=5) on top of the
#   cox_p03 tabular+text set lifts LOOCV C-index 0.7655 -> 0.8575 -- a real, bootstrap-significant
#   win (95% CI [+0.028, +0.176], 99.6% of resamples favor it), confirmed robust across k in
#   {2, 5, 10} (0.845-0.858), NOT a k=5-specific fluke. See experiments/exp_5/.
#   Follow-up calibration check (2026-08-28): a systematic check across all 19 event=1 cases found
#   predict_expectation() (the point estimate used above) is severely upward-biased under this
#   cohort's heavy censoring (+40mo mean bias). Swept predict_percentile(p) over p in {0.5,...,0.95};
#   p=0.90 both improves ranking_score (0.8575->0.8765) AND collapses the bias to -9.7mo (MAE
#   42.2->10.0mo). C-index gain is borderline (bootstrap 95% CI [-0.002,+0.046]) but never worse,
#   while the calibration gain is decisive -- promoted on that basis.
DEFAULT_MODEL_KEY = "cox_p03_prostatectomy_pca5_p90"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
