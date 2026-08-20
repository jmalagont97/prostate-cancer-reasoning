"""Cohort loading and per-modality preprocessing for Task 1 (old schema).

`Data/preprocessed_old/task1/` is the schema every one of exp_1-24 actually
reads: `patient_id`, missing values as the string `'NONE'`, splits from
`experiments/exp_4/results/mccv_design.csv` (100 MCCV columns). This module
targets that schema behind `CohortSpec` so the newer
`Data/preprocessed/task1/` schema (`case_id`, 50-split `mccv_loocv_splits.csv`)
can be a second spec later without touching `load_cohort`'s callers.

Must never import `kdm` or `src/methods` — this is harness code, not model
code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


NUM_COLS = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
CAT_COLS = ["dre"]

CONFIDENCE_CERTAINTY_MAP = {"clear": 1.00, "borderline": 0.50, "uncertain": 0.25}
CONFIDENCE_CODE_MAP = {"uncertain": 0, "borderline": 1, "clear": 2}


def resolve_data_dir(project_root: Path) -> Path:
    """Old-schema resolver, lifted from exp_23/scripts/train.py:47-55. Tries
    the (absent-in-this-checkout) lowercase path exp_1-24's scripts hardcode
    before falling back to the tracked `Data/preprocessed_old/task1/`."""
    candidates = [
        project_root / "data" / "chimera26" / "preprocessed" / "task1",
        project_root / "Data" / "preprocessed_old" / "task1",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"None of the candidate data dirs exist: {candidates}")


@dataclass
class CohortSpec:
    """Names the files/columns `load_cohort` reads, so a second schema is a
    second CohortSpec rather than a rewrite. `OLD_SCHEMA` below is the only
    instance in use so far."""

    id_col: str = "patient_id"
    num_cols: list = field(default_factory=lambda: list(NUM_COLS))
    cat_cols: list = field(default_factory=lambda: list(CAT_COLS))
    expected_n: int = 88
    expected_n_yes: int = 54
    expected_n_no: int = 34


OLD_SCHEMA = CohortSpec()


@dataclass
class Cohort:
    df_tab: pd.DataFrame
    df_mri: pd.DataFrame | None
    df_text: pd.DataFrame | None
    df_design: pd.DataFrame
    pids: np.ndarray
    y_binary: np.ndarray
    confidence: np.ndarray  # raw string labels: "clear"/"borderline"/"uncertain"
    y_conf: np.ndarray
    dre_categories: list


def load_cohort(data_dir: Path, project_root: Path, spec: CohortSpec = OLD_SCHEMA,
                 load_mri: bool = True, load_text: bool = True) -> Cohort:
    """Lifted from exp_23/scripts/train.py:79-126 (`load_cohort`), extended
    to optionally also align the MRI/text frames for multimodal use. Keeps
    exp_23's `dtype=str, keep_default_na=False` (the schema's `'NONE'`
    sentinel is not in pandas' default na_values) and its N=88 / 54-yes/34-no
    assertion, which corrects exp_13's published 56/32.
    """
    df_tab = pd.read_csv(data_dir / "clinical_data_tabular.csv", dtype=str, keep_default_na=False)
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv", dtype=str, keep_default_na=False)
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv", dtype=str, keep_default_na=False)
    df_design = pd.read_csv(project_root / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    for c in spec.num_cols:
        df_tab[c] = pd.to_numeric(df_tab[c], errors="coerce")

    pids = df_design[spec.id_col].values
    df_tab = df_tab[df_tab[spec.id_col].isin(pids)].sort_values(spec.id_col).reset_index(drop=True)
    df_dec = df_dec[df_dec[spec.id_col].isin(pids)].sort_values(spec.id_col).reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning[spec.id_col].isin(pids)].sort_values(spec.id_col).reset_index(drop=True)
    df_design = df_design.sort_values(spec.id_col).reset_index(drop=True)

    df_mri = df_text = None
    if load_mri:
        df_mri = pd.read_csv(data_dir / "mri_embeddings.csv")
        df_mri = df_mri[df_mri[spec.id_col].isin(pids)].sort_values(spec.id_col).reset_index(drop=True)
    if load_text:
        df_text = pd.read_csv(data_dir / "clinical_prompts.csv", dtype=str, keep_default_na=False)
        df_text = df_text[df_text[spec.id_col].isin(pids)].sort_values(spec.id_col).reset_index(drop=True)

    labeled_mask = df_dec["biopsy_decision"] != "NONE"

    def _mask(df):
        return None if df is None else df[labeled_mask].reset_index(drop=True)

    df_tab_labeled = _mask(df_tab)
    df_dec_labeled = _mask(df_dec)
    df_reasoning_labeled = _mask(df_reasoning)
    df_design_labeled = _mask(df_design)
    df_mri_labeled = _mask(df_mri)
    df_text_labeled = _mask(df_text)

    # Every frame was independently filtered to `pids` then sorted by id
    # before masking, so a patient-id equality check here is the assertion
    # exp_13-17 never had — alignment there was positional and silently
    # fragile.
    for name, df in [("dec", df_dec_labeled), ("reasoning", df_reasoning_labeled), ("design", df_design_labeled),
                      ("mri", df_mri_labeled), ("text", df_text_labeled)]:
        if df is None:
            continue
        if not np.array_equal(df[spec.id_col].values, df_tab_labeled[spec.id_col].values):
            raise AssertionError(f"patient_id mismatch between df_tab and {name} after masking")

    pids_labeled = df_dec_labeled[spec.id_col].values
    biopsy_label_map = {"yes": 1, "no": 0}
    y_binary = df_dec_labeled["biopsy_decision"].map(biopsy_label_map).values.astype(int)

    assert len(y_binary) == spec.expected_n, f"expected N={spec.expected_n}, got {len(y_binary)}"
    n_yes, n_no = int((y_binary == 1).sum()), int((y_binary == 0).sum())
    assert n_yes == spec.expected_n_yes and n_no == spec.expected_n_no, (
        f"class-count mismatch: yes={n_yes}, no={n_no}"
    )

    confidence = df_reasoning_labeled["confidence"].values
    y_conf_series = df_reasoning_labeled["confidence"].map(CONFIDENCE_CODE_MAP)
    assert not y_conf_series.isna().any(), "unexpected unmapped confidence value in labeled cohort"
    y_conf = y_conf_series.values.astype(int)

    dre_categories = sorted(df_tab_labeled["dre"].unique().tolist())

    return Cohort(
        df_tab=df_tab_labeled, df_mri=df_mri_labeled, df_text=df_text_labeled,
        df_design=df_design_labeled, pids=pids_labeled, y_binary=y_binary,
        confidence=confidence, y_conf=y_conf, dre_categories=dre_categories,
    )


def build_targets(y_binary: np.ndarray, confidence: np.ndarray | None = None,
                   certainty_map: dict | None = None):
    """The single place soft targets are built — replaces the byte-identical
    block copied into exp_13/14/15/16/23/24.

    `certainty_map=None` -> `y_soft = y_binary` (the former "hard" arm),
    `soft_from_confidence=False`.
    `certainty_map={"clear":1.0,"borderline":0.5,"uncertain":0.25}` (i.e.
    `CONFIDENCE_CERTAINTY_MAP`) -> `y_soft = 0.50 +/- 0.50*c` (the former
    "soft"/uncertainty-weighted arm), `soft_from_confidence=True`. Unmapped/
    missing confidence values default to fully certain (`c=1.00`) via
    `.fillna(1.00)`, matching exp_13-24 exactly — documented here rather
    than silent.
    """
    from src.methods.base import Targets  # Targets lives in methods/base.py, not evaluation

    y_binary = np.asarray(y_binary, dtype=int)
    if certainty_map is None:
        y_soft = y_binary.astype(np.float32)
        return Targets(y_binary=y_binary, y_soft=y_soft, y_conf=None, soft_from_confidence=False)

    conf_series = pd.Series(confidence)
    c_weights = conf_series.map(certainty_map).fillna(1.00).values.astype(np.float32)
    y_soft = np.where(y_binary == 1, 0.50 + 0.50 * c_weights, 0.50 - 0.50 * c_weights).astype(np.float32)
    return Targets(y_binary=y_binary, y_soft=y_soft, y_conf=None, soft_from_confidence=True)


# ---------------------------------------------------------------------------
# Tabular preprocessing — not L2-normalized (kept as exp_23 had it, so the
# reproduction gate holds; RBF-as-cosine is only meaningful for MRI/text).
# ---------------------------------------------------------------------------
def build_tabular_features(df: pd.DataFrame, train_idx, val_idx, num_cols=NUM_COLS, cat_cols=CAT_COLS,
                            dre_categories: list | None = None):
    """MinMax (numeric) + OneHotEncoder (categorical), fit on train only.

    `dre_categories=None` infers categories from the training subset (exp_13's
    reference-KNN pipeline; DO NOT use for KDM — see below). Passing the full
    known category set instead (`dre_categories=cohort.dre_categories`) is
    required for any memory-based model whose n_comp/encoded_size is fixed
    per instance: 3 of `dre`'s 5 levels are singletons in the 88-cohort, so
    49/100 MCCV splits and 3/88 LOOCV folds would infer <5 categories,
    crashing `init_kdm_layer`'s shape-checked copy_. Both variants share this
    function; do not merge the fixed/inferred cases into one default.
    """
    scaler = MinMaxScaler()
    X_tr_num = scaler.fit_transform(df.iloc[train_idx][num_cols])
    X_va_num = scaler.transform(df.iloc[val_idx][num_cols])

    if dre_categories is None:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(categories=[dre_categories], handle_unknown="ignore", sparse_output=False)
    X_tr_cat = ohe.fit_transform(df.iloc[train_idx][cat_cols])
    X_va_cat = ohe.transform(df.iloc[val_idx][cat_cols])

    X_tr = np.hstack([X_tr_num, X_tr_cat]).astype(np.float32)
    X_va = np.hstack([X_va_num, X_va_cat]).astype(np.float32)
    assert np.isfinite(X_tr).all() and np.isfinite(X_va).all(), "non-finite values in feature matrix"
    return X_tr, X_va


# ---------------------------------------------------------------------------
# MRI preprocessing — MinMax -> optional PCA -> L2 Normalizer (last step,
# since MinMax/PCA each destroy any earlier unit norm).
# ---------------------------------------------------------------------------
def build_mri_features(df: pd.DataFrame, train_idx, val_idx, pca_variance: float | None = None):
    feat_cols = [c for c in df.columns if c != "patient_id"]
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(df.iloc[train_idx][feat_cols])
    X_va = scaler.transform(df.iloc[val_idx][feat_cols])

    if pca_variance is not None:
        pca = PCA(n_components=pca_variance, random_state=42)
        X_tr = pca.fit_transform(X_tr)
        X_va = pca.transform(X_va)

    normalizer = Normalizer(norm="l2")
    X_tr = normalizer.fit_transform(X_tr).astype(np.float32)
    X_va = normalizer.transform(X_va).astype(np.float32)
    return X_tr, X_va


# ---------------------------------------------------------------------------
# Text preprocessing — spaCy lemma/stopword cleaning happens ONCE at cohort
# level (stateless per-document map, leak-free; see `clean_texts_spacy`),
# not inside every fold. TF-IDF -> MinMax -> PCA -> L2 Normalizer are fit
# per fold/split.
# ---------------------------------------------------------------------------
def clean_texts_spacy(raw_texts, model: str = "en_core_web_sm", batch_size: int = 50):
    """Runs once on the full labeled corpus before any splitting. Stateless
    per-document lemmatization is leak-free regardless of when it runs;
    running it inside every fold of a configs x 100-split Phase A (as a
    naive per-fold pipeline would) is pathological and buys nothing."""
    import spacy

    try:
        nlp = spacy.load(model, disable=["ner", "parser"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model {model!r} is not installed. Run: python -m spacy download {model}"
        ) from e
    cleaned = []
    for doc in nlp.pipe(raw_texts, batch_size=batch_size):
        tokens = [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
        cleaned.append(" ".join(tokens))
    return np.array(cleaned)


def build_text_features(cleaned_texts: np.ndarray, train_idx, val_idx, max_features=None,
                         pca_variance: float | None = 0.90):
    tfidf = TfidfVectorizer(max_features=max_features, norm="l2")
    X_tr = tfidf.fit_transform(cleaned_texts[train_idx]).toarray().astype(np.float32)
    X_va = tfidf.transform(cleaned_texts[val_idx]).toarray().astype(np.float32)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    if pca_variance is not None:
        n_comp = min(X_tr.shape[0], X_tr.shape[1])
        if n_comp >= 2:
            pca = PCA(n_components=pca_variance, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_va = pca.transform(X_va)

    normalizer = Normalizer(norm="l2")
    X_tr = normalizer.fit_transform(X_tr).astype(np.float32)
    X_va = normalizer.transform(X_va).astype(np.float32)
    return X_tr, X_va
