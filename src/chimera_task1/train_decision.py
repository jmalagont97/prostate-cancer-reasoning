"""Step 3: train + cross-validate the Task 1 biopsy-decision (yes/no) model.

Run with the project's own .venv:
    .venv/Scripts/python.exe -m src.chimera_task1.train_decision
(run from the project root, or add ``src`` to PYTHONPATH)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from chimera_task1.features import MRI_EMB_PREFIX, build_preprocessor, select_feature_frame

RANDOM_STATE = 0
N_SPLITS = 5


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(f"{data_dir}/ground_truth.csv")
    inp = pd.read_csv(f"{data_dir}/inputs.csv")
    df = inp.merge(gt, on="case_id", how="inner")
    return gt, inp, df


def load_labeled_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ground truth + inputs, filtered to the 91 cases with an actual biopsy-decision label.

    BUGFIX (found during exp_3 review, 2026-08-10): `target_biopsy_decision` is NaN for 104/195
    cases -- the exact same 104 cases missing every other ground-truth annotation (confirmed
    identical to `train_reasoning.load_annotated()`'s 91-case set). Every decision script prior
    to this fix called `load_data()` (the full 195-row merge) and computed
    `y = (df["target_biopsy_decision"] == "yes")` directly on it. Pandas' `NaN == "yes"` is
    `False`, not an error, so those 104 unlabeled cases were silently coded as `y=0` ("no")
    rather than excluded -- corrupting every decision-model F1/ROC-AUC/PR-AUC and the naive
    baselines computed before this fix (see experiments/exp_3/reports/summary.md's correction
    note, and the corresponding corrections in exp_1/exp_2's reports). Use this function, not
    `load_data()`, whenever building a decision-model target; `load_data()` itself is left
    unchanged since it's still useful for unsupervised feature engineering (e.g. `mri_pca_features`
    below should still fit on the full 195-case embedding population, not just the 91 labeled
    ones, then have its output aligned to the labeled subset by the caller).
    """
    gt, inp, df = load_data(data_dir)
    labeled_gt = gt.loc[gt["target_biopsy_decision"].notna()].reset_index(drop=True)
    labeled_inp = (
        inp.loc[inp["case_id"].isin(labeled_gt["case_id"])]
        .set_index("case_id")
        .loc[labeled_gt["case_id"]]
        .reset_index()
    )
    labeled_df = labeled_inp.merge(labeled_gt, on="case_id", how="inner")
    assert len(labeled_gt) == len(labeled_inp) == len(labeled_df), "labeled gt/inp/df must align 1:1"
    return labeled_gt, labeled_inp, labeled_df


def mri_pca_features(inp: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """PCA-reduce the 1024-dim MRI embedding, imputing missing MRI cases to 0 (post-scale mean)."""
    emb_cols = [c for c in inp.columns if c.startswith(MRI_EMB_PREFIX)]
    emb = inp[emb_cols]
    has_mri = ~emb.isna().any(axis=1)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = np.zeros((len(inp), n_components))
    coords[has_mri.values] = pca.fit_transform(emb.loc[has_mri].values)
    # Missing-MRI cases sit at the origin (mean) of the fitted PCA space, plus a flag.
    cols = [f"mri_pca_{i}" for i in range(n_components)]
    out = pd.DataFrame(coords, columns=cols, index=inp.index)
    out["mri_missing"] = (~has_mri).astype("int64")
    print(
        f"MRI PCA({n_components}): {pca.explained_variance_ratio_.sum():.1%} variance explained; "
        f"{(~has_mri).sum()} case(s) with no MRI."
    )
    return out


def naive_baselines(y: np.ndarray) -> None:
    """Trivial, feature-blind baselines any real model must beat.

    Labels are computed dynamically from the actual class balance -- previously hardcoded
    "always 'no' (majority)", which silently became wrong (and misleadingly duplicated "always
    yes"'s number) once the decision-label bugfix changed the true positive rate from 28.7% to
    61.5%, making "yes" the majority class instead of "no".
    """
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=10, random_state=RANDOM_STATE)
    majority_is_yes = y.mean() >= 0.5
    strategies = {
        f"always 'no'{' (majority)' if not majority_is_yes else ' (minority)'}": DummyClassifier(
            strategy="constant", constant=0
        ),
        f"always 'yes'{' (majority)' if majority_is_yes else ' (minority)'}": DummyClassifier(
            strategy="constant", constant=1
        ),
        "random, class-proportional": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    }
    print("Naive baselines (feature-blind):")
    for name, dummy in strategies.items():
        scores = cross_val_score(dummy, np.zeros((len(y), 1)), y, cv=cv, scoring="f1", n_jobs=-1)
        print(f"  {name:28s} F1 = {scores.mean():.3f} +/- {scores.std():.3f}")
    # Sanity check against sklearn's f1_score directly (constant predictors don't need CV).
    always_yes_f1 = f1_score(y, np.ones_like(y))
    print(f"  (closed-form always-'yes' F1 = {always_yes_f1:.3f}, matches above modulo CV noise)\n")


def threshold_independent_metrics(feature_frame: pd.DataFrame, y: np.ndarray, label: str) -> None:
    """ROC-AUC / PR-AUC via out-of-fold predicted probabilities — tells apart "no signal" from
    "signal exists but the default 0.5 threshold is wrong for F1", which the naive-baseline
    comparison alone can't distinguish."""
    preprocessor = build_preprocessor(feature_frame)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5))]
    )
    proba = cross_val_predict(pipe, feature_frame, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)

    # Diagnostic only (threshold picked post-hoc on these same out-of-fold scores, so this F1 is
    # mildly optimistic) — shows the *best case* if we tuned the decision threshold for F1.
    thresholds = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y, (proba >= t).astype(int)) for t in thresholds]
    best_t, best_f1 = thresholds[int(np.argmax(f1s))], max(f1s)

    print(
        f"[{label}] ROC-AUC = {auc:.3f}, PR-AUC = {ap:.3f} (chance = {y.mean():.3f})"
        f"   | best-threshold F1 (diagnostic, optimistic) = {best_f1:.3f} @ t={best_t:.2f}"
    )


def evaluate(feature_frame: pd.DataFrame, y: np.ndarray, label: str) -> None:
    preprocessor = build_preprocessor(feature_frame)
    # Repeated CV: N=195 makes any single 5-fold split noisy (one unlucky fold can swing the
    # mean a lot), so average over 10 different splits for a more reliable estimate.
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=10, random_state=RANDOM_STATE)

    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            max_iter=100,
            class_weight="balanced",
        ),
    }
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipe, feature_frame, y, cv=cv, scoring="f1", n_jobs=-1)
        print(f"[{label}] {name:24s} F1 = {scores.mean():.3f} +/- {scores.std():.3f}  (n={len(scores)} folds)")


def main() -> None:
    _, full_inp, _ = load_data()  # full 195-case pool, for fitting MRI-PCA on unsupervised
    gt, inp, df = load_labeled_data()  # the 91 cases with an actual decision label
    y = (df["target_biopsy_decision"] == "yes").astype(int).values
    print(f"n={len(df)}, positive rate={y.mean():.2%}\n")

    naive_baselines(y)

    clinical_only = select_feature_frame(inp)
    evaluate(clinical_only, y, "clinical only")
    threshold_independent_metrics(clinical_only, y, "clinical only")

    # Fit PCA on the full 195-case embedding population (unsupervised, doesn't need a decision
    # label), then align to the 91 labeled cases by case_id -- more data for the PCA fit itself
    # than fitting on the 91-case subset directly.
    mri_pca_full = mri_pca_features(full_inp, n_components=10)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp["case_id"]].reset_index(drop=True)
    with_mri = select_feature_frame(inp, include_mri_pca=mri_pca)
    evaluate(with_mri, y, "clinical + MRI-PCA(10)")
    threshold_independent_metrics(with_mri, y, "clinical + MRI-PCA(10)")

    # Feature importance sanity check (HGB on full data, clinical-only frame) against the
    # notebook Section 5 correlations (PI-RADS +0.29, csPCa +0.23 were the strongest signals).
    preprocessor = build_preprocessor(clinical_only)
    X = preprocessor.fit_transform(clinical_only)
    feature_names = preprocessor.get_feature_names_out()
    clf = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_iter=100,
        class_weight="balanced",
    ).fit(X, y)
    perm_like = clf.feature_importances_ if hasattr(clf, "feature_importances_") else None
    if perm_like is None:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(clf, X, y, n_repeats=10, random_state=RANDOM_STATE, scoring="f1")
        perm_like = r.importances_mean
    top = pd.Series(perm_like, index=feature_names).sort_values(ascending=False).head(10)
    print("\nTop 10 features (HGB, clinical-only, full-data fit):")
    print(top.to_string())


if __name__ == "__main__":
    main()
