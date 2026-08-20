"""Export exp_12's confidence_kdm_direct_scalar_23col model as a single deployable pickle file.

exp_12's own scripts (run_confidence_direct_scalar.py, etc.) only ever fit the model inside CV
folds / a held-out split / LOO folds -- every fitted model object was discarded once its score was
recorded, by design (the point was estimating out-of-sample performance, not producing an artifact
to ship). This script performs the same fit procedure exactly (same feature frame, same
preprocessing, same fit_kdm_backbone call, same hyperparameters) but on ALL 91 labeled cases at
once, and keeps every fitted piece (MRI-PCA, imputer, scaler, the trained KDM itself) bundled into
one `ConfidenceKDMPredictor` (see predictor.py) that a downstream user can pickle-load and call
directly on new raw cases.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_12/model/export_model.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))  # exp_12/scripts (unused directly, kept for parity)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import fit_kdm_backbone  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402
from predictor import ConfidenceKDMPredictor, mri_pca_transform  # noqa: E402

from chimera_task1.features import MRI_EMB_PREFIX, build_preprocessor
from chimera_task1.reasoning_labels import CONFIDENCE_LEVELS, CONFIDENCE_RANK, ordinal_distance
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_MRI_COMPONENTS = 2
OUT_PATH = Path(__file__).parent / "confidence_kdm_23col.pkl"


def fit_mri_pca(inp_full: pd.DataFrame, n_components: int = 2) -> PCA:
    """Fit MRI-embedding PCA on every case with an MRI embedding in the full input pool (195
    cases, labeled + unlabeled) -- exactly what exp_12's own run script did (a wider, unsupervised
    fit basis than the 91 labeled cases alone), not a new choice made for this export."""
    emb_cols = [c for c in inp_full.columns if c.startswith(MRI_EMB_PREFIX)]
    emb = inp_full[emb_cols]
    has_mri = ~emb.isna().any(axis=1)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(emb.loc[has_mri].values)
    print(f"MRI-PCA fit on {has_mri.sum()}/{len(inp_full)} cases with MRI: "
          f"{pca.explained_variance_ratio_.sum():.1%} variance explained")
    return pca


def main() -> None:
    ann, inp_ann = load_annotated()  # the 91 labeled cases
    inp_full = pd.read_csv("data/inputs.csv")  # 195 cases, MRI-PCA fit basis only
    print(f"n_labeled={len(ann)}, n_full_pool={len(inp_full)}\n")

    pca = fit_mri_pca(inp_full, n_components=N_MRI_COMPONENTS)
    mri_pca_ann = mri_pca_transform(pca, inp_ann, n_components=N_MRI_COMPONENTS)

    X_frame = select_exp8_feature_frame(inp_ann, mri_pca_ann)
    feature_columns = list(X_frame.columns)
    print(f"feature frame: {X_frame.shape}, columns: {feature_columns}\n")

    y_conf_labels = ann["target_confidence"].values
    y_conf_rank = np.array([CONFIDENCE_RANK[label] for label in y_conf_labels])

    preprocessor = build_preprocessor(X_frame)
    X_imputed = preprocessor.fit_transform(X_frame)
    X_imputed = X_imputed.toarray() if hasattr(X_imputed, "toarray") else X_imputed

    scaler = StandardScaler().fit(X_imputed)
    X_scaled = scaler.transform(X_imputed)

    print("Fitting the KDM backbone on all 91 labeled cases (same hyperparameters as exp_12: "
          "300 epochs, Adam lr=1e-2, sigma-only trainable, n_comp=91)...")
    model = fit_kdm_backbone(X_scaled, y_conf_rank, n_classes=3)

    predictor = ConfidenceKDMPredictor(
        mri_pca=pca,
        preprocessor=preprocessor,
        scaler=scaler,
        kdm_model=model,
        feature_columns=feature_columns,
        n_mri_components=N_MRI_COMPONENTS,
    )

    # In-sample sanity check ONLY -- every training row is literally one of the model's own
    # frozen prototypes, so this number is expected to look far better than the validated
    # out-of-sample numbers (CV 0.491 / LOO 0.440 / repeated-holdout 0.447 ordinal distance,
    # see MODEL_CARD.md). It confirms the exported predictor reproduces the training fit
    # correctly, nothing more.
    probs = predictor.predict_proba(inp_ann)
    preds_rank = probs.argmax(axis=1)
    pred_labels = [CONFIDENCE_LEVELS[r] for r in preds_rank]
    acc = accuracy_score(y_conf_rank, preds_rank)
    macro_f1 = f1_score(y_conf_rank, preds_rank, average="macro", labels=[0, 1, 2], zero_division=0)
    ord_dist = ordinal_distance(list(y_conf_labels), pred_labels, CONFIDENCE_RANK)
    print(f"\nIn-sample check (expected near-perfect, NOT a generalization estimate): "
          f"accuracy={acc:.3f} macro_f1={macro_f1:.3f} ordinal_distance={ord_dist:.3f}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(predictor, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
