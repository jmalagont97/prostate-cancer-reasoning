"""Deployable wrapper around exp_12's confidence_kdm_direct_scalar_23col model.

This is the class that gets pickled by `export_model.py` and unpickled by any downstream code
that loads `confidence_kdm_23col.pkl`. Keep this module import-stable (class name, module path,
attribute names) -- pickle stores objects by reference to their defining class, so renaming or
moving `ConfidenceKDMPredictor` breaks every previously-pickled file.

See MODEL_CARD.md for the input schema, output format, and validated performance numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the project's own feature-engineering code (single source of truth for how a raw
# inputs.csv-shaped row becomes a model-ready feature vector) rather than duplicating it here --
# same discipline as every other script in this project. Requires this file to stay inside the
# repo (any relative position under experiments/ works; only the four ".parent" hops below need
# adjusting if this file is moved).
_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from chimera_task1.features import MRI_EMB_PREFIX, select_official_feature_frame  # noqa: E402

CONFIDENCE_LEVELS = ["uncertain", "borderline", "clear"]


def mri_pca_transform(pca, inp: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Apply an already-fitted MRI-embedding PCA to new rows (never re-fit at inference time).

    Cases with no MRI embedding (any of the 1024 mri_emb_* columns missing, or the columns absent
    entirely) get coordinates (0, 0) -- the origin of the fitted PCA space -- plus `mri_missing=1`,
    exactly mirroring how missing-MRI cases were handled during training
    (`chimera_task1.train_decision.mri_pca_features`).
    """
    emb_cols = [c for c in inp.columns if c.startswith(MRI_EMB_PREFIX)]
    coords = np.zeros((len(inp), n_components))
    if emb_cols:
        emb = inp[emb_cols]
        has_mri = ~emb.isna().any(axis=1)
        if has_mri.any():
            coords[has_mri.values] = pca.transform(emb.loc[has_mri].values)
        missing = (~has_mri).astype("int64").values
    else:
        missing = np.ones(len(inp), dtype="int64")

    cols = [f"mri_pca_{i}" for i in range(n_components)]
    out = pd.DataFrame(coords, columns=cols, index=inp.index)
    out["mri_missing"] = missing
    return out


class ConfidenceKDMPredictor:
    """exp_12's `confidence_kdm_direct_scalar_23col` model, refit on all 91 labeled cases.

    A scalar-bandwidth Kernel Density Matrix (kdm-torch) trained *directly* on the 3-class
    confidence label, memory-based (every training row is a frozen prototype; only the RBF kernel
    bandwidth sigma is learned). See MODEL_CARD.md for the full pipeline and validated numbers.
    """

    def __init__(
        self,
        mri_pca,
        preprocessor,
        scaler,
        kdm_model,
        feature_columns: list[str],
        n_mri_components: int = 2,
    ) -> None:
        self.mri_pca = mri_pca
        self.preprocessor = preprocessor
        self.scaler = scaler
        self.kdm_model = kdm_model
        self.feature_columns = feature_columns
        self.n_mri_components = n_mri_components

    def _build_frame(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Raw inputs.csv-shaped rows -> the 23-column model-ready frame, in training column order."""
        frame = select_official_feature_frame(inp, comorbidity_treatment="flags")
        frame = frame.join(mri_pca_transform(self.mri_pca, inp, self.n_mri_components))
        frame["cli_isup"] = inp["path_hist_bx_isup"].values
        frame["vit_bmi"] = inp["vit_bmi"].values
        return frame[self.feature_columns]

    def predict_proba(self, inp: pd.DataFrame) -> np.ndarray:
        """Raw inputs.csv-shaped rows -> (n, 3) class probabilities, columns = CONFIDENCE_LEVELS order."""
        frame = self._build_frame(inp)
        X = self.preprocessor.transform(frame)
        X = X.toarray() if hasattr(X, "toarray") else X
        X = self.scaler.transform(X)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        self.kdm_model.eval()
        with torch.no_grad():
            probs = self.kdm_model(Xt).numpy()
        return probs

    def predict(self, inp: pd.DataFrame) -> list[str]:
        """Raw inputs.csv-shaped rows -> predicted confidence label per row (argmax of predict_proba)."""
        probs = self.predict_proba(inp)
        return [CONFIDENCE_LEVELS[i] for i in probs.argmax(axis=1)]

    def predict_full(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Raw inputs.csv-shaped rows -> a DataFrame with the predicted label and all 3 class
        probabilities, indexed the same as `inp` -- the most convenient entry point for most uses."""
        probs = self.predict_proba(inp)
        idx = probs.argmax(axis=1)
        return pd.DataFrame(
            {
                "predicted_confidence": [CONFIDENCE_LEVELS[i] for i in idx],
                "p_uncertain": probs[:, 0],
                "p_borderline": probs[:, 1],
                "p_clear": probs[:, 2],
            },
            index=inp.index,
        )
