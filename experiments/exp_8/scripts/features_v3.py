"""exp_8: expanded 23-column feature frame -- exp_3's official frame (with psav/psap restored,
since select_official_feature_frame never dropped them; exp_3 was the one that dropped them
afterward) plus MRI-PCA, plus two new columns found via this project's own EDA (cli_isup, vit_bmi).

cli_isup/vit_bmi get NO manual imputation here -- build_preprocessor() already applies
SimpleImputer(strategy="median") to every non-categorical column, refit per CV fold on train rows
only. Imputing here would either leak (if computed globally) or duplicate that machinery for no
reason.
"""

from __future__ import annotations

import pandas as pd

from chimera_task1.features import select_official_feature_frame


def select_exp8_feature_frame(inp: pd.DataFrame, mri_pca: pd.DataFrame) -> pd.DataFrame:
    """23-column frame: exp_3's official-flags frame (psa/psap/psav/psad kept, unlike exp_3's own
    frame which drops psap/psav) + MRI-PCA(2) + cli_isup + vit_bmi.

    `mri_pca` is precomputed by the caller via
    `chimera_task1.train_decision.mri_pca_features(inp, n_components=2)`, indexed the same as `inp`.
    """
    frame = select_official_feature_frame(inp, comorbidity_treatment="flags")
    frame = frame.join(mri_pca)
    frame["cli_isup"] = inp["path_hist_bx_isup"].values
    frame["vit_bmi"] = inp["vit_bmi"].values
    return frame
