"""exp_14 smoke test: confirm fit_kdm_regress/compute_signals_regress work cleanly on real data,
both dim_y=1 (per-factor) and dim_y=9 (joint), before committing to the full CV/held-out/LOO runs.
Uses "dre" (historically the easiest factor) on the real 23-col frame, same choice exp_13's own
smoke test made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_regress_backbone import compute_signals_regress, fit_kdm_regress  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402

from chimera_task1.features import build_preprocessor
from chimera_task1.reasoning_labels import TASK1_FACTORS, WEIGHT_RANK, weight_col
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

ann, inp_ann = load_annotated()
full_inp = pd.read_csv("data/inputs.csv")
mri_pca_full = mri_pca_features(full_inp, n_components=2)
mri_pca_full["case_id"] = full_inp["case_id"].values
mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
X_full = select_exp8_feature_frame(inp_ann, mri_pca)

preprocessor = build_preprocessor(X_full)
X_pre = preprocessor.fit_transform(X_full)
X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
from sklearn.preprocessing import StandardScaler
X_pre = StandardScaler().fit_transform(X_pre)

IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]

print("=== dim_y=1 (per-factor), factor='dre' ===")
y_labels = ann[weight_col("dre")].values
y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])
model1 = fit_kdm_regress(X_pre, y_rank.reshape(-1, 1), dim_y=1)
sig1 = compute_signals_regress(model1, X_pre)
print(f"mean shape={sig1['mean'].shape} pred_rank shape={sig1['pred_rank'].shape} "
      f"pseudo_probs shape={sig1['pseudo_probs'].shape}")
assert sig1["mean"].shape == (91, 1)
assert sig1["pred_rank"].shape == (91, 1)
assert sig1["pseudo_probs"].shape == (91, 1, 4)
prob_sums = sig1["pseudo_probs"].sum(axis=2)
print(f"pseudo_probs row sums: min={prob_sums.min():.4f} max={prob_sums.max():.4f}")
assert np.allclose(prob_sums, 1.0, atol=1e-4)
assert not np.isnan(sig1["mean"]).any()
assert sig1["pred_rank"].min() >= 0 and sig1["pred_rank"].max() <= 3
acc1 = (sig1["pred_rank"].squeeze() == y_rank).mean()
print(f"in-sample rounded accuracy (memorization expected): {acc1:.3f}")

print("\n=== dim_y=9 (joint), all in-scope factors ===")
Y_all = np.stack([
    np.array([WEIGHT_RANK[label] for label in ann[weight_col(f)].values])
    for f in IN_SCOPE_FACTORS
], axis=1)
print(f"Y_all shape={Y_all.shape}")
model9 = fit_kdm_regress(X_pre, Y_all, dim_y=9)
sig9 = compute_signals_regress(model9, X_pre)
print(f"mean shape={sig9['mean'].shape} pred_rank shape={sig9['pred_rank'].shape} "
      f"pseudo_probs shape={sig9['pseudo_probs'].shape} variance shape={sig9['variance'].shape}")
assert sig9["mean"].shape == (91, 9)
assert sig9["pred_rank"].shape == (91, 9)
assert sig9["pseudo_probs"].shape == (91, 9, 4)
prob_sums9 = sig9["pseudo_probs"].sum(axis=2)
print(f"pseudo_probs row sums (all 9 dims): min={prob_sums9.min():.4f} max={prob_sums9.max():.4f}")
assert np.allclose(prob_sums9, 1.0, atol=1e-4)
assert not np.isnan(sig9["mean"]).any()
assert sig9["pred_rank"].min() >= 0 and sig9["pred_rank"].max() <= 3
acc9 = (sig9["pred_rank"] == Y_all).mean()
print(f"in-sample rounded accuracy, all factors pooled (memorization expected): {acc9:.3f}")
print(f"variance is per-case only, shared across factors (confirmed): shape={sig9['variance'].shape}")

print("\nSMOKE TEST PASSED: both dim_y=1 and dim_y=9 work cleanly on the real 23-col frame.")
