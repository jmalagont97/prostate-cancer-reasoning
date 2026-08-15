"""exp_13 smoke test: confirm fit_kdm_backbone / fit_kdm_backbone_ard both work cleanly at
n_classes=4 (never used at 4 classes before -- exp_11/exp_12 used n_classes=3) on one real
factor/scope combo, before committing to the full 4-condition x 9-factor run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_8" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import compute_signals, fit_kdm_backbone  # noqa: E402
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_v3 import select_exp8_feature_frame  # noqa: E402

from chimera_task1.features import build_preprocessor
from chimera_task1.reasoning_labels import WEIGHT_RANK, weight_col
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

ann, inp_ann = load_annotated()
full_inp = pd.read_csv("data/inputs.csv")
mri_pca_full = mri_pca_features(full_inp, n_components=2)
mri_pca_full["case_id"] = full_inp["case_id"].values
mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
X_full = select_exp8_feature_frame(inp_ann, mri_pca)

factor = "dre"  # consistently the easiest factor in this project's history
y_labels = ann[weight_col(factor)].values
y_rank = np.array([WEIGHT_RANK[label] for label in y_labels])
print(f"factor={factor}, n={len(y_rank)}, class counts={np.bincount(y_rank, minlength=4)}")

preprocessor = build_preprocessor(X_full)
X_pre = preprocessor.fit_transform(X_full)
X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
from sklearn.preprocessing import StandardScaler
X_pre = StandardScaler().fit_transform(X_pre)

# --- scalar backbone ---
model = fit_kdm_backbone(X_pre, y_rank, n_classes=4)
sig = compute_signals(model, X_pre)
print(f"[scalar] probs shape={sig['probs'].shape}, rows sum to 1: "
      f"{np.allclose(sig['probs'].sum(axis=1), 1.0)}, probs_check_ok={sig['probs_check_ok']}")
assert sig["probs"].shape == (len(y_rank), 4)
assert np.allclose(sig["probs"].sum(axis=1), 1.0, atol=1e-4)
assert sig["probs_check_ok"]

# --- ARD backbone ---
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
model_ard = fit_kdm_backbone_ard(X_pre, y_rank, n_classes=4, **ARD_CONFIG)
sig_ard = compute_signals_ard(model_ard, X_pre)
print(f"[ard]    probs shape={sig_ard['probs'].shape}, rows sum to 1: "
      f"{np.allclose(sig_ard['probs'].sum(axis=1), 1.0)}, probs_check_ok={sig_ard['probs_check_ok']}")
assert sig_ard["probs"].shape == (len(y_rank), 4)
assert np.allclose(sig_ard["probs"].sum(axis=1), 1.0, atol=1e-4)
assert sig_ard["probs_check_ok"]

print("\nSMOKE TEST PASSED: both backbones work cleanly at n_classes=4.")
