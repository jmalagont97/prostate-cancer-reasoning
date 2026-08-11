"""exp_6: dump per-case (per-instance) signal values, not just the aggregated per-condition
metrics in results/*/metrics.json.

Uses repeat=0's 5-fold split (same RANDOM_STATE=0 as run_signals.py) so every one of the 91
cases gets a genuine out-of-fold value -- not a value from a model that saw that case in
training. Writes one row per case: decision label/probability, confidence label + Signals A/B/C,
and per-factor weight label + Signals D/E for all 9 in-scope factors.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_6/scripts/dump_per_instance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone import compute_signals, fit_kdm_backbone, kernel_distance_contribution, occlusion_delta  # noqa: E402

from chimera_task1.features import build_preprocessor, restricted_feature_group, select_exp3_feature_frame
from chimera_task1.reasoning_labels import TASK1_FACTORS, weight_col
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
OUT_PATH = Path(__file__).parent.parent / "results" / "per_instance_signals.csv"


def main() -> None:
    ann, inp_ann = load_annotated()
    case_ids = ann["case_id"].values
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)
    y_confidence_labels = ann["target_confidence"].values

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp3_feature_frame(inp_ann, mri_pca)

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

    factor_col_idx = {f: [X_frame.columns.get_loc(c) for c in restricted_feature_group(f, "flags")] for f in IN_SCOPE_FACTORS}

    n = len(y_decision)
    p_yes = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    dispersion = np.full(n, np.nan)
    participation = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    occlusion = {f: np.full(n, np.nan) for f in IN_SCOPE_FACTORS}
    kernel_dist = {f: np.full(n, np.nan) for f in IN_SCOPE_FACTORS}

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fi, (train_idx, test_idx) in enumerate(kf.split(X_pre)):
        scaler = StandardScaler().fit(X_pre[train_idx])
        X_train = scaler.transform(X_pre[train_idx])
        X_test = scaler.transform(X_pre[test_idx])

        model = fit_kdm_backbone(X_train, y_decision[train_idx], n_classes=2)
        sig = compute_signals(model, X_test)
        assert sig["probs_check_ok"]

        p_yes[test_idx] = sig["probs"][:, 1]
        entropy[test_idx] = sig["entropy"]
        dispersion[test_idx] = sig["dispersion"]
        participation[test_idx] = sig["participation"]
        fold_id[test_idx] = fi

        for factor in IN_SCOPE_FACTORS:
            col_idx = factor_col_idx[factor]
            fill = np.median(X_train[:, col_idx], axis=0)
            occlusion[factor][test_idx] = occlusion_delta(model, X_test, col_idx, fill)
            kernel_dist[factor][test_idx] = kernel_distance_contribution(model, X_test, col_idx)

    out = pd.DataFrame({
        "case_id": case_ids,
        "fold": fold_id,
        "decision_true": ann["target_biopsy_decision"].values,
        "decision_p_yes": np.round(p_yes, 4),
        "decision_pred": np.where(p_yes >= 0.5, "yes", "no"),
        "confidence_true": y_confidence_labels,
        "signal_entropy": np.round(entropy, 4),
        "signal_dispersion": np.round(dispersion, 4),
        "signal_participation": np.round(participation, 2),
    })
    for factor in IN_SCOPE_FACTORS:
        out[f"weight_true_{factor}"] = ann[weight_col(factor)].values
        out[f"occlusion_{factor}"] = np.round(occlusion[factor], 4)
        out[f"kerneldist_{factor}"] = np.round(kernel_dist[factor], 3)

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} columns to {OUT_PATH}")
    print(out.head(10).to_string())


if __name__ == "__main__":
    main()
