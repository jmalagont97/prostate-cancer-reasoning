"""exp_10: compare the ARD backbone's trained sigma_j on the full-schema frame against exp_5's
SVM-based per-factor weights results. Structural copy of exp_9/scripts/importance_comparison.py,
single frame this time (no 19-vs-23 loop) -- diagnostic script, not a scored CV condition.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_10/scripts/importance_comparison_fullschema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import fit_kdm_backbone_ard  # noqa: E402
from features_fullschema import restricted_feature_group_fullschema, select_exp10_feature_frame  # noqa: E402

from chimera_task1.reasoning_labels import TASK1_FACTORS
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}
IN_SCOPE_FACTORS = [f for f in TASK1_FACTORS if f != "fh"]
RESULTS_DIR = Path(__file__).parent.parent / "results"

# exp_5's established solvable/unsolvable split (weights_official_svm / weights_restricted_svm),
# for reference -- see experiments/exp_5/reports/summary.md.
EXP5_SOLVED = {"pirads", "bx", "dre", "age", "psa"}
EXP5_UNSOLVED = {"cspca", "comorbidity", "psad", "vol"}


def factor_importance(sigma_by_col: dict[str, float], factor: str) -> float | None:
    """Mean 1/sigma_j across a factor's column group -- higher = the kernel needs to be more
    sensitive to (i.e. finds more relevant) that factor's columns."""
    cols = restricted_feature_group_fullschema(factor)
    cols_present = [c for c in cols if c in sigma_by_col]
    if not cols_present:
        return None
    return float(np.mean([1.0 / sigma_by_col[c] for c in cols_present]))


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp10_feature_frame(inp_ann, mri_pca)
    print(f"feature frame: {X_frame.shape} (fullschema)")

    # No CV split (diagnostic, full-data fit) -- but still need train-only-style impute+scale.
    # With no held-out split here, "train" == the full 91-case set, matching exp_9's own
    # importance_comparison.py (full-data fit, no CV).
    from features_fullschema import IMPUTE_NEEDED_COLS, FULL_SCHEMA_COLUMNS
    from sklearn.preprocessing import MinMaxScaler

    X_filled = X_frame.copy()
    for col in IMPUTE_NEEDED_COLS:
        X_filled[col] = X_filled[col].fillna(X_filled[col].median())
    X_arr = X_filled[FULL_SCHEMA_COLUMNS].values.astype(np.float64)
    X_scaled = MinMaxScaler().fit_transform(X_arr)

    model = fit_kdm_backbone_ard(X_scaled, y_decision, n_classes=2, **ARD_CONFIG)
    sigma = model.kernel.sigma.detach().numpy()
    sigma_by_col = dict(zip(X_frame.columns, sigma))

    print("\n=== fullschema frame: trained sigma_j per column ===")
    for col, s in sorted(sigma_by_col.items(), key=lambda kv: kv[1]):
        print(f"  {col:28s} sigma_j={s:.4f}  (1/sigma_j={1/s:.4f})")

    factor_scores = {}
    for factor in IN_SCOPE_FACTORS:
        score = factor_importance(sigma_by_col, factor)
        if score is not None:
            factor_scores[factor] = round(score, 4)

    ranked = sorted(factor_scores.items(), key=lambda kv: kv[1], reverse=True)
    print("\n=== fullschema: factors ranked by ARD relevance (1/sigma_j, higher = more important) ===")
    for factor, score in ranked:
        tag = "solved (exp_5)" if factor in EXP5_SOLVED else "unsolved (exp_5)" if factor in EXP5_UNSOLVED else "?"
        print(f"  {factor:14s} {score:8.4f}   [{tag}]")

    top5 = {f for f, _ in ranked[:5]}
    agreement = len(top5 & EXP5_SOLVED)
    print(f"\nAgreement: {agreement}/5 of ARD's top-5-relevance factors are in exp_5's solved set {EXP5_SOLVED}")

    out_dir = RESULTS_DIR / "importance_comparison_fullschema"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "exp5_solved_factors": sorted(EXP5_SOLVED),
            "exp5_unsolved_factors": sorted(EXP5_UNSOLVED),
            "sigma_by_column": {k: round(float(v), 4) for k, v in sigma_by_col.items()},
            "factor_relevance_ranked": ranked,
            "top5_vs_exp5_solved_agreement": agreement,
        }, f, indent=2)


if __name__ == "__main__":
    main()
