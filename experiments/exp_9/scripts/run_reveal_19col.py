"""exp_9: reveal-sequence with the ARD backbone, exp_3's original 19-column frame.

Same entropy-increase-from-occlusion mechanism as exp_8's run_reveal_kdm.py, adapted for the ARD
backbone (compute_signals_ard instead of compute_signals). The only frame-specific detail:
psa_trend's feature group has no psav/psap on this frame (they don't exist here) -- trimmed to
just cli_psa/cli_psad. The 4 modeled sections (family_history/pathology_report excluded, 0
positive examples among the 91 labeled cases) are the same regardless of frame.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_9/scripts/run_reveal_19col.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.reasoning_labels import REVEAL_SECTIONS, parse_reveal_sequences, reveal_set_precision
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated, make_classifier

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"
FRAME_TAG = "19col"
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}

# psa_trend trimmed vs. exp_8's version -- psav/psap don't exist in this 19-column frame.
SECTION_FEATURE_GROUPS = {
    "psa_trend": ["cli_psa", "cli_psad"],
    "radiology_report": ["cli_pirads", "mri_pca_0", "mri_pca_1", "mri_missing"],
    "laboratory_results": ["cli_cspca", "cli_vol"],
    "previous_notes": ["cli_bx_positive", "cli_bx_missing", "cli_age"],
}


def occlusion_entropy_delta_ard(model, X: np.ndarray, col_idx: list[int], fill_values: np.ndarray) -> np.ndarray:
    original_entropy = compute_signals_ard(model, X)["entropy"]
    X_occ = X.copy()
    X_occ[:, col_idx] = fill_values
    occluded_entropy = compute_signals_ard(model, X_occ)["entropy"]
    return occluded_entropy - original_entropy


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)

    seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])
    sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]
    print(f"Modeled sections: {sections}")
    dropped = [s for s in REVEAL_SECTIONS if s not in sections]
    print(f"Dropped (0 positive examples): {dropped}\n")
    assert all(s in SECTION_FEATURE_GROUPS for s in sections)

    Y = np.array([[1 if s in seq else 0 for s in sections] for seq in seqs])

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp3_feature_frame(inp_ann, mri_pca)
    print(f"feature frame: {X_frame.shape} ({FRAME_TAG})\n")

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    section_col_idx = {s: [X_frame.columns.get_loc(c) for c in SECTION_FEATURE_GROUPS[s]] for s in sections}

    n = len(y_decision)
    n_sections = len(sections)
    repeat_precisions = []
    per_repeat_macro_f1 = []
    per_section_f1s = {s: [] for s in sections}
    per_section_stats = {s: {"tp": [], "fp": [], "fn": [], "tn": []} for s in sections}

    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        oof_pred = np.zeros((n, n_sections), dtype=int)

        for train_idx, test_idx in kf.split(X_pre):
            scaler = StandardScaler().fit(X_pre[train_idx])
            X_train = scaler.transform(X_pre[train_idx])
            X_test = scaler.transform(X_pre[test_idx])

            model = fit_kdm_backbone_ard(X_train, y_decision[train_idx], n_classes=2, **ARD_CONFIG)

            for si, section in enumerate(sections):
                col_idx = section_col_idx[section]
                fill = np.median(X_train[:, col_idx], axis=0)
                R_train = occlusion_entropy_delta_ard(model, X_train, col_idx, fill)
                R_test = occlusion_entropy_delta_ard(model, X_test, col_idx, fill)

                y_sec_train = Y[train_idx, si]
                if len(np.unique(y_sec_train)) < 2:
                    oof_pred[test_idx, si] = y_sec_train[0]
                    continue
                clf = make_classifier()
                clf.fit(R_train.reshape(-1, 1), y_sec_train)
                oof_pred[test_idx, si] = clf.predict(R_test.reshape(-1, 1))

        precisions = [
            reveal_set_precision(
                [s for s, v in zip(sections, true_row) if v],
                [s for s, v in zip(sections, pred_row) if v],
            )
            for true_row, pred_row in zip(Y, oof_pred)
        ]
        repeat_precisions.append(float(np.mean(precisions)))

        section_f1s_this_repeat = []
        for si, section in enumerate(sections):
            true_col, pred_col = Y[:, si], oof_pred[:, si]
            tp = int(np.sum((true_col == 1) & (pred_col == 1)))
            fp = int(np.sum((true_col == 0) & (pred_col == 1)))
            fn = int(np.sum((true_col == 1) & (pred_col == 0)))
            tn = int(np.sum((true_col == 0) & (pred_col == 0)))
            per_section_stats[section]["tp"].append(tp)
            per_section_stats[section]["fp"].append(fp)
            per_section_stats[section]["fn"].append(fn)
            per_section_stats[section]["tn"].append(tn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_section_f1s[section].append(f1)
            section_f1s_this_repeat.append(f1)
        per_repeat_macro_f1.append(float(np.mean(section_f1s_this_repeat)))

        print(f"repeat {repeat} done")

    per_section_report = {}
    for section in sections:
        tp = np.mean(per_section_stats[section]["tp"])
        fp = np.mean(per_section_stats[section]["fp"])
        fn = np.mean(per_section_stats[section]["fn"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        per_section_report[section] = {
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(float(np.mean(per_section_f1s[section])), 3),
            "positive_rate": round(float(Y[:, sections.index(section)].mean()), 3),
        }

    payload = {
        "condition": f"reveal_kdm_ard_{FRAME_TAG}",
        "target": "reveal_sequence",
        "features": f"exp_3 19-column frame ({FRAME_TAG})",
        "model": f"ARD-KDM backbone + per-section occlusion-entropy signal, config: {ARD_CONFIG}",
        "sections_modeled": sections,
        "sections_dropped_no_positive_examples": dropped,
        "set_precision_mean": round(float(np.mean(repeat_precisions)), 3),
        "set_precision_std": round(float(np.std(repeat_precisions)), 3),
        "macro_f1_mean": round(float(np.mean(per_repeat_macro_f1)), 3),
        "macro_f1_std": round(float(np.std(per_repeat_macro_f1)), 3),
        "n_repeats": N_REPEATS,
        "per_section": per_section_report,
        "note": "first-ever 19-column-frame reveal result (exp_8 only ran reveal on its 23-column frame); no prior baseline for this exact frame",
    }
    out_dir = RESULTS_DIR / f"reveal_kdm_ard_{FRAME_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[reveal_kdm_ard_{FRAME_TAG}] set_precision={payload['set_precision_mean']} macro_f1={payload['macro_f1_mean']}")
    for section, stats in per_section_report.items():
        print(f"  {section:20s} precision={stats['precision']} recall={stats['recall']} f1={stats['f1']}")


if __name__ == "__main__":
    main()
