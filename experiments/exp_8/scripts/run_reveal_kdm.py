"""exp_8: reveal-sequence from the same shared KDM backbone, for the first time in this project.

Signal: for each reveal section S, how much does the backbone's own decision-entropy INCREASE
when S's associated feature group is occluded (reusing exp_6/7's occlusion machinery, but reading
entropy out of compute_signals() instead of p(yes)). A section whose absence spikes uncertainty
the most is the one most "worth" revealing.

Per this session's planning-time finding: family_history and pathology_report are revealed in
0/91 labeled cases -- REVEAL_SECTIONS is filtered down to the 4 sections with at least one
positive example, exactly matching the convention every prior reveal model in this project
(exp_1-exp_5) already uses (see experiments/exp_3/scripts/run_reveal.py). SECTION_FEATURE_GROUPS
below only covers those 4.

Requires results/hyperparameter_search/winner.json to exist (run search_hyperparameters_v3.py first).

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_8/scripts/run_reveal_kdm.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_6" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_7" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from features_v3 import select_exp8_feature_frame  # noqa: E402
from kdm_backbone import compute_signals  # noqa: E402
from kdm_backbone_v2 import fit_kdm_backbone  # noqa: E402

from chimera_task1.features import build_preprocessor
from chimera_task1.reasoning_labels import REVEAL_SECTIONS, parse_reveal_sequences, reveal_set_precision
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated, make_classifier

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS = 10
RESULTS_DIR = Path(__file__).parent.parent / "results"
SEARCH_DIR = RESULTS_DIR / "hyperparameter_search"

# Only the 4 sections with >=1 positive example among the 91 labeled cases (confirmed this
# session: family_history and pathology_report are both 0/91 -- see IMPLEMENTATION.md).
SECTION_FEATURE_GROUPS = {
    "psa_trend": ["cli_psa", "cli_psad", "cli_psav", "cli_psap"],
    "radiology_report": ["cli_pirads", "mri_pca_0", "mri_pca_1", "mri_missing"],
    "laboratory_results": ["cli_cspca", "cli_vol"],
    "previous_notes": ["cli_bx_positive", "cli_bx_missing", "cli_age"],
}


def occlusion_entropy_delta(model, X: np.ndarray, col_idx: list[int], fill_values: np.ndarray) -> np.ndarray:
    """How much decision-entropy increases when col_idx's columns are occluded (set to that
    fold's training median/mode). Mirrors kdm_backbone.occlusion_delta()'s shape, reading
    'entropy' out of compute_signals() instead of 'probs'.
    """
    original_entropy = compute_signals(model, X)["entropy"]
    X_occ = X.copy()
    X_occ[:, col_idx] = fill_values
    occluded_entropy = compute_signals(model, X_occ)["entropy"]
    return occluded_entropy - original_entropy


def main() -> None:
    with open(SEARCH_DIR / "winner.json") as f:
        winner = json.load(f)
    winning_config = {
        "n_epochs": winner["n_epochs"], "lr": winner["lr"], "sigma_mult": winner["sigma_mult"],
        "optimizer": winner["optimizer"], "weight_decay": winner["weight_decay"],
    }
    print(f"Using winning config: {winning_config}\n")

    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"].values == "yes").astype(int)

    seqs = parse_reveal_sequences(ann["target_reveal_sequence_json"])
    sections = [s for s in REVEAL_SECTIONS if any(s in seq for seq in seqs)]
    print(f"Modeled sections (>=1 positive example among 91 cases): {sections}")
    dropped = [s for s in REVEAL_SECTIONS if s not in sections]
    print(f"Dropped (0 positive examples): {dropped}\n")
    assert all(s in SECTION_FEATURE_GROUPS for s in sections), "SECTION_FEATURE_GROUPS must cover every modeled section"

    Y = np.array([[1 if s in seq else 0 for s in sections] for seq in seqs])

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp8_feature_frame(inp_ann, mri_pca)
    print(f"feature frame: {X_frame.shape}\n")

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    section_col_idx = {s: [X_frame.columns.get_loc(c) for c in SECTION_FEATURE_GROUPS[s]] for s in sections}

    n = len(y_decision)
    n_sections = len(sections)
    repeat_precisions = []
    per_section_stats = {s: {"tp": [], "fp": [], "fn": [], "tn": []} for s in sections}

    for repeat in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        oof_pred = np.zeros((n, n_sections), dtype=int)

        for train_idx, test_idx in kf.split(X_pre):
            scaler = StandardScaler().fit(X_pre[train_idx])
            X_train = scaler.transform(X_pre[train_idx])
            X_test = scaler.transform(X_pre[test_idx])

            model = fit_kdm_backbone(X_train, y_decision[train_idx], n_classes=2, **winning_config)

            for si, section in enumerate(sections):
                col_idx = section_col_idx[section]
                fill = np.median(X_train[:, col_idx], axis=0)
                R_train = occlusion_entropy_delta(model, X_train, col_idx, fill)
                R_test = occlusion_entropy_delta(model, X_test, col_idx, fill)

                y_sec_train = Y[train_idx, si]
                if len(np.unique(y_sec_train)) < 2:
                    # degenerate fold (all-one-class train split for this section) -- predict the
                    # constant, not an error; recorded, not silently forced past.
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

        for si, section in enumerate(sections):
            true_col, pred_col = Y[:, si], oof_pred[:, si]
            per_section_stats[section]["tp"].append(int(np.sum((true_col == 1) & (pred_col == 1))))
            per_section_stats[section]["fp"].append(int(np.sum((true_col == 0) & (pred_col == 1))))
            per_section_stats[section]["fn"].append(int(np.sum((true_col == 1) & (pred_col == 0))))
            per_section_stats[section]["tn"].append(int(np.sum((true_col == 0) & (pred_col == 0))))

        print(f"repeat {repeat} done")

    per_section_report = {}
    for section in sections:
        tp = np.mean(per_section_stats[section]["tp"])
        fp = np.mean(per_section_stats[section]["fp"])
        fn = np.mean(per_section_stats[section]["fn"])
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        per_section_report[section] = {
            "mean_tp": round(float(tp), 2), "mean_fp": round(float(fp), 2), "mean_fn": round(float(fn), 2),
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "positive_rate": round(float(Y[:, sections.index(section)].mean()), 3),
        }

    payload = {
        "condition": "reveal_kdm_occlusion",
        "target": "reveal_sequence",
        "features": "exp_8 23-column frame",
        "model": f"KDM shared backbone (config: {winning_config}) + per-section occlusion-entropy signal + univariate binary classifiers",
        "sections_modeled": sections,
        "sections_dropped_no_positive_examples": dropped,
        "set_precision_mean": round(float(np.mean(repeat_precisions)), 3),
        "set_precision_std": round(float(np.std(repeat_precisions)), 3),
        "n_repeats": N_REPEATS,
        "per_section": per_section_report,
        "note": "compare against exp_2's reveal_flags incumbent (0.853) and naive baseline (0.783)",
    }
    out_dir = RESULTS_DIR / "reveal_kdm_occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[reveal_kdm_occlusion] set_precision={payload['set_precision_mean']} +/- {payload['set_precision_std']}")
    print("(incumbent reveal_flags=0.853, naive baseline=0.783)")
    for section, stats in per_section_report.items():
        print(f"  {section:20s} precision={stats['precision']} recall={stats['recall']} (positive_rate={stats['positive_rate']})")


if __name__ == "__main__":
    main()
