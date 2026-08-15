"""exp_8: 144-combination hyperparameter grid search for the KDM decision backbone, on the
23-column expanded frame (exp_7's identical grid/protocol, applied to a new feature set -- see
DESIGN.md Section 1b for why this is a genuine re-test, not a blind retry).

No log1p step here (exp_7-specific, not part of this experiment's hypothesis).

CLEAR_MARGIN corrected to 0.045 (exp_6's measured 10-repeat macro-F1 std) instead of exp_7's
fixed 0.02 -- exp_7's own report flagged its threshold as too small relative to the noise floor
it was being compared against.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_8/scripts/search_hyperparameters_v3.py
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
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
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS_SEARCH = 3
EXP6_DECISION_MACRO_F1 = 0.593
CLEAR_MARGIN = 0.045  # exp_6's measured 10-repeat std, per DESIGN.md's corrected methodology

EPOCHS_GRID = [150, 300, 600]
LR_GRID = [3e-3, 1e-2, 3e-2]
SIGMA_MULT_GRID = [0.5, 1.0, 1.5, 2.0]
OPT_WD_GRID = [("adam", 0.0), ("adamw", 0.0), ("adamw", 1e-4), ("adamw", 1e-3)]

RESULTS_DIR = Path(__file__).parent.parent / "results" / "hyperparameter_search"


def main() -> None:
    ann, inp_ann = load_annotated()
    y = (ann["target_biopsy_decision"].values == "yes").astype(int)

    full_inp = pd.read_csv("data/inputs.csv")
    mri_pca_full = mri_pca_features(full_inp, n_components=2)
    mri_pca_full["case_id"] = full_inp["case_id"].values
    mri_pca = mri_pca_full.set_index("case_id").loc[inp_ann["case_id"]].reset_index(drop=True)
    X_frame = select_exp8_feature_frame(inp_ann, mri_pca)

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre

    print(f"n={len(y)}, positive rate={y.mean():.4f}, feature frame: {X_frame.shape}\n")

    combos = list(itertools.product(EPOCHS_GRID, LR_GRID, SIGMA_MULT_GRID, OPT_WD_GRID))
    assert len(combos) == 144, f"expected 144 combinations, got {len(combos)}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()

    for i, (n_epochs, lr, sigma_mult, (optimizer, weight_decay)) in enumerate(combos):
        f1_scores = []
        for repeat in range(N_REPEATS_SEARCH):
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
            preds = np.empty(len(y), dtype=int)
            for train_idx, test_idx in kf.split(X_pre):
                scaler = StandardScaler().fit(X_pre[train_idx])
                X_train = scaler.transform(X_pre[train_idx])
                X_test = scaler.transform(X_pre[test_idx])
                model = fit_kdm_backbone(
                    X_train, y[train_idx], n_classes=2,
                    n_epochs=n_epochs, lr=lr, sigma_mult=sigma_mult,
                    optimizer=optimizer, weight_decay=weight_decay,
                )
                sig = compute_signals(model, X_test)
                preds[test_idx] = sig["probs"].argmax(axis=1)
            f1_scores.append(f1_score(y, preds, average="macro"))

        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        rows.append({
            "n_epochs": n_epochs, "lr": lr, "sigma_mult": sigma_mult,
            "optimizer": optimizer, "weight_decay": weight_decay,
            "macro_f1_mean": round(mean_f1, 4), "macro_f1_std": round(std_f1, 4),
        })
        if (i + 1) % 12 == 0:
            elapsed = time.time() - t_start
            print(f"  {i + 1}/144 combinations done ({elapsed:.0f}s elapsed)")

    rows.sort(key=lambda r: r["macro_f1_mean"], reverse=True)

    with open(RESULTS_DIR / "grid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    winner = rows[0]
    margin = winner["macro_f1_mean"] - EXP6_DECISION_MACRO_F1
    winner_config = {
        "n_epochs": winner["n_epochs"], "lr": winner["lr"], "sigma_mult": winner["sigma_mult"],
        "optimizer": winner["optimizer"], "weight_decay": winner["weight_decay"],
        "search_macro_f1_mean": winner["macro_f1_mean"], "search_macro_f1_std": winner["macro_f1_std"],
        "search_n_repeats": N_REPEATS_SEARCH,
        "margin_over_exp6": round(margin, 4),
        "clear_margin_threshold": CLEAR_MARGIN,
        "margin_is_clear": margin >= CLEAR_MARGIN,
    }
    with open(RESULTS_DIR / "winner.json", "w") as f:
        json.dump(winner_config, f, indent=2)

    print(f"\nTotal search time: {time.time() - t_start:.0f}s\n")
    print("=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    for r in rows[:10]:
        print(f"  epochs={r['n_epochs']:4d} lr={r['lr']:.0e} sigma_mult={r['sigma_mult']:.1f} "
              f"opt={r['optimizer']:6s} wd={r['weight_decay']:.0e}  "
              f"macro_f1={r['macro_f1_mean']:.3f} +/- {r['macro_f1_std']:.3f}")

    print("\n" + "=" * 80)
    print(f"WINNER: {winner_config}")
    print(f"Margin over exp_6's decision_kdm_backbone (0.593, 10-repeat): {margin:+.3f}")
    if margin >= CLEAR_MARGIN:
        print(f"  -> Margin clears the corrected {CLEAR_MARGIN} threshold (exp_6's own measured std) "
              f"-- worth a full 10-repeat re-evaluation and the mandatory held-out check.")
    else:
        print(f"  -> Margin does NOT clear the {CLEAR_MARGIN} threshold -- plausibly CV noise from "
              f"searching 144 configurations, or genuinely no improvement. Proceeding to full "
              f"evaluation regardless (per plan), but the report must flag this explicitly.")
    print("=" * 80)


if __name__ == "__main__":
    main()
