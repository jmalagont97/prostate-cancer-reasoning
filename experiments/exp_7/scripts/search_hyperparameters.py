"""exp_7: 144-combination hyperparameter grid search for the KDM decision backbone.

Search values (see DESIGN.md Section 2):
  N_EPOCHS   in {150, 300, 600}
  lr         in {3e-3, 1e-2, 3e-2}
  sigma_mult in {0.5, 1.0, 1.5, 2.0}
  (optimizer, weight_decay) in {(adam, 0), (adamw, 0), (adamw, 1e-4), (adamw, 1e-3)}
= 3 x 3 x 4 x 4 = 144 combinations, each scored via 5-fold x 3-repeat CV (reduced from the
project's usual 10 repeats, purely for search-phase tractability -- the winning combination gets
re-evaluated at the full 10-repeat protocol downstream in run_signals_v2.py).

Applies log1p to cli_psa/cli_psad/cli_vol before the CV loop (see kdm_backbone_v2.LOG1P_COLUMNS)
-- every combination in this search is evaluated WITH the log1p transform, per DESIGN.md's
"both combined" framing; run_ablations.py isolates whether log1p or tuning is doing the work.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_7/scripts/search_hyperparameters.py
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from kdm_backbone_v2 import LOG1P_COLUMNS, apply_log1p_transform, compute_signals, fit_kdm_backbone  # noqa: E402

from chimera_task1.features import build_preprocessor, select_exp3_feature_frame
from chimera_task1.train_decision import mri_pca_features
from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
N_SPLITS = 5
N_REPEATS_SEARCH = 3
EXP6_DECISION_MACRO_F1 = 0.593
CLEAR_MARGIN = 0.02  # per DESIGN.md Section 7/9's "clear margin" discipline

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
    X_frame = select_exp3_feature_frame(inp_ann, mri_pca)

    preprocessor = build_preprocessor(X_frame)
    X_pre = preprocessor.fit_transform(X_frame)
    X_pre = X_pre.toarray() if hasattr(X_pre, "toarray") else X_pre
    log1p_idx = [X_frame.columns.get_loc(c) for c in LOG1P_COLUMNS]
    X_pre = apply_log1p_transform(X_pre, log1p_idx)

    print(f"n={len(y)}, positive rate={y.mean():.4f}, feature frame: {X_frame.shape}, "
          f"log1p applied to columns {LOG1P_COLUMNS}\n")

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
        print(f"  -> Margin clears the {CLEAR_MARGIN} threshold -- worth a full 10-repeat re-evaluation.")
    else:
        print(f"  -> Margin does NOT clear the {CLEAR_MARGIN} threshold -- plausibly CV noise from "
              f"searching 144 configurations. Proceeding to full evaluation regardless (per plan), "
              f"but the report must flag this explicitly rather than calling it a genuine win.")
    print("=" * 80)


if __name__ == "__main__":
    main()
