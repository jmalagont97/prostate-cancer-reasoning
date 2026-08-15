"""exp_10: mandatory held-out check (per exp_7/exp_8/exp_9's established discipline) -- fits the
full-schema ARD model on the same fixed 19-case held-out split exp_3/exp_7/exp_8/exp_9 have all
used, and reports its macro-F1 alongside already-established reference numbers (not recomputed --
cited exactly as exp_9's own report cited exp_8's numbers).

More critical here than ever: this is the widest frame this project has tried (48 columns vs.
exp_9's 23), and exp_9 already showed even ARD's 19-column CV gains can evaporate on held-out.

Run with the project's own .venv (from the project root):
    .venv/Scripts/python.exe experiments/exp_10/scripts/holdout_eval_fullschema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_3" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "exp_9" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from ard_kernel import compute_signals_ard, fit_kdm_backbone_ard  # noqa: E402
from features_fullschema import fit_transform_fullschema, select_exp10_feature_frame  # noqa: E402
from holdout_eval import mri_pca_train_only  # noqa: E402

from chimera_task1.train_reasoning import load_annotated

RANDOM_STATE = 0
TEST_SIZE = 0.2
RESULTS_DIR = Path(__file__).parent.parent / "results"
ARD_CONFIG = {"n_epochs": 300, "lr": 1e-2, "sigma_mult": 1.0}

# Already-established reference numbers, NOT recomputed here -- same discipline as exp_9's report
# citing exp_8's numbers directly. See experiments/exp_9/reports/summary.md Section 3.
REFERENCE = {
    "exp9_ard_23col_held_out_macro_f1": 0.680,
    "exp9_ard_23col_cv_macro_f1": 0.608,
    "exp6_scalar_19col_held_out_macro_f1": 0.593,
}


def score(y_true, probs, label):
    preds = probs.argmax(axis=1)
    f1 = f1_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro")
    print(f"\n--- {label} ---  F1={f1:.3f}  macro-F1={macro_f1:.3f}")
    print(classification_report(y_true, preds, target_names=["no", "yes"], digits=3, zero_division=0))
    return {"f1": round(float(f1), 3), "macro_f1": round(float(macro_f1), 3)}


def main() -> None:
    ann, inp_ann = load_annotated()
    y_decision = (ann["target_biopsy_decision"] == "yes").astype(int).values

    idx = np.arange(len(ann))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_decision, random_state=RANDOM_STATE)
    print(f"n_train={len(train_idx)}, n_test={len(test_idx)} (held out, never used for fitting)\n")

    inp_train = inp_ann.iloc[train_idx].reset_index(drop=True)
    y_dec_train, y_dec_test = y_decision[train_idx], y_decision[test_idx]

    mri_pca_aligned = mri_pca_train_only(inp_train, inp_ann)
    X_all = select_exp10_feature_frame(inp_ann, mri_pca_aligned)
    print(f"feature frame: {X_all.shape} (fullschema)\n")

    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)
    X_train, X_test = fit_transform_fullschema(X_train_raw, X_test_raw)

    model = fit_kdm_backbone_ard(X_train, y_dec_train, n_classes=2, **ARD_CONFIG)
    sig = compute_signals_ard(model, X_test)
    print(f"probs_check_ok: {sig['probs_check_ok']}")

    print("=" * 70)
    print(f"HELD-OUT CHECK (n={len(test_idx)}, never used for any model selection)")
    print("=" * 70)
    result = score(y_dec_test, sig["probs"], "ARD, full-schema (48-col)")

    print("=" * 70)
    print("Reference (already established, not recomputed):")
    for k, v in REFERENCE.items():
        print(f"  {k} = {v}")
    print(f"delta vs. exp_9 ARD 23-col held-out: {result['macro_f1'] - REFERENCE['exp9_ard_23col_held_out_macro_f1']:+.3f}")
    print(f"delta vs. exp_6 scalar 19-col held-out: {result['macro_f1'] - REFERENCE['exp6_scalar_19col_held_out_macro_f1']:+.3f}")
    print("=" * 70)

    out = {
        "n_test": len(test_idx),
        "ard_fullschema": result,
        "reference": REFERENCE,
        "delta_vs_exp9_ard_23col_held_out": round(result["macro_f1"] - REFERENCE["exp9_ard_23col_held_out_macro_f1"], 3),
        "delta_vs_exp6_scalar_19col_held_out": round(result["macro_f1"] - REFERENCE["exp6_scalar_19col_held_out_macro_f1"], 3),
        "config": ARD_CONFIG,
        "probs_check_ok": sig["probs_check_ok"],
    }
    out_dir = RESULTS_DIR / "holdout_eval_fullschema"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
