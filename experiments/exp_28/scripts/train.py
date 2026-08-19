"""exp_28 — BrentMemKDM re-evaluation of the exp_5-exp_8 hard-KNN generation. See ../DESIGN.md and
../IMPLEMENTATION.md.

exp_5-exp_8 are the original hard-KNN generation of the Task-1 biopsy-decision track and still hold the
best published LOOCV Macro-F1 in the repo (exp_8's equal-weight trimodal fusion, 0.7171). This experiment
replaces each modality's KNN classifier with BrentMemKDM (src/methods/brent_mem_kdm.py) -- a MemKDM whose
only fitted quantity is the RBF bandwidth sigma per modality, chosen by a global Brent search of the mean
MCCV Macro-F1, fit once and frozen (never per fold) -- under the same two-phase MCCV/LOOCV protocol and
(where the checkout allows) the same base representations.

utils/embedding-kit/ is empty in this checkout, so exp_6's winning MRI representation (embedkit_sup) and
exp_8's fusion built on it are not reproducible targets here (DESIGN.md Sec 2.1). This script recomputes
honest KNN reference arms (knn_mri_pca, knn_fusion_equal) from the best available non-embedkit MRI
representation for those two comparisons, and reports the published-but-unreproducible numbers separately
in reports/summary.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data import (  # noqa: E402
    CONFIDENCE_CERTAINTY_MAP, OLD_SCHEMA, build_mri_features, build_tabular_features,
    build_targets, build_text_features, clean_texts_spacy, load_cohort, resolve_data_dir,
)
from src.evaluation.metrics import binary_metrics, mcnemar_exact  # noqa: E402
from src.evaluation.protocol import iter_mccv_splits, run_loocv  # noqa: E402
from src.evaluation.reporting import (  # noqa: E402
    plot_confusion_matrix, plot_grid_search_curves, plot_roc_curves, record_git_commit, write_json,
)
from src.methods.base import Targets  # noqa: E402
from src.methods.brent_mem_kdm import BrentMemKDM, Fold, run_brent_search  # noqa: E402
from src.methods.mem_kdm import simplex_grid, soft_vote  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "reports" / "figures"

MRI_REPS = [None, 0.90]  # pca_variance: raw_l2, pca90_l2
TXT_REPS = [(500, 0.90), (2000, 0.90), (None, 0.90)]


def to_2col(p1: np.ndarray) -> np.ndarray:
    return np.stack([1 - p1, p1], axis=1)


def run_loocv_folds(evaluate_fn, n: int, folds):
    """Local variant of `protocol.run_loocv` restricted to a subset of folds (for --smoke). Identical to
    `protocol.run_loocv` when `folds == range(n)`. `protocol.py` is not modified."""
    oof_pred = np.zeros(n)
    all_idx = np.arange(n)
    for i in folds:
        train_idx, val_idx = all_idx[all_idx != i], np.array([i])
        result = evaluate_fn(train_idx, val_idx)
        oof_pred[i] = result["pred"]
    return oof_pred, {}


def classifier_pred(knn: KNeighborsClassifier, X_va) -> float:
    p = knn.predict_proba(X_va)
    classes = list(knn.classes_)
    return float(p[0, classes.index(1)]) if 1 in classes else 0.0


def main(smoke: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- cohort, targets, splits
    data_dir = resolve_data_dir(PROJECT_ROOT)
    cohort = load_cohort(data_dir, PROJECT_ROOT, OLD_SCHEMA, load_mri=True, load_text=True)
    assert len(cohort.dre_categories) == 5
    n = len(cohort.y_binary)
    print(f"[exp_28] cohort N={n}, yes={int(cohort.y_binary.sum())}, no={int((1 - cohort.y_binary).sum())}")

    targets_hard = build_targets(cohort.y_binary)
    targets_soft = build_targets(cohort.y_binary, cohort.confidence, certainty_map=CONFIDENCE_CERTAINTY_MAP)
    TARGETS = {"hard": targets_hard, "soft": targets_soft}

    print("[exp_28] spaCy cleaning text corpus (once, cohort-level)...")
    cleaned_texts = clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)

    FULL_SPLITS = list(iter_mccv_splits(cohort.df_design, n_splits=100))
    SPLITS = FULL_SPLITS[:5] if smoke else FULL_SPLITS
    LOOCV_FOLDS = list(range(6)) if smoke else list(range(n))
    print(f"[exp_28] {'SMOKE MODE: ' if smoke else ''}{len(SPLITS)} MCCV splits, {len(LOOCV_FOLDS)} LOOCV folds")

    def build_modality(name, rep, train_idx, val_idx):
        if name == "tab":
            return build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
        if name == "mri":
            return build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=rep)
        if name == "txt":
            max_features, pca_variance = rep
            return build_text_features(cleaned_texts, train_idx, val_idx, max_features=max_features, pca_variance=pca_variance)
        raise ValueError(name)

    def g2_text_pipeline(train_idx, val_idx):
        """exp_7's ORIGINAL text pipeline: TF-IDF(max_features=500) -> PCA@0.90. No MinMax, no trailing
        L2 Normalizer -- deliberately NOT build_text_features (which adds both, DESIGN.md Sec 4)."""
        vec = TfidfVectorizer(max_features=500, norm="l2")
        X_tr = vec.fit_transform(cleaned_texts[train_idx]).toarray().astype(np.float32)
        X_va = vec.transform(cleaned_texts[val_idx]).toarray().astype(np.float32)
        n_comp = min(X_tr.shape[0], X_tr.shape[1])
        if n_comp > 1:
            pca = PCA(n_components=0.90, random_state=42)
            X_tr, X_va = pca.fit_transform(X_tr), pca.transform(X_va)
        return X_tr, X_va

    # ---------------------------------------------------------------- Step 0: reproduction gates
    print("[exp_28] Step 0a: verify_brent_mem_kdm.py full checks (subprocess)...")
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "verify_brent_mem_kdm.py")],
                        capture_output=True, text=True)
    g0_pass = r.returncode == 0 and "checks passed" in r.stdout and "FAIL" not in r.stdout
    print(r.stdout[-2000:])
    if not g0_pass:
        raise RuntimeError(f"G0 (verify_brent_mem_kdm.py) FAILED:\n{r.stdout}\n{r.stderr}")

    print("[exp_28] Step 0b/0c: G1 (exp_5 tabular KNN) / G2 (exp_7 text KNN) reproduction...")

    def g1_evaluate_fn(train_idx, val_idx):
        X_tr, X_va = build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=None)
        knn = KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="euclidean")
        knn.fit(X_tr, cohort.y_binary[train_idx])
        return {"pred": classifier_pred(knn, X_va)}

    g1_pred, _ = run_loocv(g1_evaluate_fn, n)
    g1_metrics = binary_metrics(cohort.y_binary, g1_pred)
    g1_pass = abs(g1_metrics["macro_f1"] - 0.6333333333333333) < 1e-12

    def g2_evaluate_fn(train_idx, val_idx):
        X_tr, X_va = g2_text_pipeline(train_idx, val_idx)
        knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="cosine")
        knn.fit(X_tr, cohort.y_binary[train_idx])
        return {"pred": classifier_pred(knn, X_va)}

    g2_pred, _ = run_loocv(g2_evaluate_fn, n)
    g2_metrics = binary_metrics(cohort.y_binary, g2_pred)
    g2_pass = abs(g2_metrics["macro_f1"] - 0.6987539367383268) < 1e-12

    reproduction_gates = {
        "G0_verify_brent_mem_kdm": {"passed": bool(g0_pass)},
        "G1_exp5_tabular_knn": {"passed": bool(g1_pass), "macro_f1": g1_metrics["macro_f1"],
                                 "target": 0.6333333333333333,
                                 "confusion": {k: g1_metrics[k] for k in ("tp", "tn", "fp", "fn")}},
        "G2_exp7_text_knn": {"passed": bool(g2_pass), "macro_f1": g2_metrics["macro_f1"],
                              "target": 0.6987539367383268,
                              "confusion": {k: g2_metrics[k] for k in ("tp", "tn", "fp", "fn")}},
    }
    write_json(reproduction_gates, RESULTS_DIR / "reproduction_gates.json")
    assert g1_pass, f"G1 FAILED: got {g1_metrics['macro_f1']!r}"
    assert g2_pass, f"G2 FAILED: got {g2_metrics['macro_f1']!r}"
    print(f"[exp_28] Step 0 PASSED: G0={g0_pass}, G1={g1_metrics['macro_f1']!r}, G2={g2_metrics['macro_f1']!r}")

    # ---------------------------------------------------------------- Phase A: per-modality Brent search
    print("[exp_28] Phase A: representation + sigma search...")

    def make_targets(idx, arm):
        return Targets(y_binary=cohort.y_binary[idx], y_soft=TARGETS[arm].y_soft[idx])

    def build_folds(name, rep, splits, arm):
        y_soft = TARGETS[arm].y_soft
        out = []
        for _, tr, va in splits:
            X_tr, X_va = build_modality(name, rep, tr, va)
            out.append(Fold(X_train={name: X_tr}, y_soft_train=y_soft[tr], X_val={name: X_va}, y_val=cohort.y_binary[va]))
        return out

    STAGE1_BEST, STAGE1_ALL = {}, {}
    for arm in ["hard", "soft"]:
        STAGE1_BEST[arm], STAGE1_ALL[arm] = {}, {}
        for name, reps in [("tab", [None]), ("mri", MRI_REPS), ("txt", TXT_REPS)]:
            rows = []
            for rep in reps:
                folds = build_folds(name, rep, SPLITS, arm)
                result = run_brent_search(folds, [name], metric="macro_f1", strategy="nested", n_prescan=15, maxiter=20)
                rows.append({"rep": repr(rep), "sigma": result.sigmas[name], "sigma_mult": result.sigma_mult[name],
                             "mean_macro_f1": result.per_fold_scores["mean"], "std_macro_f1": result.per_fold_scores["std"]})
            df = pd.DataFrame(rows).sort_values(["mean_macro_f1", "std_macro_f1"], ascending=[False, True]).reset_index(drop=True)
            STAGE1_ALL[arm][name] = df
            best = df.iloc[0]
            STAGE1_BEST[arm][name] = {
                "rep": eval(best["rep"]) if name != "tab" else None,  # noqa: S307 -- repr() of MRI_REPS/TXT_REPS entries only
                "sigma": float(best["sigma"]), "sigma_mult": float(best["sigma_mult"]),
                "mean_macro_f1": float(best["mean_macro_f1"]), "std_macro_f1": float(best["std_macro_f1"]),
            }
            print(f"[exp_28] Phase A [{arm}] {name}: rep={STAGE1_BEST[arm][name]['rep']}, "
                  f"sigma={STAGE1_BEST[arm][name]['sigma']:.5f}, mean_macro_f1={STAGE1_BEST[arm][name]['mean_macro_f1']:.4f}")

    write_json(STAGE1_BEST, RESULTS_DIR / "stage1_best_hparams.json")
    for name in ["tab", "mri", "txt"]:
        pd.concat([STAGE1_ALL[arm][name].assign(arm=arm) for arm in ["hard", "soft"]], ignore_index=True) \
            .to_csv(RESULTS_DIR / f"phasea_grid_{name}.csv", index=False)

    # ---------------------------------------------------------------- Phase B: LOOCV, unimodal + fusion
    print("[exp_28] Phase B (LOOCV): unimodal conditions...")

    def fit_predict_unimodal(name, arm, train_idx, val_idx):
        rep = STAGE1_BEST[arm][name]["rep"]
        X_tr, X_va = build_modality(name, rep, train_idx, val_idx)
        m = BrentMemKDM()
        m.sigmas_ = {name: STAGE1_BEST[arm][name]["sigma"]}
        m.modality_order = [name]
        m.fit({name: X_tr}, make_targets(train_idx, arm))
        return m.predict_proba({name: X_va})[0, 1]

    def phase_b_unimodal(name, arm):
        def evaluate_fn(train_idx, val_idx):
            return {"pred": float(fit_predict_unimodal(name, arm, train_idx, val_idx))}
        if smoke:
            oof_pred, _ = run_loocv_folds(evaluate_fn, n, LOOCV_FOLDS)
        else:
            oof_pred, _ = run_loocv(evaluate_fn, n)
        return oof_pred

    UNIMODAL_OOF = {}
    for arm in ["hard", "soft"]:
        UNIMODAL_OOF[arm] = {}
        for name in ["tab", "mri", "txt"]:
            UNIMODAL_OOF[arm][name] = phase_b_unimodal(name, arm)
            idx0 = np.array(LOOCV_FOLDS)
            f1 = f1_score(cohort.y_binary[idx0], (UNIMODAL_OOF[arm][name][idx0] >= 0.5).astype(int), average="macro", zero_division=0)
            print(f"[exp_28] Phase B unimodal_{name} [{arm}]: macro_f1={f1:.4f}")

    FUSION_EQUAL_OOF = {
        arm: soft_vote({name: to_2col(UNIMODAL_OOF[arm][name]) for name in ["tab", "mri", "txt"]},
                        {"tab": 1 / 3, "mri": 1 / 3, "txt": 1 / 3})[:, 1]
        for arm in ["hard", "soft"]
    }

    # ---------------------------------------------------------------- Reference arms (recomputed exp_6/exp_8 counterparts)
    print("[exp_28] Reference arms: knn_mri_pca, knn_fusion_equal...")

    def knn_mri_pca_evaluate_fn(train_idx, val_idx):
        X_tr, X_va = build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=0.90)
        knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean")
        knn.fit(X_tr, cohort.y_binary[train_idx])
        return {"pred": classifier_pred(knn, X_va)}

    knn_mri_pca_pred, _ = run_loocv(knn_mri_pca_evaluate_fn, n)
    knn_tab_pred, _ = run_loocv(g1_evaluate_fn, n)  # exp_5's exact winner, reused for the honest fusion reference
    knn_txt_pred, _ = run_loocv(g2_evaluate_fn, n)  # exp_7's exact winner

    knn_fusion_equal_pred = soft_vote(
        {"tab": to_2col(knn_tab_pred), "mri": to_2col(knn_mri_pca_pred), "txt": to_2col(knn_txt_pred)},
        {"tab": 1 / 3, "mri": 1 / 3, "txt": 1 / 3})[:, 1]

    # ---------------------------------------------------------------- Secondary conditions (S1/S2, hard arm only)
    print("[exp_28] S1: fusion_optimal_leakfree (Stage-1 MCCV validation probs only)...")

    def winner_val_probs(name):
        best = STAGE1_BEST["hard"][name]
        out = {}
        for split_idx, train_idx, val_idx in SPLITS:
            X_tr, X_va = build_modality(name, best["rep"], train_idx, val_idx)
            m = BrentMemKDM()
            m.sigmas_ = {name: best["sigma"]}
            m.modality_order = [name]
            m.fit({name: X_tr}, make_targets(train_idx, "hard"))
            out[split_idx] = (val_idx, m.predict_proba({name: X_va}))
        return out

    WINNER_VAL_PROBS = {name: winner_val_probs(name) for name in ["tab", "mri", "txt"]}
    grid = simplex_grid(3, step=0.05)
    scores = np.zeros(len(grid))
    for split_idx, _tr, _va in SPLITS:
        probs = {name: WINNER_VAL_PROBS[name][split_idx][1] for name in ["tab", "mri", "txt"]}
        v_idx = WINNER_VAL_PROBS["tab"][split_idx][0]
        for i, (wt, wm, wx) in enumerate(grid):
            p_fused = soft_vote(probs, dict(tab=wt, mri=wm, txt=wx))
            y_pred = (p_fused[:, 1] >= 0.50).astype(int)
            scores[i] += f1_score(cohort.y_binary[v_idx], y_pred, average="macro", zero_division=0)
    scores /= len(SPLITS)
    best_i = int(np.argmax(scores))
    fusion_weights_leakfree = {"weights": {k: float(v) for k, v in zip(["tab", "mri", "txt"], grid[best_i])},
                                "mean_macro_f1": float(scores[best_i])}
    write_json(fusion_weights_leakfree, RESULTS_DIR / "fusion_weights_leakfree.json")
    print(f"[exp_28] S1 fusion weights: {fusion_weights_leakfree}")

    FUSION_OPTIMAL_OOF = soft_vote({name: to_2col(UNIMODAL_OOF["hard"][name]) for name in ["tab", "mri", "txt"]},
                                    fusion_weights_leakfree["weights"])[:, 1]

    print("[exp_28] S2: joint_trimodal (product-kernel, coordinate search, hard arm only)...")

    def build_joint_folds(splits, arm):
        y_soft = TARGETS[arm].y_soft
        out = []
        for _, tr, va in splits:
            X_tr, X_va = {}, {}
            for name in ["tab", "mri", "txt"]:
                rep = STAGE1_BEST[arm][name]["rep"]
                X_tr[name], X_va[name] = build_modality(name, rep, tr, va)
            out.append(Fold(X_train=X_tr, y_soft_train=y_soft[tr], X_val=X_va, y_val=cohort.y_binary[va]))
        return out

    # Fixed small budget regardless of --smoke (matches verify_brent_mem_kdm.py's own trimodal smoke
    # budget, DESIGN.md Sec 4) -- not a smoke-mode reduction, a deliberately cheap secondary-condition budget.
    joint_search_splits = SPLITS[:5]
    joint_folds = build_joint_folds(joint_search_splits, "hard")
    joint_result = run_brent_search(joint_folds, ["tab", "mri", "txt"], metric="macro_f1",
                                     strategy="coordinate", n_prescan=7, maxiter=10, max_rounds=5)
    write_json({"sigmas": joint_result.sigmas, "sigma_mult": joint_result.sigma_mult,
                "score": joint_result.score, "n_evals": joint_result.n_evals}, RESULTS_DIR / "joint_trimodal_search.json")

    def joint_evaluate_fn(train_idx, val_idx):
        X_tr, X_va = {}, {}
        for name in ["tab", "mri", "txt"]:
            rep = STAGE1_BEST["hard"][name]["rep"]
            X_tr[name], X_va[name] = build_modality(name, rep, train_idx, val_idx)
        m = BrentMemKDM()
        m.sigmas_ = dict(joint_result.sigmas)
        m.modality_order = ["tab", "mri", "txt"]
        m.fit(X_tr, make_targets(train_idx, "hard"))
        return {"pred": float(m.predict_proba(X_va)[0, 1])}

    if smoke:
        joint_trimodal_pred, _ = run_loocv_folds(joint_evaluate_fn, n, LOOCV_FOLDS)
    else:
        joint_trimodal_pred, _ = run_loocv(joint_evaluate_fn, n)

    # ---------------------------------------------------------------- Metrics, McNemar, output
    print("[exp_28] Metrics + McNemar...")

    CONDITIONS_HARD = {
        "tab": UNIMODAL_OOF["hard"]["tab"], "mri": UNIMODAL_OOF["hard"]["mri"], "txt": UNIMODAL_OOF["hard"]["txt"],
        "fusion_equal": FUSION_EQUAL_OOF["hard"],
        "fusion_optimal_leakfree": FUSION_OPTIMAL_OOF, "joint_trimodal": joint_trimodal_pred,
    }
    CONDITIONS_SOFT = {
        "tab": UNIMODAL_OOF["soft"]["tab"], "mri": UNIMODAL_OOF["soft"]["mri"], "txt": UNIMODAL_OOF["soft"]["txt"],
        "fusion_equal": FUSION_EQUAL_OOF["soft"],
    }
    REFERENCE = {"knn_tab": knn_tab_pred, "knn_mri_pca": knn_mri_pca_pred, "knn_txt": knn_txt_pred,
                 "knn_fusion_equal": knn_fusion_equal_pred}

    idx = np.array(LOOCV_FOLDS)
    y_true_eval = cohort.y_binary[idx]
    loocv_metrics = {}
    for arm, conditions in [("hard", CONDITIONS_HARD), ("soft", CONDITIONS_SOFT)]:
        for cond_name, pred in conditions.items():
            key = f"{cond_name}__{arm}"
            loocv_metrics[key] = binary_metrics(y_true_eval, pred[idx])
            loocv_metrics[key]["target_informed"] = (arm == "soft")
    for ref_name, pred in REFERENCE.items():
        loocv_metrics[ref_name] = binary_metrics(y_true_eval, pred[idx])
    write_json(loocv_metrics, RESULTS_DIR / "loocv_metrics.json")

    REF_MAP = {"tab": "knn_tab", "mri": "knn_mri_pca", "txt": "knn_txt", "fusion_equal": "knn_fusion_equal",
               "fusion_optimal_leakfree": "knn_fusion_equal"}
    mcnemar_results = {}
    for arm, conditions in [("hard", CONDITIONS_HARD), ("soft", CONDITIONS_SOFT)]:
        for cond_name, pred in conditions.items():
            if cond_name not in REF_MAP:
                continue
            ref_pred = REFERENCE[REF_MAP[cond_name]][idx]
            y_pred = (pred[idx] >= 0.50).astype(int)
            y_ref = (ref_pred >= 0.50).astype(int)
            mcnemar_results[f"{cond_name}__{arm}_vs_{REF_MAP[cond_name]}"] = mcnemar_exact(y_true_eval, y_pred, y_ref)
    write_json(mcnemar_results, RESULTS_DIR / "mcnemar.json")

    df_oof = pd.DataFrame({"patient_id": cohort.pids, "ground_truth_biopsy": cohort.y_binary})
    for arm, conditions in [("hard", CONDITIONS_HARD), ("soft", CONDITIONS_SOFT)]:
        for cond_name, pred in conditions.items():
            df_oof[f"prob_{cond_name}__{arm}"] = pred
    for ref_name, pred in REFERENCE.items():
        df_oof[f"prob_{ref_name}"] = pred
    df_oof.to_csv(RESULTS_DIR / "loocv_predictions.csv", index=False)

    # ---------------------------------------------------------------- figures
    print("[exp_28] Figures...")
    try:
        for name in ["tab", "mri", "txt"]:
            df = pd.concat([STAGE1_ALL[arm][name].assign(arm=arm) for arm in ["hard", "soft"]], ignore_index=True)
            plot_grid_search_curves(df, x_col="sigma_mult", y_col="mean_macro_f1", group_cols=["arm", "rep"],
                                     title=f"exp_28 Phase A {name} sigma search", out_path=FIG_DIR / f"phasea_{name}.png")
        y_pred_fusion_hard = (CONDITIONS_HARD["fusion_equal"][idx] >= 0.50).astype(int)
        plot_confusion_matrix(y_true_eval, y_pred_fusion_hard, labels=["No", "Yes"],
                               title="exp_28 fusion_equal (hard) LOOCV", out_path=FIG_DIR / "confusion_matrix.png")
        plot_roc_curves({"fusion_equal_hard": CONDITIONS_HARD["fusion_equal"][idx],
                          "knn_fusion_equal": REFERENCE["knn_fusion_equal"][idx],
                          "tab_hard": CONDITIONS_HARD["tab"][idx]},
                         y_true_eval, title="exp_28 ROC overlay", out_path=FIG_DIR / "roc_curve.png")
    except Exception as e:  # pragma: no cover -- figures are best-effort, never block numeric results
        print(f"[exp_28] WARNING: figure generation failed: {e}")

    record_git_commit(RESULTS_DIR)
    elapsed = time.time() - t_start
    print(f"[exp_28] DONE in {elapsed / 60:.1f} min. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
