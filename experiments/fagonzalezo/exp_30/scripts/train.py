"""exp_30 -- Uncertainty prediction ability of BrentMemKDM. See ../DESIGN.md and ../IMPLEMENTATION.md.

Five consecutive KDM/MemKDM-lineage experiments (exp_23, exp_25-29) evaluated this model family only
on the binary biopsy-decision task and never beat exp_8's 0.7171 with significance. This experiment
turns to CHIMERA Task 1's OTHER objective -- the 3-class diagnostic confidence label -- which
BrentMemKDM has never been evaluated on. exp_25/26/27's confidence numbers (0.4547/0.5287/0.5630)
are ALL target_informed=True (their soft targets encode the confidence label via
CONFIDENCE_CERTAINTY_MAP); the best honest, non-target-informed result anywhere in this project is
still exp_24's 0.4368, short of exp_17's classical Composite Fuzzy ICI (0.4470). This experiment's
hard arm is structurally clean (build_targets(y_binary), sigma selected on binary macro-F1 only) --
the first clean shot at exp_17's number since exp_24.

New mechanism: `knn_k` (exp_29) makes the retrieved neighborhood literal, so a new "family C"
neighborhood signal set (src/methods/brent_mem_kdm.py's `_neighborhood_signals`) is definable for
the first time -- neighbor label disagreement (`nbr_label_entropy`, over y_binary, never
`confidence`) and k-th-neighbor distance (`nbr_kth_expo`). Pre-registered narrow grid: 3 modalities
(tab/mri/txt) + a family-D composite, `knn_k in {5, 20, None}`, both arms.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from kdm.init import _sigma_from_knn  # noqa: E402 -- for gates G3/G4, same import brent_mem_kdm.py itself uses

from src.evaluation.data import (  # noqa: E402
    CONFIDENCE_CERTAINTY_MAP, OLD_SCHEMA, build_mri_features, build_tabular_features,
    build_targets, build_text_features, clean_texts_spacy, load_cohort, resolve_data_dir,
)
from src.evaluation.metrics import confidence_metrics, mcnemar_exact  # noqa: E402
from src.evaluation.protocol import iter_mccv_splits  # noqa: E402
from src.evaluation.reporting import plot_confusion_matrix, plot_grid_search_curves, record_git_commit, write_json  # noqa: E402
from src.methods.base import Targets, apply_meta_thresholds, fit_meta_thresholds_safe, fit_predict_heldout_trees  # noqa: E402
from src.methods.brent_mem_kdm import BrentMemKDM, Fold, run_brent_search  # noqa: E402
from src.methods.mem_kdm import PARTICLE_SIGNAL_NAMES, composite_reliability_index, inter_modality_variance  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "reports" / "figures"

MRI_REPS = [None, 0.90]  # pca_variance: raw_l2, pca90_l2
TXT_REPS = [(500, 0.90), (2000, 0.90), (None, 0.90)]
KNN_K_GRID = [5, 20, None]  # narrow, pre-registered (DESIGN.md Sec 4) -- NOT exp_29's 7-point sweep


def safe_confidence_metrics(y_conf: np.ndarray, pred: np.ndarray) -> dict:
    """`confidence_metrics` wrapped with exp_25-27's NaN guard: a degenerate/constant prediction
    (all one class) makes `spearmanr` return NaN, which the strict JSON writer (`_check_finite`)
    rejects. Map non-finite rho/pvalue to `None` rather than crashing or silently allowing NaN."""
    m = confidence_metrics(y_conf, pred)
    if not np.isfinite(m["spearman_rho"]):
        m["spearman_rho"] = None
    if not np.isfinite(m["spearman_pvalue"]):
        m["spearman_pvalue"] = None
    return m


def run_loocv_signals(evaluate_fn, n: int, folds) -> dict:
    """LOOCV over `folds` (a subset of range(n) under --smoke), collecting a full dict of named
    scalar outputs per fold rather than protocol.run_loocv's single "pred" scalar -- exp_30 needs
    every uncertainty signal, not just the class-1 probability. Unevaluated folds (under --smoke)
    are left as NaN; scoring code always restricts to `folds` via an index array, matching exp_29's
    own smoke-mode pattern."""
    all_idx = np.arange(n)
    out = None
    for i in folds:
        train_idx, val_idx = all_idx[all_idx != i], np.array([i])
        result = evaluate_fn(train_idx, val_idx)
        if out is None:
            out = {k: np.full(n, np.nan) for k in result}
        for k, v in result.items():
            out[k][i] = v
    return out


def main(smoke: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- cohort, targets, splits
    data_dir = resolve_data_dir(PROJECT_ROOT)
    cohort = load_cohort(data_dir, PROJECT_ROOT, OLD_SCHEMA, load_mri=True, load_text=True)
    assert len(cohort.dre_categories) == 5
    n = len(cohort.y_binary)
    print(f"[exp_30] cohort N={n}, yes={int(cohort.y_binary.sum())}, no={int((1 - cohort.y_binary).sum())}")
    print(f"[exp_30] confidence classes: uncertain={int((cohort.y_conf == 0).sum())}, "
          f"borderline={int((cohort.y_conf == 1).sum())}, clear={int((cohort.y_conf == 2).sum())}")

    targets_hard = build_targets(cohort.y_binary)
    targets_soft = build_targets(cohort.y_binary, cohort.confidence, certainty_map=CONFIDENCE_CERTAINTY_MAP)
    TARGETS = {"hard": targets_hard, "soft": targets_soft}

    def make_targets(idx, arm):
        return Targets(y_binary=cohort.y_binary[idx], y_soft=TARGETS[arm].y_soft[idx])

    print("[exp_30] spaCy cleaning text corpus (once, cohort-level)...")
    cleaned_texts = clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)

    FULL_SPLITS = list(iter_mccv_splits(cohort.df_design, n_splits=100))
    SPLITS = FULL_SPLITS[:5] if smoke else FULL_SPLITS
    LOOCV_FOLDS = list(range(6)) if smoke else list(range(n))
    SPLITS_2T = [(tr, va) for _, tr, va in FULL_SPLITS]  # NEVER smoke-reduced (DESIGN.md Sec 9 / exp_27's invariant)
    print(f"[exp_30] {'SMOKE MODE: ' if smoke else ''}{len(SPLITS)} MCCV splits (Phase A), "
          f"{len(LOOCV_FOLDS)} LOOCV folds (Phase B), {len(SPLITS_2T)} MCCV splits (confidence heads)")

    def build_modality(name, rep, train_idx, val_idx):
        if name == "tab":
            return build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
        if name == "mri":
            return build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=rep)
        if name == "txt":
            max_features, pca_variance = rep
            return build_text_features(cleaned_texts, train_idx, val_idx, max_features=max_features, pca_variance=pca_variance)
        raise ValueError(name)

    # ---------------------------------------------------------------- Step 0: reproduction gates
    print("[exp_30] Step 0a: verify_brent_mem_kdm.py full checks (subprocess)...")
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "verify_brent_mem_kdm.py")],
                        capture_output=True, text=True)
    g0_pass = r.returncode == 0 and "checks passed" in r.stdout and "FAIL" not in r.stdout
    print(r.stdout[-1500:])
    if not g0_pass:
        raise RuntimeError(f"G0 (verify_brent_mem_kdm.py) FAILED:\n{r.stdout}\n{r.stderr}")

    print("[exp_30] Step 0b: G1 (exp_17 Composite Fuzzy ICI, target 0.4469706011059394)...")

    oof16 = pd.read_csv(PROJECT_ROOT / "experiments/fagonzalezo/exp_16/results/oof_predictions.csv")
    oof16L = oof16[oof16.patient_id.isin(cohort.pids)].sort_values("patient_id").reset_index(drop=True)
    assert (oof16L.patient_id.values == cohort.pids).all() and len(oof16L) == 88, \
        "G1: exp_16 OOF predictions don't align with this cohort's patient_id order"

    # Route A: score exp_17's own stored per-patient predictions directly.
    cm17 = {"uncertain": 0, "borderline": 1, "clear": 2}
    oof17 = pd.read_csv(PROJECT_ROOT / "experiments/fagonzalezo/exp_17/results/oof_confidence_predictions.csv")
    oof17L = oof17[oof17.patient_id.isin(cohort.pids)].sort_values("patient_id").reset_index(drop=True)
    assert (oof17L.patient_id.values == cohort.pids).all()
    g1_a_pred = oof17L.predicted_confidence.map(cm17).values
    g1_route_a = f1_score(oof17L.ground_truth_confidence.map(cm17).values, g1_a_pred, average="macro", zero_division=0)

    # Route B: full recompute from exp_16's OOF probs -- exercises the ICI formula and threshold loop.
    p1, p2, p3 = (oof16L[c].values for c in ["prob_tabular_fuzzy", "prob_mri_fuzzy", "prob_text_fuzzy"])
    pm = (p1 + p2 + p3) / 3.0
    std = np.sqrt(((p1 - pm) ** 2 + (p2 - pm) ** 2 + (p3 - pm) ** 2) / 3.0)
    ici = np.clip((2.0 * np.abs(pm - 0.50)) * (1.0 - 2.0 * std), 0.0, 1.0)
    t1s, t2s = [], []
    for i in range(100):
        tr_mask = cohort.df_design[f"split_{i}"].values == 0
        dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)
        dt.fit(ici[tr_mask].reshape(-1, 1), cohort.y_conf[tr_mask])
        th = np.sort(dt.tree_.threshold[dt.tree_.threshold != -2])
        if len(th) >= 2:
            a, b = th[0], th[1]
        elif len(th) == 1:
            a, b = th[0], th[0] + 0.1
        else:
            a, b = 0.10, 0.30
        t1s.append(a)
        t2s.append(b)
    t1, t2 = float(np.mean(t1s)), float(np.mean(t2s))
    g1_b_pred = np.where(ici < t1, 0, np.where(ici < t2, 1, 2))
    g1_route_b = f1_score(cohort.y_conf, g1_b_pred, average="macro", zero_division=0)

    g1_pass = (g1_route_a == g1_route_b == 0.4469706011059394)

    print("[exp_30] Step 0c: G2 (exp_24 a_hard__multivariate_7signal, target 0.4368347338935575)...")
    PARTICLE_S = list(PARTICLE_SIGNAL_NAMES)  # ["h_total","h_aleatoric","h_epistemic","h_weights","log_ess","w_max","log_marginal"]
    sig24 = pd.read_csv(PROJECT_ROOT / "experiments/exp_24/results/oof_particle_signals.csv")
    sig24L = sig24[sig24.patient_id.isin(cohort.pids)].sort_values("patient_id").reset_index(drop=True)
    assert (sig24L.patient_id.values == cohort.pids).all() and len(sig24L) == 88
    mat24 = np.stack([sig24L[f"a_hard_{s}"].values for s in PARTICLE_S], axis=1)
    y_conf_24 = sig24L.y_conf.values.astype(int)
    assert np.array_equal(y_conf_24, cohort.y_conf), "G2: exp_24's stored y_conf doesn't match this cohort's"
    splits_2t_full = [(np.where(cohort.df_design[f"split_{i}"].values == 0)[0],
                        np.where(cohort.df_design[f"split_{i}"].values == 1)[0]) for i in range(100)]
    g2_pred, g2_votes = fit_predict_heldout_trees(mat24, y_conf_24, splits_2t_full)
    g2_score = f1_score(y_conf_24, g2_pred, average="macro", zero_division=0)
    g2_pass = (g2_score == 0.4368347338935575) and (int(g2_votes.min()) == 11)

    print("[exp_30] Step 0d: G3 (knn_k=1 degeneracy self-check) / G4 (family-C absent at knn_k=None)...")

    def g3_evaluate_fn(train_idx, val_idx):
        X_tr, X_va = build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
        sigma = _sigma_from_knn(X_tr, 1.0)  # any positive sigma works: knn_k=1 is sigma-invariant
        m = BrentMemKDM(knn_k=1)
        m.sigmas_ = {"tab": sigma}
        m.modality_order = ["tab"]
        m.fit({"tab": X_tr}, Targets(y_binary=cohort.y_binary[train_idx], y_soft=cohort.y_binary[train_idx].astype(np.float32)))
        sig = m.uncertainty_signals({"tab": X_va})
        return {"h_weights": float(sig["h_weights"][0]), "log_ess": float(sig["log_ess"][0]),
                "w_max": float(sig["w_max"][0]), "h_total": float(sig["h_total"][0]),
                "nbr_label_entropy": float(sig["nbr_label_entropy"][0])}

    g3_sig = run_loocv_signals(g3_evaluate_fn, n, list(range(n)))
    g3_pass = bool(
        np.allclose(g3_sig["h_weights"], 0.0) and np.allclose(g3_sig["log_ess"], 0.0)
        and np.allclose(g3_sig["w_max"], 1.0) and np.allclose(g3_sig["h_total"], 0.0)
        and np.allclose(g3_sig["nbr_label_entropy"], 0.0)
    )

    X_tr0, X_va0 = build_tabular_features(cohort.df_tab, np.arange(1, n), np.array([0]), dre_categories=cohort.dre_categories)
    sigma0 = _sigma_from_knn(X_tr0, 1.0)
    m_none = BrentMemKDM(knn_k=None)
    m_none.sigmas_ = {"tab": sigma0}
    m_none.modality_order = ["tab"]
    m_none.fit({"tab": X_tr0}, Targets(y_binary=cohort.y_binary[1:], y_soft=cohort.y_binary[1:].astype(np.float32)))
    sig_none = m_none.uncertainty_signals({"tab": X_va0})
    g4_pass = "nbr_label_entropy" not in sig_none and "nbr_kth_expo" not in sig_none

    reproduction_gates = {
        "G0_verify_brent_mem_kdm": {"passed": bool(g0_pass)},
        "G1_exp17_composite_ici": {"passed": bool(g1_pass), "route_a": float(g1_route_a), "route_b": float(g1_route_b),
                                    "target": 0.4469706011059394},
        "G2_exp24_multivariate_7signal": {"passed": bool(g2_pass), "macro_f1": float(g2_score),
                                           "target": 0.4368347338935575, "min_votes": int(g2_votes.min())},
        "G3_knn_k1_degeneracy": {"passed": g3_pass},
        "G4_family_c_absent_at_knn_k_none": {"passed": bool(g4_pass)},
    }
    write_json(reproduction_gates, RESULTS_DIR / "reproduction_gates.json")
    assert g1_pass, f"G1 FAILED: route_a={g1_route_a!r}, route_b={g1_route_b!r}"
    assert g2_pass, f"G2 FAILED: got {g2_score!r}, min_votes={int(g2_votes.min())}"
    assert g3_pass, "G3 FAILED: knn_k=1 signals not degenerate as predicted"
    assert g4_pass, "G4 FAILED: family-C keys leaked into the whole-memory (knn_k=None) signal dict"
    print(f"[exp_30] Step 0 PASSED: G0={g0_pass}, G1={g1_route_a!r}, G2={g2_score!r}, G3={g3_pass}, G4={g4_pass}")

    # ---------------------------------------------------------------- Phase A1: rep search, knn_k as fixed grid
    print("[exp_30] Phase A1: representation search (binary objective) + knn_k grid sigma search...")

    def build_folds(name, rep, splits, arm):
        y_soft = TARGETS[arm].y_soft
        out = []
        for _, tr, va in splits:
            X_tr, X_va = build_modality(name, rep, tr, va)
            out.append(Fold(X_train={name: X_tr}, y_soft_train=y_soft[tr], X_val={name: X_va}, y_val=cohort.y_binary[va]))
        return out

    STAGE1_REP, STAGE1_BY_K, STAGE1_ALL = {}, {}, {}
    for arm in ["hard", "soft"]:
        STAGE1_REP[arm], STAGE1_BY_K[arm], STAGE1_ALL[arm] = {}, {}, {}
        for name, reps in [("tab", [None]), ("mri", MRI_REPS), ("txt", TXT_REPS)]:
            rows = []
            for rep in reps:
                folds = build_folds(name, rep, SPLITS, arm)
                for k in KNN_K_GRID:
                    result = run_brent_search(folds, [name], metric="macro_f1", strategy="nested",
                                               n_prescan=15, maxiter=20, knn_k=k)
                    rows.append({"rep": repr(rep), "knn_k": repr(k), "sigma": result.sigmas[name],
                                 "sigma_mult": result.sigma_mult[name],
                                 "mean_macro_f1": result.per_fold_scores["mean"],
                                 "std_macro_f1": result.per_fold_scores["std"]})
            df = pd.DataFrame(rows).sort_values(["mean_macro_f1", "std_macro_f1"], ascending=[False, True]).reset_index(drop=True)
            STAGE1_ALL[arm][name] = df
            winning_rep_repr = df.iloc[0]["rep"]
            STAGE1_REP[arm][name] = eval(winning_rep_repr) if name != "tab" else None  # noqa: S307 -- repr() of MRI_REPS/TXT_REPS entries only
            sub = df[df["rep"] == winning_rep_repr]
            STAGE1_BY_K[arm][name] = {}
            for _, row in sub.iterrows():
                STAGE1_BY_K[arm][name][row["knn_k"]] = {
                    "sigma": float(row["sigma"]), "sigma_mult": float(row["sigma_mult"]),
                    "mean_macro_f1": float(row["mean_macro_f1"]), "std_macro_f1": float(row["std_macro_f1"]),
                }
            by_k_str = {k: round(v["sigma"], 5) for k, v in STAGE1_BY_K[arm][name].items()}
            print(f"[exp_30] Phase A1 [{arm}] {name}: rep={STAGE1_REP[arm][name]}, sigma_by_knn_k={by_k_str}")

    write_json({"rep": STAGE1_REP, "by_knn_k": STAGE1_BY_K}, RESULTS_DIR / "stage1_best_hparams.json")
    for name in ["tab", "mri", "txt"]:
        pd.concat([STAGE1_ALL[arm][name].assign(arm=arm) for arm in ["hard", "soft"]], ignore_index=True) \
            .to_csv(RESULTS_DIR / f"phasea_sigma_grid_{name}.csv", index=False)

    # ---------------------------------------------------------------- Phase B: LOOCV signal extraction
    print("[exp_30] Phase B (LOOCV): per-modality signal extraction across the knn_k grid...")
    t_phase_b = time.time()

    def fit_predict_signals(name, knn_k, arm, train_idx, val_idx):
        rep = STAGE1_REP[arm][name]
        X_tr, X_va = build_modality(name, rep, train_idx, val_idx)
        cfg = STAGE1_BY_K[arm][name][repr(knn_k)]
        m = BrentMemKDM(knn_k=knn_k)
        m.sigmas_ = {name: cfg["sigma"]}
        m.modality_order = [name]
        m.fit({name: X_tr}, make_targets(train_idx, arm))
        sig = m.uncertainty_signals({name: X_va})
        out = {"pred": float(sig["probs"][0, 1])}
        for key in PARTICLE_S:
            out[key] = float(sig[key][0])
        if knn_k is not None:
            out["nbr_label_entropy"] = float(sig["nbr_label_entropy"][0])
            out["nbr_kth_expo"] = float(sig["nbr_kth_expo"][0])
        return out

    LOOCV_SIGNALS = {}
    for arm in ["hard", "soft"]:
        LOOCV_SIGNALS[arm] = {}
        for name in ["tab", "mri", "txt"]:
            LOOCV_SIGNALS[arm][name] = {}
            for k in KNN_K_GRID:
                def evaluate_fn(train_idx, val_idx, name=name, k=k, arm=arm):
                    return fit_predict_signals(name, k, arm, train_idx, val_idx)
                LOOCV_SIGNALS[arm][name][repr(k)] = run_loocv_signals(evaluate_fn, n, LOOCV_FOLDS)
            print(f"[exp_30] Phase B {name} [{arm}]: knn_k grid done ({list(LOOCV_SIGNALS[arm][name].keys())})")

    elapsed_phase_b = time.time() - t_phase_b
    if smoke:
        fold_ratio = n / len(LOOCV_FOLDS)
        print(f"[exp_30] SMOKE compute budget: Phase B took {elapsed_phase_b:.1f}s for "
              f"{len(LOOCV_FOLDS)} folds x 3 modalities x {len(KNN_K_GRID)} knn_k x 2 arms; "
              f"extrapolated full Phase B (~{n} folds): ~{elapsed_phase_b * fold_ratio / 60:.1f} min")

    # Family D -- composite ICI / inter-modality variance, derived from the three modalities' own
    # `pred` (class-1 probability) at the SAME (arm, knn_k), not a separately-fit trimodal model.
    LOOCV_COMPOSITE = {}
    for arm in ["hard", "soft"]:
        LOOCV_COMPOSITE[arm] = {}
        for k in KNN_K_GRID:
            P = np.stack([LOOCV_SIGNALS[arm][name][repr(k)]["pred"] for name in ["tab", "mri", "txt"]], axis=1)
            ici_c, _p_mean, _p_std, _margin = composite_reliability_index(P)
            var_c = inter_modality_variance(P)
            LOOCV_COMPOSITE[arm][repr(k)] = {"composite_ici": ici_c, "inter_modality_variance": var_c}

    # ---------------------------------------------------------------- loocv_signals.csv
    signal_cols = {"patient_id": cohort.pids, "y_conf": cohort.y_conf}
    for arm in ["hard", "soft"]:
        for name in ["tab", "mri", "txt"]:
            for k in KNN_K_GRID:
                for key, arr in LOOCV_SIGNALS[arm][name][repr(k)].items():
                    signal_cols[f"{name}__k{k}__{arm}__{key}"] = arr
        for k in KNN_K_GRID:
            for key, arr in LOOCV_COMPOSITE[arm][repr(k)].items():
                signal_cols[f"composite__k{k}__{arm}__{key}"] = arr
    df_signals = pd.DataFrame(signal_cols)
    df_signals.to_csv(RESULTS_DIR / "loocv_signals.csv", index=False)

    # ---------------------------------------------------------------- Phase A2: confidence heads (skipped under --smoke)
    CONFIDENCE_ROWS: list = []
    PRED_STORE: dict = {}

    if not smoke:
        print("[exp_30] Phase A2: confidence heads (1D meta-threshold + multivariate held-out trees)...")
        for arm in ["hard", "soft"]:
            for name in ["tab", "mri", "txt"]:
                for k in KNN_K_GRID:
                    sig = LOOCV_SIGNALS[arm][name][repr(k)]
                    margin = np.abs(sig["pred"] - 0.5)
                    sig_local = {"margin": margin, **{pk: sig[pk] for pk in PARTICLE_S}}
                    ab_keys = ["margin"] + PARTICLE_S
                    abc_keys = None
                    if k is not None:
                        sig_local["nbr_label_entropy"] = sig["nbr_label_entropy"]
                        sig_local["nbr_kth_expo"] = sig["nbr_kth_expo"]
                        abc_keys = ab_keys + ["nbr_label_entropy", "nbr_kth_expo"]

                    for sk in (abc_keys or ab_keys):
                        s = sig_local[sk]
                        thr = fit_meta_thresholds_safe(s, cohort.y_conf, SPLITS_2T)
                        pred = apply_meta_thresholds(s, thr)
                        row_key = f"{name}__k{k}__{arm}__1d__{sk}"
                        PRED_STORE[row_key] = pred
                        CONFIDENCE_ROWS.append({"_key": row_key, "condition": name, "knn_k": k, "arm": arm,
                                                 "head": "1d", "signal": sk, "target_informed": (arm == "soft"),
                                                 **safe_confidence_metrics(cohort.y_conf, pred)})

                    mat_ab = np.stack([sig_local[kk] for kk in ab_keys], axis=1)
                    pred_ab, votes_ab = fit_predict_heldout_trees(mat_ab, cohort.y_conf, SPLITS_2T)
                    row_key = f"{name}__k{k}__{arm}__multivariate_AB"
                    PRED_STORE[row_key] = pred_ab
                    CONFIDENCE_ROWS.append({"_key": row_key, "condition": name, "knn_k": k, "arm": arm,
                                             "head": "multivariate_AB", "signal": "+".join(ab_keys),
                                             "target_informed": (arm == "soft"), "min_votes": int(votes_ab.min()),
                                             **safe_confidence_metrics(cohort.y_conf, pred_ab)})

                    if abc_keys is not None:
                        mat_abc = np.stack([sig_local[kk] for kk in abc_keys], axis=1)
                        pred_abc, votes_abc = fit_predict_heldout_trees(mat_abc, cohort.y_conf, SPLITS_2T)
                        row_key = f"{name}__k{k}__{arm}__multivariate_ABC"
                        PRED_STORE[row_key] = pred_abc
                        CONFIDENCE_ROWS.append({"_key": row_key, "condition": name, "knn_k": k, "arm": arm,
                                                 "head": "multivariate_ABC", "signal": "+".join(abc_keys),
                                                 "target_informed": (arm == "soft"), "min_votes": int(votes_abc.min()),
                                                 **safe_confidence_metrics(cohort.y_conf, pred_abc)})

            for k in KNN_K_GRID:
                comp = LOOCV_COMPOSITE[arm][repr(k)]
                for sk in ["composite_ici", "inter_modality_variance"]:
                    s = comp[sk]
                    thr = fit_meta_thresholds_safe(s, cohort.y_conf, SPLITS_2T)
                    pred = apply_meta_thresholds(s, thr)
                    row_key = f"composite__k{k}__{arm}__1d__{sk}"
                    PRED_STORE[row_key] = pred
                    CONFIDENCE_ROWS.append({"_key": row_key, "condition": "composite", "knn_k": k, "arm": arm,
                                             "head": "1d", "signal": sk, "target_informed": (arm == "soft"),
                                             **safe_confidence_metrics(cohort.y_conf, pred)})
                mat_d = np.stack([comp["composite_ici"], comp["inter_modality_variance"]], axis=1)
                pred_d, votes_d = fit_predict_heldout_trees(mat_d, cohort.y_conf, SPLITS_2T)
                row_key = f"composite__k{k}__{arm}__multivariate_D"
                PRED_STORE[row_key] = pred_d
                CONFIDENCE_ROWS.append({"_key": row_key, "condition": "composite", "knn_k": k, "arm": arm,
                                         "head": "multivariate_D", "signal": "composite_ici+inter_modality_variance",
                                         "target_informed": (arm == "soft"), "min_votes": int(votes_d.min()),
                                         **safe_confidence_metrics(cohort.y_conf, pred_d)})

        best_hard = max((r for r in CONFIDENCE_ROWS if r["arm"] == "hard"), key=lambda r: r["macro_f1"])
        print(f"[exp_30] Best non-target-informed row: {best_hard['condition']}/k={best_hard['knn_k']}/"
              f"{best_hard['head']}/{best_hard['signal']} macro_f1={best_hard['macro_f1']:.4f} "
              f"(G1=0.4470, G2=0.4368)")
    else:
        print("[exp_30] Phase A2 (confidence heads) skipped (--smoke)")
        best_hard = None

    write_json([{k: v for k, v in row.items()} for row in CONFIDENCE_ROWS], RESULTS_DIR / "confidence_metrics.json")

    # ---------------------------------------------------------------- H3: significance vs. G1/G2, H2: family-C contribution
    significance: dict = {}
    if not smoke:
        print("[exp_30] H3: significance vs. exp_17 (G1) / exp_24 (G2)...")
        best_pred = PRED_STORE[best_hard["_key"]]

        def permutation_test_macro_f1(y_true, pred_a, pred_b, n_perm=1000, seed=0):
            rng = np.random.default_rng(seed)
            f1_a = f1_score(y_true, pred_a, average="macro", zero_division=0)
            f1_b = f1_score(y_true, pred_b, average="macro", zero_division=0)
            obs_delta = f1_a - f1_b
            n_ = len(y_true)
            count = 0
            for _ in range(n_perm):
                mask = rng.integers(0, 2, size=n_).astype(bool)
                perm_a = np.where(mask, pred_a, pred_b)
                perm_b = np.where(mask, pred_b, pred_a)
                d = (f1_score(y_true, perm_a, average="macro", zero_division=0)
                     - f1_score(y_true, perm_b, average="macro", zero_division=0))
                if abs(d) >= abs(obs_delta):
                    count += 1
            return {"f1_a": float(f1_a), "f1_b": float(f1_b), "delta": float(obs_delta),
                    "pvalue": (count + 1) / (n_perm + 1), "n_perm": n_perm}

        significance["best_hard_row"] = {k: v for k, v in best_hard.items() if k != "_key"}
        significance["vs_exp17_G1"] = {
            "mcnemar": mcnemar_exact(cohort.y_conf, best_pred, g1_b_pred),
            "permutation": permutation_test_macro_f1(cohort.y_conf, best_pred, g1_b_pred),
        }
        significance["vs_exp24_G2"] = {
            "mcnemar": mcnemar_exact(cohort.y_conf, best_pred, g2_pred),
            "permutation": permutation_test_macro_f1(cohort.y_conf, best_pred, g2_pred),
        }

        print("[exp_30] H2: family-C contribution per modality (best finite-k ABC vs knn_k=None AB)...")
        hard_rows = [r for r in CONFIDENCE_ROWS if r["arm"] == "hard"]
        h2_results = {}
        for name in ["tab", "mri", "txt"]:
            abc_rows = [r for r in hard_rows if r["condition"] == name and r["head"] == "multivariate_ABC"]
            ab_none_rows = [r for r in hard_rows if r["condition"] == name and r["head"] == "multivariate_AB" and r["knn_k"] is None]
            if abc_rows and ab_none_rows:
                best_abc = max(abc_rows, key=lambda r: r["macro_f1"])
                ab_none = ab_none_rows[0]
                h2_results[name] = {
                    "best_abc_knn_k": best_abc["knn_k"], "abc_macro_f1": best_abc["macro_f1"],
                    "ab_none_macro_f1": ab_none["macro_f1"],
                    "mcnemar": mcnemar_exact(cohort.y_conf, PRED_STORE[best_abc["_key"]], PRED_STORE[ab_none["_key"]]),
                }
        significance["H2_family_c_contribution"] = h2_results
        write_json(significance, RESULTS_DIR / "significance.json")
    else:
        print("[exp_30] H3/H2 significance skipped (--smoke)")

    # ---------------------------------------------------------------- confidence_predictions.csv
    df_conf_pred = pd.DataFrame({"patient_id": cohort.pids, "y_conf": cohort.y_conf,
                                  "g1_pred": g1_b_pred, "g2_pred": g2_pred})
    if not smoke:
        df_conf_pred["best_hard_pred"] = PRED_STORE[best_hard["_key"]]
    df_conf_pred.to_csv(RESULTS_DIR / "confidence_predictions.csv", index=False)

    # ---------------------------------------------------------------- figures (best-effort)
    print("[exp_30] Figures...")
    try:
        for name in ["tab", "mri", "txt"]:
            df = pd.concat([STAGE1_ALL[arm][name].assign(arm=arm) for arm in ["hard", "soft"]], ignore_index=True)
            plot_grid_search_curves(df, x_col="sigma_mult", y_col="mean_macro_f1", group_cols=["arm", "knn_k"],
                                     title=f"exp_30 Phase A1 {name} sigma search (rep x knn_k grid)",
                                     out_path=FIG_DIR / f"phasea_{name}.png")
        if not smoke and best_hard is not None:
            plot_confusion_matrix(cohort.y_conf, PRED_STORE[best_hard["_key"]],
                                   labels=["uncertain", "borderline", "clear"],
                                   title=f"exp_30 best hard row: {best_hard['condition']}/{best_hard['head']}",
                                   out_path=FIG_DIR / "confusion_matrix.png")
    except Exception as e:  # pragma: no cover -- figures are best-effort, never block numeric results
        print(f"[exp_30] WARNING: figure generation failed: {e}")

    record_git_commit(RESULTS_DIR)
    elapsed = time.time() - t_start
    print(f"[exp_30] DONE in {elapsed / 60:.1f} min. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
