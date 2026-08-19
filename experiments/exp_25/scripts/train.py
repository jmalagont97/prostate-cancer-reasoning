"""exp_25 — Multimodal Memory-Based KDM (MemKDM). See ../DESIGN.md and ../IMPLEMENTATION.md.

Two implementation simplifications relative to IMPLEMENTATION.md, both purely internal (no change to
DESIGN.md's protocol/hyperparameters/leak-free guarantees), noted here rather than re-litigated inline:

1. Feature rebuilding is NOT cached per (split, modality, representation) as IMPLEMENTATION.md §1.2
   sketched. exp_13-24 all rebuild scalers/encoders inside every split x config loop without caching
   (e.g. exp_13's MinMaxScaler/OneHotEncoder are refit per split per config) — that is the established
   repo convention, and §0's measured runtime already prices in a full per-config rebuild (the pilot
   measured single fits, not amortized-cache fits), so dropping the cache costs nothing against the
   budget and removes a class of staleness bugs.

2. Stage 1's fusion-weight stash (IMPLEMENTATION.md §1.4/§1.5 STAGE1_STASH, keyed by a serialized cfg
   tuple) is replaced by a small dedicated post-hoc pass: after each modality's Stage-1 winner is
   selected, that ONE winning config is re-fit across the 100 MCCV splits (a further ~300 fits, ~1-3
   min) to produce exactly the per-split validation probabilities `search_fusion_weights_local` needs.
   This avoids reconstructing a stash key from a `select_best` row's coerced dtypes (fragile) and avoids
   stashing 76x100 configs' predictions when only 3 are ever used. Still strictly leak-free — only
   Stage-1 validation folds are touched, never Phase B.

Both were flagged as build-time refinements during implementation, not new design decisions.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))  # src/evaluation/data.py does `from src.methods.base import Targets`

from src.evaluation.data import (  # noqa: E402
    CONFIDENCE_CERTAINTY_MAP, OLD_SCHEMA, build_mri_features, build_tabular_features,
    build_targets, build_text_features, clean_texts_spacy, load_cohort, resolve_data_dir,
)
from src.evaluation.metrics import binary_metrics, confidence_metrics, mcnemar_exact  # noqa: E402
from src.evaluation.protocol import iter_mccv_splits, run_mccv_grid, select_best  # noqa: E402
from src.evaluation.reporting import (  # noqa: E402
    plot_confusion_matrix, plot_grid_search_curves, plot_roc_curves, plot_signal_scatter,
    record_git_commit, write_json,
)
from src.methods.base import (  # noqa: E402
    Targets, apply_meta_thresholds, fit_meta_thresholds_safe, fit_predict_heldout_trees,
)
from src.methods.mem_kdm import (  # noqa: E402
    PARTICLE_SIGNAL_NAMES, EncoderSpec, KernelSpec, MemKDM, composite_reliability_index,
    inter_modality_variance, simplex_grid, soft_vote,
)

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "reports" / "figures"

TAB_GRID = [
    {"sigma_mult": sm, "encoder": enc, "y_train": yt}
    for sm in [0.25, 0.5, 1.0, 2.0] for enc in ["identity", "linear"] for yt in [False, True]
]
MRI_GRID = [
    {"rep": rep, "sigma_mult": sm, "encoder": enc, "y_train": yt}
    for rep in ["raw_l2", "pca90_l2"] for sm in [0.5, 1.0, 2.0]
    for enc in ["identity", "linear"] for yt in [False, True]
]
TXT_GRID = [
    {"rep": (mf, 0.90), "sigma_mult": sm, "encoder": enc, "y_train": yt}
    for mf in [500, 2000, None] for sm in [0.5, 1.0, 2.0]
    for enc in ["identity", "linear"] for yt in [False, True]
]
assert len(TAB_GRID) == 16 and len(MRI_GRID) == 24 and len(TXT_GRID) == 36

STAGE2_GRID = [
    {"sigma_scale": ss, "x_train": xt, "y_train": yt, "kernel_trainable": kt}
    for ss in [0.5, 1.0, 2.0] for xt in [False, True] for yt in [False, True] for kt in [False, True]
]
assert len(STAGE2_GRID) == 24

CONDITIONS = [["tab", "mri"], ["tab", "txt"], ["mri", "txt"], ["tab", "mri", "txt"]]
JOINT_KEYS = {"tab_mri": ["tab", "mri"], "tab_txt": ["tab", "txt"], "mri_txt": ["mri", "txt"],
              "tab_mri_txt": ["tab", "mri", "txt"]}
LATE_FUSION_NAMES = {"late_fusion_equal", "late_fusion_optimal"}


def get_out_dim(name: str, rep) -> int:
    """Representation-specific linear-encoder out_dim (IMPLEMENTATION.md §1.2 "discovered during
    implementation review" — avoids upsampling MRI's ~11-D PCA representation to 32)."""
    if name == "tab":
        return 8
    if name == "mri":
        return 32 if rep == "raw_l2" else 8
    if name == "txt":
        return 8
    raise ValueError(name)


def to_2col(p1: np.ndarray) -> np.ndarray:
    return np.stack([1 - p1, p1], axis=1)


def build_modality(cohort, cleaned_texts, name, rep, train_idx, val_idx):
    if name == "tab":
        return build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
    if name == "mri":
        pca_variance = None if rep == "raw_l2" else 0.90
        return build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=pca_variance)
    if name == "txt":
        max_features, pca_variance = rep
        return build_text_features(cleaned_texts, train_idx, val_idx, max_features=max_features, pca_variance=pca_variance)
    raise ValueError(name)


def run_loocv_folds(evaluate_fn, n: int, folds):
    """Local variant of `protocol.run_loocv` that only runs a subset of folds (for --smoke). Behaves
    identically to `protocol.run_loocv` when `folds == range(n)`. `protocol.py` is not modified."""
    oof_pred = np.zeros(n)
    oof_signals: dict = {}
    all_idx = np.arange(n)
    for i in folds:
        train_idx = all_idx[all_idx != i]
        val_idx = np.array([i])
        result = evaluate_fn(train_idx, val_idx)
        oof_pred[i] = result["pred"]
        for key, val in result.get("signals", {}).items():
            oof_signals.setdefault(key, np.zeros(n))
            oof_signals[key][i] = val
    return oof_pred, oof_signals


def main(smoke: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---------------------------------------------------------------- 1.1/1.2 setup + cohort
    data_dir = resolve_data_dir(PROJECT_ROOT)
    cohort = load_cohort(data_dir, PROJECT_ROOT, OLD_SCHEMA, load_mri=True, load_text=True)
    assert len(cohort.dre_categories) == 5
    targets = build_targets(cohort.y_binary, cohort.confidence, certainty_map=CONFIDENCE_CERTAINTY_MAP)
    n = len(cohort.y_binary)
    print(f"[exp_25] cohort N={n}, yes={int(cohort.y_binary.sum())}, no={int((1 - cohort.y_binary).sum())}")

    print("[exp_25] spaCy cleaning text corpus (once, cohort-level)...")
    cleaned_texts = clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)

    FULL_SPLITS_3T = list(iter_mccv_splits(cohort.df_design, n_splits=100))
    SPLITS_2T_CONF = [(tr, va) for _, tr, va in FULL_SPLITS_3T]  # confidence heads: always full 100 splits, even under
    # --smoke — a reduced split count breaks fit_predict_heldout_trees' "every patient gets >=1 held-out vote"
    # invariant on an 88-patient cohort (discovered running --smoke; exp_24 hardcoded n_conf_splits=100 for the
    # identical reason: "few-split runs degenerate the 1D head from small-N instability alone").
    SPLITS_3T = FULL_SPLITS_3T[:5] if smoke else FULL_SPLITS_3T  # Stage 1/2 model search + fusion-weight search
    LOOCV_FOLDS = list(range(6)) if smoke else list(range(n))
    print(f"[exp_25] {'SMOKE MODE: ' if smoke else ''}{len(SPLITS_3T)} MCCV splits, {len(LOOCV_FOLDS)} LOOCV folds")

    def make_targets(idx):
        return Targets(y_binary=cohort.y_binary[idx], y_soft=targets.y_soft[idx])

    def bm(name, rep, tr, va):
        return build_modality(cohort, cleaned_texts, name, rep, tr, va)

    # ---------------------------------------------------------------- 1.3 Step 0: reproduction gate + degeneracy probe
    print("[exp_25] Step 0: reproduction gate...")

    def repro_evaluate_fn(train_idx, val_idx):
        X_tr, X_va = bm("tab", None, train_idx, val_idx)
        m = MemKDM(kernels={"tab": KernelSpec(sigma_mult=2.0)}, encoders={"tab": EncoderSpec("identity")},
                   x_train=False, y_train=False, epochs=300, lr=1e-3, seed=0)
        m.fit({"tab": X_tr}, make_targets(train_idx))
        p = m.predict_proba({"tab": X_va})
        sig = m.uncertainty_signals({"tab": X_va})
        return {"pred": float(p[0, 1]), "signals": {k: float(v[0]) for k, v in sig.items() if k != "probs"}}

    repro_pred, repro_signals = run_loocv_folds(repro_evaluate_fn, n, list(range(n)))  # always full 88, even under --smoke
    repro_metrics = binary_metrics(cohort.y_binary, repro_pred)
    repro_pass = abs(repro_metrics["macro_f1"] - 0.6694214876033058) < 1e-6
    h_al = repro_signals["h_aleatoric"]
    reproduction_check = {
        "reproduction_macro_f1": repro_metrics["macro_f1"],
        "reproduction_target": 0.6694214876033058,
        "reproduction_passed": bool(repro_pass),
        "frac_nonzero_h_aleatoric": float((h_al > 1e-6).mean()),
        "median_h_aleatoric": float(np.median(h_al)),
    }
    write_json(reproduction_check, RESULTS_DIR / "reproduction_check.json")
    assert repro_pass, f"exp_23 Arm B reproduction FAILED: got {repro_metrics['macro_f1']}, expected 0.6694214876033058"
    print(f"[exp_25] Step 0 PASSED: macro_f1={repro_metrics['macro_f1']!r}, "
          f"frac_nonzero_h_aleatoric={reproduction_check['frac_nonzero_h_aleatoric']:.3f}")

    for name, rep in [("tab", None), ("mri", "pca90_l2"), ("txt", (2000, 0.90))]:
        idx = np.arange(n)
        X_full, _ = bm(name, rep, idx, idx[:1])
        enc = EncoderSpec("identity")
        m = MemKDM(kernels={name: KernelSpec(sigma_mult=1.0)}, encoders={name: enc},
                   x_train=False, y_train=False, epochs=0, lr=1e-3, seed=0, check_roundtrip=True)
        m.fit({name: X_full}, make_targets(idx))  # raises AssertionError internally if the roundtrip fails
    print("[exp_25] Step 0: check_roundtrip passed for tab / mri(pca90_l2) / txt(2000,0.9)")

    # ---------------------------------------------------------------- 1.4 Stage 1 Phase A
    print("[exp_25] Stage 1 Phase A...")

    def stage1_evaluate_factory(name):
        def evaluate_fn(cfg, train_idx, val_idx):
            rep = cfg.get("rep")
            X_tr, X_va = bm(name, rep, train_idx, val_idx)
            enc_spec = EncoderSpec("identity") if cfg["encoder"] == "identity" else EncoderSpec("linear", out_dim=get_out_dim(name, rep))
            m = MemKDM(kernels={name: KernelSpec(sigma_mult=cfg["sigma_mult"])}, encoders={name: enc_spec},
                       x_train=False, y_train=cfg["y_train"], epochs=300, lr=1e-3, seed=0)
            m.fit({name: X_tr}, make_targets(train_idx))
            y_pred = (m.predict_proba({name: X_va})[:, 1] >= 0.50).astype(int)
            return {"macro_f1": f1_score(cohort.y_binary[val_idx], y_pred, average="macro", zero_division=0)}
        return evaluate_fn

    def extract_stage1_best(name, best_row):
        d = {
            "sigma_mult": float(best_row["sigma_mult"]),
            "encoder": str(best_row["encoder"]),
            "y_train": bool(best_row["y_train"]),
            "mean_macro_f1": float(best_row["mean_macro_f1"]),
            "std_macro_f1": float(best_row["std_macro_f1"]),
            "cfg_id": int(best_row["cfg_id"]),
        }
        rep = best_row.get("rep") if name != "tab" else None
        if isinstance(rep, (list, tuple)):
            rep = tuple(rep)
        d["rep"] = rep
        return d

    STAGE1_BEST = {}
    STAGE1_GRIDS = []
    for name, grid in [("tab", TAB_GRID), ("mri", MRI_GRID), ("txt", TXT_GRID)]:
        df_grid = run_mccv_grid(grid, stage1_evaluate_factory(name), SPLITS_3T)
        best_row = select_best(df_grid, primary_metric="macro_f1")
        STAGE1_BEST[name] = extract_stage1_best(name, best_row)
        STAGE1_GRIDS.append(df_grid.assign(modality=name))
        print(f"[exp_25] Stage 1 {name}: best mean_macro_f1={STAGE1_BEST[name]['mean_macro_f1']:.4f}, cfg={STAGE1_BEST[name]}")

    write_json(STAGE1_BEST, RESULTS_DIR / "stage1_best_hparams.json")
    pd.concat(STAGE1_GRIDS, ignore_index=True).to_csv(RESULTS_DIR / "stage1_grid_search.csv", index=False)

    # ---------------------------------------------------------------- fusion-weight search (leak-free, Stage-1 val folds only)
    print("[exp_25] Fusion-weight search (post-hoc re-fit of the 3 Stage-1 winners across 100 splits)...")

    def winner_val_probs(name):
        best = STAGE1_BEST[name]
        rep = best["rep"]
        enc_spec = EncoderSpec("identity") if best["encoder"] == "identity" else EncoderSpec("linear", out_dim=get_out_dim(name, rep))
        out = {}
        for split_idx, train_idx, val_idx in SPLITS_3T:
            X_tr, X_va = bm(name, rep, train_idx, val_idx)
            m = MemKDM(kernels={name: KernelSpec(sigma_mult=best["sigma_mult"])}, encoders={name: enc_spec},
                       x_train=False, y_train=best["y_train"], epochs=300, lr=1e-3, seed=0)
            m.fit({name: X_tr}, make_targets(train_idx))
            out[split_idx] = (val_idx, m.predict_proba({name: X_va}))
        return out

    WINNER_VAL_PROBS = {name: winner_val_probs(name) for name in ["tab", "mri", "txt"]}

    def search_fusion_weights_local():
        grid = simplex_grid(3, step=0.05)
        scores = np.zeros(len(grid))
        for split_idx, _train_idx, _val_idx in SPLITS_3T:
            probs = {name: WINNER_VAL_PROBS[name][split_idx][1] for name in ["tab", "mri", "txt"]}
            v_idx = WINNER_VAL_PROBS["tab"][split_idx][0]
            for i, (wt, wm, wx) in enumerate(grid):
                p_fused = soft_vote(probs, dict(tab=wt, mri=wm, txt=wx))
                y_pred = (p_fused[:, 1] >= 0.50).astype(int)
                scores[i] += f1_score(cohort.y_binary[v_idx], y_pred, average="macro", zero_division=0)
        scores /= len(SPLITS_3T)
        best_i = int(np.argmax(scores))
        return {"weights": {k: float(v) for k, v in zip(["tab", "mri", "txt"], grid[best_i])},
                "mean_macro_f1": float(scores[best_i])}

    fusion_weights = search_fusion_weights_local()
    write_json(fusion_weights, RESULTS_DIR / "fusion_weights.json")
    print(f"[exp_25] fusion weights: {fusion_weights}")

    # ---------------------------------------------------------------- 1.5 Stage 2 Phase A
    print("[exp_25] Stage 2 Phase A...")

    def build_joint_kernels_encoders(subset, cfg):
        kernels, encoders = {}, {}
        for mod in subset:
            base_mult = STAGE1_BEST[mod]["sigma_mult"]
            scale = 1.0 if mod == "tab" else cfg["sigma_scale"]
            kernels[mod] = KernelSpec(sigma_mult=base_mult * scale, trainable=cfg["kernel_trainable"], sigma=None)
            rep = STAGE1_BEST[mod]["rep"]
            enc = STAGE1_BEST[mod]["encoder"]
            encoders[mod] = EncoderSpec("identity") if enc == "identity" else EncoderSpec("linear", out_dim=get_out_dim(mod, rep))
        return kernels, encoders

    def stage2_evaluate_factory(subset, label_smoothing=0.0):
        def evaluate_fn(cfg, train_idx, val_idx):
            X_tr, X_va = {}, {}
            for mod in subset:
                rep = STAGE1_BEST[mod]["rep"]
                X_tr[mod], X_va[mod] = bm(mod, rep, train_idx, val_idx)
            kernels, encoders = build_joint_kernels_encoders(subset, cfg)
            m = MemKDM(kernels=kernels, encoders=encoders, x_train=cfg["x_train"], y_train=cfg["y_train"],
                       label_smoothing=label_smoothing, epochs=300, lr=1e-3, seed=0)
            m.fit(X_tr, make_targets(train_idx))
            y_pred = (m.predict_proba(X_va)[:, 1] >= 0.50).astype(int)
            return {"macro_f1": f1_score(cohort.y_binary[val_idx], y_pred, average="macro", zero_division=0)}
        return evaluate_fn

    def extract_stage2_best(subset, best_row):
        return {
            "sigma_scale": float(best_row["sigma_scale"]),
            "x_train": bool(best_row["x_train"]),
            "y_train": bool(best_row["y_train"]),
            "kernel_trainable": bool(best_row["kernel_trainable"]),
            "mean_macro_f1": float(best_row["mean_macro_f1"]),
            "std_macro_f1": float(best_row["std_macro_f1"]),
            "cfg_id": int(best_row["cfg_id"]),
            "modalities": list(subset),
        }

    STAGE2_BEST = {}
    STAGE2_GRIDS = []
    for subset in CONDITIONS:
        key = "_".join(subset)
        df_grid = run_mccv_grid(STAGE2_GRID, stage2_evaluate_factory(subset), SPLITS_3T)
        STAGE2_BEST[key] = extract_stage2_best(subset, select_best(df_grid, "macro_f1"))
        STAGE2_GRIDS.append(df_grid.assign(condition=key))
        print(f"[exp_25] Stage 2 {key}: best mean_macro_f1={STAGE2_BEST[key]['mean_macro_f1']:.4f}")

    df_grid_c = run_mccv_grid(STAGE2_GRID, stage2_evaluate_factory(["tab", "mri", "txt"], label_smoothing=0.10), SPLITS_3T)
    STAGE2_BEST["confidence_arm"] = extract_stage2_best(["tab", "mri", "txt"], select_best(df_grid_c, "macro_f1"))
    STAGE2_GRIDS.append(df_grid_c.assign(condition="confidence_arm"))
    print(f"[exp_25] Stage 2 confidence_arm: best mean_macro_f1={STAGE2_BEST['confidence_arm']['mean_macro_f1']:.4f}")

    write_json({"stage1": STAGE1_BEST, "stage2": STAGE2_BEST}, RESULTS_DIR / "best_hparams.json")
    pd.concat(STAGE2_GRIDS, ignore_index=True).to_csv(RESULTS_DIR / "stage2_grid_search.csv", index=False)

    # ---------------------------------------------------------------- 1.6 Phase B — LOOCV, all conditions
    print("[exp_25] Phase B (LOOCV)...")

    def phase_b_condition(subset, cfg, n_seeds, label_smoothing=0.0):
        def make_evaluate_fn(seed):
            def evaluate_fn(train_idx, val_idx):
                X_tr, X_va = {}, {}
                for mod in subset:
                    rep = STAGE1_BEST[mod]["rep"]
                    X_tr[mod], X_va[mod] = bm(mod, rep, train_idx, val_idx)
                kernels, encoders = build_joint_kernels_encoders(subset, cfg)
                m = MemKDM(kernels=kernels, encoders=encoders, x_train=cfg["x_train"], y_train=cfg["y_train"],
                           label_smoothing=label_smoothing, epochs=300, lr=1e-3, seed=seed)
                m.fit(X_tr, make_targets(train_idx))
                p = m.predict_proba(X_va)[0]
                sig = m.uncertainty_signals(X_va)
                return {"pred": float(p[1]), "signals": {k: float(v[0]) for k, v in sig.items() if k != "probs"}}
            return evaluate_fn

        per_seed_pred, per_seed_signals = [], []
        for seed in range(n_seeds):
            oof_pred, oof_signals = run_loocv_folds(make_evaluate_fn(seed), n, LOOCV_FOLDS)
            per_seed_pred.append(oof_pred)
            per_seed_signals.append(oof_signals)

        p_mean = np.mean(per_seed_pred, axis=0)
        sig_mean = {k: np.mean([s[k] for s in per_seed_signals], axis=0) for k in per_seed_signals[0]}
        idx = np.array(LOOCV_FOLDS)
        per_seed_macro_f1 = [
            float(f1_score(cohort.y_binary[idx], (p[idx] >= 0.5).astype(int), average="macro", zero_division=0))
            for p in per_seed_pred
        ]
        if n_seeds > 1:
            y_pred_mean = (p_mean[idx] >= 0.5).astype(int)
            votes = np.stack([(p[idx] >= 0.5).astype(int) for p in per_seed_pred], axis=0)
            mode_vote = (votes.mean(axis=0) >= 0.5).astype(int)
            mode_vote_agreement = float(np.mean(y_pred_mean == mode_vote))
            macro_f1_std = float(np.std(per_seed_macro_f1))
        else:
            mode_vote_agreement, macro_f1_std = 1.0, 0.0

        return {
            "oof_pred": p_mean, "oof_signals": sig_mean, "n_seeds": n_seeds, "deterministic": n_seeds == 1,
            "per_seed_macro_f1": per_seed_macro_f1, "macro_f1_std_across_seeds": macro_f1_std,
            "mode_vote_agreement": mode_vote_agreement,
        }

    unimodal_oof = {}
    for mod in ["tab", "mri", "txt"]:
        cfg = {"x_train": False, "y_train": STAGE1_BEST[mod]["y_train"], "sigma_scale": 1.0, "kernel_trainable": True}
        n_seeds = 10 if STAGE1_BEST[mod]["encoder"] == "linear" else 1
        if smoke:
            n_seeds = min(n_seeds, 3)
        unimodal_oof[mod] = phase_b_condition([mod], cfg, n_seeds)
        print(f"[exp_25] Phase B unimodal_{mod}: n_seeds={n_seeds}, "
              f"macro_f1={np.mean(unimodal_oof[mod]['per_seed_macro_f1']):.4f}")

    joint_oof = {}
    for key, subset in JOINT_KEYS.items():
        cfg = STAGE2_BEST[key]
        n_seeds = 10 if any(STAGE1_BEST[mod]["encoder"] == "linear" for mod in subset) else 1
        if smoke:
            n_seeds = min(n_seeds, 3)
        joint_oof[key] = phase_b_condition(subset, cfg, n_seeds)
        print(f"[exp_25] Phase B {key}: n_seeds={n_seeds}, macro_f1={np.mean(joint_oof[key]['per_seed_macro_f1']):.4f}")

    cfg_c = STAGE2_BEST["confidence_arm"]
    n_seeds_c = 10 if any(STAGE1_BEST[mod]["encoder"] == "linear" for mod in ["tab", "mri", "txt"]) else 1
    if smoke:
        n_seeds_c = min(n_seeds_c, 3)
    confidence_arm_oof = phase_b_condition(["tab", "mri", "txt"], cfg_c, n_seeds_c, label_smoothing=0.10)
    print(f"[exp_25] Phase B confidence_arm: n_seeds={n_seeds_c}")

    late_fusion_equal_probs = soft_vote({m: to_2col(unimodal_oof[m]["oof_pred"]) for m in ["tab", "mri", "txt"]},
                                         {"tab": 1 / 3, "mri": 1 / 3, "txt": 1 / 3})
    late_fusion_optimal_probs = soft_vote({m: to_2col(unimodal_oof[m]["oof_pred"]) for m in ["tab", "mri", "txt"]},
                                           fusion_weights["weights"])

    ALL_CONDITIONS = {
        "unimodal_tab": to_2col(unimodal_oof["tab"]["oof_pred"]),
        "unimodal_mri": to_2col(unimodal_oof["mri"]["oof_pred"]),
        "unimodal_txt": to_2col(unimodal_oof["txt"]["oof_pred"]),
        "joint_tab_mri": to_2col(joint_oof["tab_mri"]["oof_pred"]),
        "joint_tab_txt": to_2col(joint_oof["tab_txt"]["oof_pred"]),
        "joint_mri_txt": to_2col(joint_oof["mri_txt"]["oof_pred"]),
        "joint_trimodal": to_2col(joint_oof["tab_mri_txt"]["oof_pred"]),
        "late_fusion_equal": late_fusion_equal_probs,
        "late_fusion_optimal": late_fusion_optimal_probs,
        "confidence_arm": to_2col(confidence_arm_oof["oof_pred"]),
    }

    idx = np.array(LOOCV_FOLDS)
    y_true_eval = cohort.y_binary[idx]
    loocv_metrics = {}
    n_seeds_lookup = {
        "unimodal_tab": unimodal_oof["tab"], "unimodal_mri": unimodal_oof["mri"], "unimodal_txt": unimodal_oof["txt"],
        "joint_tab_mri": joint_oof["tab_mri"], "joint_tab_txt": joint_oof["tab_txt"],
        "joint_mri_txt": joint_oof["mri_txt"], "joint_trimodal": joint_oof["tab_mri_txt"],
        "confidence_arm": confidence_arm_oof,
    }
    for name, probs in ALL_CONDITIONS.items():
        m = binary_metrics(y_true_eval, probs[idx, 1])
        if name in n_seeds_lookup:
            src = n_seeds_lookup[name]
            m.update({"deterministic": src["deterministic"], "n_seeds": src["n_seeds"],
                      "per_seed_macro_f1": src["per_seed_macro_f1"],
                      "macro_f1_std_across_seeds": src["macro_f1_std_across_seeds"],
                      "mode_vote_agreement": src["mode_vote_agreement"]})
        else:  # late_fusion_* — derived from the 3 unimodal conditions, no independent seed loop
            member_seeds = {mm: unimodal_oof[mm]["n_seeds"] for mm in ["tab", "mri", "txt"]}
            m.update({"deterministic": all(v == 1 for v in member_seeds.values()), "member_n_seeds": member_seeds})
        loocv_metrics[name] = m

    y_pred_trimodal = (ALL_CONDITIONS["joint_trimodal"][idx, 1] >= 0.50).astype(int)
    y_pred_late_opt = (ALL_CONDITIONS["late_fusion_optimal"][idx, 1] >= 0.50).astype(int)
    mcnemar_results = {"joint_trimodal_vs_late_fusion_optimal": mcnemar_exact(y_true_eval, y_pred_trimodal, y_pred_late_opt)}
    try:
        exp23_oof = pd.read_csv(PROJECT_ROOT / "experiments" / "exp_23" / "results" / "oof_predictions.csv")
        exp23_oof = exp23_oof.set_index("patient_id").loc[cohort.pids]
        exp23_soft_pred = exp23_oof["kdm_soft_pred"].values.astype(int)[idx]
        mcnemar_results["joint_trimodal_vs_exp23_soft"] = mcnemar_exact(y_true_eval, y_pred_trimodal, exp23_soft_pred)
    except Exception as e:  # pragma: no cover — context comparison only, not load-bearing for H1
        mcnemar_results["joint_trimodal_vs_exp23_soft"] = {"error": str(e)}
    loocv_metrics["_mcnemar"] = mcnemar_results
    write_json(loocv_metrics, RESULTS_DIR / "loocv_metrics.json")

    df_oof = pd.DataFrame({"patient_id": cohort.pids, "ground_truth_biopsy": cohort.y_binary,
                            "confidence_annotation": cohort.confidence})
    for name, probs in ALL_CONDITIONS.items():
        df_oof[f"prob_{name}"] = probs[:, 1]
        df_oof[f"pred_{name}"] = (probs[:, 1] >= 0.50).astype(int)
    df_oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)

    # ---------------------------------------------------------------- 1.7 Confidence task
    print("[exp_25] Confidence task...")
    y_conf = cohort.y_conf  # not targets.y_conf — build_targets never populates it (known src/ gap)

    P_unimodal = np.stack([unimodal_oof[m]["oof_pred"] for m in ["tab", "mri", "txt"]], axis=1)
    ici, p_mean_arr, p_std_arr, margin_arr = composite_reliability_index(P_unimodal)
    inter_var = inter_modality_variance(P_unimodal)
    late_fusion_signals = {f"{m}__{k}": v for m in ["tab", "mri", "txt"] for k, v in unimodal_oof[m]["oof_signals"].items()}
    late_fusion_signals.update({"composite_ici": ici, "inter_modality_variance": inter_var,
                                 "p_mean": p_mean_arr, "p_std": p_std_arr, "margin": margin_arr})

    CONDITION_SIGNALS = {
        "unimodal_tab": unimodal_oof["tab"]["oof_signals"], "unimodal_mri": unimodal_oof["mri"]["oof_signals"],
        "unimodal_txt": unimodal_oof["txt"]["oof_signals"], "joint_tab_mri": joint_oof["tab_mri"]["oof_signals"],
        "joint_tab_txt": joint_oof["tab_txt"]["oof_signals"], "joint_mri_txt": joint_oof["mri_txt"]["oof_signals"],
        "joint_trimodal": joint_oof["tab_mri_txt"]["oof_signals"],
        "late_fusion_equal": late_fusion_signals, "late_fusion_optimal": late_fusion_signals,
        "confidence_arm": confidence_arm_oof["oof_signals"],
    }

    def safe_confidence_metrics(y_c, pred):
        """`confidence_metrics`'s `spearmanr` returns NaN when `pred` collapses to a single class (a
        real possibility on a degenerate head, not just a --smoke artifact — the exact bare-NaN bug
        `reporting._StrictEncoder` exists to catch, per `exp_24/results/confidence_metrics.json`).
        `write_json` correctly refuses to serialize NaN; the fix belongs here, at the source, not as an
        `allow_nan=True` bypass at the write call."""
        m = confidence_metrics(y_c, pred)
        for k in ("spearman_rho", "spearman_pvalue"):
            if not np.isfinite(m[k]):
                m[k] = None
        return m

    confidence_rows = []
    for name, signals in CONDITION_SIGNALS.items():
        is_late_fusion = name in LATE_FUSION_NAMES
        keys_1d = [k for k in signals if k not in ("p_mean", "p_std", "margin")] if is_late_fusion else list(PARTICLE_SIGNAL_NAMES)
        best_1d = None
        for key in keys_1d:
            thr = fit_meta_thresholds_safe(signals[key], y_conf, SPLITS_2T_CONF)
            pred = apply_meta_thresholds(signals[key], thr)
            m = safe_confidence_metrics(y_conf, pred)
            row = {"condition": name, "head": "1d", "signal": key, "target_informed": True, **m,
                   "meta_threshold_1": thr["meta_threshold_1"], "meta_threshold_2": thr["meta_threshold_2"],
                   "direction": thr["direction"], "degenerate_fallback": thr["degenerate_fallback"]}
            confidence_rows.append(row)
            if best_1d is None or m["macro_f1"] > best_1d["macro_f1"]:
                best_1d = row

        S = np.stack([signals[k] for k in keys_1d], axis=1)
        pred, votes = fit_predict_heldout_trees(S, y_conf, SPLITS_2T_CONF)
        m = safe_confidence_metrics(y_conf, pred)
        confidence_rows.append({"condition": name, "head": "multivariate_full", "keys": keys_1d,
                                 "target_informed": True, "min_votes": int(votes.min()), **m})

        ablation_keys = ["composite_ici", "inter_modality_variance"] if is_late_fusion else ["h_total", "log_marginal"]
        S_ab = np.stack([signals[k] for k in ablation_keys], axis=1)
        pred_ab, votes_ab = fit_predict_heldout_trees(S_ab, y_conf, SPLITS_2T_CONF)
        m_ab = safe_confidence_metrics(y_conf, pred_ab)
        confidence_rows.append({"condition": name, "head": "multivariate_ablation", "keys": ablation_keys,
                                 "target_informed": True, "min_votes": int(votes_ab.min()), **m_ab})

    confidence_output = {
        "rows": confidence_rows,
        "context_baselines_not_like_for_like": {
            "exp23_entropy_soft_target_informed": 0.4164,
            "exp24_best_non_target_informed": 0.4368,
            "exp17_composite_ici_non_target_informed": 0.4470,
        },
    }
    write_json(confidence_output, RESULTS_DIR / "confidence_metrics.json")

    df_sig = pd.DataFrame({"patient_id": cohort.pids, "confidence_annotation": cohort.confidence, "y_conf": cohort.y_conf})
    for name, signals in CONDITION_SIGNALS.items():
        for k, v in signals.items():
            df_sig[f"{name}__{k}"] = v
    df_sig.to_csv(RESULTS_DIR / "oof_particle_signals.csv", index=False)

    # ---------------------------------------------------------------- figures
    print("[exp_25] Figures...")
    try:
        for name, df_grid in [("tab", STAGE1_GRIDS[0]), ("mri", STAGE1_GRIDS[1]), ("txt", STAGE1_GRIDS[2])]:
            plot_grid_search_curves(df_grid, x_col="sigma_mult", y_col="mean_macro_f1", group_cols=["encoder", "y_train"],
                                     title=f"exp_25 Stage 1 {name} grid search", out_path=FIG_DIR / f"stage1_grid_search_curves_{name}.png")
        for df_grid in STAGE2_GRIDS:
            cond = df_grid["condition"].iloc[0]
            plot_grid_search_curves(df_grid, x_col="sigma_scale", y_col="mean_macro_f1", group_cols=["kernel_trainable", "y_train"],
                                     title=f"exp_25 Stage 2 {cond} grid search", out_path=FIG_DIR / f"stage2_grid_search_curves_{cond}.png")
        plot_confusion_matrix(y_true_eval, y_pred_trimodal, labels=["No", "Yes"],
                               title="exp_25 joint_trimodal LOOCV", out_path=FIG_DIR / "confusion_matrix.png")
        plot_roc_curves({"joint_trimodal": ALL_CONDITIONS["joint_trimodal"][idx, 1],
                          "late_fusion_optimal": ALL_CONDITIONS["late_fusion_optimal"][idx, 1],
                          "unimodal_tab": ALL_CONDITIONS["unimodal_tab"][idx, 1]},
                         y_true_eval, title="exp_25 ROC overlay", out_path=FIG_DIR / "roc_curve.png")
        plot_signal_scatter(CONDITION_SIGNALS["joint_trimodal"]["h_aleatoric"][idx], CONDITION_SIGNALS["joint_trimodal"]["h_epistemic"][idx],
                             cohort.confidence[idx], colors={"clear": "tab:green", "borderline": "tab:orange", "uncertain": "tab:red"},
                             xlabel="h_aleatoric", ylabel="h_epistemic", title="exp_25 joint_trimodal particle signals",
                             out_path=FIG_DIR / "particle_signal_scatter.png")
        best_conf_row = max((r for r in confidence_rows if r["head"] != "1d" or True), key=lambda r: r["macro_f1"])
        print(f"[exp_25] best confidence row (for reference, not re-plotted): {best_conf_row['condition']}/{best_conf_row['head']} "
              f"macro_f1={best_conf_row['macro_f1']:.4f}")
    except Exception as e:  # pragma: no cover — figures are best-effort, never block the numeric results
        print(f"[exp_25] WARNING: figure generation failed: {e}")

    record_git_commit(RESULTS_DIR)
    elapsed = time.time() - t_start
    print(f"[exp_25] DONE in {elapsed / 60:.1f} min. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
