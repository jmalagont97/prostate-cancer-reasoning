import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, brier_score_loss,
)
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr, binomtest

from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer
from kdm.utils import pure2dm, dm2comp


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_24"
RESULTS_DIR = EXP_DIR / "results"
REPORTS_DIR = EXP_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXP23_RESULTS = PROJECT_ROOT / "experiments" / "exp_23" / "results"

for d in (RESULTS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def resolve_data_dir():
    candidates = [
        PROJECT_ROOT / "data" / "chimera26" / "preprocessed" / "task1",
        PROJECT_ROOT / "Data" / "preprocessed_old" / "task1",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"None of the candidate data dirs exist: {candidates}")


NUM_COLS = ["age", "psa", "vol", "pirads", "psad", "psav", "psap"]
CAT_COLS = ["dre"]

SIGMA_MULTS_C = [0.5, 1.0, 2.0]
EPS_LIST_C = [0.05, 0.10, 0.20]
GRID_C = [
    {"sigma_mult": sm, "eps": eps, "encoder": enc, "x_train": False, "y_train": True}
    for sm in SIGMA_MULTS_C
    for eps in EPS_LIST_C
    for enc in ["identity", "linear"]
]
assert len(GRID_C) == 18

SIGNAL_NAMES = ["h_total", "h_aleatoric", "h_epistemic", "h_weights", "log_ess", "w_max", "log_marginal"]


# ---------------------------------------------------------------------------
# 1.1 Load, align, and validate the cohort — copy exp_23/scripts/train.py:79-126 verbatim
# ---------------------------------------------------------------------------
def load_cohort(data_dir):
    df_tab = pd.read_csv(data_dir / "clinical_data_tabular.csv", dtype=str, keep_default_na=False)
    df_dec = pd.read_csv(data_dir / "biopsy_decision.csv", dtype=str, keep_default_na=False)
    df_reasoning = pd.read_csv(data_dir / "clinical_reasoning.csv", dtype=str, keep_default_na=False)
    df_design = pd.read_csv(PROJECT_ROOT / "experiments" / "exp_4" / "results" / "mccv_design.csv")

    for c in NUM_COLS:
        df_tab[c] = pd.to_numeric(df_tab[c], errors="coerce")

    pids = df_design["patient_id"].values
    df_tab = df_tab[df_tab["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_dec = df_dec[df_dec["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_reasoning = df_reasoning[df_reasoning["patient_id"].isin(pids)].sort_values("patient_id").reset_index(drop=True)
    df_design = df_design.sort_values("patient_id").reset_index(drop=True)

    labeled_mask = df_dec["biopsy_decision"] != "NONE"
    df_tab_labeled = df_tab[labeled_mask].reset_index(drop=True)
    df_dec_labeled = df_dec[labeled_mask].reset_index(drop=True)
    df_reasoning_labeled = df_reasoning[labeled_mask].reset_index(drop=True)
    df_design_labeled = df_design[labeled_mask].reset_index(drop=True)

    pids_labeled = df_dec_labeled["patient_id"].values
    biopsy_label_map = {"yes": 1, "no": 0}
    y_binary = df_dec_labeled["biopsy_decision"].map(biopsy_label_map).values.astype(int)

    assert len(y_binary) == 88, f"expected N=88, got {len(y_binary)}"
    n_yes, n_no = int((y_binary == 1).sum()), int((y_binary == 0).sum())
    assert n_yes == 54 and n_no == 34, f"class-count mismatch: yes={n_yes}, no={n_no}"

    confidence_certainty_map = {"clear": 1.00, "borderline": 0.50, "uncertain": 0.25}
    c_weights = df_reasoning_labeled["confidence"].map(confidence_certainty_map).fillna(1.00).values.astype(float)
    y_soft = np.where(y_binary == 1, 0.50 + 0.50 * c_weights, 0.50 - 0.50 * c_weights).astype(np.float32)

    confidence_map = {"uncertain": 0, "borderline": 1, "clear": 2}
    y_conf_series = df_reasoning_labeled["confidence"].map(confidence_map)
    assert not y_conf_series.isna().any(), "unexpected unmapped confidence value in labeled cohort"
    y_conf = y_conf_series.values.astype(int)

    return dict(
        df_tab_labeled=df_tab_labeled,
        df_design_labeled=df_design_labeled,
        pids_labeled=pids_labeled,
        y_binary=y_binary,
        y_soft=y_soft,
        c_weights=c_weights,
        confidence_annotation=df_reasoning_labeled["confidence"].values,
        y_conf=y_conf,
    )


# ---------------------------------------------------------------------------
# 1.2 Feature construction — copy exp_23/scripts/train.py:148-173 verbatim
# ---------------------------------------------------------------------------
def build_features_fixed_categories(df, train_idx, val_idx, dre_categories):
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

    scaler = MinMaxScaler()
    X_tr_num = scaler.fit_transform(df.iloc[train_idx][NUM_COLS])
    X_va_num = scaler.transform(df.iloc[val_idx][NUM_COLS])

    ohe = OneHotEncoder(categories=[dre_categories], handle_unknown="ignore", sparse_output=False)
    X_tr_cat = ohe.fit_transform(df.iloc[train_idx][CAT_COLS])
    X_va_cat = ohe.transform(df.iloc[val_idx][CAT_COLS])

    X_tr = np.hstack([X_tr_num, X_tr_cat]).astype(np.float32)
    X_va = np.hstack([X_va_num, X_va_cat]).astype(np.float32)

    assert np.isfinite(X_tr).all() and np.isfinite(X_va).all(), "non-finite values in feature matrix"
    expected_dim = len(NUM_COLS) + len(dre_categories)
    assert X_tr.shape[1] == expected_dim and X_va.shape[1] == expected_dim, \
        f"unexpected feature dim: X_tr={X_tr.shape[1]}, X_va={X_va.shape[1]}, expected={expected_dim}"
    return X_tr, X_va


# ---------------------------------------------------------------------------
# 1.3 Target/amplitude encodings — to_amplitude_hard/soft copied verbatim; smoothed variant new
# ---------------------------------------------------------------------------
def to_amplitude_hard(y_binary_subset):
    t = torch.as_tensor(y_binary_subset, dtype=torch.long)
    return F.one_hot(t, 2).float().numpy()


def to_amplitude_soft(y_soft_subset):
    y_soft_subset = np.clip(y_soft_subset, 0.0, 1.0)
    return np.stack([np.sqrt(1 - y_soft_subset), np.sqrt(y_soft_subset)], axis=1).astype(np.float32)


def to_amplitude_hard_smoothed(y_binary_subset, eps):
    """eps=0.0 reduces exactly to to_amplitude_hard — used as the H3 structural identity."""
    y = y_binary_subset.astype(np.float32)
    p1 = y * (1 - eps) + eps / 2
    p0 = 1 - p1
    return np.stack([np.sqrt(p0), np.sqrt(p1)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 1.4 KDM model builder and training step — copy exp_23/scripts/train.py:262-321 verbatim
# ---------------------------------------------------------------------------
def build_kdm(cfg, n_comp):
    if cfg["encoder"] == "identity":
        encoder, encoded_size = nn.Identity(), 12
    else:
        encoder, encoded_size = nn.Linear(12, 8), 8
    return KDMClassModel(
        encoded_size=encoded_size, dim_y=2, encoder=encoder, n_comp=n_comp,
        sigma=0.5, sigma_trainable=True,
        x_train=cfg["x_train"], y_train=cfg["y_train"], w_train=True,
    )


def init_and_check(model, X_tr, c_y_tr, sigma_mult, check_roundtrip=False):
    with torch.no_grad():
        enc_sub = model.encoder(torch.as_tensor(X_tr, dtype=torch.float32))
    init_kdm_layer(
        model.kdm, enc_sub.detach(), torch.as_tensor(c_y_tr, dtype=torch.float32),
        init_sigma=True, sigma_mult=sigma_mult,
    )
    if check_roundtrip:
        import copy
        from scipy.spatial.distance import pdist
        probe = copy.deepcopy(model)
        probe.eval()
        with torch.no_grad():
            enc_np = model.encoder(torch.as_tensor(X_tr, dtype=torch.float32)).numpy()
        pairwise = pdist(enc_np)
        pairwise = pairwise[pairwise > 0]
        min_d = float(pairwise.min()) if len(pairwise) > 0 else 1.0
        probe_sigma = max(float(probe.kernel.min_sigma) * 1.5, min_d / 50.0)
        probe.kernel.sigma = probe_sigma
        with torch.no_grad():
            probs = probe(torch.as_tensor(X_tr, dtype=torch.float32))
        target = c_y_tr[:, 1] ** 2
        err = float(np.abs(probs[:, 1].numpy() - target).max())
        assert err < 1e-4, (
            f"amplitude round-trip failed: max err {err} "
            f"(probe_sigma={probe_sigma:.2e}, min_pairwise_dist={min_d:.4f})"
        )


def train_kdm(model, X_tr, arm, y_binary_tr=None, y_soft_tr=None, lr=1e-3, epochs=300):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X_tr, dtype=torch.float32)
    for _ in range(epochs):
        probs = model(Xt)
        if arm == "hard":
            loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), torch.as_tensor(y_binary_tr, dtype=torch.long))
        else:
            t = torch.as_tensor(np.stack([1 - y_soft_tr, y_soft_tr], axis=1), dtype=torch.float32)
            loss = -(t * torch.log(probs.clamp_min(1e-7))).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# 1.5 Particle-set signal extraction (new)
# ---------------------------------------------------------------------------
def extract_particle_signals(model, X, eps=1e-7):
    """Per-sample signals from the KDM's output density matrix, before dm2discrete collapses it.
    Reproduces dm2discrete's own probs exactly (own_libs/kdm/kdm/utils.py:58-71: normalized weights,
    L2-normalized+squared vectors), so h_total is bit-identical to H(model(x))."""
    model.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(X, dtype=torch.float32)
        enc = model.encoder(Xt)
        rho_x = pure2dm(enc)
        rho_y = model.kdm(rho_x)                                  # (n, n_comp, dim_y+1)
        w, v = dm2comp(rho_y)
        w = w / w.sum(-1, keepdim=True)
        p = F.normalize(v, p=2, dim=-1, eps=1e-12) ** 2            # (n, n_comp, dim_y)

        p_mean = (w.unsqueeze(-1) * p).sum(dim=1)                  # == dm2discrete(rho_y)
        h_total = -(p_mean * torch.log(p_mean.clamp_min(eps))).sum(-1)

        h_particles = -(p * torch.log(p.clamp_min(eps))).sum(-1)   # (n, n_comp)
        h_aleatoric = (w * h_particles).sum(dim=1)
        h_epistemic = h_total - h_aleatoric

        h_weights = -(w * torch.log(w.clamp_min(eps))).sum(-1)
        log_ess = -torch.log((w ** 2).sum(-1).clamp_min(eps))
        w_max = w.max(dim=-1).values

        log_marginal = model.kdm.log_marginal(rho_x)
    return {
        "probs": p_mean.numpy(), "h_total": h_total.numpy(), "h_aleatoric": h_aleatoric.numpy(),
        "h_epistemic": h_epistemic.numpy(), "h_weights": h_weights.numpy(), "log_ess": log_ess.numpy(),
        "w_max": w_max.numpy(), "log_marginal": log_marginal.numpy(),
    }


# ---------------------------------------------------------------------------
# 1.6 Phase A — Arm C grid search (100 MCCV splits)
# ---------------------------------------------------------------------------
def kdm_phase_a_arm_c(df_tab_labeled, df_design_labeled, y_binary, dre_categories, n_splits=100, grid=None):
    grid = GRID_C if grid is None else grid
    grid_scores = {i: [] for i in range(len(grid))}
    grid_extra = {i: {"acc": [], "sens": [], "spec": []} for i in range(len(grid))}

    for split_idx in range(n_splits):
        split_vals = df_design_labeled[f"split_{split_idx}"].values
        train_idx = np.where(split_vals == 0)[0]
        val_idx = np.where(split_vals == 1)[0]

        X_tr, X_va = build_features_fixed_categories(df_tab_labeled, train_idx, val_idx, dre_categories)
        y_binary_tr = y_binary[train_idx]
        y_va_true = y_binary[val_idx]

        for cfg_idx, cfg in enumerate(grid):
            c_y_tr = to_amplitude_hard_smoothed(y_binary_tr, cfg["eps"])
            torch.manual_seed(42)  # same init for every config within a split -> paired comparison
            model = build_kdm(cfg, n_comp=len(train_idx))
            init_and_check(model, X_tr, c_y_tr, cfg["sigma_mult"], check_roundtrip=False)
            train_kdm(model, X_tr, "hard", y_binary_tr=y_binary_tr)

            model.eval()
            with torch.no_grad():
                probs_va = model(torch.as_tensor(X_va, dtype=torch.float32)).numpy()
            y_val_pred = (probs_va[:, 1] >= 0.50).astype(int)

            macro_f1 = f1_score(y_va_true, y_val_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_va_true, y_val_pred)
            tn, fp, fn, tp = confusion_matrix(y_va_true, y_val_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            grid_scores[cfg_idx].append(macro_f1)
            grid_extra[cfg_idx]["acc"].append(acc)
            grid_extra[cfg_idx]["sens"].append(sens)
            grid_extra[cfg_idx]["spec"].append(spec)

        if (split_idx + 1) % 10 == 0:
            print(f"    [arm_c] MCCV splits completed: {split_idx + 1}/{n_splits}")

    rows = []
    for cfg_idx, cfg in enumerate(grid):
        rows.append({
            "cfg_id": cfg_idx, **cfg,
            "mean_macro_f1": float(np.mean(grid_scores[cfg_idx])),
            "std_macro_f1": float(np.std(grid_scores[cfg_idx])),
            "mean_acc": float(np.mean(grid_extra[cfg_idx]["acc"])),
            "mean_sens": float(np.mean(grid_extra[cfg_idx]["sens"])),
            "mean_spec": float(np.mean(grid_extra[cfg_idx]["spec"])),
        })
    df_grid = pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    best = df_grid.iloc[0]
    best_cfg = {
        "sigma_mult": float(best["sigma_mult"]), "eps": float(best["eps"]),
        "x_train": False, "y_train": True, "encoder": str(best["encoder"]),
        "mccv_mean_macro_f1": float(best["mean_macro_f1"]), "mccv_std_macro_f1": float(best["std_macro_f1"]),
        "mccv_mean_accuracy": float(best["mean_acc"]),
    }
    return best_cfg, df_grid


# ---------------------------------------------------------------------------
# 1.7 Phase B — LOOCV (88 folds), shared across all three arms, with particle signals
# ---------------------------------------------------------------------------
def kdm_phase_b_particles(df_tab_labeled, y_binary, y_soft, loss_arm, c_y_builder, cfg, dre_categories,
                           n_seeds=10, track_drift=False):
    """Generalizes exp_23's kdm_phase_b: same LOOCV/seed/mode-vote/metrics structure, but c_y init is
    supplied by c_y_builder(y_binary_tr, y_soft_tr) so Arms A/B/C share one loop, and
    extract_particle_signals replaces the entropy/log_marginal-only extraction."""
    is_deterministic = (cfg["encoder"] == "identity")
    seeds = [0] if is_deterministic else list(range(n_seeds))
    n = len(y_binary)

    oof_p_soft = np.zeros((len(seeds), n))
    oof_signals = {s: np.zeros((len(seeds), n)) for s in SIGNAL_NAMES}
    max_cy_drift = 0.0

    loo = LeaveOneOut()
    for train_idx, val_idx in loo.split(y_binary):
        val_idx0 = int(val_idx[0])
        X_tr, X_va = build_features_fixed_categories(df_tab_labeled, train_idx, val_idx, dre_categories)
        y_binary_tr = y_binary[train_idx]
        y_soft_tr = y_soft[train_idx]
        c_y_tr = c_y_builder(y_binary_tr, y_soft_tr)

        for s_idx, seed in enumerate(seeds):
            torch.manual_seed(seed)  # BEFORE build_kdm — the only randomness is nn.Linear's init inside it
            model = build_kdm(cfg, n_comp=len(train_idx))
            init_and_check(model, X_tr, c_y_tr, cfg["sigma_mult"], check_roundtrip=False)
            c_y_init = model.kdm.c_y.detach().clone() if track_drift else None
            train_kdm(model, X_tr, loss_arm, y_binary_tr=y_binary_tr, y_soft_tr=y_soft_tr)
            if track_drift:
                drift = (model.kdm.c_y.detach() - c_y_init).abs().max().item()
                max_cy_drift = max(max_cy_drift, drift)

            sig = extract_particle_signals(model, X_va)
            oof_p_soft[s_idx, val_idx0] = sig["probs"][0, 1]
            for s in SIGNAL_NAMES:
                oof_signals[s][s_idx, val_idx0] = sig[s][0]

        if (val_idx0 + 1) % 10 == 0:
            print(f"    [{loss_arm}] LOOCV folds completed: {val_idx0 + 1}/{n}")

    p_mean = oof_p_soft.mean(axis=0)
    signals_mean = {s: oof_signals[s].mean(axis=0) for s in SIGNAL_NAMES}
    y_pred = (p_mean >= 0.50).astype(int)

    hard_preds_per_seed = (oof_p_soft >= 0.50).astype(int)
    mode_vote = (hard_preds_per_seed.mean(axis=0) >= 0.5).astype(int)
    agreement = float(np.mean(y_pred == mode_vote))

    macro_f1 = f1_score(y_binary, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_binary, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_binary, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    auroc = roc_auc_score(y_binary, p_mean)
    brier = brier_score_loss(y_binary, p_mean)

    metrics = {
        "macro_f1": float(macro_f1), "accuracy": float(acc),
        "sensitivity": float(sens), "specificity": float(spec),
        "auroc": float(auroc), "brier_score": float(brier),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_cases": int(n),
        "deterministic": is_deterministic,
        "n_seeds": len(seeds),
        "mode_vote_agreement": agreement,
    }
    if not is_deterministic:
        per_seed_f1 = []
        for s_idx in range(len(seeds)):
            yp_s = (oof_p_soft[s_idx] >= 0.5).astype(int)
            per_seed_f1.append(f1_score(y_binary, yp_s, average="macro", zero_division=0))
        metrics["per_seed_macro_f1"] = [float(x) for x in per_seed_f1]
        metrics["macro_f1_std_across_seeds"] = float(np.std(per_seed_f1))

    return dict(
        metrics=metrics, p_mean=p_mean, y_pred=y_pred,
        signals_mean=signals_mean, oof_signals_per_seed=oof_signals,
        max_cy_drift=max_cy_drift,
    )


# ---------------------------------------------------------------------------
# 1.8 Structural check (H3)
# ---------------------------------------------------------------------------
def degeneracy_check(df_tab_labeled, y_binary, dre_categories, encoder="linear", sigma_mult=2.0, epochs=300):
    """Confirms DESIGN.md §2.1: a one-hot c_y is a gradient fixed point even with y_train=True and
    gradients flowing. Trains one full-cohort model (n_comp=88, structural check only — never scored,
    no leakage concern) at eps=0.0 (== to_amplitude_hard exactly) with y_train=True and asserts c_y does
    not move."""
    idx = np.arange(len(y_binary))
    X_tr, _ = build_features_fixed_categories(df_tab_labeled, idx, idx[:1], dre_categories)
    cfg = {"encoder": encoder, "x_train": False, "y_train": True}
    c_y_tr = to_amplitude_hard_smoothed(y_binary, eps=0.0)
    torch.manual_seed(0)
    model = build_kdm(cfg, n_comp=len(idx))
    init_and_check(model, X_tr, c_y_tr, sigma_mult, check_roundtrip=False)
    c_y_before = model.kdm.c_y.detach().clone()
    train_kdm(model, X_tr, "hard", y_binary_tr=y_binary, epochs=epochs)
    max_drift = (model.kdm.c_y.detach() - c_y_before).abs().max().item()
    return {"max_cy_drift_eps0": max_drift, "passed": bool(max_drift < 1e-9)}


def full_cohort_drift(df_tab_labeled, y_binary, dre_categories, cfg, c_y_tr, sigma_mult, loss_arm,
                       y_soft=None, epochs=300, seed=0):
    """Fits one full-cohort model and returns the per-particle |c_y_after - c_y_before| vector, for
    the cy_drift.png figure (Arm-A-style eps=0 model vs. Arm C's actual selected config)."""
    idx = np.arange(len(y_binary))
    X_tr, _ = build_features_fixed_categories(df_tab_labeled, idx, idx[:1], dre_categories)
    torch.manual_seed(seed)
    model = build_kdm(cfg, n_comp=len(idx))
    init_and_check(model, X_tr, c_y_tr, sigma_mult, check_roundtrip=False)
    c_y_before = model.kdm.c_y.detach().clone()
    train_kdm(model, X_tr, loss_arm, y_binary_tr=y_binary, y_soft_tr=y_soft, epochs=epochs)
    drift = (model.kdm.c_y.detach() - c_y_before).abs().max(dim=-1).values.numpy()
    return drift


# ---------------------------------------------------------------------------
# McNemar's exact test — copy exp_23/scripts/train.py:490-500 verbatim
# ---------------------------------------------------------------------------
def mcnemar_exact(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0, "pvalue": 1.0}
    stat = min(b, c)
    pvalue = binomtest(stat, n, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "statistic": int(stat), "pvalue": float(pvalue)}


# ---------------------------------------------------------------------------
# 1.9 Confidence heads
# ---------------------------------------------------------------------------
# 1D per-signal — copy exp_23/scripts/train.py:506-587 verbatim
def fit_1d_confidence_signal(signal_oof, y_conf, df_design_labeled, n_splits=100):
    thresholds_t1, thresholds_t2, directions = [], [], []
    fallback_count = 0
    degenerate_count = 0
    nonmonotone_count = 0

    lo, hi = float(signal_oof.min()), float(signal_oof.max())
    sweep = np.linspace(lo, hi, 50).reshape(-1, 1)

    for split_idx in range(n_splits):
        split_vals = df_design_labeled[f"split_{split_idx}"].values
        train_mask = split_vals == 0

        X_tr = signal_oof[train_mask].reshape(-1, 1)
        y_tr = y_conf[train_mask]

        dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)
        dt.fit(X_tr, y_tr)

        tree_thresholds = dt.tree_.threshold[dt.tree_.threshold != -2]
        tree_thresholds = np.sort(tree_thresholds)

        if len(tree_thresholds) >= 2:
            t1, t2 = float(tree_thresholds[0]), float(tree_thresholds[1])
        elif len(tree_thresholds) == 1:
            fallback_count += 1
            t1 = float(tree_thresholds[0])
            p67 = float(np.percentile(X_tr, 67))
            t2 = max(p67, t1 + (float(X_tr.max()) - t1) / 2)
        else:
            fallback_count += 1
            t1, t2 = (float(v) for v in np.percentile(X_tr, [33, 67]))

        thresholds_t1.append(t1)
        thresholds_t2.append(t2)

        pred_sweep = dt.predict(sweep)
        if len(np.unique(pred_sweep)) == 1:
            degenerate_count += 1
            continue
        rho, _ = spearmanr(np.arange(len(sweep)), pred_sweep)
        if np.isnan(rho):
            degenerate_count += 1
            continue
        if abs(rho) < 0.1:
            nonmonotone_count += 1
        directions.append(np.sign(rho))

    meta_t1 = float(np.mean(thresholds_t1))
    meta_t2 = float(np.mean(thresholds_t2))
    if len(directions) == 0:
        raise RuntimeError("all splits degenerate — cannot determine signal direction")
    direction = int(np.sign(np.nansum(directions)))
    assert direction in (1, -1), f"ambiguous direction: {direction}"

    return {
        "meta_threshold_1": meta_t1, "meta_threshold_2": meta_t2,
        "direction": direction,
        "fallback_count_of_100": fallback_count,
        "degenerate_count_of_100": degenerate_count,
        "nonmonotone_count_of_100": nonmonotone_count,
    }


def apply_1d_confidence_signal(signal_oof, thr):
    t1, t2, direction = thr["meta_threshold_1"], thr["meta_threshold_2"], thr["direction"]
    if direction == 1:
        pred = np.where(signal_oof < t1, 0, np.where(signal_oof < t2, 1, 2))
    else:
        pred = np.where(signal_oof < t1, 2, np.where(signal_oof < t2, 1, 0))
    return pred


def fit_1d_confidence_signal_safe(signal_oof, y_conf, df_design_labeled, n_splits=100):
    """Wraps fit_1d_confidence_signal (verbatim exp_23 logic, left unmodified above) with a fallback
    for two rare edge cases it raises on rather than resolves: every split's tree collapsing to a
    constant sweep prediction (RuntimeError), or an exact tie in the per-split direction vote
    (AssertionError on np.sign(0)). Both are legitimate possible outcomes for a real signal on N=88 —
    not something exp_23 ever had to handle because none of its four signals happened to hit them —
    and crashing an otherwise-complete ~20-30 min run over one signal is worse than a documented,
    flagged fallback (33rd/67th percentile thresholds, direction=+1)."""
    try:
        thr = fit_1d_confidence_signal(signal_oof, y_conf, df_design_labeled, n_splits=n_splits)
        thr["degenerate_fallback"] = False
        return thr
    except (RuntimeError, AssertionError) as e:
        print(f"    WARNING: fit_1d_confidence_signal fell back ({type(e).__name__}: {e})")
        t1, t2 = (float(v) for v in np.percentile(signal_oof, [33, 67]))
        return {
            "meta_threshold_1": t1, "meta_threshold_2": t2, "direction": 1,
            "fallback_count_of_100": n_splits, "degenerate_count_of_100": n_splits,
            "nonmonotone_count_of_100": 0, "degenerate_fallback": True,
        }


def score_confidence(y_conf, pred):
    macro_f1 = f1_score(y_conf, pred, average="macro", zero_division=0)
    acc = accuracy_score(y_conf, pred)
    rho, pval = spearmanr(y_conf, pred)
    return {
        "macro_f1": float(macro_f1), "accuracy": float(acc),
        "spearman_rho": float(rho), "spearman_pvalue": float(pval),
        "total_cases": int(len(y_conf)),
    }


# Multivariate frozen tree ensemble (new)
def fit_multivariate_confidence_head(signal_matrix, y_conf, df_design_labeled, n_splits=100,
                                      max_depth=3, random_state=42):
    """One DecisionTreeClassifier per MCCV split, trained on that split's train rows only. Patient i's
    Phase-B prediction is the majority vote over ONLY the trees whose split had i in validation
    (split_i == 1) — i.e. trees that never saw patient i during fitting. Required because a depth-3
    tree over several signals has far more memorization capacity than exp_17's 2-scalar 1D threshold,
    so exp_17's whole-cohort application (tolerated there because dilution over 100 splits makes any
    one patient's influence negligible) would leak here."""
    n = signal_matrix.shape[0]
    votes = [[] for _ in range(n)]
    for split_idx in range(n_splits):
        split_vals = df_design_labeled[f"split_{split_idx}"].values
        train_mask = split_vals == 0
        val_idx = np.where(split_vals == 1)[0]
        dt = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=random_state)
        dt.fit(signal_matrix[train_mask], y_conf[train_mask])
        preds = dt.predict(signal_matrix[val_idx])
        for i, p in zip(val_idx, preds):
            votes[i].append(int(p))
    assert all(len(v) > 0 for v in votes), "a patient received zero held-out votes"
    final_pred = np.array([int(np.bincount(v, minlength=3).argmax()) for v in votes])
    votes_per_patient = np.array([len(v) for v in votes])
    return final_pred, votes_per_patient


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny run to catch bugs, not for timing")
    args = parser.parse_args()

    t_start = time.time()
    n_splits = 5 if args.smoke else 100
    n_seeds = 5 if args.smoke else 10

    print("Resolving data directory...")
    data_dir = resolve_data_dir()
    print(f"Using data_dir = {data_dir}")

    cohort = load_cohort(data_dir)
    df_tab_labeled = cohort["df_tab_labeled"]
    df_design_labeled = cohort["df_design_labeled"]
    pids_labeled = cohort["pids_labeled"]
    y_binary = cohort["y_binary"]
    y_soft = cohort["y_soft"]
    confidence_annotation = cohort["confidence_annotation"]
    y_conf = cohort["y_conf"]

    print(f"Cohort: N={len(y_binary)}, yes={int((y_binary == 1).sum())}, no={int((y_binary == 0).sum())}")

    dre_categories = sorted(df_tab_labeled["dre"].unique().tolist())
    print(f"dre categories (fixed for KDM one-hot width): {dre_categories}")
    assert len(dre_categories) == 5, f"expected 5 dre categories, got {dre_categories}"

    # -----------------------------------------------------------------
    # Load exp_23's frozen Arm A/B configs and Fuzzy KNN reference (no re-sweep, no re-run)
    # -----------------------------------------------------------------
    print("\n=== Loading exp_23 frozen configs & reference (no re-sweep) ===")
    with open(EXP23_RESULTS / "best_hparams.json") as f:
        exp23_hparams = json.load(f)
    cfg_a = {k: exp23_hparams["kdm"]["hard"][k] for k in ["sigma_mult", "x_train", "y_train", "encoder"]}
    cfg_b = {k: exp23_hparams["kdm"]["soft"][k] for k in ["sigma_mult", "x_train", "y_train", "encoder"]}
    print(f"Arm A (hard, frozen from exp_23): {cfg_a}")
    print(f"Arm B (soft, frozen from exp_23): {cfg_b}")

    with open(EXP23_RESULTS / "loocv_metrics.json") as f:
        exp23_loocv = json.load(f)
    fuzzy_knn_reference = exp23_loocv["fuzzy_knn_reference"]
    df_oof_exp23 = pd.read_csv(EXP23_RESULTS / "oof_predictions.csv")
    oof_y_pred_reference = df_oof_exp23["reference_knn_pred"].values

    # -----------------------------------------------------------------
    # H3 structural check — degeneracy of the one-hot fixed point
    # -----------------------------------------------------------------
    print("\n=== H3 structural check: one-hot c_y fixed point ===")
    deg_check = degeneracy_check(df_tab_labeled, y_binary, dre_categories,
                                  encoder="linear", sigma_mult=2.0,
                                  epochs=30 if args.smoke else 300)
    print(f"Degeneracy check: {deg_check}")
    assert deg_check["passed"], (
        f"H3 structural claim failed: max c_y drift at eps=0 was {deg_check['max_cy_drift_eps0']:.3e}, "
        "expected < 1e-9 — the one-hot fixed-point argument in DESIGN.md §2.1 does not hold as stated."
    )
    with open(RESULTS_DIR / "degeneracy_check.json", "w") as f:
        json.dump(deg_check, f, indent=4)

    # -----------------------------------------------------------------
    # KDM Phase A — Arm C only
    # -----------------------------------------------------------------
    print("\n=== KDM Phase A: Arm C grid search (18 configs) ===")
    best_cfg_c, df_grid_c = kdm_phase_a_arm_c(
        df_tab_labeled, df_design_labeled, y_binary, dre_categories, n_splits=n_splits)
    print(f"Best Arm C config: {best_cfg_c}")
    df_grid_c.to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)

    with open(RESULTS_DIR / "best_hparams.json", "w") as f:
        json.dump({
            "arm_a_hard": {**cfg_a, "source": "experiments/exp_23/results/best_hparams.json (frozen, not re-swept)"},
            "arm_b_soft": {**cfg_b, "source": "experiments/exp_23/results/best_hparams.json (frozen, not re-swept)"},
            "arm_c_smoothed_hard": best_cfg_c,
        }, f, indent=4)

    print(f"\nElapsed since start: {(time.time() - t_start) / 60:.1f} min")

    # -----------------------------------------------------------------
    # KDM Phase B — all three arms, particle signals
    # -----------------------------------------------------------------
    print("\n=== KDM Phase B: LOOCV (88 folds), 3 arms, particle-set signals ===")

    print("-- Arm A (hard, frozen) --")
    res_a = kdm_phase_b_particles(
        df_tab_labeled, y_binary, y_soft, "hard",
        lambda yb, ys: to_amplitude_hard(yb),
        cfg_a, dre_categories, n_seeds=n_seeds, track_drift=False)
    print(f"Arm A metrics: {res_a['metrics']}")

    print("-- Arm B (soft, frozen) --")
    res_b = kdm_phase_b_particles(
        df_tab_labeled, y_binary, y_soft, "soft",
        lambda yb, ys: to_amplitude_soft(ys),
        cfg_b, dre_categories, n_seeds=n_seeds, track_drift=False)
    print(f"Arm B metrics: {res_b['metrics']}")

    print("-- Arm C (smoothed-hard, selected) --")
    res_c = kdm_phase_b_particles(
        df_tab_labeled, y_binary, y_soft, "hard",
        lambda yb, ys: to_amplitude_hard_smoothed(yb, best_cfg_c["eps"]),
        best_cfg_c, dre_categories, n_seeds=n_seeds, track_drift=True)
    print(f"Arm C metrics: {res_c['metrics']}")
    print(f"Arm C max c_y drift across all LOOCV folds/seeds: {res_c['max_cy_drift']:.4e}")

    print(f"\nElapsed since start: {(time.time() - t_start) / 60:.1f} min")

    # -----------------------------------------------------------------
    # H3 enforcement on Arm A's actual Phase B output (not just the isolated check)
    # -----------------------------------------------------------------
    max_h_ale_a = float(np.max(res_a["signals_mean"]["h_aleatoric"]))
    print(f"\nArm A max h_aleatoric across all 88 patients: {max_h_ale_a:.3e} (expect < 1e-6)")
    assert max_h_ale_a < 1e-6, (
        f"Arm A h_aleatoric is not degenerate as claimed: max={max_h_ale_a:.3e} — DESIGN.md §2.1/§7 "
        "must be revisited before reporting any Arm A aleatoric/epistemic row."
    )
    assert np.allclose(res_a["signals_mean"]["h_epistemic"], res_a["signals_mean"]["h_total"], atol=1e-6), (
        "Arm A h_epistemic does not equal h_total within tolerance, contradicting the claimed degeneracy."
    )

    # spot-check extract_particle_signals against model(x) directly (catches axis/sign bugs)
    with torch.no_grad():
        model_spot = build_kdm(cfg_b, n_comp=87)
        idx = np.arange(88)
        X_tr_spot, X_va_spot = build_features_fixed_categories(df_tab_labeled, idx[1:], idx[:1], dre_categories)
        c_y_spot = to_amplitude_soft(y_soft[idx[1:]])
        init_and_check(model_spot, X_tr_spot, c_y_spot, cfg_b["sigma_mult"], check_roundtrip=False)
        probs_direct = model_spot(torch.as_tensor(X_va_spot, dtype=torch.float32)).numpy()
    sig_spot = extract_particle_signals(model_spot, X_va_spot)
    assert np.allclose(sig_spot["probs"], probs_direct, atol=1e-5), "extract_particle_signals probs mismatch vs model(x)"
    h_direct = -(probs_direct * np.log(np.clip(probs_direct, 1e-7, 1))).sum(-1)
    assert np.allclose(sig_spot["h_total"], h_direct, atol=1e-5), "h_total mismatch vs H(model(x))"
    print("Spot-check passed: extract_particle_signals reproduces model(x) exactly.")

    # -----------------------------------------------------------------
    # cy_drift.png data — full-cohort fits, Arm-A-style (eps=0) vs Arm C (selected eps)
    # -----------------------------------------------------------------
    print("\n=== Full-cohort drift comparison (for cy_drift.png) ===")
    drift_a = full_cohort_drift(
        df_tab_labeled, y_binary, dre_categories,
        cfg={"encoder": "linear", "x_train": False, "y_train": True},
        c_y_tr=to_amplitude_hard_smoothed(y_binary, eps=0.0), sigma_mult=2.0, loss_arm="hard",
        epochs=30 if args.smoke else 300)
    drift_c = full_cohort_drift(
        df_tab_labeled, y_binary, dre_categories,
        cfg={"encoder": best_cfg_c["encoder"], "x_train": False, "y_train": True},
        c_y_tr=to_amplitude_hard_smoothed(y_binary, eps=best_cfg_c["eps"]),
        sigma_mult=best_cfg_c["sigma_mult"], loss_arm="hard",
        epochs=30 if args.smoke else 300)
    print(f"Full-cohort drift: eps=0 max={drift_a.max():.3e}, Arm C max={drift_c.max():.3e}")

    # -----------------------------------------------------------------
    # McNemar (Arm C vs. exp_23's recomputed Fuzzy KNN reference)
    # -----------------------------------------------------------------
    mcnemar_c_vs_knn = mcnemar_exact(y_binary, res_c["y_pred"], oof_y_pred_reference)
    print(f"\nMcNemar (Arm C vs. Fuzzy KNN reference): {mcnemar_c_vs_knn}")

    loocv_metrics = {
        "kdm_arm_a_hard": res_a["metrics"],
        "kdm_arm_b_soft": res_b["metrics"],
        "kdm_arm_c_smoothed_hard": res_c["metrics"],
        "fuzzy_knn_reference": fuzzy_knn_reference,
        "exp23_kdm_hard_reproduction_target": 0.5636363636363637,
        "exp23_kdm_soft_reproduction_target": 0.6694214876033058,
        "mcnemar_arm_c_vs_fuzzy_knn": mcnemar_c_vs_knn,
    }

    if not args.smoke:
        assert abs(res_b["metrics"]["macro_f1"] - 0.6694214876033058) < 1e-6, (
            f"Arm B (deterministic) failed to reproduce exp_23: got {res_b['metrics']['macro_f1']}"
        )
        assert abs(res_a["metrics"]["macro_f1"] - 0.5636363636363637) < 1e-4, (
            f"Arm A failed to reproduce exp_23 within tolerance: got {res_a['metrics']['macro_f1']}"
        )
        print("Arm A/B reproduction checks passed.")

    with open(RESULTS_DIR / "loocv_metrics.json", "w") as f:
        json.dump(loocv_metrics, f, indent=4)

    df_oof_signals = pd.DataFrame({"patient_id": pids_labeled, "confidence_annotation": confidence_annotation,
                                    "y_conf": y_conf, "ground_truth_biopsy": y_binary})
    for arm_name, res in [("a_hard", res_a), ("b_soft", res_b), ("c_smoothed", res_c)]:
        df_oof_signals[f"{arm_name}_p_biopsy"] = res["p_mean"]
        df_oof_signals[f"{arm_name}_pred"] = res["y_pred"]
        for s in SIGNAL_NAMES:
            df_oof_signals[f"{arm_name}_{s}"] = res["signals_mean"][s]
    df_oof_signals.to_csv(RESULTS_DIR / "oof_particle_signals.csv", index=False)

    # -----------------------------------------------------------------
    # Confidence heads
    # -----------------------------------------------------------------
    print("\n=== Secondary objective: confidence prediction from particle-set signals ===")
    # Decision-tree fits here are cheap regardless of split count (unlike the KDM training loops
    # above), and few-split runs are prone to `fit_1d_confidence_signal` degenerating (all sweep
    # predictions constant) purely from small-N instability — always use the full 100-split protocol.
    n_conf_splits = 100
    confidence_results = {}
    entropy_family = {"h_total", "h_epistemic"}
    target_informed_arms = {"b_soft": True, "a_hard": False, "c_smoothed": False}

    signal_matrices = {}
    for arm_name, res in [("a_hard", res_a), ("b_soft", res_b), ("c_smoothed", res_c)]:
        mat = np.stack([res["signals_mean"][s] for s in SIGNAL_NAMES], axis=1)
        signal_matrices[arm_name] = mat

        for s in SIGNAL_NAMES:
            raw = res["signals_mean"][s]
            signal_oof = -raw if s in entropy_family else raw
            key = f"{arm_name}__{s}"
            n_unique = len(np.unique(np.round(signal_oof, 8)))
            print(f"[{key}] range=[{signal_oof.min():.4g}, {signal_oof.max():.4g}] n_unique={n_unique}/88")
            thr = fit_1d_confidence_signal_safe(signal_oof, y_conf, df_design_labeled, n_splits=n_conf_splits)
            pred = apply_1d_confidence_signal(signal_oof, thr)
            scored = score_confidence(y_conf, pred)
            scored.update(thr)
            scored["target_informed"] = target_informed_arms[arm_name]
            confidence_results[key] = scored

        dt_pred_full, votes_full = fit_multivariate_confidence_head(
            mat, y_conf, df_design_labeled, n_splits=n_conf_splits)
        scored_full = score_confidence(y_conf, dt_pred_full)
        scored_full["target_informed"] = target_informed_arms[arm_name]
        scored_full["min_votes_per_patient"] = int(votes_full.min())
        confidence_results[f"{arm_name}__multivariate_7signal"] = scored_full

        ablation_cols = [SIGNAL_NAMES.index("h_total"), SIGNAL_NAMES.index("log_marginal")]
        mat_ablation = mat[:, ablation_cols]
        dt_pred_abl, votes_abl = fit_multivariate_confidence_head(
            mat_ablation, y_conf, df_design_labeled, n_splits=n_conf_splits)
        scored_abl = score_confidence(y_conf, dt_pred_abl)
        scored_abl["target_informed"] = target_informed_arms[arm_name]
        scored_abl["min_votes_per_patient"] = int(votes_abl.min())
        confidence_results[f"{arm_name}__ablation_h_total_log_marginal"] = scored_abl

        print(f"[{arm_name}] multivariate_7signal: {scored_full}")
        print(f"[{arm_name}] ablation: {scored_abl}")

    confidence_results["exp_17_baseline"] = {
        "macro_f1": 0.4470, "accuracy": 0.5795, "spearman_rho": 0.2790, "spearman_pvalue": 0.0085,
    }
    confidence_results["exp_23_best_nontarget_informed_baseline"] = {
        "signal": "log_marginal_hard", "macro_f1": 0.33404570881360285,
        "spearman_rho": 0.3368115081618016, "spearman_pvalue": 0.001333387130464484,
    }

    with open(RESULTS_DIR / "confidence_metrics.json", "w") as f:
        json.dump(confidence_results, f, indent=4)

    print(f"\nElapsed since start: {(time.time() - t_start) / 60:.1f} min")

    if args.smoke:
        print("\nSmoke test complete — skipping figure generation.")
        return

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    print("\n=== Generating figures ===")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, enc in zip(axes, ["identity", "linear"]):
        sub = df_grid_c[df_grid_c["encoder"] == enc]
        for eps_val, grp in sub.groupby("eps"):
            grp_sorted = grp.sort_values("sigma_mult")
            ax.plot(grp_sorted["sigma_mult"], grp_sorted["mean_macro_f1"], marker="o",
                     label=f"eps={eps_val}", alpha=0.85)
        ax.set_title(f"Arm C — encoder={enc}")
        ax.set_xlabel("sigma_mult")
        ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean Validation Macro-F1")
    plt.suptitle("Arm C (smoothed-hard) 100-Split MCCV Grid Search", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "arm_c_grid_search_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(drift_a, bins=20, alpha=0.6, label=f"eps=0 (Arm-A-style), max={drift_a.max():.1e}", color="steelblue")
    plt.hist(drift_c, bins=20, alpha=0.6, label=f"Arm C (eps={best_cfg_c['eps']}), max={drift_c.max():.1e}", color="darkorange")
    plt.xlabel(r"$\max_k |c_{y,\mathrm{after}} - c_{y,\mathrm{before}}|$ per particle")
    plt.ylabel("Count (of 88 particles)")
    plt.title("c_y Drift: One-Hot Fixed Point (eps=0) vs. Smoothed Init (Arm C)", fontweight="bold")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cy_drift.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    colors_map = {"clear": "green", "borderline": "orange", "uncertain": "red"}
    for label, color in colors_map.items():
        mask = confidence_annotation == label
        plt.scatter(res_c["signals_mean"]["h_aleatoric"][mask], res_c["signals_mean"]["h_epistemic"][mask],
                    c=color, label=label, alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.xlabel("h_aleatoric (Arm C)")
    plt.ylabel("h_epistemic (Arm C)")
    plt.title("Arm C Particle-Set Decomposition vs. Clinician Confidence", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "particle_signal_scatter.png", dpi=300)
    plt.close()

    sig_df_c = pd.DataFrame({s: res_c["signals_mean"][s] for s in SIGNAL_NAMES})
    corr = sig_df_c.corr()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson r")
    plt.xticks(range(len(SIGNAL_NAMES)), SIGNAL_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(SIGNAL_NAMES)), SIGNAL_NAMES)
    for i in range(len(SIGNAL_NAMES)):
        for j in range(len(SIGNAL_NAMES)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Arm C — 7-Signal Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "signal_correlation_heatmap.png", dpi=300)
    plt.close()

    best_key = max(
        (k for k in confidence_results if isinstance(confidence_results[k], dict) and "macro_f1" in confidence_results[k]
         and k not in ("exp_17_baseline", "exp_23_best_nontarget_informed_baseline")),
        key=lambda k: confidence_results[k]["macro_f1"],
    )
    print(f"Best confidence head overall: {best_key} -> {confidence_results[best_key]}")

    arm_name_best, sig_key_best = best_key.split("__", 1)
    mat_best = signal_matrices[arm_name_best]
    if sig_key_best == "multivariate_7signal":
        best_pred, _ = fit_multivariate_confidence_head(mat_best, y_conf, df_design_labeled, n_splits=n_conf_splits)
    elif sig_key_best == "ablation_h_total_log_marginal":
        cols = [SIGNAL_NAMES.index("h_total"), SIGNAL_NAMES.index("log_marginal")]
        best_pred, _ = fit_multivariate_confidence_head(mat_best[:, cols], y_conf, df_design_labeled, n_splits=n_conf_splits)
    else:
        s = sig_key_best
        raw = res_a["signals_mean"][s] if arm_name_best == "a_hard" else (
            res_b["signals_mean"][s] if arm_name_best == "b_soft" else res_c["signals_mean"][s])
        signal_oof = -raw if s in entropy_family else raw
        thr = {k: confidence_results[best_key][k] for k in ["meta_threshold_1", "meta_threshold_2", "direction"]}
        best_pred = apply_1d_confidence_signal(signal_oof, thr)

    cm3 = confusion_matrix(y_conf, best_pred, labels=[0, 1, 2])
    disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=["Uncertain", "Borderline", "Clear"])
    disp3.plot(cmap=plt.cm.Greens)
    plt.title(f"Confidence Prediction ({best_key}) 3-Class CM "
              f"(Macro-F1: {confidence_results[best_key]['macro_f1']:.4f})",
              fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_confusion_matrix.png", dpi=300)
    plt.close()

    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")
    print("Done. (reports/summary.md is written separately via the ml-experiment-reporter skill.)")


if __name__ == "__main__":
    main()
