import copy
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

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, brier_score_loss, roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr, binomtest
from scipy.spatial.distance import pdist

from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer
from kdm.utils import pure2dm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_23"
RESULTS_DIR = EXP_DIR / "results"
REPORTS_DIR = EXP_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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

SIGMA_MULTS = [0.25, 0.5, 1.0, 2.0]
GRID = [
    {"sigma_mult": sm, "x_train": xt, "y_train": yt, "encoder": enc}
    for sm in SIGMA_MULTS
    for xt in [True, False]
    for yt in [True, False]
    for enc in ["identity", "linear"]
]
assert len(GRID) == 32

KNN_K_LIST = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25]
KNN_WEIGHTS_LIST = ["uniform", "distance"]
KNN_METRICS_LIST = ["euclidean", "manhattan", "cosine"]


# ---------------------------------------------------------------------------
# 1.2 Load, align, and validate the cohort
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
# 1.3 Feature & target construction
# ---------------------------------------------------------------------------
def build_features(df, train_idx, val_idx):
    scaler = MinMaxScaler()
    X_tr_num = scaler.fit_transform(df.iloc[train_idx][NUM_COLS])
    X_va_num = scaler.transform(df.iloc[val_idx][NUM_COLS])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr_cat = ohe.fit_transform(df.iloc[train_idx][CAT_COLS])
    X_va_cat = ohe.transform(df.iloc[val_idx][CAT_COLS])

    X_tr = np.hstack([X_tr_num, X_tr_cat]).astype(np.float32)
    X_va = np.hstack([X_va_num, X_va_cat]).astype(np.float32)

    assert np.isfinite(X_tr).all() and np.isfinite(X_va).all(), "non-finite values in feature matrix"
    return X_tr, X_va


def build_features_fixed_categories(df, train_idx, val_idx, dre_categories):
    """Like build_features, but the OneHotEncoder is given the full known category set up
    front instead of inferring it from the training subset. exp_13's plain KNN tolerates a
    per-split-variable feature width because each split is a self-contained fit/predict pair,
    but KDM's n_comp/encoded_size are fixed per model — three of the five `dre` values are
    singletons in the 88-cohort, so 49/100 MCCV splits and 3/88 LOOCV folds have fewer than 5
    categories in their training subset under plain per-split inference, which would otherwise
    crash init_kdm_layer's shape-checked copy_ call. Used only by the KDM loops; the reference
    KNN pipeline keeps using build_features unmodified so its exp_13-reproduction check stays valid.
    """
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


def to_amplitude_hard(y_binary_subset):
    t = torch.as_tensor(y_binary_subset, dtype=torch.long)
    return F.one_hot(t, 2).float().numpy()


def to_amplitude_soft(y_soft_subset):
    y_soft_subset = np.clip(y_soft_subset, 0.0, 1.0)
    return np.stack([np.sqrt(1 - y_soft_subset), np.sqrt(y_soft_subset)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 1.4 Recomputed Fuzzy KNN reference (exp_13's exact pipeline, inline)
# ---------------------------------------------------------------------------
def knn_mccv_grid_search(df_tab_labeled, df_design_labeled, y_binary, y_soft, n_splits=100):
    grid = [{"k": k, "weights": w, "metric": m}
            for k in KNN_K_LIST for w in KNN_WEIGHTS_LIST for m in KNN_METRICS_LIST]
    grid_scores = {i: {"macro_f1": [], "acc": [], "sens": [], "spec": []} for i in range(len(grid))}

    for split_idx in range(n_splits):
        split_vals = df_design_labeled[f"split_{split_idx}"].values
        train_idx = np.where(split_vals == 0)[0]
        val_idx = np.where(split_vals == 1)[0]

        X_tr, X_va = build_features(df_tab_labeled, train_idx, val_idx)
        y_tr_soft = y_soft[train_idx]
        y_va_true = y_binary[val_idx]

        for cfg_idx, cfg in enumerate(grid):
            knn = KNeighborsRegressor(n_neighbors=cfg["k"], weights=cfg["weights"], metric=cfg["metric"])
            knn.fit(X_tr, y_tr_soft)
            p_val_soft = knn.predict(X_va)
            y_val_pred = (p_val_soft >= 0.50).astype(int)

            macro_f1 = f1_score(y_va_true, y_val_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_va_true, y_val_pred)
            tn, fp, fn, tp = confusion_matrix(y_va_true, y_val_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            grid_scores[cfg_idx]["macro_f1"].append(macro_f1)
            grid_scores[cfg_idx]["acc"].append(acc)
            grid_scores[cfg_idx]["sens"].append(sens)
            grid_scores[cfg_idx]["spec"].append(spec)

    rows = []
    for cfg_idx, cfg in enumerate(grid):
        rows.append({
            "cfg_id": cfg_idx, **cfg,
            "mean_macro_f1": float(np.mean(grid_scores[cfg_idx]["macro_f1"])),
            "std_macro_f1": float(np.std(grid_scores[cfg_idx]["macro_f1"])),
            "mean_acc": float(np.mean(grid_scores[cfg_idx]["acc"])),
            "mean_sens": float(np.mean(grid_scores[cfg_idx]["sens"])),
            "mean_spec": float(np.mean(grid_scores[cfg_idx]["spec"])),
        })
    df_grid = pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    best = df_grid.iloc[0]
    best_hparams = {"k": int(best["k"]), "weights": str(best["weights"]), "metric": str(best["metric"])}
    return best_hparams, df_grid


def knn_loocv_macrof1_and_cm(df_tab_labeled, y_binary, y_soft, k, weights, metric):
    """Runs exp_13's Phase B LOOCV loop for one fixed (k, weights, metric)."""
    loo = LeaveOneOut()
    oof_p_soft = []
    oof_y_true = []
    for train_idx, val_idx in loo.split(y_binary):
        X_tr, X_va = build_features(df_tab_labeled, train_idx, val_idx)
        y_tr_soft = y_soft[train_idx]
        knn = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric)
        knn.fit(X_tr, y_tr_soft)
        p_val_soft = knn.predict(X_va)[0]
        oof_p_soft.append(p_val_soft)
        oof_y_true.append(y_binary[val_idx[0]])

    oof_p_soft = np.array(oof_p_soft)
    oof_y_true = np.array(oof_y_true)
    oof_y_pred = (oof_p_soft >= 0.50).astype(int)

    macro_f1 = f1_score(oof_y_true, oof_y_pred, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(oof_y_true, oof_y_pred, labels=[0, 1]).ravel()
    return macro_f1, (int(tp), int(tn), int(fp), int(fn)), oof_p_soft, oof_y_pred


# ---------------------------------------------------------------------------
# 1.5 KDM model builder and training step
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
        probe = copy.deepcopy(model)
        probe.eval()
        # Narrow sigma far enough below the data's own minimum pairwise distance (in the
        # encoded space the kernel actually operates on) that each training point's own
        # prototype overwhelmingly dominates its neighbors' kernel weight. A fixed constant
        # is not safe in general: on this real clinical feature space (MinMax-scaled [0,1]^7
        # one-hot^5, ~70 points) the measured minimum pairwise distance was ~0.033, at which
        # a naively "small" sigma=0.05 still gives ~80% neighbor leakage and fails the check.
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
# 1.6 Phase A — MCCV grid search
# ---------------------------------------------------------------------------
def kdm_phase_a(df_tab_labeled, df_design_labeled, y_binary, y_soft, arm, dre_categories,
                 n_splits=100, grid=None):
    grid = GRID if grid is None else grid
    grid_scores = {i: [] for i in range(len(grid))}
    grid_extra = {i: {"acc": [], "sens": [], "spec": []} for i in range(len(grid))}

    for split_idx in range(n_splits):
        split_vals = df_design_labeled[f"split_{split_idx}"].values
        train_idx = np.where(split_vals == 0)[0]
        val_idx = np.where(split_vals == 1)[0]

        X_tr, X_va = build_features_fixed_categories(df_tab_labeled, train_idx, val_idx, dre_categories)
        y_binary_tr = y_binary[train_idx]
        y_soft_tr = y_soft[train_idx]
        y_va_true = y_binary[val_idx]

        c_y_tr = to_amplitude_hard(y_binary_tr) if arm == "hard" else to_amplitude_soft(y_soft_tr)

        for cfg_idx, cfg in enumerate(grid):
            torch.manual_seed(42)  # same init for every config within a split -> paired comparison
            model = build_kdm(cfg, n_comp=len(train_idx))
            check_rt = (split_idx == 0 and cfg_idx == 0)
            init_and_check(model, X_tr, c_y_tr, cfg["sigma_mult"], check_roundtrip=check_rt)
            train_kdm(model, X_tr, arm, y_binary_tr=y_binary_tr, y_soft_tr=y_soft_tr)

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
            print(f"    [{arm}] MCCV splits completed: {split_idx + 1}/{n_splits}")

    rows = []
    for cfg_idx, cfg in enumerate(grid):
        rows.append({
            "arm": arm, "cfg_id": cfg_idx, **cfg,
            "mean_macro_f1": float(np.mean(grid_scores[cfg_idx])),
            "std_macro_f1": float(np.std(grid_scores[cfg_idx])),
            "mean_acc": float(np.mean(grid_extra[cfg_idx]["acc"])),
            "mean_sens": float(np.mean(grid_extra[cfg_idx]["sens"])),
            "mean_spec": float(np.mean(grid_extra[cfg_idx]["spec"])),
        })
    df_grid = pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)
    best = df_grid.iloc[0]
    best_cfg = {
        "sigma_mult": float(best["sigma_mult"]), "x_train": bool(best["x_train"]),
        "y_train": bool(best["y_train"]), "encoder": str(best["encoder"]),
        "mccv_mean_macro_f1": float(best["mean_macro_f1"]), "mccv_std_macro_f1": float(best["std_macro_f1"]),
        "mccv_mean_accuracy": float(best["mean_acc"]),
    }
    return best_cfg, df_grid


# ---------------------------------------------------------------------------
# 1.7 Phase B — LOOCV, R=10 seeds
# ---------------------------------------------------------------------------
def kdm_phase_b(df_tab_labeled, y_binary, y_soft, arm, best_cfg, dre_categories, n_seeds=10):
    is_deterministic = (best_cfg["encoder"] == "identity")
    seeds = [0] if is_deterministic else list(range(n_seeds))
    n = len(y_binary)

    oof_p_soft = np.zeros((len(seeds), n))
    oof_entropy = np.zeros((len(seeds), n))
    oof_log_marginal = np.zeros((len(seeds), n))
    in_sample_signals = {} if arm == "hard" else None  # {(val_idx0, seed): (entropy_tr, log_p_x_tr, train_idx)}

    loo = LeaveOneOut()
    for train_idx, val_idx in loo.split(y_binary):
        val_idx0 = int(val_idx[0])
        X_tr, X_va = build_features_fixed_categories(df_tab_labeled, train_idx, val_idx, dre_categories)
        y_binary_tr = y_binary[train_idx]
        y_soft_tr = y_soft[train_idx]
        c_y_tr = to_amplitude_hard(y_binary_tr) if arm == "hard" else to_amplitude_soft(y_soft_tr)

        for s_idx, seed in enumerate(seeds):
            torch.manual_seed(seed)  # BEFORE build_kdm — the only randomness is nn.Linear's init inside it
            model = build_kdm(best_cfg, n_comp=len(train_idx))
            init_and_check(model, X_tr, c_y_tr, best_cfg["sigma_mult"], check_roundtrip=False)
            train_kdm(model, X_tr, arm, y_binary_tr=y_binary_tr, y_soft_tr=y_soft_tr)

            model.eval()
            with torch.no_grad():
                Xva_t = torch.as_tensor(X_va, dtype=torch.float32)
                probs_va = model(Xva_t)
                entropy_va = -(probs_va * torch.log(probs_va.clamp_min(1e-7))).sum(-1)
                log_p_x_va = model.kdm.log_marginal(pure2dm(model.encoder(Xva_t)))

            oof_p_soft[s_idx, val_idx0] = probs_va[0, 1].item()
            oof_entropy[s_idx, val_idx0] = entropy_va[0].item()
            oof_log_marginal[s_idx, val_idx0] = log_p_x_va[0].item()

            if arm == "hard":
                with torch.no_grad():
                    Xtr_t = torch.as_tensor(X_tr, dtype=torch.float32)
                    probs_tr = model(Xtr_t)
                    entropy_tr = -(probs_tr * torch.log(probs_tr.clamp_min(1e-7))).sum(-1)
                    log_p_x_tr = model.kdm.log_marginal(pure2dm(model.encoder(Xtr_t)))
                in_sample_signals[(val_idx0, seed)] = (
                    entropy_tr.numpy(), log_p_x_tr.numpy(), train_idx,
                )

        if (val_idx0 + 1) % 10 == 0:
            print(f"    [{arm}] LOOCV folds completed: {val_idx0 + 1}/{n}")

    p_mean = oof_p_soft.mean(axis=0)
    entropy_mean = oof_entropy.mean(axis=0)
    log_marginal_mean = oof_log_marginal.mean(axis=0)
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
        metrics=metrics,
        p_mean=p_mean, y_pred=y_pred,
        entropy_mean=entropy_mean, log_marginal_mean=log_marginal_mean,
        oof_entropy_per_seed=oof_entropy, oof_log_marginal_per_seed=oof_log_marginal,
        in_sample_signals=in_sample_signals,
    )


# ---------------------------------------------------------------------------
# McNemar's exact test
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
# 1.8 Secondary objective — confidence prediction from native uncertainty
# ---------------------------------------------------------------------------
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


def score_confidence(y_conf, pred):
    macro_f1 = f1_score(y_conf, pred, average="macro", zero_division=0)
    acc = accuracy_score(y_conf, pred)
    rho, pval = spearmanr(y_conf, pred)
    return {
        "macro_f1": float(macro_f1), "accuracy": float(acc),
        "spearman_rho": float(rho), "spearman_pvalue": float(pval),
        "total_cases": int(len(y_conf)),
    }


def fit_predict_2d_confidence(in_sample_signals, oof_entropy_hard, oof_log_marginal_hard, y_conf,
                               n, seeds, is_deterministic):
    preds_per_seed = -np.ones((len(seeds), n), dtype=int)
    for (val_idx0, seed), (entropy_tr, log_p_x_tr, train_idx) in in_sample_signals.items():
        s_idx = seeds.index(seed)
        X_tr2d = np.stack([entropy_tr, log_p_x_tr], axis=1)
        y_tr2d = y_conf[train_idx]
        dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
        dt.fit(X_tr2d, y_tr2d)
        x_val = np.array([[oof_entropy_hard[s_idx, val_idx0], oof_log_marginal_hard[s_idx, val_idx0]]])
        preds_per_seed[s_idx, val_idx0] = int(dt.predict(x_val)[0])

    final_pred = np.zeros(n, dtype=int)
    for i in range(n):
        vals = preds_per_seed[:, i]
        vals = vals[vals >= 0]
        counts = np.bincount(vals, minlength=3)
        final_pred[i] = int(np.argmax(counts))
    return final_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    print("Resolving data directory...")
    data_dir = resolve_data_dir()
    print(f"Using data_dir = {data_dir}")

    cohort = load_cohort(data_dir)
    df_tab_labeled = cohort["df_tab_labeled"]
    df_design_labeled = cohort["df_design_labeled"]
    pids_labeled = cohort["pids_labeled"]
    y_binary = cohort["y_binary"]
    y_soft = cohort["y_soft"]
    c_weights = cohort["c_weights"]
    confidence_annotation = cohort["confidence_annotation"]
    y_conf = cohort["y_conf"]

    print(f"Cohort: N={len(y_binary)}, yes={int((y_binary == 1).sum())}, no={int((y_binary == 0).sum())}")

    dre_categories = sorted(df_tab_labeled["dre"].unique().tolist())
    print(f"dre categories (fixed for KDM one-hot width): {dre_categories}")
    assert len(dre_categories) == 5, f"expected 5 dre categories, got {dre_categories}"

    # -----------------------------------------------------------------
    # Recomputed Fuzzy KNN reference
    # -----------------------------------------------------------------
    print("\n=== Recomputing Fuzzy KNN reference (exp_13 pipeline) ===")
    best_hparams_reference, df_grid_reference = knn_mccv_grid_search(
        df_tab_labeled, df_design_labeled, y_binary, y_soft)
    macro_f1_ref, cm_ref, oof_p_soft_reference, oof_y_pred_reference = knn_loocv_macrof1_and_cm(
        df_tab_labeled, y_binary, y_soft, **best_hparams_reference)
    tp_r, tn_r, fp_r, fn_r = cm_ref
    print(f"Reference best hparams: {best_hparams_reference}")
    print(f"Reference LOOCV Macro-F1: {macro_f1_ref:.4f}, (tp,tn,fp,fn)=({tp_r},{tn_r},{fp_r},{fn_r})")

    if abs(macro_f1_ref - 0.6364) >= 0.01:
        raise AssertionError(
            f"Recomputed Fuzzy KNN Macro-F1 {macro_f1_ref:.4f} diverges from exp_13's published 0.6364 "
            "— data-path substitution not input-identical")

    if cm_ref != (40, 18, 16, 14):
        print(f"MISMATCH: selected config = {best_hparams_reference}, got (tp,tn,fp,fn)={cm_ref}")
        print(df_grid_reference.head(5).to_string())
        _, cm_forced, _, _ = knn_loocv_macrof1_and_cm(
            df_tab_labeled, y_binary, y_soft, k=1, weights="uniform", metric="euclidean")
        if cm_forced != (40, 18, 16, 14):
            raise AssertionError(
                f"reference confusion matrix mismatch even with exp_13's exact published config forced: "
                f"{cm_forced} — data-path substitution not input-identical")
        print("Forced exp_13 config reproduces (40,18,16,14) — mismatch was benign tie-breaking, continuing.")

    sens_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0
    spec_r = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0
    reference_metrics = {
        "macro_f1": float(macro_f1_ref), "accuracy": float((tp_r + tn_r) / 88),
        "sensitivity": float(sens_r), "specificity": float(spec_r),
        "auroc": float(roc_auc_score(y_binary, oof_p_soft_reference)),
        "brier_score": float(brier_score_loss(y_binary, oof_p_soft_reference)),
        "tp": tp_r, "tn": tn_r, "fp": fp_r, "fn": fn_r, "total_cases": 88,
        "best_hparams": best_hparams_reference,
        "published_reference_macro_f1": 0.6364,
    }

    # -----------------------------------------------------------------
    # KDM Phase A
    # -----------------------------------------------------------------
    print("\n=== KDM Phase A: MCCV grid search (32 configs x 2 arms) ===")
    best_cfg = {}
    df_grid_kdm_all = []
    for arm in ["hard", "soft"]:
        print(f"-- Arm: {arm} --")
        bc, df_grid_arm = kdm_phase_a(df_tab_labeled, df_design_labeled, y_binary, y_soft, arm, dre_categories)
        best_cfg[arm] = bc
        df_grid_kdm_all.append(df_grid_arm)
        print(f"Best config for {arm}: {bc}")

    df_grid_kdm = pd.concat(df_grid_kdm_all, ignore_index=True)
    df_grid_kdm.to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)

    with open(RESULTS_DIR / "best_hparams.json", "w") as f:
        json.dump({"kdm": best_cfg, "fuzzy_knn_reference": best_hparams_reference}, f, indent=4)

    print(f"\nElapsed since start: {(time.time() - t_start) / 60:.1f} min")

    # -----------------------------------------------------------------
    # KDM Phase B
    # -----------------------------------------------------------------
    print("\n=== KDM Phase B: LOOCV (88 folds), R=10 seeds ===")
    phase_b_results = {}
    for arm in ["hard", "soft"]:
        print(f"-- Arm: {arm} --")
        res = kdm_phase_b(df_tab_labeled, y_binary, y_soft, arm, best_cfg[arm], dre_categories, n_seeds=10)
        phase_b_results[arm] = res
        print(f"{arm} LOOCV metrics: {res['metrics']}")

    print(f"\nElapsed since start: {(time.time() - t_start) / 60:.1f} min")

    # -----------------------------------------------------------------
    # McNemar's test (both arms vs. reference)
    # -----------------------------------------------------------------
    mcnemar_results = {}
    for arm in ["hard", "soft"]:
        mcnemar_results[arm] = mcnemar_exact(y_binary, phase_b_results[arm]["y_pred"], oof_y_pred_reference)
    print(f"McNemar results (KDM arm vs. recomputed Fuzzy KNN reference): {mcnemar_results}")

    loocv_metrics = {
        "kdm_hard": phase_b_results["hard"]["metrics"],
        "kdm_soft": phase_b_results["soft"]["metrics"],
        "fuzzy_knn_reference": reference_metrics,
        "mcnemar": mcnemar_results,
    }
    with open(RESULTS_DIR / "loocv_metrics.json", "w") as f:
        json.dump(loocv_metrics, f, indent=4)

    df_oof = pd.DataFrame({
        "patient_id": pids_labeled,
        "ground_truth_biopsy": y_binary,
        "confidence_annotation": confidence_annotation,
        "certainty_weight": c_weights,
        "kdm_hard_p_mean": phase_b_results["hard"]["p_mean"],
        "kdm_hard_pred": phase_b_results["hard"]["y_pred"],
        "kdm_soft_p_mean": phase_b_results["soft"]["p_mean"],
        "kdm_soft_pred": phase_b_results["soft"]["y_pred"],
        "reference_knn_p": oof_p_soft_reference,
        "reference_knn_pred": oof_y_pred_reference,
    })
    df_oof.to_csv(RESULTS_DIR / "oof_predictions.csv", index=False)

    # -----------------------------------------------------------------
    # Secondary objective: confidence prediction from native uncertainty
    # -----------------------------------------------------------------
    print("\n=== Secondary objective: confidence prediction from native uncertainty ===")
    confidence_results = {}

    signals = {
        "entropy_hard": -phase_b_results["hard"]["entropy_mean"],
        "log_marginal_hard": phase_b_results["hard"]["log_marginal_mean"],
        "entropy_soft": -phase_b_results["soft"]["entropy_mean"],
        "log_marginal_soft": phase_b_results["soft"]["log_marginal_mean"],
    }
    target_informed = {
        "entropy_hard": False, "log_marginal_hard": False,
        "entropy_soft": True, "log_marginal_soft": True,
    }

    for name, signal_oof in signals.items():
        thr = fit_1d_confidence_signal(signal_oof, y_conf, df_design_labeled)
        pred = apply_1d_confidence_signal(signal_oof, thr)
        scored = score_confidence(y_conf, pred)
        scored.update(thr)
        scored["target_informed"] = target_informed[name]
        confidence_results[name] = scored
        print(f"{name}: {scored}")

    hard_seeds = [0] if phase_b_results["hard"]["metrics"]["deterministic"] else list(range(10))
    pred_2d = fit_predict_2d_confidence(
        phase_b_results["hard"]["in_sample_signals"],
        phase_b_results["hard"]["oof_entropy_per_seed"],
        phase_b_results["hard"]["oof_log_marginal_per_seed"],
        y_conf, n=len(y_binary), seeds=hard_seeds,
        is_deterministic=phase_b_results["hard"]["metrics"]["deterministic"],
    )
    scored_2d = score_confidence(y_conf, pred_2d)
    scored_2d["target_informed"] = False
    confidence_results["joint_2d_hard"] = scored_2d
    print(f"joint_2d_hard: {scored_2d}")

    confidence_results["exp_17_baseline"] = {
        "macro_f1": 0.4470, "accuracy": 0.5795, "spearman_rho": 0.2790, "spearman_pvalue": 0.0085,
    }

    with open(RESULTS_DIR / "confidence_metrics.json", "w") as f:
        json.dump(confidence_results, f, indent=4)

    print(f"\nElapsed since start: {(time.time() - t_start) / 60:.1f} min")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    print("\n=== Generating figures ===")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, arm in zip(axes, ["hard", "soft"]):
        sub_all = df_grid_kdm[df_grid_kdm["arm"] == arm]
        for (xt, yt, enc), grp in sub_all.groupby(["x_train", "y_train", "encoder"]):
            grp_sorted = grp.sort_values("sigma_mult")
            ax.plot(grp_sorted["sigma_mult"], grp_sorted["mean_macro_f1"], marker="o",
                     label=f"x={xt},y={yt},{enc}", alpha=0.8)
        ax.set_title(f"Arm: {arm}")
        ax.set_xlabel("sigma_mult")
        ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Mean Validation Macro-F1")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.suptitle("Tabular KDM 100-Split MCCV Grid Search Performance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "grid_search_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    best_arm = "hard" if phase_b_results["hard"]["metrics"]["macro_f1"] >= phase_b_results["soft"]["metrics"]["macro_f1"] else "soft"
    m = phase_b_results[best_arm]["metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Biopsy", "Biopsy"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Tabular KDM ({best_arm}) LOOCV Confusion Matrix (Macro-F1: {m['macro_f1']:.4f})",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    for arm, color in [("hard", "darkorange"), ("soft", "green")]:
        fpr, tpr, _ = roc_curve(y_binary, phase_b_results[arm]["p_mean"])
        auc_val = phase_b_results[arm]["metrics"]["auroc"]
        plt.plot(fpr, tpr, color=color, lw=2, label=f"KDM {arm} (AUC={auc_val:.4f})")
    fpr_r, tpr_r, _ = roc_curve(y_binary, oof_p_soft_reference)
    plt.plot(fpr_r, tpr_r, color="navy", lw=2, linestyle="-.",
              label=f"Fuzzy KNN reference (AUC={reference_metrics['auroc']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("KDM vs. Recomputed Fuzzy KNN — LOOCV ROC", fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.close()

    signal_names_1d = list(signals.keys())
    best_signal_name = max(signal_names_1d, key=lambda nm: confidence_results[nm]["macro_f1"])
    best_signal_oof = signals[best_signal_name]
    best_thr = {k: confidence_results[best_signal_name][k]
                for k in ["meta_threshold_1", "meta_threshold_2", "direction"]}
    best_pred = apply_1d_confidence_signal(best_signal_oof, best_thr)
    cm3 = confusion_matrix(y_conf, best_pred, labels=[0, 1, 2])
    disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=["Uncertain", "Borderline", "Clear"])
    disp3.plot(cmap=plt.cm.Greens)
    plt.title(f"Confidence Prediction ({best_signal_name}) 3-Class CM "
              f"(Macro-F1: {confidence_results[best_signal_name]['macro_f1']:.4f})",
              fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confidence_confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    colors_map = {"clear": "green", "borderline": "orange", "uncertain": "red"}
    for label, color in colors_map.items():
        mask = confidence_annotation == label
        plt.scatter(
            phase_b_results["hard"]["entropy_mean"][mask],
            phase_b_results["hard"]["log_marginal_mean"][mask],
            c=color, label=label, alpha=0.7, edgecolors="k", linewidths=0.3,
        )
    plt.xlabel("Predictive Entropy (Arm A / hard)")
    plt.ylabel("log P(x) — log_marginal (Arm A / hard)")
    plt.title("KDM Native Uncertainty vs. Clinician Confidence Annotation", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "uncertainty_scatter.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
    # summary.md
    # -----------------------------------------------------------------
    print("\n=== Writing summary report ===")
    best_arm_metrics = phase_b_results[best_arm]["metrics"]

    summary_path = REPORTS_DIR / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# Tabular KDM Biopsy Decision Prediction (exp_23) Summary Report\n\n")
        f.write("**Date**: 2026-08-18  \n")
        f.write("**Model**: `KDMClassModel` (Kernel Density Matrix classifier), `n_comp = n_train`, "
                "two target arms (hard / uncertainty-guided soft)  \n")
        f.write(r"**Dataset**: Labeled Complete-Case Tabular Clinical Data "
                r"($N_{\mathrm{labeled}} = 88$, 54 yes / 34 no)  " + "\n\n")

        f.write("## Phase A: 100-Split MCCV Grid Search (32 configs x 2 arms)\n")
        for arm in ["hard", "soft"]:
            bc = best_cfg[arm]
            f.write(f"- **Arm `{arm}`** best config: `sigma_mult={bc['sigma_mult']}, x_train={bc['x_train']}, "
                    f"y_train={bc['y_train']}, encoder={bc['encoder']}` — Mean Validation Macro-F1: "
                    f"**{bc['mccv_mean_macro_f1']:.4f}** (std={bc['mccv_std_macro_f1']:.4f})  \n")
        f.write(f"- **Fuzzy KNN reference** best config: `{best_hparams_reference}` — recomputed inline on "
                f"identical rows/splits/folds.  \n\n")

        f.write("## Phase B: LOOCV (88 folds), R=10 seeds\n")
        f.write("| Model | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Brier | Deterministic |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for arm in ["hard", "soft"]:
            mm = phase_b_results[arm]["metrics"]
            f.write(f"| KDM `{arm}` | **{mm['macro_f1']:.4f}** | {mm['accuracy'] * 100:.2f}% | "
                    f"{mm['sensitivity']:.4f} | {mm['specificity']:.4f} | {mm['auroc']:.4f} | "
                    f"{mm['brier_score']:.4f} | {mm['deterministic']} |\n")
        rm = reference_metrics
        f.write(f"| Fuzzy KNN reference (recomputed) | {rm['macro_f1']:.4f} | {rm['accuracy'] * 100:.2f}% | "
                f"{rm['sensitivity']:.4f} | {rm['specificity']:.4f} | {rm['auroc']:.4f} | "
                f"{rm['brier_score']:.4f} | — |\n")
        f.write("| exp_13 published (other checkout, reference only) | 0.6364 | 65.91% | 0.7407 | 0.5294 | "
                "0.6304 | 0.2908 | — |\n\n")

        f.write(f"### 2x2 Confusion Matrix (best KDM arm: `{best_arm}`)\n")
        f.write("| Ground Truth \\ Predicted | No Biopsy | Biopsy |\n")
        f.write("|:---|:---:|:---:|\n")
        f.write(f"| **No Biopsy** ($N=34$) | **{best_arm_metrics['tn']}** | {best_arm_metrics['fp']} |\n")
        f.write(f"| **Biopsy** ($N=54$) | {best_arm_metrics['fn']} | **{best_arm_metrics['tp']}** |\n\n")

        f.write("### McNemar's Exact Test (KDM vs. recomputed Fuzzy KNN reference — both arms reported, "
                "two comparisons)\n")
        f.write("| Arm | Discordant b (KDM right/KNN wrong) | Discordant c (KDM wrong/KNN right) | p-value |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        for arm in ["hard", "soft"]:
            mc = mcnemar_results[arm]
            f.write(f"| `{arm}` | {mc['b']} | {mc['c']} | {mc['pvalue']:.4f} |\n")
        f.write("\n")

        f.write("**Note on the soft-target ceiling**: only the 32 non-`clear` patients can differ between "
                "the hard and soft arms (ỹ takes 6 distinct values, mostly 0/1) — a small Arm A/B gap should "
                "not be over-read.\n\n")

        f.write("## Secondary Objective: Diagnostic Confidence from Native Uncertainty\n")
        f.write("| Signal | Target-informed | Macro-F1 | Accuracy | Spearman rho | p-value | Direction | "
                "Fallback/100 |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for name in signal_names_1d:
            r = confidence_results[name]
            f.write(f"| `{name}` | {r['target_informed']} | {r['macro_f1']:.4f} | {r['accuracy'] * 100:.2f}% | "
                    f"{r['spearman_rho']:.4f} | {r['spearman_pvalue']:.4e} | {r['direction']:+d} | "
                    f"{r['fallback_count_of_100']} |\n")
        r2d = confidence_results["joint_2d_hard"]
        f.write(f"| `joint_2d_hard` (local-fit, exp_12/22 pattern) | False | {r2d['macro_f1']:.4f} | "
                f"{r2d['accuracy'] * 100:.2f}% | {r2d['spearman_rho']:.4f} | {r2d['spearman_pvalue']:.4e} | "
                f"— | — |\n")
        f.write("| **`exp_17` baseline (Composite Fuzzy ICI)** | — | **0.4470** | **57.95%** | **0.2790** | "
                "0.0085 | — | — |\n\n")

        f.write("## Known-Pitfall Checklist (see DESIGN.md Sec.6)\n")
        f.write("- Class totals reported as 54 yes / 34 no throughout (not exp_13's incorrect 56/32).\n")
        f.write("- `results/git_commit.txt` written before this run.\n")
        f.write("- Amplitude round-trip assertion passed during Phase A (checked once per arm, "
                "split 0 / config 0).\n")

    print(f"\nSummary report written to: {summary_path}")
    print(f"\nTotal elapsed: {(time.time() - t_start) / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
