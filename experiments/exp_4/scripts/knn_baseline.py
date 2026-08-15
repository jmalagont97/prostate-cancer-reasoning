# KNN baseline for CHIMERA Task 1.1 (exp_4) — feature-subset ablation.
# Same pipeline as exp_3, with ONE controlled modification: variables with >50%
# missingness on the usable_labeled cohort are dropped (essential clinical vars
# always retained). 128 configs x 50 frozen MCCV splits. Selection by Macro-F1.
# LOO (88 folds) for the selected config as a sanity check.
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "chimera26" / "preprocessed" / "task1"
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_4"
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "reports" / "figures"
SELECTED_DIR = RESULTS_DIR / "selected_config"

INPUTS_TABULAR = DATA_DIR / "inputs_tabular.csv"
GROUND_TRUTH = DATA_DIR / "ground_truth.csv"
SPLITS = DATA_DIR / "mccv_loocv_splits.csv"

N_SPLITS = 50
N_JOBS = 22
SEED = 42
THRESHOLD = 0.5
EPS = 1e-12
MISSINGNESS_THRESHOLD = 0.5

CONTINUOUS_COLS = [
    "cli_age", "cli_psa", "cli_psap", "cli_psav", "cli_psad", "cli_vol",
    "cli_months", "cli_pirads", "cli_cspca", "cli_comorbidity_count",
    "cli_allergies_count", "cli_ipss_score", "vit_weight_kg", "vit_height_cm",
    "vit_bmi", "vit_bp_systolic", "vit_bp_diastolic", "vit_heart_rate_bpm",
    "vit_smoking_pack_years", "path_hist_bx_isup", "path_hist_bx_gl_prim",
    "path_hist_bx_gl_sec", "path_hist_bx_gl_tert", "cli_fh_binary",
    "psa_tr_count", "psa_tr_first_val", "psa_tr_last_val", "psa_tr_min",
    "psa_tr_max", "psa_tr_mean", "psa_tr_delta", "psa_tr_slope",
    "lab_creatinine_mg_dl", "lab_hemoglobin_g_dl", "lab_free_psa_ng_ml",
    "lab_free_total_ratio",
]
CATEGORICAL_COLS = ["cli_bx", "cli_dre", "vit_smoking_status"]
ALL_FEATURES = CONTINUOUS_COLS + CATEGORICAL_COLS

ESSENTIAL_FEATURES = {
    "cli_age", "cli_fh_binary", "cli_cspca", "cli_pirads", "cli_vol",
    "cli_psa", "cli_comorbidity_count", "cli_psad", "cli_dre", "cli_bx",
}

CONFIDENCE_MAP = {2.0: 1.0, 1.0: 0.5, 0.0: 0.25}
DISTANCES = ["euclidean", "manhattan", "minkowski_p3", "cosine"]
RULES = ["rigid", "fuzzy_confidence"]
WEIGHTS = ["uniform", "inverse_distance"]
KS = [1, 3, 5, 7, 9, 11, 15, 21]

METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "precision_yes", "F1_yes", "F1_no", "Macro_F1", "MCC", "Brier",
    "ROC_AUC", "PR_AUC",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def build_grid() -> list[dict]:
    grid = []
    for d in DISTANCES:
        for r in RULES:
            for w in WEIGHTS:
                for k in KS:
                    grid.append({"distance": d, "rule": r, "weight": w, "k": k})
    return grid


def config_id(cfg: dict) -> str:
    return f"{cfg['distance']}_{cfg['rule']}_{cfg['weight']}_k{cfg['k']}"


def load_data() -> pd.DataFrame:
    feats = read_csv(INPUTS_TABULAR)
    gt = read_csv(GROUND_TRUTH)[["case_id", "target_biopsy_decision", "target_confidence_code"]]
    splits = read_csv(SPLITS)

    u = splits.loc[splits["cohort_status"].eq("usable_labeled")].copy()
    u = u.merge(feats, on="case_id", how="left", validate="one_to_one")
    u = u.merge(gt, on="case_id", how="left", validate="one_to_one")

    y = u["target_biopsy_decision"].map({"yes": 1, "no": 0}).astype(int)
    conf_code = pd.to_numeric(u["target_confidence_code"], errors="coerce")
    conf_w = conf_code.map(CONFIDENCE_MAP).astype(float)

    assert len(u) == 88, f"usable_labeled != 88: {len(u)}"
    assert y.value_counts().to_dict() == {1: 54, 0: 34}, y.value_counts().to_dict()
    for i in range(N_SPLITS):
        col = f"mccv_split_{i:02d}"
        n_val = int(u[col].eq("1").sum())
        assert n_val == 18, f"{col}: val != 18 ({n_val})"
    assert u["loocv_fold"].nunique() == 88

    u["y"] = y.to_numpy()
    u["conf_w"] = conf_w.to_numpy()
    return u.reset_index(drop=True)


def build_feature_mask(dfu: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    n = len(dfu)
    rows = []
    removed, retained = [], []
    for c in ALL_FEATURES:
        n_missing = int(dfu[c].eq("").sum())
        rate = n_missing / n
        is_essential = c in ESSENTIAL_FEATURES
        keep = (rate <= MISSINGNESS_THRESHOLD) or is_essential
        rows.append({
            "variable": c, "n_missing": n_missing, "missing_rate": rate,
            "is_essential": is_essential, "retained": keep,
        })
        if keep:
            retained.append(c)
        else:
            removed.append(c)
    miss_df = pd.DataFrame(rows)
    assert removed == ["path_hist_bx_gl_tert", "lab_hemoglobin_g_dl"], removed
    assert len(retained) == 37, f"retained != 37: {len(retained)}"
    assert all(c in retained for c in ESSENTIAL_FEATURES)
    assert all(r > MISSINGNESS_THRESHOLD for r in miss_df.loc[
        miss_df["variable"].isin(removed), "missing_rate"])
    return retained, removed, miss_df


def write_feature_selection_manifest(retained, removed) -> None:
    pd.DataFrame([{
        "rule": "drop if missing_rate > 0.5 on usable_labeled (88), unless essential",
        "threshold": MISSINGNESS_THRESHOLD,
        "cohort": "usable_labeled", "n_cases": 88,
        "n_features_total": len(ALL_FEATURES), "n_retained": len(retained),
        "n_removed": len(removed), "n_essential": len(ESSENTIAL_FEATURES),
        "removed_features": ";".join(removed),
        "retained_features": ";".join(retained),
        "essential_features": ";".join(sorted(ESSENTIAL_FEATURES)),
    }]).to_csv(RESULTS_DIR / "feature_selection_manifest.csv", index=False)


class TabularTransformer:
    def __init__(self, continuous_cols: list[str], categorical_cols: list[str],
                 all_used_cols: list[str]):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.all_used_cols = all_used_cols
        self._cont_min: dict[str, float] = {}
        self._cont_max: dict[str, float] = {}
        self._cat_cats: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "TabularTransformer":
        for c in self.continuous_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            obs = v.dropna()
            if len(obs) == 0:
                self._cont_min[c] = float("nan")
                self._cont_max[c] = float("nan")
            else:
                self._cont_min[c] = float(obs.min())
                self._cont_max[c] = float(obs.max())
        for c in self.categorical_cols:
            s = df[c].astype(str)
            cats = sorted(s.loc[~s.eq("")].unique())
            self._cat_cats[c] = cats
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        n = len(df)
        blocks: list[np.ndarray] = []
        cols: list[str] = []

        for c in self.continuous_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            observed = ~np.isnan(v)
            out = np.zeros(n, dtype=float)
            mn = self._cont_min[c]
            mx = self._cont_max[c]
            if not np.isnan(mn):
                if mx > mn:
                    scaled = (v - mn) / (mx - mn)
                    out[observed] = np.clip(scaled[observed], 0.0, 1.0)
                else:
                    out[observed] = 1.0
            blocks.append(out)
            cols.append(c)

        for c in self.categorical_cols:
            s = df[c].astype(str).to_numpy()
            for cat in self._cat_cats[c]:
                blocks.append((s == cat).astype(float))
                cols.append(f"{c}={cat}")

        for c in self.all_used_cols:
            if c in self.continuous_cols:
                m = np.isnan(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
            else:
                m = df[c].astype(str).eq("").to_numpy()
            blocks.append(m.astype(float))
            cols.append(f"missing_{c}")

        X = np.column_stack(blocks).astype(np.float64)
        assert not np.isnan(X).any(), "NaN in transformed matrix"
        assert np.isfinite(X).all(), "non-finite in transformed matrix"
        return X, cols


def compute_distances(Xva: np.ndarray, Xtr: np.ndarray, distance: str) -> np.ndarray:
    if distance == "euclidean":
        return cdist(Xva, Xtr, metric="euclidean")
    if distance == "manhattan":
        return cdist(Xva, Xtr, metric="cityblock")
    if distance == "minkowski_p3":
        return cdist(Xva, Xtr, metric="minkowski", p=3)
    if distance == "cosine":
        nrm_v = np.sqrt((Xva ** 2).sum(axis=1))
        nrm_t = np.sqrt((Xtr ** 2).sum(axis=1))
        assert (nrm_v > 0).all() and (nrm_t > 0).all(), "zero-norm row for cosine"
        return cdist(Xva, Xtr, metric="cosine")
    raise ValueError(distance)


def knn_predict(
    Xtr: np.ndarray, Xva: np.ndarray, ytr: np.ndarray, conf_tr: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    k = cfg["k"]
    dmat = compute_distances(Xva, Xtr, cfg["distance"])
    idx = np.argsort(dmat, axis=1, kind="stable")[:, :k]
    d_k = np.take_along_axis(dmat, idx, axis=1)
    if cfg["weight"] == "uniform":
        dw = np.ones_like(d_k)
    else:
        dw = 1.0 / np.maximum(d_k, EPS)
    if cfg["rule"] == "fuzzy_confidence":
        dw = dw * conf_tr[idx]
    y_k = ytr[idx]
    num = np.sum(dw * y_k, axis=1)
    den = np.sum(dw, axis=1)
    p_yes = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    y_pred = (p_yes >= THRESHOLD).astype(int)
    return p_yes, y_pred


def fold_metrics(y_true: np.ndarray, p_yes: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    ba = balanced_accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec_yes = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_yes = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_no = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    macro = (f1_yes + f1_no) / 2.0
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, p_yes)
    if len(np.unique(y_true)) > 1 and len(np.unique(p_yes)) > 1:
        roc = roc_auc_score(y_true, p_yes)
        prauc = average_precision_score(y_true, p_yes)
    else:
        roc, prauc = float("nan"), float("nan")
    return {
        "accuracy": acc, "balanced_accuracy": ba, "sensitivity": sens,
        "specificity": spec, "precision_yes": prec_yes, "F1_yes": f1_yes,
        "F1_no": f1_no, "Macro_F1": macro, "MCC": mcc, "Brier": brier,
        "ROC_AUC": roc, "PR_AUC": prauc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _transform_fold(df_train: pd.DataFrame, df_val: pd.DataFrame):
    tf = TabularTransformer(CONTINUOUS_USED, CATEGORICAL_USED, ALL_FEATURES_USED).fit(df_train)
    Xtr, _ = tf.transform(df_train)
    Xva, _ = tf.transform(df_val)
    return Xtr, Xva


def run_mccv_config(cfg: dict, dfu: pd.DataFrame) -> dict:
    t0 = time.time()
    cid = config_id(cfg)
    fold_metrics_rows = []
    oof_rows = []
    cm_rows = []
    for i in range(N_SPLITS):
        col = f"mccv_split_{i:02d}"
        val_mask = dfu[col].eq("1").to_numpy()
        train_mask = dfu[col].eq("0").to_numpy()
        df_tr = dfu.loc[train_mask]
        df_va = dfu.loc[val_mask]
        Xtr, Xva = _transform_fold(df_tr, df_va)
        p_yes, y_pred = knn_predict(
            Xtr, Xva, df_tr["y"].to_numpy(), df_tr["conf_w"].to_numpy(), cfg
        )
        y_va = df_va["y"].to_numpy()
        m = fold_metrics(y_va, p_yes, y_pred)
        fold_metrics_rows.append({"config_id": cid, "split_id": i, **m})
        case_ids = df_va["case_id"].to_numpy()
        oof_rows.extend(
            zip([cid] * len(case_ids), [i] * len(case_ids), case_ids,
                y_va.tolist(), p_yes.tolist(), y_pred.tolist())
        )
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        cm_rows.append((cid, i, 0, 0, tn))
        cm_rows.append((cid, i, 0, 1, fp))
        cm_rows.append((cid, i, 1, 0, fn))
        cm_rows.append((cid, i, 1, 1, tp))
    return {
        "config_id": cid, "fold_metrics_rows": fold_metrics_rows,
        "oof_rows": oof_rows, "cm_rows": cm_rows, "elapsed_sec": time.time() - t0,
    }


def summarize_folds(fold_metrics_rows: list[dict]) -> dict:
    df = pd.DataFrame(fold_metrics_rows)
    out = {}
    for name in METRIC_NAMES:
        vals = df[name]
        out[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_valid": int(vals.notna().sum()),
        }
    return out


def run_baselines(dfu: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bname, bval in [("always_yes", 1), ("always_no", 0)]:
        for i in range(N_SPLITS):
            col = f"mccv_split_{i:02d}"
            val_mask = dfu[col].eq("1").to_numpy()
            y_va = dfu.loc[val_mask, "y"].to_numpy()
            y_pred = np.full(len(y_va), bval, dtype=int)
            p = np.full(len(y_va), float(bval))
            m = fold_metrics(y_va, p, y_pred)
            rows.append({"baseline": bname, "protocol": "mccv", "split_id": i, **m})
        y_all = dfu["y"].to_numpy()
        y_pred = np.full(len(y_all), bval, dtype=int)
        p = np.full(len(y_all), float(bval))
        m = fold_metrics(y_all, p, y_pred)
        rows.append({"baseline": bname, "protocol": "loo_pooled", "split_id": -1, **m})
    return pd.DataFrame(rows)


def run_loo(dfu: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cid = config_id(cfg)
    rows = []
    pooled_t = []
    pooled_p = []
    for fold in range(88):
        case_id_f = dfu.loc[dfu["loocv_fold"].astype(int).eq(fold), "case_id"].iloc[0]
        val_mask = dfu["case_id"].eq(case_id_f).to_numpy()
        train_mask = ~val_mask
        df_tr = dfu.loc[train_mask]
        df_va = dfu.loc[val_mask]
        Xtr, Xva = _transform_fold(df_tr, df_va)
        p_yes, y_pred = knn_predict(
            Xtr, Xva, df_tr["y"].to_numpy(), df_tr["conf_w"].to_numpy(), cfg
        )
        y_va = df_va["y"].to_numpy()
        m = fold_metrics(y_va, p_yes, y_pred)
        rows.append({"config_id": cid, "fold": fold, "case_id": case_id_f,
                     "y_true": int(y_va[0]), "p_yes": float(p_yes[0]),
                     "y_pred": int(y_pred[0]), **m})
        pooled_t.append(int(y_va[0]))
        pooled_p.append(float(p_yes[0]))
    pooled_t = np.array(pooled_t)
    pooled_p = np.array(pooled_p)
    pooled_metrics = {
        "ROC_AUC": float(roc_auc_score(pooled_t, pooled_p)),
        "PR_AUC": float(average_precision_score(pooled_t, pooled_p)),
    }
    loo_df = pd.DataFrame(rows)
    cm = confusion_matrix(pooled_t, loo_df["y_pred"].to_numpy(), labels=[0, 1])
    cm_df = pd.DataFrame(
        [(cid, tr, pr, int(cm[tr, pr])) for tr in range(2) for pr in range(2)],
        columns=["config_id", "true_label", "pred_label", "count"],
    )
    return loo_df, cm_df, pooled_metrics


def write_data_manifest(dfu: pd.DataFrame, grid_meta: dict) -> None:
    rows = []
    for path in [INPUTS_TABULAR, GROUND_TRUTH, SPLITS]:
        rows.append({
            "kind": "file", "key": path.name,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size, "sha256": sha256(path),
        })
    for key, val in grid_meta.items():
        rows.append({"kind": "param", "key": key, "value": str(val)})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "data_manifest.csv", index=False)


def main() -> None:
    warnings.filterwarnings("ignore", message="A single label was found")
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    global CONTINUOUS_USED, CATEGORICAL_USED, ALL_FEATURES_USED
    dfu = load_data()
    retained, removed, miss_df = build_feature_mask(dfu)
    CONTINUOUS_USED = [c for c in CONTINUOUS_COLS if c in retained]
    CATEGORICAL_USED = [c for c in CATEGORICAL_COLS if c in retained]
    ALL_FEATURES_USED = CONTINUOUS_USED + CATEGORICAL_USED
    assert len(CONTINUOUS_USED) == 34, len(CONTINUOUS_USED)
    assert len(CATEGORICAL_USED) == 3, len(CATEGORICAL_USED)
    assert len(ALL_FEATURES_USED) == 37, len(ALL_FEATURES_USED)

    miss_df.to_csv(RESULTS_DIR / "feature_missingness.csv", index=False)
    write_feature_selection_manifest(retained, removed)
    print(f"[filter] removed={removed} retained={len(retained)}")

    grid = build_grid()
    assert len(grid) == 128

    grid_meta = {
        "cohort": "usable_labeled", "n_cases": 88, "n_yes": 54, "n_no": 34,
        "n_mccv_splits": 50, "n_loo_folds": 88, "threshold": THRESHOLD,
        "selection_metric": "Macro_F1", "tiebreak_1": "F1_yes",
        "tiebreak_2": "balanced_accuracy", "n_configs": len(grid),
        "n_features": len(ALL_FEATURES_USED),
        "n_features_total": len(ALL_FEATURES),
        "continuous_cols": len(CONTINUOUS_USED), "categorical_cols": len(CATEGORICAL_USED),
        "imputation": "none", "pruning": "missingness_gt50",
        "missingness_threshold": MISSINGNESS_THRESHOLD,
        "removed_features": ";".join(removed),
        "essential_features": ";".join(sorted(ESSENTIAL_FEATURES)),
        "confidence_map": json.dumps(CONFIDENCE_MAP),
    }
    write_data_manifest(dfu, grid_meta)
    print(f"[data] usable={len(dfu)} configs={len(grid)} features={len(ALL_FEATURES_USED)}")

    print("[mccv] running 128 configs x 50 splits ...")
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_mccv_config)(cfg, dfu) for cfg in grid
    )
    print(f"[mccv] done in {time.time() - t0:.1f}s")

    fold_rows = []
    oof_rows = []
    cm_rows = []
    summary_rows = []
    for r in results:
        fold_rows.extend(r["fold_metrics_rows"])
        oof_rows.extend(r["oof_rows"])
        cm_rows.extend(r["cm_rows"])
        summ = summarize_folds(r["fold_metrics_rows"])
        cfg = next(c for c in grid if config_id(c) == r["config_id"])
        summary_rows.append({
            "config_id": r["config_id"], "distance": cfg["distance"],
            "rule": cfg["rule"], "weight": cfg["weight"], "k": cfg["k"],
            **{f"{k}_mean": v["mean"] for k, v in summ.items()},
            **{f"{k}_std": v["std"] for k, v in summ.items()},
            **{f"{k}_min": v["min"] for k, v in summ.items()},
            **{f"{k}_max": v["max"] for k, v in summ.items()},
            **{f"{k}_n_valid": v["n_valid"] for k, v in summ.items()},
            "elapsed_sec": r["elapsed_sec"],
        })

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows, columns=["config_id", "split_id", "case_id",
                                             "y_true", "p_yes", "y_pred"])
    cm_df = pd.DataFrame(cm_rows, columns=["config_id", "split_id", "true_label",
                                           "pred_label", "count"])
    summ_df = pd.DataFrame(summary_rows)

    assert len(summ_df) == 128
    assert len(fold_df) == 128 * 50
    assert len(oof_df) == 128 * 50 * 18
    assert len(cm_df) == 128 * 50 * 4

    summ_df.to_csv(RESULTS_DIR / "mccv_summary.csv", index=False)
    fold_df.to_csv(RESULTS_DIR / "mccv_fold_metrics.csv", index=False)
    oof_df.to_csv(RESULTS_DIR / "mccv_oof_predictions.csv", index=False)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrices_mccv.csv", index=False)
    print(f"[mccv] wrote summary/fold/oof/cm CSVs ({time.time() - t0:.1f}s)")

    bl = run_baselines(dfu)
    bl.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    print("[mccv] baselines written")

    sel = summ_df.sort_values(
        ["Macro_F1_mean", "F1_yes_mean", "balanced_accuracy_mean"],
        ascending=[False, False, False],
    ).iloc[0]
    sel_cfg = {"distance": sel["distance"], "rule": sel["rule"],
               "weight": sel["weight"], "k": int(sel["k"])}
    sel_id = sel["config_id"]
    print(f"[select] winner = {sel_id} (Macro_F1={sel['Macro_F1_mean']:.4f})")

    with open(SELECTED_DIR / "hyperparameters.json", "w") as f:
        json.dump({
            "selected_config": sel_cfg,
            "selection_metric": "Macro_F1",
            "tiebreak_1": "F1_yes",
            "tiebreak_2": "balanced_accuracy",
            "source": "mccv",
            "config_id": sel_id,
        }, f, indent=2)

    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H %s"], capture_output=True, text=True,
        cwd=PROJECT_ROOT,
    ).stdout.strip()
    (SELECTED_DIR / "git_commit.txt").write_text(commit + "\n")

    sel_metrics = summarize_folds(
        fold_df.loc[fold_df["config_id"].eq(sel_id)].to_dict("records")
    )
    with open(SELECTED_DIR / "metrics_mccv.json", "w") as f:
        json.dump({"config_id": sel_id, "mccv_summary": sel_metrics}, f, indent=2)

    print("[loo] running 88 folds for selected config ...")
    loo_df, loo_cm_df, pooled_metrics = run_loo(dfu, sel_cfg)
    loo_df.to_csv(RESULTS_DIR / "loo_predictions.csv", index=False)
    loo_cm_df.to_csv(RESULTS_DIR / "confusion_matrix_loo.csv", index=False)

    loo_sum = summarize_folds(loo_df.to_dict("records"))
    with open(SELECTED_DIR / "metrics_loo.json", "w") as f:
        json.dump({
            "config_id": sel_id,
            "loo_summary": loo_sum,
            "loo_pooled": pooled_metrics,
        }, f, indent=2)

    print("[loo] done")

    oof_sel = oof_df.loc[oof_df["config_id"].eq(sel_id)]
    cr_rows = []
    for protocol, yt, yp in [
        ("mccv_pooled", oof_sel["y_true"].to_numpy(), oof_sel["y_pred"].to_numpy()),
        ("loo_pooled", loo_df["y_true"].to_numpy(), loo_df["y_pred"].to_numpy()),
    ]:
        rep = classification_report(yt, yp, target_names=["no", "yes"],
                                    output_dict=True, zero_division=0)
        for label in ["no", "yes", "macro avg", "weighted avg"]:
            cr_rows.append({
                "protocol": protocol, "label": label,
                "precision": rep[label]["precision"], "recall": rep[label]["recall"],
                "f1": rep[label]["f1-score"], "support": rep[label]["support"],
            })
    pd.DataFrame(cr_rows).to_csv(RESULTS_DIR / "classification_report.csv", index=False)
    print("[done] all outputs written")


if __name__ == "__main__":
    main()
