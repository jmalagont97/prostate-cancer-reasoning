#!/usr/bin/env python3
"""Run exp_13 Task 3 Cox-Ridge replication under the frozen MCCV-30 protocol.

The runner is intentionally verbose and flushes every progress message so it can be
executed inside tmux:0 while a second terminal tails the log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "task3"
RESULTS = ROOT / "experiments" / "exp_13" / "results"
CHIMERA_METRICS = ROOT / "utils" / "pkg" / "chimera_metrics"
if str(CHIMERA_METRICS) not in sys.path:
    sys.path.insert(0, str(CHIMERA_METRICS))
from chimera_metrics import evaluate_task3_cohort  # noqa: E402

SEED = 0
PENALIZER = 0.3
PCA_COMPONENTS = 5
MAX_MONTHS = 330.0
EVENT_HORIZON_MONTHS = 60.0
EVENT_THRESHOLD = 0.7
CONDITIONS = (
    "clinical_pathology",
    "prostatectomy_pca5",
    "prostatectomy_pca5_p90",
    "prostatectomy_pca5_mean",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict[str, str]:
    def run(args: list[str]) -> str:
        return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    return {"commit": run(["git", "rev-parse", "HEAD"]), "status": run(["git", "status", "--short"])}


def write_status(status: str, completed: list[str]) -> None:
    (RESULTS / "execution_status.json").write_text(json.dumps({
        "experiment": "exp_13", "status": status,
        "completed_conditions": completed, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    }, indent=2) + "\n")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    features = pd.read_csv(DATA / "clinical_pathology_features.csv")
    labels = pd.read_csv(DATA / "survival_labels.csv")
    splits = pd.read_csv(DATA / "mccv_splits_30.csv")
    emb = np.load(DATA / "embeddings.npz", allow_pickle=False)
    ids = features.case_id.astype(str).tolist()
    for frame, name in ((labels, "labels"), (splits, "splits")):
        if set(ids) != set(frame.case_id.astype(str)):
            raise ValueError(f"case_id mismatch in {name}")
    if set(ids) != set(emb["case_ids"].astype(str).tolist()):
        raise ValueError("case_id mismatch in embeddings")
    if len(splits.columns) != 31 or len(splits) != len(ids):
        raise ValueError("MCCV file must contain 75 rows and 30 split columns")
    features = features.set_index("case_id").loc[ids].reset_index()
    labels = labels.set_index("case_id").loc[ids].reset_index()
    emb_by_id = {}
    for i, cid in enumerate(emb["case_ids"]):
        count = int(emb["prostatectomy_counts"][i])
        emb_by_id[str(cid)] = np.asarray(emb["prostatectomy"][i, :count], dtype=float)
    return features, labels, {"splits": splits, "prostatectomy": emb_by_id}


def numeric_columns(features: pd.DataFrame) -> list[str]:
    cols = []
    for col in features.columns:
        if col == "case_id":
            continue
        values = pd.to_numeric(features[col], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            cols.append(col)
    if not cols:
        raise ValueError("No non-constant numeric clinical/pathology features")
    return cols


class FoldModel:
    def __init__(self, use_embedding: bool, percentile: float | None):
        self.use_embedding = use_embedding
        self.percentile = percentile
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.pca: PCA | None = None
        self.model: CoxPHFitter | None = None
        self.columns: list[str] = []

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, emb: dict[str, np.ndarray]) -> "FoldModel":
        self.columns = numeric_columns(x)
        base = x[self.columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        parts = [base]
        if self.use_embedding:
            matrix = np.asarray([np.nanmean(emb[str(cid)], axis=0) for cid in x.case_id], dtype=float)
            if not np.isfinite(matrix).all():
                raise ValueError("Non-finite values remain in prostatectomy embeddings")
            n = min(PCA_COMPONENTS, matrix.shape[0] - 1, matrix.shape[1])
            self.pca = PCA(n_components=n, random_state=SEED).fit(matrix)
            parts.append(pd.DataFrame(self.pca.transform(matrix), index=base.index,
                                      columns=[f"prostatectomy_pca_{i}" for i in range(n)]))
        design = pd.concat(parts, axis=1)
        design = pd.DataFrame(self.imputer.fit_transform(design), columns=design.columns)
        cox = design.copy()
        cox["months_to_recurrence"] = y.months_to_recurrence.to_numpy()
        cox["event"] = y.event.astype(int).to_numpy()
        self.model = CoxPHFitter(penalizer=PENALIZER)
        self.model.fit(cox, duration_col="months_to_recurrence", event_col="event", show_progress=False)
        return self

    def predict(self, x: pd.DataFrame, emb: dict[str, np.ndarray]) -> np.ndarray:
        base = x[self.columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        parts = [base]
        if self.use_embedding:
            if self.pca is None:
                raise RuntimeError("PCA was not fitted")
            matrix = np.asarray([np.nanmean(emb[str(cid)], axis=0) for cid in x.case_id], dtype=float)
            if not np.isfinite(matrix).all():
                raise ValueError("Non-finite values remain in prostatectomy embeddings")
            parts.append(pd.DataFrame(self.pca.transform(matrix), index=base.index,
                                      columns=[f"prostatectomy_pca_{i}" for i in range(self.pca.n_components_)]))
        design = pd.concat(parts, axis=1)
        design = pd.DataFrame(self.imputer.transform(design), columns=design.columns)
        if self.model is None:
            raise RuntimeError("Cox model was not fitted")
        if self.percentile is None:
            raw = self.model.predict_expectation(design)
        else:
            raw = self.model.predict_percentile(design, p=self.percentile)
        pred = np.atleast_1d(np.asarray(raw, dtype=float)).reshape(-1)
        pred = np.where(np.isfinite(pred), pred, MAX_MONTHS)
        return np.clip(pred, 0.0, MAX_MONTHS)

    def predict_event(self, x: pd.DataFrame, emb: dict[str, np.ndarray]) -> np.ndarray:
        """Predict BCR status from the fitted Cox survival curve at 60 months."""
        base = x[self.columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        parts = [base]
        if self.use_embedding:
            if self.pca is None:
                raise RuntimeError("PCA was not fitted")
            matrix = np.asarray([np.nanmean(emb[str(cid)], axis=0) for cid in x.case_id], dtype=float)
            if not np.isfinite(matrix).all():
                raise ValueError("Non-finite values remain in prostatectomy embeddings")
            parts.append(pd.DataFrame(self.pca.transform(matrix), index=base.index,
                                      columns=[f"prostatectomy_pca_{i}" for i in range(self.pca.n_components_)]))
        design = pd.concat(parts, axis=1)
        design = pd.DataFrame(self.imputer.transform(design), columns=design.columns)
        if self.model is None:
            raise RuntimeError("Cox model was not fitted")
        survival = self.model.predict_survival_function(
            design, times=[EVENT_HORIZON_MONTHS]
        )
        s_at_horizon = np.asarray(survival.values, dtype=float).reshape(-1)
        return (s_at_horizon < EVENT_THRESHOLD).astype(int)


def condition_config(condition: str) -> tuple[bool, float | None]:
    if condition == "clinical_pathology":
        return False, None
    if condition == "prostatectomy_pca5":
        return True, None
    if condition == "prostatectomy_pca5_p90":
        return True, 0.90
    if condition == "prostatectomy_pca5_mean":
        return True, None
    raise ValueError(condition)


def run_condition(condition: str, features: pd.DataFrame, labels: pd.DataFrame, data: dict[str, np.ndarray], splits: pd.DataFrame) -> None:
    out = RESULTS / condition
    out.mkdir(parents=True, exist_ok=True)
    use_emb, percentile = condition_config(condition)
    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    log(f"START condition={condition} embedding={use_emb} percentile={percentile}")
    for fold in range(30):
        split_col = f"split_{fold}"
        train_ids = splits.loc[splits[split_col] == "train", "case_id"].astype(str).tolist()
        val_ids = splits.loc[splits[split_col] == "val", "case_id"].astype(str).tolist()
        tr_mask = features.case_id.astype(str).isin(train_ids)
        va_mask = features.case_id.astype(str).isin(val_ids)
        tr_x, va_x = features.loc[tr_mask].reset_index(drop=True), features.loc[va_mask].reset_index(drop=True)
        tr_y, va_y = labels.loc[tr_mask].reset_index(drop=True), labels.loc[va_mask].reset_index(drop=True)
        model = FoldModel(use_emb, percentile).fit(tr_x, tr_y, data["prostatectomy"])
        pred_months = model.predict(va_x, data["prostatectomy"])
        pred_events = model.predict_event(va_x, data["prostatectomy"])
        gt = [{"months_to_recurrence": float(t), "event": int(e)} for t, e in zip(va_y.months_to_recurrence, va_y.event)]
        pred = [{"months_to_recurrence": float(t), "event": int(e)}
                for t, e in zip(pred_months, pred_events)]
        case_results, cohort = evaluate_task3_cohort(gt, pred, case_ids=va_x.case_id.astype(str).tolist())
        row = {"fold": fold, "n_train": len(tr_x), "n_val": len(va_x), "n_events_val": int(va_y.event.sum()),
               "c_index": cohort.c_index, "mean_event_score": cohort.mean_event_score,
               "mean_time_score": cohort.mean_time_score, "td_auc": json.dumps(cohort.td_auc, sort_keys=True)}
        metrics_rows.append(row)
        for cid, t, e, pm, pe in zip(va_x.case_id, va_y.months_to_recurrence, va_y.event, pred_months, pred_events):
            prediction_rows.append({"case_id": str(cid), "fold": fold, "months_to_recurrence": float(t),
                                    "event": int(e), "predicted_months_to_recurrence": float(pm),
                                    "predicted_event": int(pe)})
        pd.DataFrame(metrics_rows).to_csv(out / "fold_metrics.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(out / "predictions.csv", index=False)
        log(f"PROGRESS condition={condition} fold={fold + 1}/30 c_index={cohort.c_index} events_val={int(va_y.event.sum())}")
    numeric = pd.DataFrame(metrics_rows)
    summary = {"condition": condition, "n_folds": 30, "config": {"penalizer": PENALIZER, "pca_components": PCA_COMPONENTS,
               "percentile": percentile, "use_prostatectomy": use_emb},
               "metrics": {"c_index_mean": float(numeric.c_index.mean()), "c_index_median": float(numeric.c_index.median()),
                           "c_index_std": float(numeric.c_index.std(ddof=1)), "c_index_folds": numeric.c_index.tolist(),
                           "mean_event_score_mean": float(numeric.mean_event_score.mean()),
                           "mean_time_score_mean": float(numeric.mean_time_score.mean())}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "version.txt").write_text(json.dumps({"git": git_state(), "data_hash": sha256(DATA / "mccv_splits_30.csv")}, indent=2) + "\n")
    log(f"DONE condition={condition} c_index_mean={summary['metrics']['c_index_mean']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    features, labels, data = load_data()
    splits = data["splits"]
    log(f"EXP13 START cases={len(features)} events={int(labels.event.sum())} folds=30")
    conditions = (args.condition,) if args.condition else CONDITIONS
    completed: list[str] = []
    write_status("running", completed)
    for condition in conditions:
        run_condition(condition, features, labels, data, splits)
        completed.append(condition)
        write_status("running", completed)
    manifest = {
        "experiment": "exp_13",
        "conditions": list(conditions),
        "n_cases": int(len(features)),
        "n_events": int(labels.event.sum()),
        "n_censored": int((labels.event == 0).sum()),
        "n_folds": 30,
        "fold_shape": {"train": 60, "validation": 15, "validation_events": 4},
        "configuration": {"penalizer": PENALIZER, "pca_components": PCA_COMPONENTS, "seed": SEED},
        "data_hashes": {name: sha256(DATA / name) for name in
                        ("clinical_pathology_features.csv", "survival_labels.csv",
                         "embeddings.npz", "mccv_splits_30.csv")},
        "git": git_state(),
        "evaluation_entrypoint": "chimera_metrics.evaluate_task3_cohort",
        "output_event_policy": {"horizon_months": EVENT_HORIZON_MONTHS,
                                 "survival_threshold": EVENT_THRESHOLD,
                                 "rule": "predicted_event=1 if S(horizon|x) < threshold"},
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_status("completed", completed)
    log("EXP13 COMPLETE")


if __name__ == "__main__":
    main()
