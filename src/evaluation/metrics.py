"""Scoring functions shared across experiments. Byte-identical blocks
deduplicated from exp_23/scripts/train.py:446-478 (binary metrics),
exp_23:490-503 (McNemar), and exp_23:579-587 (confidence metrics). Must
never import `kdm`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, p_soft: np.ndarray, threshold: float = 0.50) -> dict:
    """macro-F1/accuracy/sensitivity/specificity/AUROC/Brier + confusion
    counts, from continuous predictions thresholded at `threshold`."""
    y_pred = (np.asarray(p_soft) >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "macro_f1": float(macro_f1), "accuracy": float(acc),
        "sensitivity": float(sens), "specificity": float(spec),
        "auroc": float(roc_auc_score(y_true, p_soft)),
        "brier_score": float(brier_score_loss(y_true, np.clip(p_soft, 0.0, 1.0))),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "total_cases": int(len(y_true)),
    }


def confidence_metrics(y_conf: np.ndarray, pred: np.ndarray) -> dict:
    """3-class macro-F1/accuracy/Spearman rho, for the diagnostic-confidence
    task ({0,1,2} = uncertain/borderline/clear)."""
    macro_f1 = f1_score(y_conf, pred, average="macro", zero_division=0)
    acc = accuracy_score(y_conf, pred)
    rho, pval = spearmanr(y_conf, pred)
    return {
        "macro_f1": float(macro_f1), "accuracy": float(acc),
        "spearman_rho": float(rho), "spearman_pvalue": float(pval),
        "total_cases": int(len(y_conf)),
    }


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact binomial McNemar test on paired predictions. `statsmodels` is
    absent from the `pytorch` conda env, which is why exp_23 hand-rolled
    this via `scipy.stats.binomtest`."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0, "pvalue": 1.0}
    stat = min(b, c)
    pvalue = binomtest(stat, n, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "statistic": int(stat), "pvalue": float(pvalue)}
