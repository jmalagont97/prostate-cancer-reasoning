"""The two-phase MCCV/LOOCV harness (see CLAUDE.md "Two-phase leak-free
evaluation protocol"), decoupled from any particular method or data schema.

Callers supply small closures (`evaluate_fn`) that close over their own X/y;
this module only owns the split iteration, aggregation, and the explicit
tie-break rule that exp_13-24 never had. Must never import `kdm`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def iter_mccv_splits(df_design: pd.DataFrame, n_splits: int = 100):
    """Yields (split_idx, train_idx, val_idx) from a `split_<i>` column
    convention (0=train, 1=val, other=excluded) — the exp_4 MCCV design
    schema. Mirrors exp_13-24's `split_vals = df[f"split_{i}"].values;
    train_idx = where(vals==0); val_idx = where(vals==1)` idiom, in one place."""
    for split_idx in range(n_splits):
        vals = df_design[f"split_{split_idx}"].values
        train_idx = np.where(vals == 0)[0]
        val_idx = np.where(vals == 1)[0]
        yield split_idx, train_idx, val_idx


def run_mccv_grid(grid: list, evaluate_fn, splits, primary_metric: str = "macro_f1") -> pd.DataFrame:
    """Phase A. `evaluate_fn(cfg, train_idx, val_idx) -> dict[str, float]`
    (one split's scalar metrics for one config); `splits` is any iterable of
    (split_idx, train_idx, val_idx) or (train_idx, val_idx) tuples (the
    split_idx is unused here, so `iter_mccv_splits`'s 3-tuples work directly).

    Returns one row per config with `mean_<metric>`/`std_<metric>` for every
    key `evaluate_fn` returns, sorted descending by `mean_<primary_metric>`.
    """
    splits = list(splits)
    per_cfg = [[] for _ in grid]
    for split in splits:
        train_idx, val_idx = split[-2], split[-1]
        for cfg_idx, cfg in enumerate(grid):
            per_cfg[cfg_idx].append(evaluate_fn(cfg, train_idx, val_idx))

    rows = []
    for cfg_idx, cfg in enumerate(grid):
        metric_dicts = per_cfg[cfg_idx]
        row = {"cfg_id": cfg_idx, **cfg}
        for key in metric_dicts[0].keys():
            vals = [d[key] for d in metric_dicts]
            row[f"mean_{key}"] = float(np.mean(vals))
            row[f"std_{key}"] = float(np.std(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"mean_{primary_metric}", ascending=False).reset_index(drop=True)


def select_best(df_grid: pd.DataFrame, primary_metric: str = "macro_f1", tol: float = 1e-9):
    """Explicit tie-break, absent from exp_13-24 (there, ties silently
    resolved by pandas' stable sort order). Among configs within `tol` of the
    best mean primary metric, picks the lowest std (most stable across
    splits), then lowest `cfg_id`. Matters because winners' std_macro_f1
    (~0.11 in exp_13/23) dwarfs typical inter-config gaps (<0.02)."""
    mean_col, std_col = f"mean_{primary_metric}", f"std_{primary_metric}"
    best = df_grid[mean_col].max()
    candidates = df_grid[df_grid[mean_col] >= best - tol]
    sort_cols = [std_col, "cfg_id"] if std_col in candidates.columns else ["cfg_id"]
    return candidates.sort_values(sort_cols).iloc[0]


def run_loocv(evaluate_fn, n: int):
    """Phase B. `evaluate_fn(train_idx, val_idx) -> dict` with at least
    `"pred"` (a scalar prediction for the single held-out row) and optionally
    `"signals"` (dict[str, float]). `evaluate_fn` is a factory-style closure
    — it must fit fresh on `train_idx` each call, never reuse a
    full-cohort-fitted model, so "never re-fit during Phase B" (CLAUDE.md)
    only applies to hyperparameters, not to the per-fold model fit itself.

    Returns (oof_pred: (n,), oof_signals: dict[str, (n,)]).
    """
    oof_pred = np.zeros(n)
    oof_signals: dict = {}
    all_idx = np.arange(n)
    for i in range(n):
        train_idx = all_idx[all_idx != i]
        val_idx = np.array([i])
        result = evaluate_fn(train_idx, val_idx)
        oof_pred[i] = result["pred"]
        for key, val in result.get("signals", {}).items():
            oof_signals.setdefault(key, np.zeros(n))
            oof_signals[key][i] = val
    return oof_pred, oof_signals
