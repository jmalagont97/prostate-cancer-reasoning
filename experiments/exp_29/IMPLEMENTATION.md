# Implementation Plan: BrentMemKDM k-NN truncation sweep over the exp_28 hard-KNN generation (exp_29)
**Experiment**: experiments/exp_29/ · **Status**: Draft

See `DESIGN.md` for motivation, scope, and decision rules. `experiments/exp_28/scripts/train.py`
(410 lines) is the direct base — this plan is written as a diff against it, since exp_29 reuses
exp_28's harness, Step-0 gates, reference arms, and S1/S2 secondary conditions almost verbatim.
Everything below was verified against the live repo state during planning:

- `src/methods/brent_mem_kdm.py`'s `knn_k` option (added this session) is implemented and
  verified: `python scripts/verify_brent_mem_kdm.py` — **56/56 checks pass** (the original 32 plus
  24 new knn-mode checks: k=1 closed-form identity across sigma regimes, `knn_k=None`
  bit-exactness, `k >= n_train` equivalence, fast-vs-exact-per-query agreement at the
  lower/center/upper sigma bounds, determinism).
- `BrentMemKDM(knn_k=1)` reproduces `sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,
  weights="uniform", metric="euclidean")` **exactly**, per-fold, checked directly against the real
  cohort's `tab` representation over 15 MCCV splits (0 prediction mismatches) — the basis for gate
  G3 below.
- A 100-MCCV-split Phase-A-style sweep of `BrentMemKDM(knn_k=k)` on `tab`, soft arm, reproduced
  exp_28's own `results/stage1_best_hparams.json` `tab`/`soft` entry bit-for-bit at `knn_k=None`
  (`sigma_mult=0.43333`, `mean_macro_f1=0.57660`) and found `knn_k=3` peaking at `0.6346` — the
  basis for `KNN_K_GRID` below (DESIGN.md §4; this is a one-modality, Phase-A-only signal, not
  itself an H1/H2 result).
- The `knn_k` addition and its verification checks are **currently uncommitted** working-tree
  changes (`git status`: `M src/methods/brent_mem_kdm.py`, `M scripts/verify_brent_mem_kdm.py`).

## 0. Prerequisite commit (before the full run, not before drafting `train.py`)

Per DESIGN.md §2.4/§10: commit `src/methods/brent_mem_kdm.py`'s `knn_k` addition and
`scripts/verify_brent_mem_kdm.py`'s new knn-mode checks as one commit, so `results/git_commit.txt`
names the code this experiment actually ran against. Drafting and smoke-testing `train.py` does
not require this commit first; the full run does.

## 1. `experiments/exp_29/scripts/train.py`

Self-contained, built directly on `src/evaluation`/`src/methods` — same convention as exp_28.
Structure mirrors `exp_28/scripts/train.py`'s shape (imports → cohort → Step 0 gates → Phase A per
condition → Phase B LOOCV → reference arms → S1/S2 → metrics/McNemar → figures) with one added
Phase-A dimension (`knn_k`), one added gate (G3), and one added metrics section (H2 vs. exp_28).

### 1.1 Imports and setup

Identical to `exp_28/scripts/train.py:17-53`, plus one import for gate G3:

```python
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

from kdm.init import _sigma_from_knn  # NEW: for gate G3 — same import brent_mem_kdm.py itself uses

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

MRI_REPS = [None, 0.90]                              # pca_variance: raw_l2, pca90_l2
TXT_REPS = [(500, 0.90), (2000, 0.90), (None, 0.90)]
KNN_K_GRID = [1, 3, 5, 10, 20, 40, None]              # NEW — DESIGN.md §4
```

`to_2col`, `run_loocv_folds`, `classifier_pred` are copied verbatim from
`exp_28/scripts/train.py:56-75` (identical local helpers, no changes needed).

Everything from §1.2 onward lives inside `def main(smoke: bool = False):`, exactly like
`exp_28/scripts/train.py`'s structure.

### 1.2 Cohort, targets, splits

Identical to `exp_28/scripts/train.py:83-100` — same cohort load, same `targets_hard`/
`targets_soft`, same `build_modality`/`g2_text_pipeline` closures, same `FULL_SPLITS`/`SPLITS`/
`LOOCV_FOLDS` smoke-reduction convention. No changes.

### 1.3 Step 0 — reproduction gates (DESIGN.md §6)

G0/G1/G2 identical to `exp_28/scripts/train.py:124-167`. **New G3**, appended after G2, before the
`reproduction_gates` dict is assembled:

```python
print("[exp_29] Step 0d: G3 (BrentMemKDM(knn_k=1) == plain 1-NN on C1's own representation)...")

def g3_bmk_evaluate_fn(train_idx, val_idx):
    X_tr, X_va = build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
    sigma = _sigma_from_knn(X_tr, 1.0)  # any positive sigma works: knn_k=1 is sigma-invariant
    m = BrentMemKDM(knn_k=1)
    m.sigmas_ = {"tab": sigma}
    m.modality_order = ["tab"]
    m.fit({"tab": X_tr}, Targets(y_binary=cohort.y_binary[train_idx], y_soft=cohort.y_binary[train_idx].astype(np.float32)))
    return {"pred": float(m.predict_proba({"tab": X_va})[0, 1])}

def g3_knn1_evaluate_fn(train_idx, val_idx):
    X_tr, X_va = build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
    knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean")
    knn.fit(X_tr, cohort.y_binary[train_idx])
    return {"pred": classifier_pred(knn, X_va)}

g3_bmk_pred, _ = run_loocv(g3_bmk_evaluate_fn, n)
g3_knn1_pred, _ = run_loocv(g3_knn1_evaluate_fn, n)
g3_mismatches = int(np.sum((g3_bmk_pred >= 0.5).astype(int) != (g3_knn1_pred >= 0.5).astype(int)))
g3_pass = g3_mismatches == 0
```

`g3_bmk_evaluate_fn` deliberately uses the **same** `dre_categories=cohort.dre_categories`-fixed
representation this experiment's own C1 condition uses (not G1's original un-fixed pipeline) — G3
checks `BrentMemKDM(knn_k=1)` against a 1-NN fit on *its own* representation, not against G1's
number (DESIGN.md §6's stated reasoning: the two encodings can legitimately break nearest-neighbor
ties differently).

```python
reproduction_gates = {
    "G0_verify_brent_mem_kdm": {"passed": bool(g0_pass)},
    "G1_exp5_tabular_knn": {"passed": bool(g1_pass), "macro_f1": g1_metrics["macro_f1"],
                             "target": 0.6333333333333333,
                             "confusion": {k: g1_metrics[k] for k in ("tp", "tn", "fp", "fn")}},
    "G2_exp7_text_knn": {"passed": bool(g2_pass), "macro_f1": g2_metrics["macro_f1"],
                          "target": 0.6987539367383268,
                          "confusion": {k: g2_metrics[k] for k in ("tp", "tn", "fp", "fn")}},
    "G3_knn_k1_equals_1nn": {"passed": bool(g3_pass), "mismatches": g3_mismatches},  # NEW
}
write_json(reproduction_gates, RESULTS_DIR / "reproduction_gates.json")
assert g1_pass, f"G1 FAILED: got {g1_metrics['macro_f1']!r}"
assert g2_pass, f"G2 FAILED: got {g2_metrics['macro_f1']!r}"
assert g3_pass, f"G3 FAILED: {g3_mismatches} prediction mismatches between BrentMemKDM(knn_k=1) and 1-NN"
print(f"[exp_29] Step 0 PASSED: G0={g0_pass}, G1={g1_metrics['macro_f1']!r}, "
      f"G2={g2_metrics['macro_f1']!r}, G3 mismatches={g3_mismatches}")
```

G0/G1/G2/G3 all run in full regardless of `--smoke` — same convention as exp_28 (a reproduction
gate that gets skipped isn't one).

### 1.4 Representation × `knn_k` grid and per-modality Brent search (Phase A)

```python
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
            folds = build_folds(name, rep, SPLITS, arm)  # built once per (name, rep, arm); knn_k doesn't affect folds
            for k in KNN_K_GRID:
                result = run_brent_search(folds, [name], metric="macro_f1", strategy="nested",
                                           n_prescan=15, maxiter=20, knn_k=k)
                rows.append({"rep": repr(rep), "knn_k": repr(k), "sigma": result.sigmas[name],
                             "sigma_mult": result.sigma_mult[name],
                             "mean_macro_f1": result.per_fold_scores["mean"],
                             "std_macro_f1": result.per_fold_scores["std"]})
        df = pd.DataFrame(rows).sort_values(["mean_macro_f1", "std_macro_f1"], ascending=[False, True]).reset_index(drop=True)
        STAGE1_ALL[arm][name] = df
        best = df.iloc[0]
        STAGE1_BEST[arm][name] = {
            "rep": eval(best["rep"]) if name != "tab" else None,  # noqa: S307 -- repr() of MRI_REPS/TXT_REPS entries only
            "knn_k": eval(best["knn_k"]),  # noqa: S307 -- repr() of KNN_K_GRID entries only
            "sigma": float(best["sigma"]), "sigma_mult": float(best["sigma_mult"]),
            "mean_macro_f1": float(best["mean_macro_f1"]), "std_macro_f1": float(best["std_macro_f1"]),
        }
        print(f"[exp_29] Phase A [{arm}] {name}: rep={STAGE1_BEST[arm][name]['rep']}, "
              f"knn_k={STAGE1_BEST[arm][name]['knn_k']}, sigma={STAGE1_BEST[arm][name]['sigma']:.5f}, "
              f"mean_macro_f1={STAGE1_BEST[arm][name]['mean_macro_f1']:.4f}")

write_json(STAGE1_BEST, RESULTS_DIR / "stage1_best_hparams.json")
for name in ["tab", "mri", "txt"]:
    pd.concat([STAGE1_ALL[arm][name].assign(arm=arm) for arm in ["hard", "soft"]], ignore_index=True) \
        .to_csv(RESULTS_DIR / f"phasea_grid_{name}.csv", index=False)
```

The only differences from `exp_28/scripts/train.py:169-207`: the inner `for k in KNN_K_GRID` loop
(one `run_brent_search(..., knn_k=k)` call per `(rep, k)` pair, `folds` built once per `(name,
rep, arm)` and reused across all 7 `k` values — no redundant feature-building), the `"knn_k":
repr(k)` field, and `"knn_k": eval(best["knn_k"])` in `STAGE1_BEST`. Search count: tab 1×7=7, mri
2×7=14, txt 3×7=21 per arm (84 total vs. exp_28's 12) — still cheap on the fast Nadaraya-Watson
path (§4/verification below times this empirically via `--smoke`).

### 1.5 Phase B — LOOCV, frozen `(rep, knn_k, sigma)`, per condition

```python
def make_targets(idx, arm):
    return Targets(y_binary=cohort.y_binary[idx], y_soft=TARGETS[arm].y_soft[idx])

def fit_predict_unimodal(name, arm, train_idx, val_idx):
    rep = STAGE1_BEST[arm][name]["rep"]
    X_tr, X_va = build_modality(name, rep, train_idx, val_idx)
    m = BrentMemKDM(knn_k=STAGE1_BEST[arm][name]["knn_k"])  # CHANGED: was BrentMemKDM()
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
        print(f"[exp_29] Phase B unimodal_{name} [{arm}]: macro_f1={f1:.4f}")

FUSION_EQUAL_OOF = {
    arm: soft_vote({name: to_2col(UNIMODAL_OOF[arm][name]) for name in ["tab", "mri", "txt"]},
                    {"tab": 1 / 3, "mri": 1 / 3, "txt": 1 / 3})[:, 1]
    for arm in ["hard", "soft"]
}
```

Identical to `exp_28/scripts/train.py:209-243` except the single `knn_k=...` addition in
`fit_predict_unimodal` (marked `CHANGED` above) — `BrentMemKDM.fit`/`.predict_proba` already
dispatch on `self.knn_k` internally, so no other line changes.

### 1.6 Reference arms — recomputed exp_6/exp_8 counterparts (DESIGN.md §5)

Identical to `exp_28/scripts/train.py:245-260`, verbatim — KNN reference arms have no `knn_k`
(they already *are* the `k=1`/whole-cohort-average endpoints this experiment interpolates around,
DESIGN.md §5).

### 1.7 Secondary conditions (DESIGN.md §3, S1/S2 — hard arm only)

S1 (`fusion_optimal_leakfree`) — identical to `exp_28/scripts/train.py:262-295` except
`winner_val_probs`'s `BrentMemKDM()` construction:

```python
def winner_val_probs(name):
    best = STAGE1_BEST["hard"][name]
    out = {}
    for split_idx, train_idx, val_idx in SPLITS:
        X_tr, X_va = build_modality(name, best["rep"], train_idx, val_idx)
        m = BrentMemKDM(knn_k=best["knn_k"])  # CHANGED: was BrentMemKDM()
        m.sigmas_ = {name: best["sigma"]}
        m.modality_order = [name]
        m.fit({name: X_tr}, make_targets(train_idx, "hard"))
        out[split_idx] = (val_idx, m.predict_proba({name: X_va}))
    return out
```

Everything else in S1 (the simplex grid search over Stage-1 MCCV validation probs,
`fusion_weights_leakfree`, `FUSION_OPTIMAL_OOF`) is unchanged.

S2 (`joint_trimodal`) — **identical to `exp_28/scripts/train.py:297-333`, verbatim, no `knn_k`
threading at all.** Stays whole-memory (`BrentMemKDM()` with default `knn_k=None`) per DESIGN.md
§3: a joint per-query neighbor retrieval across a product kernel is a materially different design
question (which modality's distance dominates neighbor selection) and is out of scope here.

### 1.8 Metrics, McNemar (H1 + new H2), output

```python
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
KNN_K_BY_COND = {  # NEW -- for recording knn_k alongside each condition's metrics
    "hard": {**{name: STAGE1_BEST["hard"][name]["knn_k"] for name in ["tab", "mri", "txt"]},
             "fusion_equal": {name: STAGE1_BEST["hard"][name]["knn_k"] for name in ["tab", "mri", "txt"]}},
    "soft": {**{name: STAGE1_BEST["soft"][name]["knn_k"] for name in ["tab", "mri", "txt"]},
             "fusion_equal": {name: STAGE1_BEST["soft"][name]["knn_k"] for name in ["tab", "mri", "txt"]}},
}
for arm, conditions in [("hard", CONDITIONS_HARD), ("soft", CONDITIONS_SOFT)]:
    for cond_name, pred in conditions.items():
        key = f"{cond_name}__{arm}"
        loocv_metrics[key] = binary_metrics(y_true_eval, pred[idx])
        loocv_metrics[key]["target_informed"] = (arm == "soft")
        if cond_name in KNN_K_BY_COND[arm]:  # NEW -- not set for fusion_optimal_leakfree/joint_trimodal (no single knn_k)
            loocv_metrics[key]["knn_k"] = KNN_K_BY_COND[arm][cond_name]
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

df_oof = pd.DataFrame({"patient_id": cohort.pids, "ground_truth_biopsy": cohort.y_binary})
for arm, conditions in [("hard", CONDITIONS_HARD), ("soft", CONDITIONS_SOFT)]:
    for cond_name, pred in conditions.items():
        df_oof[f"prob_{cond_name}__{arm}"] = pred
for ref_name, pred in REFERENCE.items():
    df_oof[f"prob_{ref_name}"] = pred
df_oof.to_csv(RESULTS_DIR / "loocv_predictions.csv", index=False)

# ---------------------------------------------------------------- NEW: H2 -- vs. exp_28's own knn_k=None predictions
print("[exp_29] H2: McNemar vs. exp_28's own knn_k=None predictions...")
exp28_pred_path = PROJECT_ROOT / "experiments" / "exp_28" / "results" / "loocv_predictions.csv"
if not smoke and exp28_pred_path.exists():
    exp28_df = pd.read_csv(exp28_pred_path)
    exp28_prob_cols = [c for c in exp28_df.columns if c.startswith("prob_")]
    exp28_probs = exp28_df[["patient_id"] + exp28_prob_cols].rename(
        columns={c: f"{c}__exp28" for c in exp28_prob_cols})
    merged = df_oof.merge(exp28_probs, on="patient_id", how="inner")
    assert len(merged) == n, f"patient_id alignment against exp_28 lost rows: {len(merged)} != {n}"

    H2_CONDITIONS = ["tab", "mri", "txt", "fusion_equal"]
    for arm in ["hard", "soft"]:
        for cond in H2_CONDITIONS:
            col_new, col_old = f"prob_{cond}__{arm}", f"prob_{cond}__{arm}__exp28"
            if col_old not in merged.columns:
                continue
            y_new = (merged[col_new].values >= 0.50).astype(int)
            y_old = (merged[col_old].values >= 0.50).astype(int)
            mcnemar_results[f"{cond}__{arm}_vs_exp28_knn_k_none"] = mcnemar_exact(
                merged["ground_truth_biopsy"].values, y_new, y_old)
else:
    print("[exp_29] H2 skipped (--smoke, or exp_28/results/loocv_predictions.csv not found)")

write_json(mcnemar_results, RESULTS_DIR / "mcnemar.json")
```

H2 is skipped under `--smoke` (the reduced `LOOCV_FOLDS` subset wouldn't align meaningfully
against exp_28's full-cohort predictions) and guarded by `exp28_pred_path.exists()` so a missing
exp_28 results directory degrades to a warning, not a crash. `mcnemar.json` ends up holding both
H1 rows (`*_vs_knn_*`, unchanged from exp_28) and the new H2 rows (`*_vs_exp28_knn_k_none`) in one
file, per DESIGN.md §7's file layout.

### 1.9 Figures and git commit

Confusion matrix / ROC overlay: identical to `exp_28/scripts/train.py:384-398`. **New**: a
Macro-F1-vs-`knn_k` curve per modality, since `plot_grid_search_curves`
(`src/evaluation/reporting.py`, not modified) sorts `x_col` numerically and can't place a `None`
grid point — a small local helper instead:

```python
def plot_knn_k_curves(df_grid, title, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 6))
    explicit_ks = sorted(k for k in df_grid["knn_k"].unique() if k is not None)
    none_x = explicit_ks[-1] * 2  # placeholder x-position for knn_k=None (2x the largest explicit k)
    for (arm, rep), grp in df_grid.groupby(["arm", "rep"]):
        grp = grp.copy()
        grp["x"] = grp["knn_k"].apply(lambda k: none_x if k is None else k)
        grp = grp.sort_values("x")
        plt.plot(grp["x"], grp["mean_macro_f1"], marker="o", alpha=0.8, label=f"arm={arm},rep={rep}")
    plt.xticks(explicit_ks + [none_x], [str(k) for k in explicit_ks] + ["None\n(whole-memory)"])
    plt.xlabel("knn_k")
    plt.ylabel("mean_macro_f1")
    plt.title(title, fontweight="bold")
    plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

try:
    for name in ["tab", "mri", "txt"]:
        df = pd.concat([STAGE1_ALL[arm][name].assign(arm=arm) for arm in ["hard", "soft"]], ignore_index=True)
        df["knn_k"] = df["knn_k"].apply(eval)  # noqa: S307 -- repr() of KNN_K_GRID entries only
        df["rep"] = df["rep"].apply(eval)      # noqa: S307 -- repr() of MRI_REPS/TXT_REPS/[None] entries only
        plot_grid_search_curves(df, x_col="sigma_mult", y_col="mean_macro_f1", group_cols=["arm", "rep", "knn_k"],
                                 title=f"exp_29 Phase A {name} sigma search (by knn_k)", out_path=FIG_DIR / f"phasea_{name}.png")
        plot_knn_k_curves(df, title=f"exp_29 Phase A {name}: mean Macro-F1 vs knn_k",
                           out_path=FIG_DIR / f"phasea_knn_curve_{name}.png")
    y_pred_fusion_hard = (CONDITIONS_HARD["fusion_equal"][idx] >= 0.50).astype(int)
    plot_confusion_matrix(y_true_eval, y_pred_fusion_hard, labels=["No", "Yes"],
                           title="exp_29 fusion_equal (hard) LOOCV", out_path=FIG_DIR / "confusion_matrix.png")
    plot_roc_curves({"fusion_equal_hard": CONDITIONS_HARD["fusion_equal"][idx],
                      "knn_fusion_equal": REFERENCE["knn_fusion_equal"][idx],
                      "tab_hard": CONDITIONS_HARD["tab"][idx]},
                     y_true_eval, title="exp_29 ROC overlay", out_path=FIG_DIR / "roc_curve.png")
except Exception as e:  # pragma: no cover -- figures are best-effort, never block numeric results
    print(f"[exp_29] WARNING: figure generation failed: {e}")

record_git_commit(RESULTS_DIR)
elapsed = time.time() - t_start
print(f"[exp_29] DONE in {elapsed / 60:.1f} min. Results in {RESULTS_DIR}")
```

Note `plot_grid_search_curves`'s existing sigma-vs-macro_f1 plot now groups by `["arm", "rep",
"knn_k"]` instead of exp_28's `["arm", "rep"]` (one more line per group, since each `(rep, arm)`
now sweeps 7 `knn_k` values) — passed `sigma_mult` and `mean_macro_f1` as before, no change to
that shared function itself.

## 2. Anti-leak assertions (DESIGN.md §8, "verification")

Identical to `exp_28/scripts/train.py`'s implicit invariants: `SPLITS` always sourced from
`iter_mccv_splits`, never from LOOCV folds; `sigmas_`/`modality_order` set exactly once per fresh
`BrentMemKDM` instance inside each `evaluate_fn`/`phase_b_*` closure (never reused across folds),
by construction. Add one new explicit check for the H2 merge (already inlined in §1.8 above):
`assert len(merged) == n` after the `patient_id` merge against exp_28's predictions, so a silent
row-dropping join failure aborts the run instead of producing a wrong H2 number.

## 3. Command Lines

```bash
cd /Users/fgonza/Documents/research/code/prostate-cancer-reasoning
conda activate pytorch   # NOT histo-DL — see project memory

# smoke test first (5 MCCV splits for Stage 1 / unimodal Phase B, 6 LOOCV folds)
python experiments/exp_29/scripts/train.py --smoke

# full run
python experiments/exp_29/scripts/train.py
```

## 4. `--smoke` flag

```python
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
main(smoke=args.smoke)
```

Reduces `SPLITS` (Phase A, unimodal Phase B loop bound) and `LOOCV_FOLDS`; does **not** reduce
G0/G1/G2/G3 (§1.3, run in full always) or S2's joint-search split count (already fixed small,
independent of `--smoke`, unchanged from exp_28). H2 (§1.8) is skipped entirely under `--smoke`.
`--smoke`'s main additional job here is timing the `KNN_K_GRID` multiplier empirically — watch the
Phase-A wall-clock time printed per `(arm, name)` block against exp_28's own (undocumented but
observably fast) Phase-A timing before committing to the full 100-split × 7-`k` run.

## 5. Post-Execution (after results exist and are reviewed)

- Confirm `results/reproduction_gates.json` shows G0/G1/G2/**G3** all `passed: true` (G3
  specifically: `mismatches: 0`).
- Compare `results/loocv_metrics.json`'s `fusion_equal__hard`/`__soft` to exp_8's 0.7171 and to
  `knn_fusion_equal` (H1, DESIGN.md §8), reporting `mcnemar.json`'s `fusion_equal__*_vs_knn_fusion_equal`.
- Compare every condition's best-`knn_k` number to exp_28's own `knn_k=None` number for that exact
  condition/arm (H2), reporting `mcnemar.json`'s `*_vs_exp28_knn_k_none` rows — not just the point
  estimate.
- Report the winning `knn_k` per condition/arm (`stage1_best_hparams.json`) alongside `sigma_mult`
  (as exp_28 did for sigma alone), and note where the grid pinned an edge (`knn_k=1` or `knn_k=40`)
  vs. an interior value — an edge pin motivates widening `KNN_K_GRID` in a follow-up, an interior
  peak does not.
- Given `std_macro_f1 ≈ 0.11` across MCCV splits (DESIGN.md §8), report any Phase-B improvement
  without significant McNemar as **not established**, not a win.
- Write `reports/summary.md` (ml-experiment-reporter conventions), framed against `DESIGN.md` §8's
  decision rules.
- Flip `DESIGN.md`'s and `experiments/INDEX.md`'s `Status` to **In Progress** once runs are
  launched (`CLAUDE.md`/skill convention — the reporter sets the terminal state later).
- Commit `src/methods/brent_mem_kdm.py`, `scripts/verify_brent_mem_kdm.py`, and
  `experiments/exp_29` together (§0) — only after the user has reviewed results, per `CLAUDE.md`'s
  workflow.
- Append `.logbook.md`/`.discussion.md` entries once the user has reviewed and approved the
  results — only record points the user has explicitly approved.
