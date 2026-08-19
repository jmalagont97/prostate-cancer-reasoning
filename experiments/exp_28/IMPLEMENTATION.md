# Implementation Plan: BrentMemKDM re-evaluation of the exp_5–exp_8 hard-KNN generation (exp_28)
**Experiment**: experiments/exp_28/ · **Status**: Complete

See `DESIGN.md` for motivation, scope, and decision rules. Everything below was verified against the
live repo state during planning (results quoted are real, not projected):

- `python3 -c "... build_tabular_features(dre_categories=None) + KNeighborsClassifier(k=3,...) through
  run_loocv ..."` reproduces exp_5's exact `0.6333333333333333` / 46-14-20-8 using `load_cohort` +
  `build_tabular_features` from the shared harness (`dre_categories=None`, matching exp_5's inferred-category
  behavior) — confirms G1 is achievable through `src/evaluation` without copying exp_5's bare-`read_csv` code.
- The same check for exp_7's text pipeline (TF-IDF(max_features=500) → PCA@0.90, **no** MinMax, **no**
  trailing L2 — i.e. NOT `build_text_features`, which adds both) reproduces exp_7's exact
  `0.6987539367383268` / 42-21-13-12. G2's text pipeline is therefore hand-built in `train.py`, not
  `build_text_features`.
- `BrentMemKDM.search()` → `.sigmas_` → fresh `BrentMemKDM` instance with `sigmas_`/`modality_order` set
  manually → `.fit()`/`.predict_proba()` through `run_loocv` runs end-to-end on the tabular modality and is
  bit-for-bit deterministic across two independent `.fit()` calls on the same fold.
- `python scripts/verify_brent_mem_kdm.py --quick` — **30/30 checks pass** (fast-vs-torch exactness at
  bounds/center for 1/2/3-modality kernels, vectorized macro-F1 vs sklearn, nested-search trace
  well-formedness). This *is* Step-0 gate G0's `--quick` half; the full run (checks 3/5, the 100-split
  searches) runs once more inside `train.py`'s own Step-0 block before it trusts any Brent output.
- `experiments/exp_6/results/grid_search_results.csv` row for `representation=pca, n_neighbors=1,
  weights=uniform, metric=euclidean`: `mean_macro_f1=0.536742` (confirms DESIGN.md §2.1's reference number).

## 0. Prerequisite commit (before the full run, not before drafting `train.py`)

Per `DESIGN.md` §2.4/§10: commit `src/methods/mem_kdm.py`'s `has_trainable` guard (currently uncommitted,
already verified working above), `src/methods/brent_mem_kdm.py`, `scripts/verify_brent_mem_kdm.py`, and
`experiments/exp_25`–`exp_27` (all currently untracked/modified) as one commit, so
`results/git_commit.txt` names the code this experiment actually ran against. Drafting and smoke-testing
`train.py` does not require this commit first; the full run does.

## 1. `experiments/exp_28/scripts/train.py`

Self-contained, built directly on `src/evaluation`/`src/methods` (not copied from any exp_5–8 `train.py`,
which hardcode a nonexistent data path). Structure mirrors `exp_27/scripts/train.py`'s shape
(imports → cohort → Step 0 gates → Phase A per condition → Phase B LOOCV → reporting) with no
confidence-task section (out of scope per `DESIGN.md` §9) and no particle-signal CSV (`BrentMemKDM.fit()`
still exposes `uncertainty_signals()` since it delegates to a real `MemKDM`, but nothing in this
experiment's scope consumes it).

### 1.1 Imports and setup

```python
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data import (
    CONFIDENCE_CERTAINTY_MAP, OLD_SCHEMA, build_mri_features, build_tabular_features,
    build_targets, build_text_features, clean_texts_spacy, load_cohort, resolve_data_dir,
)
from src.evaluation.metrics import binary_metrics, mcnemar_exact
from src.evaluation.protocol import iter_mccv_splits, run_loocv
from src.evaluation.reporting import (
    plot_confusion_matrix, plot_grid_search_curves, plot_roc_curves, record_git_commit, write_json,
)
from src.methods.base import Targets
from src.methods.brent_mem_kdm import BrentMemKDM, Fold, run_brent_search
from src.methods.mem_kdm import soft_vote, simplex_grid

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "reports" / "figures"
```

Note: `select_best`/`run_mccv_grid` from `protocol.py` are **not** imported — the representation grid for
each modality is small enough (1/2/3 candidates) that Phase A loops representations directly and calls
`run_brent_search` once per representation, picking the winner by mean Phase-A Macro-F1 with the same
std-then-lowest-index tie-break `select_best` uses (implemented inline, §1.4, since `run_brent_search`'s
output isn't a `run_mccv_grid`-shaped DataFrame row).

Everything from §1.2 onward lives inside `def main(smoke: bool = False):`, exactly like
`exp_27/scripts/train.py`'s structure — `smoke` is a `main()` parameter, not a module global, and
`if __name__ == "__main__":` at the bottom does `argparse` → `main(smoke=args.smoke)` (§4). Also copied
verbatim from `exp_27/scripts/train.py:86-100` into `train.py` (not added to `src/evaluation/protocol.py`):

```python
def run_loocv_folds(evaluate_fn, n: int, folds):
    """Local variant of protocol.run_loocv restricted to a subset of folds (for --smoke). Identical to
    protocol.run_loocv when folds == range(n)."""
    oof_pred = np.zeros(n)
    all_idx = np.arange(n)
    for i in folds:
        train_idx, val_idx = all_idx[all_idx != i], np.array([i])
        result = evaluate_fn(train_idx, val_idx)
        oof_pred[i] = result["pred"]
    return oof_pred, {}
```

### 1.2 Cohort, targets, splits

```python
data_dir = resolve_data_dir(PROJECT_ROOT)
cohort = load_cohort(data_dir, PROJECT_ROOT, OLD_SCHEMA, load_mri=True, load_text=True)
assert len(cohort.dre_categories) == 5
n = len(cohort.y_binary)
print(f"[exp_28] cohort N={n}, yes={int(cohort.y_binary.sum())}, no={int((1 - cohort.y_binary).sum())}")

targets_hard = build_targets(cohort.y_binary)  # y_soft = y_binary
targets_soft = build_targets(cohort.y_binary, cohort.confidence, certainty_map=CONFIDENCE_CERTAINTY_MAP)
TARGETS = {"hard": targets_hard, "soft": targets_soft}

print("[exp_28] spaCy cleaning text corpus (once, cohort-level)...")
cleaned_texts = clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)

FULL_SPLITS = list(iter_mccv_splits(cohort.df_design, n_splits=100))  # (split_idx, train_idx, val_idx)
SPLITS = FULL_SPLITS[:5] if smoke else FULL_SPLITS
LOOCV_FOLDS = list(range(6)) if smoke else list(range(n))
```

`smoke` is the `--smoke` CLI flag (§4), same convention as `exp_25`–`27`.

### 1.3 Step 0 — reproduction gates (DESIGN.md §6)

```python
print("[exp_28] Step 0a: verify_brent_mem_kdm.py full checks (subprocess)...")
import subprocess
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
    p = knn.predict_proba(X_va)
    classes = list(knn.classes_)
    return {"pred": float(p[0, classes.index(1)]) if 1 in classes else 0.0}

g1_pred, _ = run_loocv(g1_evaluate_fn, n)
g1_metrics = binary_metrics(cohort.y_binary, g1_pred)
g1_pass = abs(g1_metrics["macro_f1"] - 0.6333333333333333) < 1e-12

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

def g2_evaluate_fn(train_idx, val_idx):
    X_tr, X_va = g2_text_pipeline(train_idx, val_idx)
    knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="cosine")
    knn.fit(X_tr, cohort.y_binary[train_idx])
    p = knn.predict_proba(X_va)
    classes = list(knn.classes_)
    return {"pred": float(p[0, classes.index(1)]) if 1 in classes else 0.0}

g2_pred, _ = run_loocv(g2_evaluate_fn, n)
g2_metrics = binary_metrics(cohort.y_binary, g2_pred)
g2_pass = abs(g2_metrics["macro_f1"] - 0.6987539367383268) < 1e-12

reproduction_gates = {
    "G0_verify_brent_mem_kdm": {"passed": bool(g0_pass)},
    "G1_exp5_tabular_knn": {"passed": bool(g1_pass), "macro_f1": g1_metrics["macro_f1"],
                             "target": 0.6333333333333333, "confusion": {k: g1_metrics[k] for k in ("tp","tn","fp","fn")}},
    "G2_exp7_text_knn": {"passed": bool(g2_pass), "macro_f1": g2_metrics["macro_f1"],
                          "target": 0.6987539367383268, "confusion": {k: g2_metrics[k] for k in ("tp","tn","fp","fn")}},
}
write_json(reproduction_gates, RESULTS_DIR / "reproduction_gates.json")
assert g1_pass, f"G1 FAILED: got {g1_metrics['macro_f1']!r}"
assert g2_pass, f"G2 FAILED: got {g2_metrics['macro_f1']!r}"
print(f"[exp_28] Step 0 PASSED: G0={g0_pass}, G1={g1_metrics['macro_f1']!r}, G2={g2_metrics['macro_f1']!r}")
```

`--smoke` still runs G0/G1/G2 in full (they're independent of `SPLITS`/`LOOCV_FOLDS` reduction) — cheap
relative to the rest of the run and the whole point of a reproduction gate is that it doesn't get skipped.

### 1.4 Representation grids and per-modality Brent search (Phase A)

```python
MRI_REPS = [None, 0.90]              # pca_variance: raw_l2, pca90_l2
TXT_REPS = [(500, 0.90), (2000, 0.90), (None, 0.90)]

def build_modality(name, rep, train_idx, val_idx):
    if name == "tab":
        return build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
    if name == "mri":
        return build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=rep)
    if name == "txt":
        max_features, pca_variance = rep
        return build_text_features(cleaned_texts, train_idx, val_idx, max_features=max_features, pca_variance=pca_variance)
    raise ValueError(name)

def build_folds(name, rep, splits, arm):
    y_soft = TARGETS[arm].y_soft
    out = []
    for _, tr, va in splits:
        X_tr, X_va = build_modality(name, rep, tr, va)
        out.append(Fold(X_train={name: X_tr}, y_soft_train=y_soft[tr], X_val={name: X_va}, y_val=cohort.y_binary[va]))
    return out

def search_unimodal(name, reps, arm):
    """One representation (tab) or a small list (mri/txt); pick the winner by Phase-A mean Macro-F1,
    tie-break lowest std then first-listed (mirrors protocol.select_best; DESIGN.md Sec 4)."""
    best = None
    for rep in reps:
        folds = build_folds(name, rep, SPLITS, arm)
        result = run_brent_search(folds, [name], metric="macro_f1", strategy="nested",
                                   n_prescan=15, maxiter=20, label_smoothing=0.0)
        row = {"rep": rep, "sigma": result.sigmas[name], "sigma_mult": result.sigma_mult[name],
               "mean_macro_f1": result.per_fold_scores["mean"], "std_macro_f1": result.per_fold_scores["std"],
               "n_evals": result.n_evals, "trace": result.trace}
        if best is None or row["mean_macro_f1"] > best["mean_macro_f1"] + 1e-9 or \
           (abs(row["mean_macro_f1"] - best["mean_macro_f1"]) <= 1e-9 and row["std_macro_f1"] < best["std_macro_f1"]):
            best = row
    return best

STAGE1_BEST = {}   # {arm: {modality: best_row}}
STAGE1_ALL = {}    # {arm: {modality: [row, ...]}}  -- every representation tried, for phasea_grid_*.csv
for arm in ["hard", "soft"]:
    STAGE1_BEST[arm], STAGE1_ALL[arm] = {}, {}
    for name, reps in [("tab", [None]), ("mri", MRI_REPS), ("txt", TXT_REPS)]:
        rows = []
        for rep in reps:
            folds = build_folds(name, rep, SPLITS, arm)
            result = run_brent_search(folds, [name], metric="macro_f1", strategy="nested", n_prescan=15, maxiter=20)
            rows.append({"rep": repr(rep), "sigma": result.sigmas[name], "sigma_mult": result.sigma_mult[name],
                         "mean_macro_f1": result.per_fold_scores["mean"], "std_macro_f1": result.per_fold_scores["std"]})
        df = pd.DataFrame(rows).sort_values(["mean_macro_f1", "std_macro_f1"], ascending=[False, True])
        STAGE1_ALL[arm][name] = df
        best = df.iloc[0]
        STAGE1_BEST[arm][name] = {"rep": eval(best["rep"]) if name != "tab" else None,
                                   "sigma": float(best["sigma"]), "sigma_mult": float(best["sigma_mult"]),
                                   "mean_macro_f1": float(best["mean_macro_f1"]), "std_macro_f1": float(best["std_macro_f1"])}
        print(f"[exp_28] Phase A [{arm}] {name}: rep={STAGE1_BEST[arm][name]['rep']}, "
              f"sigma={STAGE1_BEST[arm][name]['sigma']:.5f}, mean_macro_f1={STAGE1_BEST[arm][name]['mean_macro_f1']:.4f}")

write_json(STAGE1_BEST, RESULTS_DIR / "stage1_best_hparams.json")
for name in ["tab", "mri", "txt"]:
    pd.concat([STAGE1_ALL[arm][name].assign(arm=arm) for arm in ["hard", "soft"]], ignore_index=True) \
        .to_csv(RESULTS_DIR / f"phasea_grid_{name}.csv", index=False)
```

`eval(best["rep"])` round-trips the `repr()` of `None`/a float/a tuple written into the CSV — safe here
since the value always originates from `MRI_REPS`/`TXT_REPS`, never from external input.

### 1.5 Phase B — LOOCV, frozen sigma, per condition

```python
def fit_predict_unimodal(name, arm, train_idx, val_idx):
    rep = STAGE1_BEST[arm][name]["rep"]
    X_tr, X_va = build_modality(name, rep, train_idx, val_idx)
    m = BrentMemKDM()
    m.sigmas_ = {name: STAGE1_BEST[arm][name]["sigma"]}
    m.modality_order = [name]
    y_tgt = Targets(y_binary=cohort.y_binary[train_idx], y_soft=TARGETS[arm].y_soft[train_idx])
    m.fit({name: X_tr}, y_tgt)
    return m.predict_proba({name: X_va})[0, 1]

def phase_b_unimodal(name, arm):
    def evaluate_fn(train_idx, val_idx):
        return {"pred": float(fit_predict_unimodal(name, arm, train_idx, val_idx))}
    oof_pred, _ = run_loocv(evaluate_fn, n) if not smoke else run_loocv_folds(evaluate_fn, n, LOOCV_FOLDS)
    return oof_pred
```

`run_loocv_folds` is the same reduced-fold local helper `exp_27` uses (`run_loocv` with a `folds` subset,
`src/evaluation/protocol.py` unmodified) — copied verbatim into `train.py`, not re-added to `protocol.py`.

```python
UNIMODAL_OOF = {arm: {name: phase_b_unimodal(name, arm) for name in ["tab", "mri", "txt"]} for arm in ["hard", "soft"]}

def to_2col(p1): return np.stack([1 - p1, p1], axis=1)

FUSION_EQUAL_OOF = {
    arm: soft_vote({name: to_2col(UNIMODAL_OOF[arm][name]) for name in ["tab", "mri", "txt"]},
                    {"tab": 1/3, "mri": 1/3, "txt": 1/3})[:, 1]
    for arm in ["hard", "soft"]
}
```

### 1.6 Reference arms — recomputed exp_6/exp_8 counterparts (DESIGN.md §5)

```python
def knn_mri_pca_evaluate_fn(train_idx, val_idx):
    X_tr, X_va = build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=0.90)  # exp_6's own pipeline: MinMax->PCA, no L2
    knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean")
    knn.fit(X_tr, cohort.y_binary[train_idx])
    p = knn.predict_proba(X_va)
    classes = list(knn.classes_)
    return {"pred": float(p[0, classes.index(1)]) if 1 in classes else 0.0}

knn_mri_pca_pred, _ = run_loocv(knn_mri_pca_evaluate_fn, n)

def exp5_knn_evaluate_fn(train_idx, val_idx):  # exp_5's exact winner, reused for the honest fusion reference
    X_tr, X_va = build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=None)
    knn = KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="euclidean")
    knn.fit(X_tr, cohort.y_binary[train_idx])
    p = knn.predict_proba(X_va)
    classes = list(knn.classes_)
    return {"pred": float(p[0, classes.index(1)]) if 1 in classes else 0.0}

knn_tab_pred, _ = run_loocv(exp5_knn_evaluate_fn, n)

def exp7_knn_evaluate_fn(train_idx, val_idx):  # exp_7's exact winner
    X_tr, X_va = g2_text_pipeline(train_idx, val_idx)
    knn = KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="cosine")
    knn.fit(X_tr, cohort.y_binary[train_idx])
    p = knn.predict_proba(X_va)
    classes = list(knn.classes_)
    return {"pred": float(p[0, classes.index(1)]) if 1 in classes else 0.0}

knn_txt_pred, _ = run_loocv(exp7_knn_evaluate_fn, n)

knn_fusion_equal_pred = soft_vote(
    {"tab": to_2col(knn_tab_pred), "mri": to_2col(knn_mri_pca_pred), "txt": to_2col(knn_txt_pred)},
    {"tab": 1/3, "mri": 1/3, "txt": 1/3})[:, 1]
```

`exp5_knn_evaluate_fn`/`exp7_knn_evaluate_fn` are identical in content to `g1_evaluate_fn`/`g2_evaluate_fn`
(§1.3) — kept as separate named closures rather than reused directly so the Step-0 gate's pass/fail
bookkeeping stays independent of the reference-arm predictions used downstream in `mcnemar_exact`.

### 1.7 Secondary conditions (DESIGN.md §3, S1/S2 — hard arm only)

```python
# S1: fusion_optimal_leakfree — simplex weights chosen on Stage-1 MCCV validation probs only (never LOOCV)
def winner_val_probs(name, arm):
    rep = STAGE1_BEST[arm][name]["rep"]
    sigma = STAGE1_BEST[arm][name]["sigma"]
    out = {}
    for split_idx, train_idx, val_idx in SPLITS:
        X_tr, X_va = build_modality(name, rep, train_idx, val_idx)
        m = BrentMemKDM(); m.sigmas_ = {name: sigma}; m.modality_order = [name]
        y_tgt = Targets(y_binary=cohort.y_binary[train_idx], y_soft=TARGETS[arm].y_soft[train_idx])
        m.fit({name: X_tr}, y_tgt)
        out[split_idx] = (val_idx, m.predict_proba({name: X_va}))
    return out

WINNER_VAL_PROBS = {name: winner_val_probs(name, "hard") for name in ["tab", "mri", "txt"]}
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
fusion_weights_leakfree = {"weights": {k: float(v) for k, v in zip(["tab","mri","txt"], grid[best_i])},
                            "mean_macro_f1": float(scores[best_i])}
write_json(fusion_weights_leakfree, RESULTS_DIR / "fusion_weights_leakfree.json")

FUSION_OPTIMAL_OOF = soft_vote({name: to_2col(UNIMODAL_OOF["hard"][name]) for name in ["tab","mri","txt"]},
                                fusion_weights_leakfree["weights"])[:, 1]

# S2: joint_trimodal — single product-kernel BrentMemKDM, coordinate strategy, hard arm only
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

joint_search_splits = SPLITS[:5] if not smoke else SPLITS  # coordinate search budget, not a smoke-only cut
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
    m = BrentMemKDM(); m.sigmas_ = dict(joint_result.sigmas); m.modality_order = ["tab", "mri", "txt"]
    y_tgt = Targets(y_binary=cohort.y_binary[train_idx], y_soft=TARGETS["hard"].y_soft[train_idx])
    m.fit(X_tr, y_tgt)
    return {"pred": float(m.predict_proba(X_va)[0, 1])}

joint_trimodal_pred, _ = run_loocv(joint_evaluate_fn, n) if not smoke else run_loocv_folds(joint_evaluate_fn, n, LOOCV_FOLDS)
```

`joint_search_splits` uses only 5 MCCV splits for the coordinate search (matching
`scripts/verify_brent_mem_kdm.py`'s own trimodal smoke budget, DESIGN.md §4) **regardless of `--smoke`** —
this is a fixed, deliberately-cheap budget for an already-secondary condition, not a smoke-mode reduction.

### 1.8 Metrics, McNemar, output

```python
loocv_metrics = {}
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
for arm, conditions in [("hard", CONDITIONS_HARD), ("soft", CONDITIONS_SOFT)]:
    for cond_name, pred in conditions.items():
        key = f"{cond_name}__{arm}"
        loocv_metrics[key] = binary_metrics(y_true_eval, pred[idx])
        loocv_metrics[key]["target_informed"] = (arm == "soft")
for ref_name, pred in REFERENCE.items():
    loocv_metrics[ref_name] = binary_metrics(y_true_eval, pred[idx])

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
write_json(loocv_metrics, RESULTS_DIR / "loocv_metrics.json")

df_oof = pd.DataFrame({"patient_id": cohort.pids, "ground_truth_biopsy": cohort.y_binary})
for arm, conditions in [("hard", CONDITIONS_HARD), ("soft", CONDITIONS_SOFT)]:
    for cond_name, pred in conditions.items():
        df_oof[f"prob_{cond_name}__{arm}"] = pred
for ref_name, pred in REFERENCE.items():
    df_oof[f"prob_{ref_name}"] = pred
df_oof.to_csv(RESULTS_DIR / "loocv_predictions.csv", index=False)
```

Published-but-unreproducible reference numbers (`exp_6` 0.5335, `exp_8` 0.7171) are **not** computed here
(nothing to compute — they require `embedkit`) — `reports/summary.md` states them as literal quoted
constants from `experiments/exp_6/results/loocv_metrics.json` / `experiments/exp_8/results/loocv_metrics.json`,
labeled not-reproducible-in-this-checkout, alongside this experiment's own `knn_mri_pca`/`knn_fusion_equal`.

### 1.9 Figures and git commit

```python
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
```

## 2. Anti-leak assertions (DESIGN.md §4, "verification")

```python
assert all(len(f.y_val) >= 1 for f in folds)          # sanity: folds built from MCCV, not degenerate
assert len(SPLITS) <= 100 and all(s[0] < 100 for s in SPLITS)  # SPLITS always sourced from iter_mccv_splits
```

`sigmas_`/`modality_order` are set exactly once per `BrentMemKDM` instance created inside each
`evaluate_fn`/`phase_b_*` closure (a fresh instance per fold, never reused across folds) — by
construction, not by an added runtime check, since each closure creates its own `BrentMemKDM()`.

## 3. Command Lines

```bash
cd /Users/fgonza/Documents/research/code/prostate-cancer-reasoning
conda activate pytorch   # verified: torch, kdm, sklearn, spacy, pandas all importable here

# smoke test first (5 MCCV splits for Stage 1 / unimodal Phase B, 6 LOOCV folds)
python experiments/exp_28/scripts/train.py --smoke

# full run
python experiments/exp_28/scripts/train.py
```

## 4. `--smoke` flag

```python
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
smoke = args.smoke
```

Reduces `SPLITS` (Phase A, unimodal Phase B loop bound) and `LOOCV_FOLDS`; does **not** reduce G0/G1/G2
(§1.3, run in full always) or S2's joint-search split count (§1.7, already fixed small independent of
`--smoke`).

## 5. Post-Execution (after results exist and are reviewed)

- Confirm `results/reproduction_gates.json` shows G0/G1/G2 all `passed: true`.
- Compare `results/loocv_metrics.json`'s `fusion_equal__hard` to exp_8's 0.7171 and to
  `knn_fusion_equal` (this run's own honest reference) — report both deltas, per `DESIGN.md` §8 H1.
- Compare `tab__hard` to 0.6333 and `txt__hard` to 0.6988 (H2, like-for-like); `mri__hard` to
  `knn_mri_pca` only.
- Report `sigma_mult` for every condition against exp_27's frozen grid winners (`tab` 0.5, `mri` 1.0,
  `txt` 0.5) and exp_25's edge-of-grid `sigma_scale=2.0` pattern (DESIGN.md §9's secondary readout).
- Write `reports/summary.md` (ml-experiment-reporter conventions), framed against `DESIGN.md` §8's
  decision rules — explicit about the `std_macro_f1 ≈ 0.11` caveat before calling anything a win.
- Add exp_28's row to `experiments/INDEX.md`.
- Commit `src/methods/mem_kdm.py`, `src/methods/brent_mem_kdm.py`, `scripts/verify_brent_mem_kdm.py`,
  `experiments/exp_25`–`27`, and `experiments/exp_28` together (§0) — only after the user has reviewed
  results, per `CLAUDE.md`'s workflow.
- Append `.logbook.md`/`.discussion.md` entries once the user has reviewed and approved the results —
  only record points the user has explicitly approved.
