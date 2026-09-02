# Implementation Plan: Bayesian Evidence-Combination Fusion — Log-Linear Pooling and a Non-Parametric KDM Reliability Gate (exp_31)
**Experiment**: experiments/exp_31/ · **Status**: In Progress (results in, pending review)

See `DESIGN.md` for motivation, scope, and decision rules. Unlike exp_29 (a close diff against
exp_28's script), this experiment has no direct prior script to diff against — it consumes exp_29's
**results** (`stage1_best_hparams.json`, `loocv_metrics.json`, `fusion_weights_leakfree.json`) as
frozen inputs, not its code. `scripts/train.py` is self-contained, built the same way every prior
experiment in this lineage is: `src/evaluation/*` and `src/methods/{base,mem_kdm,brent_mem_kdm}.py`
reused unmodified (§9 of DESIGN.md — nothing there changes).

**Prerequisite check (before writing code):** `experiments/fagonzalezo/exp_29/results/
stage1_best_hparams.json`, `loocv_metrics.json`, and `fusion_weights_leakfree.json` must exist and be
readable — they are load-bearing inputs, not just context. Confirmed present (`experiments/fagonzalezo/
exp_29/results/`, read during design).

## 1. `experiments/exp_31/scripts/train.py`

### 1.1 Imports and setup

```python
"""exp_31 — Bayesian evidence-combination fusion. See ../DESIGN.md.

Reloads experiments/fagonzalezo/exp_29's frozen hard-arm per-modality winners
(stage1_best_hparams.json) verbatim -- no per-modality (rep, knn_k, sigma) search happens in this
script. Only the fusion layer is new: C1 (arithmetic pool, matched-input reference), C2 (log-linear
pool), C3 (non-parametric per-patient KDM reliability gate).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from kdm.init import _sigma_from_knn  # noqa: E402 -- gate bandwidth default, DESIGN.md §2.4

from src.evaluation.data import (  # noqa: E402
    OLD_SCHEMA, build_mri_features, build_tabular_features, build_text_features,
    clean_texts_spacy, load_cohort, resolve_data_dir,
)
from src.evaluation.metrics import binary_metrics, mcnemar_exact  # noqa: E402
from src.evaluation.protocol import iter_mccv_splits, run_loocv  # noqa: E402
from src.evaluation.reporting import (  # noqa: E402
    plot_confusion_matrix, plot_roc_curves, record_git_commit, write_json,
)
from src.methods.base import Targets  # noqa: E402
from src.methods.brent_mem_kdm import BrentMemKDM  # noqa: E402
from src.methods.mem_kdm import simplex_grid, soft_vote  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
EXP29_RESULTS = PROJECT_ROOT / "experiments" / "fagonzalezo" / "exp_29" / "results"
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "reports" / "figures"

LABEL_SMOOTHING = 0.05  # DESIGN.md §2.2 -- fixed a priori, not searched
MODALITIES = ["tab", "mri", "txt"]


def to_2col(p1: np.ndarray) -> np.ndarray:
    return np.stack([1 - p1, p1], axis=1)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def logit_pool(probs: dict, weights: dict) -> np.ndarray:
    """Weighted geometric mean in probability space (DESIGN.md §3.2). probs[m]: (n, 2)."""
    fused_logit = sum(weights[m] * logit(probs[m][:, 1]) for m in probs)
    p1 = 1.0 / (1.0 + np.exp(-fused_logit))
    return np.stack([1 - p1, p1], axis=1)


def run_loocv_folds(evaluate_fn, n: int, folds):
    """Local `--smoke` variant of `protocol.run_loocv`, restricted to `folds` -- same idiom exp_29
    uses (`experiments/fagonzalezo/exp_29/scripts/train.py:63-72`)."""
    oof_pred = np.zeros(n)
    all_idx = np.arange(n)
    for i in folds:
        train_idx, val_idx = all_idx[all_idx != i], np.array([i])
        result = evaluate_fn(train_idx, val_idx)
        oof_pred[i] = result["pred"]
    return oof_pred
```

### 1.2 Cohort, splits, frozen per-modality winners

```python
def main(smoke: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    data_dir = resolve_data_dir(PROJECT_ROOT)
    cohort = load_cohort(data_dir, PROJECT_ROOT, OLD_SCHEMA, load_mri=True, load_text=True)
    n = len(cohort.y_binary)
    y_bin = cohort.y_binary
    print(f"[exp_31] cohort N={n}, yes={int(y_bin.sum())}, no={int((1 - y_bin).sum())}")

    cleaned_texts = clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)

    FULL_SPLITS = list(iter_mccv_splits(cohort.df_design, n_splits=100))
    SPLITS = FULL_SPLITS[:5] if smoke else FULL_SPLITS
    LOOCV_FOLDS = list(range(6)) if smoke else list(range(n))
    print(f"[exp_31] {'SMOKE MODE: ' if smoke else ''}{len(SPLITS)} MCCV splits (Phase A), "
          f"{len(FULL_SPLITS)} MCCV splits (gates, always full), {len(LOOCV_FOLDS)} LOOCV folds")

    def build_modality(name, rep, train_idx, val_idx):
        if name == "tab":
            return build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
        if name == "mri":
            return build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=rep)
        if name == "txt":
            max_features, pca_variance = rep
            return build_text_features(cleaned_texts, train_idx, val_idx, max_features=max_features, pca_variance=pca_variance)
        raise ValueError(name)

    with open(EXP29_RESULTS / "stage1_best_hparams.json") as f:
        STAGE1_BEST_HARD = json.load(f)["hard"]
    for name in MODALITIES:  # JSON round-trips MRI/txt `rep` tuples as lists -- restore tuples
        rep = STAGE1_BEST_HARD[name]["rep"]
        STAGE1_BEST_HARD[name]["rep"] = tuple(rep) if isinstance(rep, list) else rep
    with open(EXP29_RESULTS / "loocv_metrics.json") as f:
        EXP29_METRICS = json.load(f)
    with open(EXP29_RESULTS / "fusion_weights_leakfree.json") as f:
        EXP29_S1_WEIGHTS = json.load(f)
    print(f"[exp_31] Loaded exp_29 frozen winners: {STAGE1_BEST_HARD}")

    def fit_predict_unimodal(name, train_idx, val_idx, label_smoothing):
        best = STAGE1_BEST_HARD[name]
        X_tr, X_va = build_modality(name, best["rep"], train_idx, val_idx)
        m = BrentMemKDM(knn_k=best["knn_k"], label_smoothing=label_smoothing)
        m.sigmas_ = {name: best["sigma"]}
        m.modality_order = [name]
        m.fit({name: X_tr}, Targets(y_binary=y_bin[train_idx], y_soft=y_bin[train_idx].astype(np.float32)))
        return float(m.predict_proba({name: X_va})[0, 1])

    def phase_b_unimodal(name, label_smoothing):
        def evaluate_fn(train_idx, val_idx):
            return {"pred": fit_predict_unimodal(name, train_idx, val_idx, label_smoothing)}
        if smoke:
            return run_loocv_folds(evaluate_fn, n, LOOCV_FOLDS)
        return run_loocv(evaluate_fn, n)[0]

    def winner_val_probs(splits, label_smoothing):
        """{split_idx: (val_idx, {modality: (n_val, 2) probs})} -- input to both the C1/C2 weight
        search and C3's evidence table (DESIGN.md §3.1)."""
        out = {}
        for split_idx, train_idx, val_idx in splits:
            probs = {}
            for name in MODALITIES:
                best = STAGE1_BEST_HARD[name]
                X_tr, X_va = build_modality(name, best["rep"], train_idx, val_idx)
                m = BrentMemKDM(knn_k=best["knn_k"], label_smoothing=label_smoothing)
                m.sigmas_ = {name: best["sigma"]}
                m.modality_order = [name]
                m.fit({name: X_tr}, Targets(y_binary=y_bin[train_idx], y_soft=y_bin[train_idx].astype(np.float32)))
                probs[name] = m.predict_proba({name: X_va})
            out[split_idx] = (val_idx, probs)
        return out

    def search_weights(val_probs_by_split, pool_fn):
        """Simplex-grid weight search (DESIGN.md §2.4/§3.2), scored by mean Macro-F1 over the given
        MCCV splits -- structurally identical to exp_29 S1's own loop, `pool_fn` swapped in."""
        grid = simplex_grid(3, step=0.05)
        scores = np.zeros(len(grid))
        for _split_idx, (val_idx, probs) in val_probs_by_split.items():
            for i, (wt, wm, wx) in enumerate(grid):
                p_fused = pool_fn(probs, dict(tab=wt, mri=wm, txt=wx))
                y_pred = (p_fused[:, 1] >= 0.50).astype(int)
                scores[i] += f1_score(y_bin[val_idx], y_pred, average="macro", zero_division=0)
        scores /= len(val_probs_by_split)
        best_i = int(np.argmax(scores))
        wt, wm, wx = grid[best_i]
        return {"weights": {"tab": float(wt), "mri": float(wm), "txt": float(wx)},
                "mean_macro_f1": float(scores[best_i])}
```

### 1.3 Step 0 — reproduction gates (DESIGN.md §5)

Always run against `FULL_SPLITS`/full `run_loocv`, regardless of `--smoke` (same convention as
exp_29's G0–G3: a gate that gets skipped under `--smoke` isn't one).

```python
    print("[exp_31] G0: reload exp_29's frozen winners, refit at label_smoothing=0, compare to exp_29's stored numbers...")
    g0_results = {}
    for name in MODALITIES:
        def evaluate_fn(train_idx, val_idx, name=name):
            return {"pred": fit_predict_unimodal(name, train_idx, val_idx, label_smoothing=0.0)}
        pred, _ = run_loocv(evaluate_fn, n)
        got = binary_metrics(y_bin, pred)["macro_f1"]
        expected = EXP29_METRICS[f"{name}__hard"]["macro_f1"]
        g0_results[name] = {"got": got, "expected": expected, "passed": abs(got - expected) < 1e-9}
    g0_pass = all(v["passed"] for v in g0_results.values())

    print("[exp_31] G1: reproduce exp_29 S1's simplex weight search and fusion_optimal_leakfree number at label_smoothing=0...")
    WINNER_VAL_PROBS_0 = winner_val_probs(FULL_SPLITS, label_smoothing=0.0)
    g1_weights = search_weights(WINNER_VAL_PROBS_0, soft_vote)
    g1_weights_pass = all(
        abs(g1_weights["weights"][m] - EXP29_S1_WEIGHTS["weights"][m]) < 1e-9 for m in MODALITIES
    )
    # G1's Phase-B check must run the full 88-fold LOOCV regardless of --smoke (phase_b_unimodal is
    # --smoke-aware and would give a 6-fold subset otherwise), so it is not reused here:
    UNIMODAL_OOF_0_FULL = {
        name: run_loocv(lambda tr, va, name=name: {"pred": fit_predict_unimodal(name, tr, va, 0.0)}, n)[0]
        for name in MODALITIES
    }
    fusion_g1_pred = soft_vote({m: to_2col(UNIMODAL_OOF_0_FULL[m]) for m in MODALITIES}, g1_weights["weights"])[:, 1]
    g1_macro_f1 = binary_metrics(y_bin, fusion_g1_pred)["macro_f1"]
    g1_phaseb_pass = abs(g1_macro_f1 - EXP29_METRICS["fusion_optimal_leakfree__hard"]["macro_f1"]) < 1e-9
    g1_pass = g1_weights_pass and g1_phaseb_pass

    reproduction_gates = {
        "G0_reload_exp29_winners": {"passed": bool(g0_pass), "per_modality": g0_results},
        "G1_reproduce_exp29_s1": {
            "passed": bool(g1_pass), "weights_match": bool(g1_weights_pass), "phaseb_match": bool(g1_phaseb_pass),
            "got_weights": g1_weights["weights"], "expected_weights": EXP29_S1_WEIGHTS["weights"],
            "got_macro_f1": g1_macro_f1, "expected_macro_f1": EXP29_METRICS["fusion_optimal_leakfree__hard"]["macro_f1"],
        },
    }
    write_json(reproduction_gates, RESULTS_DIR / "reproduction_gates.json")
    assert g0_pass, f"G0 FAILED: {g0_results}"
    assert g1_pass, f"G1 FAILED: {reproduction_gates['G1_reproduce_exp29_s1']}"
    print(f"[exp_31] Step 0 PASSED: G0={g0_pass}, G1={g1_pass}")
```

**Verified**: a design-time diagnostic already ran exactly this G0/G1 sequence against the live cohort
(throwaway pilot script, not this file) — `g0_pass=True` for all three modalities (exact match to
`1e-16`), `g1_weights_pass=True` (`(0.4, 0.1, 0.5)` exactly), `g1_phaseb_pass=True`
(`0.7067941474721136` exactly). G0/G1 are expected to pass on the first real run, not merely hoped to.

### 1.4 Phase A + Phase B — C1, C2 (DESIGN.md §3.1–3.2)

```python
    print("[exp_31] Phase A: C1/C2 evidence table + weight search (label_smoothing=0.05)...")
    WINNER_VAL_PROBS = winner_val_probs(SPLITS, label_smoothing=LABEL_SMOOTHING)

    c1_result = search_weights(WINNER_VAL_PROBS, soft_vote)
    write_json(c1_result, RESULTS_DIR / "fusion_weights_arithmetic.json")
    print(f"[exp_31] C1 weights: {c1_result}")

    c2_result = search_weights(WINNER_VAL_PROBS, logit_pool)
    write_json(c2_result, RESULTS_DIR / "fusion_weights_loglinear.json")
    print(f"[exp_31] C2 weights: {c2_result}")

    print("[exp_31] Phase B: unimodal experts (label_smoothing=0.05)...")
    UNIMODAL_SMOOTHED_OOF = {name: phase_b_unimodal(name, LABEL_SMOOTHING) for name in MODALITIES}
    for name in MODALITIES:
        idx0 = np.array(LOOCV_FOLDS)
        f1 = f1_score(y_bin[idx0], (UNIMODAL_SMOOTHED_OOF[name][idx0] >= 0.5).astype(int), average="macro", zero_division=0)
        print(f"[exp_31] unimodal_{name}_smoothed: macro_f1={f1:.4f}")

    C1_PRED = soft_vote({m: to_2col(UNIMODAL_SMOOTHED_OOF[m]) for m in MODALITIES}, c1_result["weights"])[:, 1]
    C2_PRED = logit_pool({m: to_2col(UNIMODAL_SMOOTHED_OOF[m]) for m in MODALITIES}, c2_result["weights"])[:, 1]
```

### 1.5 C3 — the KDM reliability gate (DESIGN.md §3.3)

```python
    print("[exp_31] C3: building evidence table and reliability gate...")

    def build_evidence_table(val_probs_by_split):
        rows = []
        for split_idx, (val_idx, probs) in val_probs_by_split.items():
            for pos, patient_idx in enumerate(val_idx):
                e = {m: float(probs[m][pos, 1]) for m in MODALITIES}
                y_true = int(y_bin[patient_idx])
                row = {"patient_idx": int(patient_idx), "split_idx": int(split_idx), "y_true": y_true,
                       "e_tab": e["tab"], "e_mri": e["mri"], "e_txt": e["txt"]}
                for m in MODALITIES:
                    # >= 0.5, matching binary_metrics' threshold convention -- NOT round(), which is
                    # banker's rounding and maps e_m == 0.5 to 0 (DESIGN.md §3.3).
                    row[f"correct_{m}"] = int((e[m] >= 0.5) == (y_true == 1))
                rows.append(row)
        return pd.DataFrame(rows)

    EVIDENCE_TABLE = build_evidence_table(WINNER_VAL_PROBS)

    # Gate bandwidth: computed on a DEDUPLICATED (one row per patient, mean evidence vector) table, not
    # the raw duplicated one. DESIGN.md §3.3 -- a design-time diagnostic found the raw-table bandwidth
    # collapses to an effective sample size of ~1-12 (out of ~1700 rows) once a query patient's own rows
    # are excluded, because same-patient rows across splits are near-duplicates and dominate the
    # nearest-neighbor distance the heuristic is based on. The deduplicated bandwidth gave ESS ~76-357
    # on the same check -- a stable neighborhood. The averaging memory itself still uses every
    # split-appearance as a distinct row; only the bandwidth comes from the deduplicated table.
    E_DEDUP = EVIDENCE_TABLE.groupby("patient_idx")[["e_tab", "e_mri", "e_txt"]].mean().values.astype(np.float64)
    GATE_SIGMA = float(_sigma_from_knn(E_DEDUP, sigma_mult=1.0))  # DESIGN.md §2.4 -- not Brent-searched
    write_json({"gate_sigma": GATE_SIGMA, "n_evidence_rows": int(len(EVIDENCE_TABLE)),
                "n_dedup_patients": int(len(E_DEDUP))}, RESULTS_DIR / "gate_sigma.json")
    print(f"[exp_31] evidence table: {len(EVIDENCE_TABLE)} rows over {len(E_DEDUP)} patients, "
          f"gate_sigma={GATE_SIGMA:.5f} (from deduplicated table)")

    def modality_reliability(e_query, exclude_patient, modality):
        rows = EVIDENCE_TABLE[EVIDENCE_TABLE.patient_idx != exclude_patient]
        E = rows[["e_tab", "e_mri", "e_txt"]].values.astype(np.float64)
        d2 = ((E - e_query) ** 2).sum(axis=1)
        w = np.exp(-d2 / (2 * GATE_SIGMA ** 2))
        if w.sum() < 1e-300:
            return float(rows[f"correct_{modality}"].mean())
        return float(np.average(rows[f"correct_{modality}"].values, weights=w))

    def gate_weights_for_patient(patient_idx):
        e_query = np.array([UNIMODAL_SMOOTHED_OOF[m][patient_idx] for m in MODALITIES])
        r = np.array([modality_reliability(e_query, patient_idx, m) for m in MODALITIES])
        if r.sum() < 1e-8:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        return r / r.sum()

    GATE_WEIGHTS = np.full((n, 3), np.nan)
    for i in LOOCV_FOLDS:
        GATE_WEIGHTS[i] = gate_weights_for_patient(i)

    C3_PRED = np.full(n, np.nan)
    for i in LOOCV_FOLDS:
        wt, wm, wx = GATE_WEIGHTS[i]
        C3_PRED[i] = (wt * UNIMODAL_SMOOTHED_OOF["tab"][i] + wm * UNIMODAL_SMOOTHED_OOF["mri"][i]
                      + wx * UNIMODAL_SMOOTHED_OOF["txt"][i])

    idx = np.array(LOOCV_FOLDS)
    gate_df = pd.DataFrame({"patient_id": np.array(cohort.pids)[idx], "w_tab": GATE_WEIGHTS[idx, 0],
                             "w_mri": GATE_WEIGHTS[idx, 1], "w_txt": GATE_WEIGHTS[idx, 2]})
    gate_df.to_csv(RESULTS_DIR / "gate_weights_per_patient.csv", index=False)
    write_json({
        "mean": {"tab": float(gate_df.w_tab.mean()), "mri": float(gate_df.w_mri.mean()), "txt": float(gate_df.w_txt.mean())},
        "std": {"tab": float(gate_df.w_tab.std()), "mri": float(gate_df.w_mri.std()), "txt": float(gate_df.w_txt.std())},
        "exp29_fixed_weights": EXP29_S1_WEIGHTS["weights"],
    }, RESULTS_DIR / "gate_weights_summary.json")
```

`GATE_WEIGHTS`/`C3_PRED` are `np.nan`-initialized (not zero) so an accidental miss in the `LOOCV_FOLDS`
loop is loud (NaN propagates into `binary_metrics`/plots and fails visibly) rather than silently
scoring as a wrong-but-valid prediction.

### 1.6 Metrics, McNemar (DESIGN.md §7), output

```python
    y_true_eval = y_bin[idx]
    CONDITIONS = {
        "fusion_arithmetic_smoothed": C1_PRED, "fusion_loglinear": C2_PRED, "fusion_kdm_gate": C3_PRED,
        "unimodal_tab_smoothed": UNIMODAL_SMOOTHED_OOF["tab"], "unimodal_mri_smoothed": UNIMODAL_SMOOTHED_OOF["mri"],
        "unimodal_txt_smoothed": UNIMODAL_SMOOTHED_OOF["txt"],
    }
    loocv_metrics = {f"{k}__hard": binary_metrics(y_true_eval, v[idx]) for k, v in CONDITIONS.items()}
    write_json(loocv_metrics, RESULTS_DIR / "loocv_metrics.json")

    y_c1 = (C1_PRED[idx] >= 0.50).astype(int)
    y_c2 = (C2_PRED[idx] >= 0.50).astype(int)
    y_c3 = (C3_PRED[idx] >= 0.50).astype(int)
    mcnemar_results = {
        "H1_loglinear_vs_arithmetic": mcnemar_exact(y_true_eval, y_c2, y_c1),
        "H2_gate_vs_arithmetic": mcnemar_exact(y_true_eval, y_c3, y_c1),
        "context_gate_vs_loglinear": mcnemar_exact(y_true_eval, y_c3, y_c2),
    }
    write_json(mcnemar_results, RESULTS_DIR / "mcnemar.json")

    df_oof = pd.DataFrame({"patient_id": cohort.pids, "ground_truth_biopsy": y_bin})
    for k, v in CONDITIONS.items():
        df_oof[f"prob_{k}"] = v
    df_oof.to_csv(RESULTS_DIR / "loocv_predictions.csv", index=False)

    print(f"[exp_31] C1 (arithmetic_smoothed): {loocv_metrics['fusion_arithmetic_smoothed__hard']['macro_f1']:.4f}")
    print(f"[exp_31] C2 (loglinear):           {loocv_metrics['fusion_loglinear__hard']['macro_f1']:.4f}")
    print(f"[exp_31] C3 (kdm_gate):            {loocv_metrics['fusion_kdm_gate__hard']['macro_f1']:.4f}")
    print(f"[exp_31] exp_29 fusion_optimal_leakfree (context, not directly comparable): "
          f"{EXP29_METRICS['fusion_optimal_leakfree__hard']['macro_f1']:.4f}")
    print(f"[exp_31] H1 p={mcnemar_results['H1_loglinear_vs_arithmetic']['pvalue']:.4f}, "
          f"H2 p={mcnemar_results['H2_gate_vs_arithmetic']['pvalue']:.4f}")
```

### 1.7 Figures and git commit

```python
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.boxplot([gate_df.w_tab, gate_df.w_mri, gate_df.w_txt], tick_labels=["tab", "mri", "txt"])
        for pos, m in enumerate(["tab", "mri", "txt"], start=1):
            plt.scatter([pos], [EXP29_S1_WEIGHTS["weights"][m]], color="red", marker="D", zorder=5,
                        label="exp_29 fixed weight" if pos == 1 else None)
        plt.ylabel("gate weight (per LOOCV patient)")
        plt.title("exp_31 C3: per-patient gate weights vs. exp_29's fixed (0.4, 0.1, 0.5)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "gate_weights_boxplot.png", dpi=300)
        plt.close()

        plot_confusion_matrix(y_true_eval, y_c3, labels=["No", "Yes"],
                               title="exp_31 fusion_kdm_gate (hard) LOOCV", out_path=FIG_DIR / "confusion_matrix.png")
        plot_roc_curves({"fusion_arithmetic_smoothed": C1_PRED[idx], "fusion_loglinear": C2_PRED[idx],
                          "fusion_kdm_gate": C3_PRED[idx]}, y_true_eval,
                         title="exp_31 ROC overlay", out_path=FIG_DIR / "roc_curve.png")
    except Exception as e:  # pragma: no cover -- figures are best-effort, never block numeric results
        print(f"[exp_31] WARNING: figure generation failed: {e}")

    record_git_commit(RESULTS_DIR)
    elapsed = time.time() - t_start
    print(f"[exp_31] DONE in {elapsed / 60:.1f} min. Results in {RESULTS_DIR}")
```

### 1.8 `--smoke` flag

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(smoke=args.smoke)
```

Reduces `SPLITS` (C1/C2/C3's Phase-A weight/evidence-table construction) and `LOOCV_FOLDS` (Phase-B
unimodal experts and C3's gate loop). Does **not** reduce G0/G1 (§1.3, always full `FULL_SPLITS`/
`run_loocv`) — same convention as exp_29.

## 2. Anti-leak assertions

- `WINNER_VAL_PROBS`/`EVIDENCE_TABLE` are built from `SPLITS` (MCCV), never from `LOOCV_FOLDS` — by
  construction, no shared code path between the two.
- `modality_reliability` filters `EVIDENCE_TABLE` by `patient_idx != exclude_patient` on every call —
  this is the one line that makes C3's gate leak-free at the memory level (DESIGN.md §2.4); if this
  filter is ever accidentally dropped, add an explicit assertion:
  `assert exclude_patient not in rows["patient_idx"].values` immediately after the filter, as a
  standing regression check.
- G0/G1 (§1.3) fail loudly (`assert`) rather than warn — a silent reload/reproduction mismatch would
  invalidate every downstream number.

## 3. Command Lines

```bash
cd /Users/fgonza/Documents/research/code/prostate-cancer-reasoning
conda activate pytorch   # NOT histo-DL -- see project memory

# smoke test first (5 MCCV splits for Phase A, 6 LOOCV folds; G0/G1 always run full)
python experiments/fagonzalezo/exp_31/scripts/train.py --smoke

# full run
python experiments/fagonzalezo/exp_31/scripts/train.py
```

## 4. Post-Execution (after results exist and are reviewed)

- Confirm `results/reproduction_gates.json` shows G0 and G1 both `passed: true` (G1 specifically:
  `weights_match` and `phaseb_match` both `true`, matching exp_29's `(0.4, 0.1, 0.5)` / `0.7068`
  exactly) before trusting anything else in `results/`.
- Report `loocv_metrics.json`'s `fusion_arithmetic_smoothed__hard` next to exp_29's `0.7068` — expect
  close but not identical (DESIGN.md §2.2); a large divergence would suggest `label_smoothing=0.05`
  is doing more than intended and is worth investigating before trusting H1/H2.
- Report H1 (`mcnemar.json`'s `H1_loglinear_vs_arithmetic`) and H2 (`H2_gate_vs_arithmetic`) exactly as
  DESIGN.md §7 specifies — no result without significance is a win, matching this lineage's standard.
- Report `gate_weights_summary.json`'s mean ± std next to exp_29's `(0.4, 0.1, 0.5)`, and flag the
  DESIGN.md §2.4 leakage-asymmetry caveat explicitly if C3 outperforms C1/C2.
- Write `reports/summary.md` (`ml-experiment-reporter` conventions), framed against `DESIGN.md` §7.
- Flip `DESIGN.md`'s `Status` to **In Progress** once the full run launches, then **Complete** once
  reviewed (matches exp_29's own convention).
- Append `.logbook.md`/`.discussion.md` entries only after the user has reviewed and approved the
  results (`CLAUDE.md` workflow) — not before.
- Add a row to `experiments/INDEX.md` only if/when this experiment is folded into the main-track index
  (exp_25–30 are not currently listed there either — confirm with the user whether exp_31 should be,
  before adding it).
