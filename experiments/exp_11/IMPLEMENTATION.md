"# Implementation Details: Trimodal Linear-SVM Late Fusion (Hard SVC + Soft-SVR) for Task 2 (exp_11)

**Experiment**: `experiments/exp_11/`
**Base**: `exp_10` runner structure (adapted)
**Protocol Governance**: `chimera_metrics.task2.decision.compute_treatment_decision_gate`
**Evaluation Standard**: sklearn homologation for Macro/Micro/Weighted-F1, AUROC OVR, Brier multiclass
**Date**: 2026-08-25
**Status**: Approved (DESIGN.md) — pending implementation approval

---

## 1. Architecture & Code Structure

```
experiments/exp_11/
├── DESIGN.md                  <- research design (approved)
├── IMPLEMENTATION.md          <- this file
├── scripts/
│   └── run_exp11_mccv.py      <- single runner (Phase A + Phase B + metrics)
├── results/
│   ├── version.txt
│   ├── aggregate_metrics.json
│   ├── split_metrics.json
│   ├── best_hyperparams.json
│   └── wilcoxon_tests.json
└── reports/
    └── summary.md             <- final write-up (experiment-reporter)
```

**No new preprocessing scripts needed**: reuses `data/processed/task2/` CSVs and `mccv_splits_100.csv` from `exp_10`.

---

## 2. Script: `run_exp11_mccv.py`

### 2.1 Reused components from `exp_10` / `exp_4`

| Component | Action | Detail |
| :--- | :--- | :--- |
| `compute_split_metrics()` | **Direct reuse** | From `exp_10` runner (Macro/Micro/Weighted-F1, AUROC OVR, Brier, gate_acc, per_class_f1) |
| `load_task2_splits()` | **Direct reuse** | Custom loader for `mccv_splits_100.csv` |
| `prepare_fold_cache()` | **Adapted** | `Task2TabularPreprocessor` uses `StandardScaler` (not `MinMaxScaler`); MRI gets `StandardScaler` |
| `run_fusion_search()` | **Direct reuse** | Simplex Δ² step 0.05, 231 weights |
| `CONFIDENCE_C`, `CLASS_ORDER`, soft-target formula | **Direct reuse** | `y_k = 0.25 + 0.75c`, `y_j≠k = (1-y_k)/3` |

### 2.2 New components (SVM replacement)

```
class HardLinearSVM:
    """SVC(kernel='linear', probability=True) wrapper for multiclass 4-class.
    
    predict_proba -> Platt-scaled probabilities (n_val, 4)
    """

class SoftLinearSVR:
    """MultiOutputRegressor(LinearSVR) on 4-d soft targets.
    
    fit(X_train, y_multiclass, y_confidence):
        build 4-d soft targets (same formula as exp_10 fuzzy)
    predict_proba(X_val) -> (n_val, 4):
        raw = self.model.predict(X_val)
        clipped = np.clip(raw, 0, 1)
        return L1-normalized rows (simplex)
    """
```

### 2.3 Phase A: Grid definitions (per family)

| Modality | Hard (SVC) | Fuzzy (SVR) |
| :--- | :--- | :--- |
| Tabular | C {0.1, 1.0, 10.0} × class_weight {None, 'balanced'} = 6 | C {0.1, 1.0, 10.0} × epsilon {0.05, 0.1} = 6 |
| MRI | C {0.1, 1.0, 10.0} × class_weight {None, 'balanced'} = 6 | C {0.1, 1.0, 10.0} × epsilon {0.05, 0.1} = 6 |
| Texto | C {0.1, 1.0, 10.0} × class_weight {None, 'balanced'} = 6 | C {0.1, 1.0, 10.0} × epsilon {0.05, 0.1} = 6 |

**Total**: 18 configs × 100 splits = 1,800 unimodal evals per family. Much faster than KNN.

### 2.4 Fallback mechanisms (pre-registered)

| Condition | Fallback | When |
| :--- | :--- | :--- |
| `LinearSVC` fails to converge on MRI 1024D (N≈49) | `PCA(n_components=64)` fitted on train fold, then `LinearSVC` on reduced space | `solver` warning or `ConvergenceWarning` |
| Platt probabilities poorly calibrated (high Brier) | Use `decision_function` → softmax (temperature=1.0) instead of `probability=True` probabilities | If Brier > 0.60 in validation (vs 0.52 exp_10) |
| `LinearSVR` outputs out-of-[0,1] | `np.clip(raw, 0, 1)` + L1 row normalization | Always applied (not optional) |

### 2.5 Fusion & Phase B

- `run_fusion_search()`: identical to exp_10 (231 weight vectors per family, select by mean val Macro-F1).
- `run_loocv()`: identical workflow; for each holdout: fit StandardScaler + SVM on 69 train, predict on 1 val.
- **Cross-experiment tests**: load `exp_10/results/split_metrics.json` in memory, run Wilcoxon paired between exp_11 vs exp_10 per family.

### 2.6 Metrics & statistical tests (unchanged from exp_10)

- `compute_split_metrics()` — same function.
- Wilcoxon paired (two-sided, α=0.05):
  1. H1 (SVM-hard) vs H2 (SVR-fuzzy) internal.
  2. exp_11-H1 vs exp_10-H1 (cross).
  3. exp_11-H2 vs exp_10-H2 (cross).
- McNemar exact (4×4) only if Wilcoxon p ∈ [0.05, 0.10] for any paired comparison.

---

## 3. Storage Footprint (Minimal)

Identical to exp_10:
- `aggregate_metrics.json` (~3 KB), `split_metrics.json` (100 rows × 2 conditions ≈ 90 KB), `wilcoxon_tests.json` (~2 KB), `best_hyperparams.json` (~1 KB), `version.txt`.
- NO prediction CSVs; probability arrays in-memory only.

---

## 4. Governance Compliance

1. Decision gate per-case via `chimera_metrics.task2.compute_treatment_decision_gate`.
2. Fixed partitions reused from `exp_10` (`mccv_splits_100.csv`) — cross-experiment paired tests valid.
3. Zero leakage: `active_treatment_flag` excluded; `StandardScaler`/TF-IDF fit only on train fold.
4. Metric homologation documented (same mapping as exp_10).
5. Fallbacks pre-registered (PCA-64, softmax from decision_function) — any use is logged in `version.txt`.

---

## 5. Execution Order

```bash
# from repo root
python experiments/exp_11/scripts/run_exp11_mccv.py
```

Estimated wall-time: **< 5 min** CPU (16 threads) — SVM linear on N≈49 is fast.

---

## 6. Acceptance Criteria

- [ ] Runner completes Phase A + Phase B without convergence warnings unhandled.
- [ ] `best_hyperparams.json` contains 6 entries (3 hard + 3 fuzzy modality configs) + 2 fusion weight vectors.
- [ ] `aggregate_metrics.json` contains all 6 metrics per condition + per-class F1 + LOOCV results.
- [ ] `wilcoxon_tests.json` contains 4 paired tests (internal H1/H2, cross H1/H2 vs exp_10).
- [ ] Storage minimal (no prediction CSVs).
- [ ] Any fallback used (PCA-64, softmax) is explicitly logged in `version.txt`.

---

## 7. Next Steps

1. **PI approval of this IMPLEMENTATION.md.**
2. Build `run_exp11_mccv.py` (adapt exp_10 runner).
3. Execute in background (no tmux available).
4. Run `experiment-reporter` → `reports/summary.md`.
5. Update `.logbook.md` + `.discussion.md` after consensus.

---