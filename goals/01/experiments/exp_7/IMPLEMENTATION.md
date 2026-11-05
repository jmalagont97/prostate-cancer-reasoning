# Implementation Plan: KNN on Standardized MRI Embedding (exp_7)
**Experiment**: experiments/exp_7/ · **Project**: pathology-reasoning · **Date**: 2026-08-16 · **Status**: Approved

---

## 1. Overview

Adapt `exp_6` runner by inserting a `StandardScaler` step inside each fold (MCCV and LOO).
The scaler is fit exclusively on training data and applied to validation/test data.
No other change to the grid, metrics, or figure generation.

## 2. Key differences from `exp_6`

| Aspect | `exp_6` | `exp_7` |
|---|---|---|
| Preprocessing | None (raw embedding) | `StandardScaler` per fold |
| Sklearn import | — | `from sklearn.preprocessing import StandardScaler` |
| MCCV inner loop | `X_train = X_emb[train_idx]` | `scaler.fit(X_train); X_train = scaler.transform(X_train); X_val = scaler.transform(X_val)` |
| LOO inner loop | same as MCCV but raw | same scaler pattern |
| Config name | `knn_n{k}_metric{m}_weights{w}_variant{v}` | identical format |
| Validation report | `input_shape`, `no_nan`, etc. | same checks + `scaler_applied: true` |

## 3. Script location

```text
experiments/exp_7/scripts/run_knn_std_mri_experiment.py
```

## 4. Execution

```bash
conda activate histo-DL && python3 experiments/exp_7/scripts/run_knn_std_mri_experiment.py
```

Must be launched on **tmux session 0** with real-time stdout output.

## 5. Hard validations

Same as `exp_6` plus:
- `StandardScaler` mean per training fold ≈ 0, std ≈ 1 (logged but not asserted, since validation fold has only 18 cases).

## 6. Artefacts

Identical structure to `exp_6`, under `experiments/exp_7/results/`.
