# Implementation Plan — Experiment 9

**Experiment**: KNN + PCA Dimensionality Reduction on MRI Embedding
**Script**: `experiments/exp_9/scripts/run_mri_pca_knn_experiment.py`
**Runtime**: tmux session 0, real-time stdout
**Expected time**: ~5–15 min (PCA is fast on 70×1024; KNN grid is the bottleneck)

---

## 1. Script Structure

Adapted from `exp_8/scripts/run_mri_embedding_pruning_experiment.py`
(Spearman pruning) + `exp_6/scripts/run_knn_image_embedding_experiment.py`
(MRI embedding loading + confusion matrix figures).

### Key Differences from exp_8

| Aspect | exp_8 | exp_9 |
|--------|-------|-------|
| Dimensionality reduction | Spearman correlation pruning (feature selection) | PCA (linear projection) |
| Conditions | `no_prune` + 4 τ thresholds | `no_pca` + 8 `n_components` values |
| LOO | Fixed intersection of MCCV-selected dims | No intersection — PCA refit per LOO fold |
| Extra artefact | pruning_report.json, clusters JSON | pca_report.json, explained_variance CSV/PNG |

### Key Differences from exp_6

| Aspect | exp_6 | exp_9 |
|--------|-------|-------|
| Preprocessing | None (raw 1024D) | PCA per fold |
| Conditions | 1 | 9 (no_pca + 8 PCA sizes) |
| LOO | Direct 1024D | PCA refit per fold with frozen n_components |

## 2. PCA Pipeline (per MCCV fold)

```python
# X_emb_train: (70, 1024) raw, no scaling
pca = PCA(n_components=d, svd_solver="full", whiten=False)
X_train_pca = pca.fit_transform(X_emb_train)   # (70, d)
X_val_pca   = pca.transform(X_emb_val)         # (18, d)
# pca.explained_variance_ratio_ available for reporting
```

## 3. KNN Evaluation

Same as exp_8: 72 configs, ConfidenceWeightedKNN for fuzzy variant, same metrics
suite. Applied to the PCA-transformed (or raw, for `no_pca`) embedding matrix.

## 4. LOO Evaluation

- `n_components` frozen from MCCV selection.
- Per LOO fold: fit PCA on 87 training cases → transform train+test.
- No intersection across folds (PCA bases differ; intersection is undefined).
- The `pca_69` condition uses 69 components (max rank in MCCV = 69); in LOO
  with 87 training cases, the rank is 86, so 69 components are valid.

## 5. Artefacts

- `config_log.json` with per-config mean metrics (all conditions).
- `pca_report.json` with per-condition explained variance stats.
- `explained_variance_<condition>.csv` with per-fold explained variance ratios.
- `summary_selection.json` with global best config, MCCV and LOO metrics.
- Confusion matrix PNG + PDF for the global best config.
- Explained variance bar plot PNG + PDF.
- Validation report with all sanity checks.
- Git commit hash.

## 6. Launch

```bash
conda activate histo-DL
tmux new-session -d -s 0 "python3 experiments/exp_9/scripts/run_mri_pca_knn_experiment.py 2>&1 | tee experiments/exp_9/run_output.log"
```

Progress printed to stdout in real time (every 10 splits per condition).
