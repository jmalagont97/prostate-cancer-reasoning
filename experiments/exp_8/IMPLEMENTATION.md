# Implementation Plan — Experiment 8

**Experiment**: KNN + Spearman Correlation Pruning on MRI Embedding  
**Script**: `experiments/exp_8/scripts/run_mri_embedding_pruning_experiment.py`  
**Runtime**: tmux session 0, real-time stdout  
**Expected time**: ~10–30 min (Spearman matrix is O(1024²) per fold × 250 folds)

---

## 1. Script Structure

Adapted from `exp_5/scripts/run_knn_pruning_experiment.py` (pruning logic) +
`exp_6/scripts/run_knn_image_embedding_experiment.py` (MRI embedding loading +
confusion matrix figures).

### Key Differences from exp_5

| Aspect | exp_5 | exp_8 |
|--------|-------|-------|
| Input | main_tabular.csv (27 mixed features) | images.csv (1024 numeric dims) |
| Association | Mixed-type Spearman (num×num, cat×num, cat×cat) | Pure numeric `spearmanr(X)` |
| Essential vars | 10 clinical vars always kept | None |
| Missingness | >50% NaN → drop | Not applicable (no NaN) |
| Categoricals | OHE with sentinel handling | Not applicable |
| Scaling | MinMaxScaler on numerics | None (raw embedding) |
| Missingness indicators | `__is_missing` columns | Not applicable |

### Key Differences from exp_6

| Aspect | exp_6 | exp_8 |
|--------|-------|-------|
| Pruning | None | Spearman + hierarchical clustering |
| Conditions | 1 (1024D raw) | 5 (no_prune + 4 τ thresholds) |
| LOO | Direct 1024D | Fixed intersection of MCCV-selected dims |

## 2. Pruning Pipeline (per MCCV fold)

```python
# X_emb_train: (70, 1024) raw, no scaling
rho, _ = spearmanr(X_emb_train)   # (1024, 1024)
A = np.abs(rho)
D = 1.0 - A
np.fill_diagonal(D, 0)
D = np.maximum(D, 0)
condensed = squareform(D, checks=False)
Z = linkage(condensed, method="complete")
labels = fcluster(Z, t=(1 - tau), criterion="distance")
# For each cluster → select medoid → collect retained dim indices
```

## 3. KNN Evaluation

Same as exp_6: 72 configs, ConfidenceWeightedKNN for fuzzy variant, same metrics
suite. Applied to the pruned (reduced) embedding matrix.

## 4. LOO Evaluation

- Compute intersection of retained dimensions across 50 MCCV folds.
- If intersection is empty → skip LOO for that condition, report as non-evaluable.
- Otherwise, apply the fixed intersection as column mask on the full 1024D embedding
  for each LOO fold's train/test split.

## 5. Artefacts

- `config_log.json` with per-config mean metrics (all conditions).
- `pruning_report.json` with per-condition set sizes, union, intersection.
- `feature_frequency_<condition>.csv` with per-dimension retention frequency.
- `summary_selection.json` with global best config, MCCV and LOO metrics.
- Confusion matrix PNG + PDF for the global best config.
- Validation report with all sanity checks.
- Git commit hash.

## 6. Launch

```bash
conda activate histo-DL
tmux new-session -d -s 0 "python3 experiments/exp_8/scripts/run_mri_embedding_pruning_experiment.py 2>&1 | tee experiments/exp_8/run_output.log"
```

Progress printed to stdout in real time (every 10 splits per condition).
