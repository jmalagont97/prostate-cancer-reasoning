# exp_14: ICI Confidence Tree — Corrected (v4)

**Status:** Pending execution
**Date:** 2026-08-18
**Objective:** Predict `target_confidence` (uncertain / borderline / clear) using a `DecisionTreeClassifier` on ICI, with the corrected frozen ensemble (ConfidenceWeightedKNN trained on `target_biopsy_decision_binary`).

### Critical corrections from v3

1. **Base models use `ConfidenceWeightedKNN`** (not `StandardKNN`), matching the winning configurations from exp_4–exp_11.
2. **KNNs are trained on `target_biopsy_decision_binary`** (binary yes/no for biopsy decision), NOT on `target_confidence`. The clinical confidence values are used ONLY as neighbor weights in the fuzzy variant.
3. **No circularity:** The evaluated sample's own confidence is never used in its prediction.

---

## Hypothesis

A `DecisionTreeClassifier` trained on ICI (computed from ConfidenceWeightedKNN base models) predicts `target_confidence` better than the majority baseline.

### Null Hypothesis

The ICI-based tree does not improve over predicting the majority class (`clear`) for all cases when evaluated on MOE_abs.

---

## Data

### Cohort
- **Subset:** `usable_labeled` cases with GT.
- **Cases:** 88.
- **Ground truth column:** `target_confidence`.

### Distribution
| Category     | Count | Proportion |
|-------------|-------|-----------|
| clear       | 56    | 63.6%     |
| borderline  | 18    | 20.5%     |
| uncertain   | 14    | 15.9%     |

---

## Frozen Multimodal Ensemble

All three KNN base models use `ConfidenceWeightedKNN` trained on `target_biopsy_decision_binary`.

| Modality | Preprocessing | KNN Parameters |
|----------|--------------|----------------|
| Tabular (T) | 21 frozen variables, zero-fill + absence indicators, OHE, MinMax | k=1, cosine, uniform, confidence_weighted |
| MRI (M) | 1024-dim embedding, PCA n_components=1 (fit per fold) | k=1, euclidean, distance, confidence_weighted |
| Text (X) | TF-IDF max_features=2000, corrected preprocessing (lowercase, remove special chars, remove numeric tokens, remove stopwords with negation protection, lemmatize via spaCy en_core_web_sm) | k=3, cosine, distance, confidence_weighted |

### Confidence Weights (for neighbor weighting only)

```text
clear      = 1.00
borderline = 0.50
uncertain  = 0.25
```

Weight formula: `q_j = 0.5 + c_j * (y_j - 0.5)`

---

## ICI Formula

\[
ICI = 2 \cdot |\bar{p} - 0.5| \cdot (1 - 2\sigma)
\]

where:
- $\bar{p} = \frac{p_T + p_M + p_X}{3}$
- $\sigma = \text{std}(p_T, p_M, p_X)$ (population std, ddof=0)

---

## Model

### Architecture
- **Model:** `DecisionTreeClassifier`
- **Input:** ICI (1D feature)
- **Output:** Predicted category {0=uncertain, 1=borderline, 2=clear}
- **Fixed params:** `max_depth=2`, `max_leaf_nodes=3`, `min_samples_leaf=5`, `random_state=42`
- **Hyperparameter:** `class_weight` ∈ {None, "balanced"}

### Structural verification
After each fold:
- Tree has exactly 2 internal nodes and 3 leaves
- Two thresholds exist: t1 < t2

---

## Protocol

### Outer Loop (MCCV Selection)
- 50 MCCV splits from `mccv_loocv_splits.csv` (0=train, 1=val).
- ~70% train / ~18% val per split.

### Per Fold
1. Inner OOF: 3-fold CV within training set to generate ICI for all 70 training cases.
   - Each inner fold: train 3 ConfidenceWeightedKNNs on `target_biopsy_decision_binary`, weight neighbors by `target_confidence`.
2. Train 3 base models on full 70-case training set.
3. Predict p_T, p_M, p_X for 18 validation cases.
4. Compute ICI for training and validation.
5. Train DecisionTreeClassifier on (ICI_train → target_confidence).
6. Predict confidence for validation cases.
7. Compute MOE_abs and F1_macro.

### LOO (Sanity Check)
- 88 folds, same inner OOF protocol.
- Only the best MCCV config is evaluated.
- Never used for selection.

---

## Evaluation

### Primary metric
\[
MOE_{abs} = \frac{1}{3} \sum_c \text{mean}_{i:y_i=c}\left(\frac{|\hat{y}_i - y_i|}{2}\right)
\]
Range [0,1], lower = better.

### Tiebreaker
F1_macro (higher = better).

### Baseline
Always predict `clear` using real MCCV labels (exactly 900 predictions).

### Selection cascade
1. valid_structure_rate == 100%
2. MOE_abs < baseline
3. No zero recall
4. Minimize MOE_abs
5. Tiebreak: maximize F1_macro

---

## Hyperparameter Sweep

| Config | `class_weight` |
|--------|---------------|
| `tree_none` | `None` |
| `tree_balanced` | `"balanced"` |

---

## Outputs

- `results/summary.json`
- `results/evaluation_scorecard.csv`
- `results/predictions_mccv.csv`
- `results/predictions_loo.csv`
- `results/per_fold.csv`
- `results/confusion_matrices.json`
- `results/figures/confusion_matrices_mccv.png`
- `results/figures/confusion_matrix_loo_selected.png`
- `results/figures/confusion_matrix_loo_selected_normalized.png`
- `results/figures/confusion_matrix_loo_baseline.png`
