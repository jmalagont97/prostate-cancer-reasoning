# Experiment Design: Tabular KNN Hyperparameter Sweep (MCCV) & LOOCV Evaluation
**Experiment**: experiments/exp_5/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Complete

---

## 1. Hypothesis
A K-Nearest Neighbors (KNN) classifier operating on MinMaxScaler-scaled numerical clinical features and OneHotEncoder-transformed categorical variables will achieve its optimal generalization boundary when neighborhood size $k$ is tuned via 100-split Monte Carlo Cross-Validation (MCCV). Furthermore, evaluating the selected optimal model under Leave-One-Out Cross-Validation (LOOCV) will provide a high-resolution, unbiased assessment of generalizability.

## 2. Experimental Setup
- **Dataset**:
  - Tabular features: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`
  - Targets: `data/chimera26/preprocessed/task1/biopsy_decision.csv`
  - Complete cases: 190 patients (excluding the 5 audit failures: `PT-pseudo_3646e0a2ae13`, `PT-pseudo_4d54f04e26ae`, `PT-pseudo_4bfd4ec864d8`, `PT-pseudo_7dbdcd6f9064`, `PT-pseudo_8636aa471ef7`).
- **Validation Splitting**:
  - Phase A: 100-split Monte Carlo Cross-Validation (`experiments/exp_4/results/mccv_design.csv`). Labeled cohort ($N=88$ complete cases) partitioned into 70 train and 18 validation.
  - Phase B: Leave-One-Out Cross-Validation (88 folds) over the 88 labeled complete cases.
- **Preprocessing Pipeline**:
  - **Numerical columns**: `age`, `psa`, `vol`, `pirads`, `psad`, `psav`, `psap`.
    - Scaling method: `MinMaxScaler` (bounds [0, 1]).
  - **Categorical columns**: `dre`.
    - Encoding method: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
  - To prevent data leakage, preprocessing parameters (MinMax min/max values and OneHot categories) must be fit strictly on the training partition of each split/fold and applied to the validation/test partition.
- **Hyperparameter Sweep Space (KNN)**:
  - Neighbor size $k \in \{1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25\}$.
  - Neighbor weights: `['uniform', 'distance']`.
  - Distance metrics: `['euclidean', 'manhattan', 'cosine']`.
- **Hardware/Cost**:
  - Run time is minimal ($\le 10$ seconds on CPU).

## 3. File Layout for This Experiment
```
experiments/exp_5/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← grid search + LOOCV script
├── results/
│   ├── grid_search_results.csv ← metrics per parameter configuration averaged over 100 splits
│   ├── best_hparams.json       ← best selected hyperparameters
│   ├── loocv_metrics.json      ← final out-of-fold metrics of Phase B
│   └── loocv_predictions.csv   ← final out-of-fold predictions
└── reports/
    ├── figures/
    │   ├── grid_search_curves.png  ← validation metric curves across neighbors
    │   └── confusion_matrix.png     ← confusion matrix of final LOOCV
    └── summary.md             ← write-up of results and optimal hparams
```

## 4. Baselines
| Baseline | Config file | Expected metric range |
|----------|------------|----------------------|
| Unimodal Tabular KNN (legacy exp_15 5-fold) | N/A | Macro-F1 $\sim$ 0.75 |

## 5. Proposed Conditions
| Condition ID | Model | Validation Split Strategy | Search Type | Parameters Swept |
|:---|:---|:---|:---|:---|
| **COND-01-MCCV-Sweep** | Tabular KNN | MCCV (100 splits) | Grid Search | $k$, weights, metric |
| **COND-02-LOOCV-Eval** | Tabular KNN | LOOCV (88 folds) | Final Evaluation | Frozen optimal hparams |

## 6. Evaluation Protocol
- **Primary Metric**: Macro-F1 score (due to class imbalance: 56 Yes, 32 No).
- **Secondary Metrics**: Accuracy, Sensitivity (recall for 'yes'), Specificity (recall for 'no').
- **Evaluation Loop (Phase A)**:
  - For each hyperparameter combination:
    - Train on 70 train cases of split $b$, predict on 18 val cases.
    - Average validation metrics across all 100 splits.
  - Choose configuration maximizing average validation Macro-F1.
- **Evaluation Loop (Phase B)**:
  - Freeze the optimal configuration.
  - Run LOOCV: for each of the 88 labeled cases, train on the other 87, predict on the held-out case.
  - Consolidate predictions and calculate out-of-fold metrics.

## 7. Expected Results & Decision Rules
- The optimal neighborhood size $k$ is expected to be odd (e.g. $k=5$ or $k=7$) to avoid ties under uniform voting, and Cosine distance might perform differently than Euclidean/Manhattan distance under MinMax normalization.
- The LOOCV evaluation provides the final generalization score that will serve as the tabular unimodal benchmark.

## 8. Risks & Mitigations
- **Risk: Categorical Category Mismatches**: The categorical variable `dre` has limited categories ('Normal' and potentially others). If a category appears in the validation split but not training split, the encoder might crash.
  - *Mitigation*: Configure `OneHotEncoder` with `handle_unknown='ignore'` to handle unseen values gracefully.
- **Risk: Preprocessing Data Leakage**: Scaling features using the global min/max instead of fit-on-train will lead to optimistic validation metrics.
  - *Mitigation*: Strictly fit the `MinMaxScaler` and `OneHotEncoder` on the training subset of each pliegue/fold.

## 9. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Training script placed under `scripts/`
- [ ] Output metrics and plots saved to `results/` and `reports/`
- [ ] Environment frozen
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before running the training script

## 10. Next Steps
1. Review and accept this experiment plan (hypothesis, KNN search space, preprocessing, LOOCV final evaluation).
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and run it to search parameters and perform LOOCV evaluation.
