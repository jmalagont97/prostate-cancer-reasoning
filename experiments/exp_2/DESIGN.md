# Experiment Design: Clean Cohort Selection, Missingness Exclusion & Rigorous Validation Protocol (MCCV + LOOCV)
**Experiment**: experiments/exp_2/  
**Project**: pathology-reasoning  
**Date**: 2026-08-09  
**Author**: Antigravity & Principal Investigator  
**Status**: Draft

---

## 1. Hypothesis
Excluding unannotated cases ($N=102$), cases missing 1024D MRI embeddings ($N=4$), and the single case missing PI-RADS ($N=1$) defines a clean, fully usable labeled cohort of $N=88$ patient cases. Establishing a 50-repeat Monte Carlo Cross-Validation (MCCV) scheme for hyperparameter optimization paired with an 88-fold Leave-One-Out Cross-Validation (LOOCV) for out-of-fold generalization guarantees deterministic, zero-leakage validation splits for all downstream ML models.

---

## 2. Cohort Selection & Exclusion Criteria (Part 1)

### 2.1 Exhaustive Exclusion Rules
Analysis of `data/chimera26/preprocessed/task1/inputs.csv` ($195 \times 1077$, per `exp_1` validation report) and `ground_truth.csv` ($195 \times 27$) establishes the following strict exclusion hierarchy:

1. **Exclusion Rule A (Missing MRI Embeddings):** Exclude $N=4$ patient cases lacking 1024D foundation model MRI embeddings (`mri_emb_*`):
   - `PT-pseudo_4bfd4ec864d8`
   - `PT-pseudo_4d54f04e26ae`
   - `PT-pseudo_7dbdcd6f9064`
   - `PT-pseudo_8636aa471ef7`
2. **Exclusion Rule B (Missing PI-RADS Score):** Exclude $N=1$ patient case missing the core PI-RADS clinical score (`cli_pirads`):
   - `PT-pseudo_3646e0a2ae13`
3. **Exclusion Rule C (Unlabeled / Held-out Test Cases):** Exclude $N=102$ unannotated cases lacking ground truth target labels (`NaN` in `ground_truth.csv`).

### 2.2 Cohort Summary Table ($N=195$)

| Cohort Status Category | Case Count ($N$) | Percentage (%) | Experimental Role & Handling |
|---|---|---|---|
| **`usable_labeled`** | **88** | **45.1%** | **Clean Labeled Cohort.** Used for hyperparameter tuning (MCCV) and out-of-fold evaluation (LOOCV). |
| **`unlabeled_test`** | **102** | **52.3%** | **Unannotated Test Set.** Reserved strictly for unannotated inference pipelines. |
| **`excluded_missing_mri`** | **4** | **2.1%** | **Excluded.** Missing 1024D image representation vectors. |
| **`excluded_missing_pirads`** | **1** | **0.5%** | **Excluded.** Missing primary PI-RADS radiological score. |
| **Total Population** | **195** | **100.0%** | — |

### 2.3 Downstream Imputation Policy
All other non-excluded missing data across the $N=88$ usable labeled cohort will be handled via downstream imputation during model training:
- **`cli_bx` (Biopsies Previas, 26.4% `NaN` en etiquetados):** Mapeado explícitamente como categoría válida `"No Prior Biopsy"` (paciente virgen de biopsia).
- **`cli_fh` (Antecedente Familiar, 3.3% `NaN` en etiquetados):** Imputado con la categoría `"Unknown / 0"`.

---

## 3. Experimental Validation Protocol (Part 2)

### 3.1 Monte Carlo Cross-Validation (MCCV) for Hyperparameter Search
- **Protocol**: 50 Repeated Random Stratified Splits ($80\%$ Train / $20\%$ Validation) on the $N=88$ usable labeled cohort.
- **Stratification Target**: `target_biopsy_decision_binary` (preserves 0/1 class balance across all 50 splits).
- **Purpose**: Selects optimal model hyperparameters (e.g. learning rate, regularization, tree depth, weights) that maximize average Macro-F1 across 50 independent random validation folds without validation leak.

### 3.2 Leave-One-Out Cross-Validation (LOOCV) for Final Performance Evaluation
- **Protocol**: 88-fold LOOCV across all $N=88$ usable labeled cases (Train on $N-1=87$, test on 1).
- **Hyperparameter Injection**: In each LOOCV fold, models execute using the optimal hyperparameters selected via MCCV.
- **Purpose**: Delivers unbiased out-of-fold predictions for every single usable labeled patient case.

### 3.3 Permanent Split CSV Storage
The explicit split assignments, fold indices, and exclusion statuses are permanently stored at:
[`data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`](file:///home/jmalagont/project/pathology-reasoning/data/chimera26/preprocessed/task1/mccv_loocv_splits.csv) ($195 \times 56$).

Columns:
- `case_id`: Patient identifier
- `cohort_status`: `usable_labeled` (88), `unlabeled_test` (102), `excluded_missing_mri` (4), `excluded_missing_pirads` (1)
- `has_gt`, `has_mri`, `has_pirads`: Flags (1/0)
- `loocv_fold`: Fold index $0 \dots 87$ (-1 for excluded)
- `mccv_split_00` ... `mccv_split_49`: Train (0) vs Validation (1) assignments (-1 for excluded) for 50 repeats.

---

## 4. Reproducibility Checklist
- [x] Strict exclusion rules applied ($N=4$ missing MRI, $N=1$ missing PI-RADS).
- [x] Usable clean cohort established at $N=88$ cases.
- [x] Imputation policy defined for non-excluded variables (`bx`, `fh`).
- [x] Random seed fixed (42) for reproducible StratifiedShuffleSplit.
- [x] Split CSV generated and stored at `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`.
- [x] Git commit hash recorded in `experiments/exp_2/results/git_commit.txt` (`702fc02`, HEAD at execution).

---

## 5. Next Steps
1. Review and accept this clean experiment design (`DESIGN.md`).
2. Proceed to model baseline implementations using `mccv_loocv_splits.csv`.
