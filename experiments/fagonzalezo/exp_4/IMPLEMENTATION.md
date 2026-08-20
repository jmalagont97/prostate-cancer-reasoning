# Implementation Plan: MCCV 100-Split Design Generation
**Experiment**: experiments/exp_4/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Approved

---

## 1. Code Changes & Additions

### New Script: `experiments/exp_4/scripts/generate_folds.py`
This script will implement the data auditing and Monte Carlo partition generation logic:
1. **Load Preprocessed Data**: Ingest `clinical_prompts.csv`, `biopsy_decision.csv`, `clinical_data_tabular.csv`, and `mri_embeddings.csv` from `data/chimera26/preprocessed/task1/`.
2. **Audit Complete Cases**:
   - Programmatically flag patients with missing data (`'NONE'` or NaN) in prompts, tabular features, or MRI features.
   - Verify that exactly 5 patients are problematic: `PT-pseudo_3646e0a2ae13`, `PT-pseudo_4d54f04e26ae`, `PT-pseudo_4bfd4ec864d8`, `PT-pseudo_7dbdcd6f9064`, `PT-pseudo_8636aa471ef7`.
   - Remove these 5 patients, leaving a cohort of 190 complete cases.
3. **Partition Cohort**:
   - **Test Cohort**: 102 patients where `biopsy_decision` is `'NONE'`. In all 100 splits, these cases are assigned index `-1`.
   - **Labeled Cohort**: 88 patients where `biopsy_decision` is `'yes'` or `'no'`. Map labels to binary class values for stratification.
4. **Generate 100 MCCV Splits**:
   - Seed the random state with `random_state=42`.
   - Using `StratifiedShuffleSplit(n_splits=100, train_size=70, test_size=18, random_state=42)` on the 88 labeled cases.
   - For each split $b \in \{0, \dots, 99\}$:
     - Assign index `0` to the selected 70 training cases.
     - Assign index `1` to the selected 18 validation cases.
5. **Serialize Output**:
   - Combine the labeled splits and test splits into a single unified DataFrame of 190 rows, ordered by `patient_id`.
   - Verify that all rows contain valid indices: `0` (train), `1` (val), `-1` (test).
   - Write to `experiments/exp_4/results/mccv_design.csv`.
6. **Generate reports & visualizations**:
   - Print partition sanity statistics: average class ratio in train and validation splits.
   - Generate distribution plots under `experiments/exp_4/reports/figures/split_distributions.png` comparing target classes and split sizes.
   - Write a summary report to `experiments/exp_4/reports/summary.md`.

---

## 2. Command Lines

### Execution Command
```bash
/home/jmalagont/miniconda3/envs/histo-DL/bin/python3 experiments/exp_4/scripts/generate_folds.py
```

### Git Checkpointing (Reproducibility)
```bash
git status
git log -1 --format="%H %s" > experiments/exp_4/results/git_commit.txt
```
