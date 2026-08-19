# Experiment Summary: exp_2 (Clean Cohort Selection & Rigorous Validation Protocol)

## Key Results & Validation Summary
- **Source Matrices**: `inputs.csv` (195 × 1077) and `ground_truth.csv` (195 × 27) loaded from `data/chimera26/preprocessed/task1/`.
- **Cohort Assignment** (verified against DESIGN): `usable_labeled` **88** (45.1%), `unlabeled_test` **102** (52.3%), `excluded_missing_mri` **4** (2.1%), `excluded_missing_pirads` **1** (0.5%).
- **Excluded Cases**: exactly match DESIGN lists — MRI-missing `{4bfd4ec864d8, 4d54f04e26ae, 7dbdcd6f9064, 8636aa471ef7}`, PI-RADS-missing `{3646e0a2ae13}`.
- **Class Balance (usable)**: 54 yes / 34 no (`target_biopsy_decision_binary`).
- **MCCV Protocol**: 50 stratified splits (80/20, seed=42); every split is 70/18 with both classes present (43/27 train, 11/7 val).
- **LOOCV Protocol**: folds 0–87 assigned lexicographically by `case_id`; excluded cases `-1` in `loocv_fold` and all `mccv_split_*`.
- **Consistency with `inputs.csv`**: case-ID sets align; every usable row has non-null MRI embeddings and PI-RADS.
- **Determinism**: in-process double run produces identical splits; regenerated artifact is byte-identical to the previously stored one and its SHA-256 (`eda3391e…`) matches the historical `data_manifest.csv`.
- **Artifacts**: `mccv_loocv_splits.csv`, `results/validation_report.json`, `results/data_manifest.csv`, `results/git_commit.txt`.

## Verdict
**SUCCESS**: The clean labeled cohort (N=88) and the zero-leakage validation protocol (MCCV 50 splits for selection + 88-fold LOOCV for evaluation) are established and frozen as the standard for all downstream Task 1 experiments.
