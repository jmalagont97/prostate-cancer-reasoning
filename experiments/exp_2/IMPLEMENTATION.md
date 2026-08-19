# Implementation Plan: Clean Cohort Selection & Rigorous Validation Protocol (exp_2)
**Experiment**: experiments/exp_2/ · **Project**: pathology-reasoning · **Date**: 2026-08-09 · **Status**: Approved

---

## 1. Overview & Objective

This implementation plan details the split-generation script `experiments/exp_2/scripts/generate_splits.py`. It reads the canonical master matrices produced by `exp_1` (`data/chimera26/preprocessed/task1/inputs.csv` and `ground_truth.csv`), applies the exhaustive exclusion rules defined in `DESIGN.md`, and emits the deterministic, versionable validation artifact `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv` (195 × 56).

The artifact freezes for all downstream Task 1 experiments:

1. **Cohort status** per case: `usable_labeled` (N=88), `unlabeled_test` (N=102), `excluded_missing_mri` (N=4), `excluded_missing_pirads` (N=1).
2. **50 Monte Carlo CV splits** (80% train / 20% val, stratified on `target_biopsy_decision_binary`, seed=42) for hyperparameter selection by Macro-F1.
3. **88 LOOCV fold indices** (0–87, lexicographic by `case_id`) for out-of-fold evaluation.

---

## 2. File & Script Structure

```
pathology-reasoning/
├── data/
│   └── chimera26/
│       └── preprocessed/
│           └── task1/
│               ├── inputs.csv              ← Source feature matrix (195 × 1077)
│               ├── ground_truth.csv        ← Source target matrix (195 × 27)
│               └── mccv_loocv_splits.csv   ← Generated validation artifact (195 × 56)
└── experiments/
    └── exp_2/
        ├── DESIGN.md                       ← Approved research design
        ├── IMPLEMENTATION.md               ← This implementation plan
        ├── scripts/
        │   └── generate_splits.py          ← Main execution script & validator
        ├── results/
        │   ├── validation_report.json      ← Automated protocol validation audit
        │   ├── data_manifest.csv           ← SHA-256 hashes of source & output CSVs
        │   └── git_commit.txt              ← Git commit hash at execution time
        └── reports/
            └── summary.md                  ← Final summary writeup
```

---

## 3. Data Flow

1. **Load sources** (`inputs.csv`, `ground_truth.csv`) and assert canonical shapes
   (195 × 1077 and 195 × 27 respectively, per `exp_1` validation report).
2. **Compute per-case flags** from the source matrices (no assumptions hardcoded
   beyond column naming conventions):
   - `has_gt`: `target_biopsy_decision` non-null in `ground_truth.csv`.
   - `has_mri`: not all 1024 `mri_emb_*` columns null in `inputs.csv`.
   - `has_pirads`: `cli_pirads` non-null in `inputs.csv`.
3. **Assign `cohort_status`** with strict precedence (per DESIGN §2):
   `excluded_missing_mri` → `excluded_missing_pirads` → `unlabeled_test` → `usable_labeled`.
   The script asserts the resulting excluded case-ID sets equal the DESIGN lists.
4. **MCCV splits**: `StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)`
   over the 88 usable cases using `target_biopsy_decision_binary`.
5. **LOOCV folds**: indices 0–87 assigned in lexicographic `case_id` order among usable cases.
6. **Emit artifact**: single CSV sorted by `case_id` (columns: `case_id`,
   `cohort_status`, `has_gt`, `has_mri`, `has_pirads`, `loocv_fold`,
   `mccv_split_00` … `mccv_split_49`). Excluded cases are `-1` in `loocv_fold` and all `mccv_split_*`.

---

## 4. Automated Validations (hard-asserted)

| Check | Rule | Expected |
|---|---|---|
| IO shapes | `inputs.csv` / `ground_truth.csv` | (195, 1077) / (195, 27) |
| Output shape | `mccv_loocv_splits.csv` | (195, 56) |
| Cohort counts | `cohort_status.value_counts()` | 88 / 102 / 4 / 1 |
| Exclusion sets | MRI- and PI-RADS-missing `case_id`s | exactly the DESIGN lists |
| Class balance | usable `target_biopsy_decision_binary` | yes=54 / no=34 |
| MCCV integrity | per split, usable rows | train=70 / val=18, both classes present |
| Excluded handling | non-usable rows in `loocv_fold` & `mccv_split_*` | all `-1` |
| LOOCV integrity | usable rows | folds 0–87, each exactly once |
| Consistency w/ inputs | usable rows must have MRI and PI-RADS | all true |
| Determinism | two independent runs in-process | identical splits |

---

## 5. Reproducibility

- Fixed random seed (42) in `StratifiedShuffleSplit`.
- Lexicographic `case_id` ordering for LOOCV assignment and output layout.
- The script regenerates the artifact from the sources in a single pass; running it
  twice must produce an identical CSV (verified in-process via double-run check).
- Input hashes recorded in `results/data_manifest.csv`; execution commit recorded
  in `results/git_commit.txt`.

---

## 6. Execution

```bash
python3 experiments/exp_2/scripts/generate_splits.py
```

Expected outputs: `mccv_loocv_splits.csv`, `validation_report.json`,
`data_manifest.csv`, `git_commit.txt`. Exit code 0 iff every validation passes.
