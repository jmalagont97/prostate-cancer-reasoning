# Implementation Plan: Multimodal Preprocessing & Canonical Ingestion for Task 1 (`exp_2`)

**Experiment**: experiments/exp_2/  
**Project**: prostate-cancer-reasoning  
**Author**: Juan Sebastian Malagón Torres  
**Date**: 2026-08-22  
**Status**: Updated — BX integration

---

## 1. Overview & Objectives
Implement a robust, modular Python preprocessing pipeline (`experiments/exp_2/scripts/preprocess_task1.py`) that extracts and canonicalizes all 195 Task 1 cases from `data/raw/train_release/task1/` into 4 standardized CSV files under `data/processed/task1/`:
1. `clinical_tabular.csv`
2. `clinical_texts.csv`
3. `mri_embeddings.csv`
4. `ground_truth_task1.csv`

---

## 2. Extraction & Transformation Modules
- **Module 1: Tabular Parser (`extract_tabular`)**:
  - Ingests `structured-prompt.json`.
  - Extracts numeric and categorical features (`age`, `psa`, `vol`, `psad`, `psav`, `psap`, `pirads`, `dre`, `bx`, `cspca`, `ipss`, demographics, vitals parsed into `bp_sys`, `bp_dia`, `bmi`, `hr`, `smoking`, etc.).
  - Reads `bx` exclusively from `structured-prompt.json`; normalizes `Positive` → `1`, `Negative` → `0`, and raw `None` → `np.nan`.
  - Handles missing entries by populating with IEEE 754 `np.nan`.
- **Module 2: Clinical Narrative Parser (`extract_texts`)**:
  - Ingests `structured-prompt.json` exclusively (ignoring `clinical-data.json`).
  - Unrolls `note_sections` (`Chief complaint`, `History`, `Physical examination`) and unifies encounter metadata into longitudinal clinical texts.
- **Module 3: Neural Embeddings Extractor (`extract_embeddings`)**:
  - Ingests `prostate-modality-level-neural-representations.json`.
  - Extracts the 1024-dimensional feature vector under key `'MRI image'`.
  - Enforces explicit exclusion rule: `PT-pseudo_0bd72f429a8b` file is ignored and its 1024 features set to `np.nan`.
- **Module 4: Supervised Ground Truth Builder (`extract_ground_truth`)**:
  - Ingests `prostate-biopsy-decision.json` and `prostate-biopsy-decision-reasoning.json` where present.
  - Extracts binary biopsy decision, confidence rating, individual weights for all 10 clinical variables, decisive/important factor set flags, tool reveal sequences, and pathologist rationales.
  - Imputes `np.nan` for the 104 unlabelled cases, keeping all 195 rows aligned by `pid`.

---

## 3. Verification & Quality Gates
- **Row count check**: Exactly 195 rows per CSV.
- **Bijective Key Alignment**: Exact match of sorted `pid` across all 4 files.
- **Nullity & Dimensionality Check**: `(195, 1024)` matrix for embeddings, five rows with `NaN`s (one explicitly excluded case plus four missing raw representation files), exactly 91 non-null labels in ground truth.
- **BX Check**: `bx` is restricted to `{0, 1, NaN}` after normalization and all 195 cases must retain a row.
- **Leakage Check**: Tool-gated clinical-data files are not read by the preprocessor. Text fields remain those supplied by `structured-prompt.json`.
