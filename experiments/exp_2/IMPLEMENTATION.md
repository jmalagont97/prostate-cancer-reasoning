# Implementation Plan: Task 1 Multimodal Preprocessing (exp_2)

This document details the build plan for the multimodal preprocessing and tabularization pipeline.

---

## 1. Scope & Output Paths

We will create a Python script at `experiments/exp_2/scripts/preprocess.py` that outputs 5 CSV files to `data/chimera26/preprocessed/task1/`:
1.  `mri_embeddings.csv` (features: `patient_id` + `mri_feat_0` to `mri_feat_1023`)
2.  `clinical_prompts.csv` (features: `patient_id` + `clinical_prompt_text`)
3.  `clinical_data_tabular.csv` (features: `patient_id` + standard clinical variables)
4.  `clinical_reasoning.csv` (features: `patient_id` + weights + reasoning text)
5.  `biopsy_decision.csv` (features: `patient_id` + `biopsy_decision` target)

---

## 2. Directory & Files to Create

```
experiments/exp_2/
├── scripts/
│   └── preprocess.py          ← Preprocessing and output generation script
├── results/
│   └── preprocessing_metrics.json ← Verification shapes and columns metrics
└── reports/
    └── summary.md             ← Markdown summary of the preprocessed tabular cohort
```

---

## 3. Preprocessing Script Details (`preprocess.py`)

### A. Scanning & Sorting
*   Iterate through all directories matching `PT-pseudo_*` under `data/chimera26/raw/task1/`.
*   Sort patient IDs alphabetically to ensure consistent row ordering across all 5 files.

### B. Extracting Features per Modality
*   **MRI Embeddings:**
    *   Load `prostate-modality-level-neural-representations.json`.
    *   If representations file exists and has `"MRI image"`, extract the 1024 floats.
    *   If missing, assign `NONE` to all 1024 columns.
*   **Clinical Prompts:**
    *   Load `structured-prompt.json`.
    *   If notes exist, concatenate `"Chief complaint"`, `"History"`, and `"Physical examination"` text.
    *   If missing, assign `NONE`.
*   **Clinical Tabular:**
    *   Load `structured-prompt.json`.
    *   Extract: `age`, `psa`, `vol`, `pirads`, `psad`, `psav`, `psap`, `dre`.
    *   If any variable is null/missing, assign the string `'NONE'`.
*   **Clinical Reasoning:**
    *   Load `prostate-biopsy-decision-reasoning.json`.
    *   Extract: `free_text` (reasoning text), `confidence`, and weights (`weight_psad`, `weight_vol`, `weight_pirads`, `weight_dre`, `weight_fh`, `weight_comorbidity`, `weight_cspca`, `weight_age`, `weight_bx`, `weight_psa`).
    *   If reasoning file does not exist (test cases), assign `'NONE'` to all variables.
*   **Biopsy Decision:**
    *   Load `prostate-biopsy-decision.json`.
    *   Extract target string (`"yes"` or `"no"`).
    *   If missing (test cases), assign `'NONE'`.

### C. Formatting and Exporting
*   Convert lists to Pandas DataFrames.
*   Validate row count is exactly 195 for each DataFrame.
*   Save to `data/chimera26/preprocessed/task1/` using comma separator and proper text quoting.

---

## 4. Run Command

Execute in the active Conda environment:
```bash
python3 experiments/exp_2/scripts/preprocess.py \
    --data_dir data/chimera26/raw/task1 \
    --output_dir data/chimera26/preprocessed/task1 \
    --results_dir experiments/exp_2/results \
    --reports_dir experiments/exp_2/reports
```
