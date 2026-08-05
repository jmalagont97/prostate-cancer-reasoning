# Implementation Plan: Task 1 Cohort Composition Analysis (exp_1)

This document details the build plan for the simplified exploratory data analysis (EDA) script and reports of **exp_1**.

---

## 1. Scope & Script Location

We will create a Python script at `experiments/exp_1/scripts/eda.py` which:
1.  Scans the Task 1 cohort directory (`data/chimera26/raw/task1/`).
2.  Audits file completeness per patient directory.
3.  Parses representation and prompt files to count active information sources.
4.  Persists results to `experiments/exp_1/results/eda_metrics.json`.
5.  Generates two simple bar charts via `matplotlib` in technical English, saved to `experiments/exp_1/reports/figures/`.
6.  Generates a Markdown summary report saved to `experiments/exp_1/reports/summary.md`.

---

## 2. Directory & Files to Create

```
experiments/exp_1/
├── scripts/
│   └── eda.py                 ← Main analysis and plotting script
├── results/
│   └── eda_metrics.json       ← Extracted counts (written by eda.py)
└── reports/
    ├── figures/
    │   ├── file_presence.png  ← Matplotlib bar chart (written by eda.py)
    │   └── active_sources.png ← Matplotlib bar chart (written by eda.py)
    └── summary.md             ← Markdown report summarizing counts (written by eda.py)
```

---

## 3. Detailed Logic of `eda.py`

### A. Data Scanning
*   Ingest the base path `data/chimera26/raw/task1/`.
*   Iterate over subdirectories matching `PT-pseudo_*`.
*   Identify:
    *   `total_cases`: Total number of folders found.
    *   For each folder, check existence of:
        *   `prostate-biopsy-decision-clinical-data.json`
        *   `prostate-biopsy-decision-reasoning.json`
        *   `prostate-biopsy-decision.json`
        *   `prostate-modality-level-neural-representations.json`
        *   `structured-prompt.json`

### B. Modality Parsing
*   If `prostate-modality-level-neural-representations.json` exists:
    *   Load JSON and check if `"MRI image"` is a list and has values. Increment `mri_present`.
*   If `structured-prompt.json` exists:
    *   Load JSON and check if `"note_sections"` contains non-empty lists. Increment `clinical_notes_present`.
    *   Check if `"psa"`, `"age"`, `"pirads"` or other tabular elements are present/non-null. Increment `tabular_data_present`.
*   If `prostate-biopsy-decision.json` exists:
    *   Load JSON value. Count `"yes"` vs `"no"`. Increment `labeled_cases`.

### C. Plotting with Matplotlib
Generate simple, clean bar charts with a white background and classic styling (labeled in technical English):
1.  **File Presence:**
    *   X-axis: File Names (e.g. `biopsy-decision.json`, `clinical-data.json`, etc.).
    *   Y-axis: Count of patients (0 to 195).
    *   Annotations: Show exact count and percentage above each bar.
    *   Title: "File Ingestion Completeness - Task 1 Cohort"
2.  **Active Sources:**
    *   X-axis: Modality / Information Source (e.g. `MRI Representation`, `Structured Clinical Text`, `Laboratory & Demographics`, `Biopsy Label`).
    *   Y-axis: Count of active patients (0 to 195).
    *   Title: "Available Modalities and Targets - Task 1 Cohort"

### D. Output Formats
*   `eda_metrics.json` will contain a structured dictionary of counts.
*   `summary.md` will list:
    *   Total cases and proportion labeled/unlabeled.
    *   File presence counts and rates.
    *   Modality completeness rates.
    *   Paths to the generated bar charts.

---

## 4. Run Command

The script will be executed inside the `histo-DL` conda environment:
```bash
python3 experiments/exp_1/scripts/eda.py \
    --data_dir data/chimera26/raw/task1 \
    --results_dir experiments/exp_1/results \
    --reports_dir experiments/exp_1/reports
```

---

## 5. Review & Approval

*   **Next step:** The user reviews this implementation plan. Once approved, the co-investigator will proceed to create `experiments/exp_1/scripts/eda.py` and run it to produce the outputs.
