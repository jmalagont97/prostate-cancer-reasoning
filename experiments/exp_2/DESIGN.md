# Experiment Design: Task 1 Multimodal Preprocessing and Tabularization

**Experiment**: experiments/exp_2/  
**Project**: pathology-reasoning  
**Date**: 2026-07-20  
**Author**: Co-Investigator (Gemini Expert on Digital Pathology & Deep Learning)  
**Status**: Draft

---

## 1. Hypothesis
Preprocessing and tabularizing all multimodal sources (MRI embeddings, structured text prompts, lab tables, targets, and reasoning justifications) into five synchronized CSV files containing all 195 cases (using `NONE` for missing data placeholders) will establish a standardized tabular framework that facilitates imputation analysis and subsequent machine learning modeling.

## 2. Experimental Setup
*   **Dataset**: Chimera26 Task 1 raw dataset located at `data/chimera26/raw/task1/`.
*   **Preprocessed Target Directory**: `data/chimera26/preprocessed/task1/`
*   **Analysis Code**: Python preprocessing script to be implemented in `experiments/exp_2/scripts/preprocess.py`.
*   **Hardware**: CPU for reading and parsing JSON, building Pandas DataFrames, and exporting to CSV.

## 3. File Layout for This Experiment
```
experiments/exp_2/
├── DESIGN.md                  ← this file (experiment design only)
├── scripts/
│   └── preprocess.py          ← Preprocessing script (decided in implementation plan)
├── results/
│   └── preprocessing_metrics.json ← output check metrics (shape, file counts, verification status)
└── reports/
    └── summary.md             ← final preprocessing summary report
```

All preprocessing scripts and outputs reference paths relative to the experiment root.

## 4. Baselines
*   **Baseline**: N/A (Data Preprocessing).

## 5. Proposed Conditions (Preprocessing Outputs)
We will build a pipeline script to ingest the 195 folders in `data/chimera26/raw/task1` and output 5 synchronized CSV files in `data/chimera26/preprocessed/task1/`. All CSVs must contain exactly 195 rows (plus header), sorted by patient ID, with missing entries explicitly marked as `NONE`.

*   **File 1: `mri_embeddings.csv`**
    *   **Structure**: Columns: `patient_id`, followed by 1024 columns (`mri_feat_0` to `mri_feat_1023`).
    *   **Details**: Flat numerical representation of the 1024-D MRI embedding. If missing, all feature columns will contain `NONE`.
*   **File 2: `clinical_prompts.csv`**
    *   **Structure**: Columns: `patient_id`, `clinical_prompt_text`.
    *   **Details**: Unstructured patient clinical narrative (concatenated note sections from `structured-prompt.json` chief complaint, history, and physical examination) in a single column.
*   **File 3: `clinical_data_tabular.csv`**
    *   **Structure**: Columns: `patient_id`, `age`, `psa`, `vol`, `pirads`, `psad`, `psav`, `psap`, `dre`.
    *   **Details**: Standard patient variables extracted from `structured-prompt.json` or `prostate-biopsy-decision-clinical-data.json`.
*   **File 4: `clinical_reasoning.csv`**
    *   **Structure**: Columns: `patient_id`, `reasoning_text`, `confidence`, and weights for key variables (e.g. `weight_psad`, `weight_vol`, `weight_pirads`, `weight_dre`, `weight_fh`, `weight_comorbidity`, `weight_cspca`, `weight_age`, `weight_bx`, `weight_psa`).
    *   **Details**: Justification rationale, weights, and confidence levels extracted from `prostate-biopsy-decision-reasoning.json`. Only available for the 91 labeled cases.
*   **File 5: `biopsy_decision.csv`**
    *   **Structure**: Columns: `patient_id`, `biopsy_decision`.
    *   **Details**: Binary target value (`yes` or `no`). Test set (unlabeled) cases will contain `NONE`.

## 6. Evaluation Protocol
We will programmatically verify the generated files inside `data/chimera26/preprocessed/task1/`:
*   **Row Verification**: Check that each file has exactly 195 lines (excluding header).
*   **Index Alignment**: Verify that the set of `patient_id` matches exactly across all 5 files.
*   **Missing Value Representation**: Verify that missing fields contain the string `'NONE'`.
*   **Data Outputs**: Write verification status, size on disk, and column schemas into `results/preprocessing_metrics.json`.

## 7. Expected Results & Decision Rules
*   **Success Criterion**: All 5 CSV files are successfully written with exactly 195 rows, identical patient ID order, and proper missingness representations. This will allow `exp_3` to execute model design and imputation protocols seamlessly.

## 8. Risks & Mitigations
*   **Risk**: Text formatting anomalies (commas, double-quotes, or newlines in clinical narratives breaking standard CSV row parsers).  
    *   **Mitigation**: Use Pandas `DataFrame.to_csv` with standard double-quoting (`quoting=csv.QUOTE_MINIMAL`) to handle text escaping automatically.

## 9. Reproducibility Checklist
- [ ] Preprocessing code preserved as `preprocess.py`
- [ ] Output verification metrics saved under `results/preprocessing_metrics.json`
- [ ] Working tree clean at run time
- [ ] **Git commit hash recorded** to `results/git_commit.txt` before execution

## 10. Next Steps
1.  Review and accept this experiment design plan.
2.  Once approved, produce an **implementation plan** (in plan mode) to create `experiments/exp_2/scripts/preprocess.py` to tabularize the 5 files.
