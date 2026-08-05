# Tabular Preprocessing Summary Report — Task 1

**Date**: 2026-07-20  
**Raw Ingestion Folders Scanned**: 195  
**Generated Synchronized CSV Files**: 5  

## CSV Schema and Dimensions

| File Name | Rows | Columns | Missing ('NONE') Fields | Status |
| :--- | :---: | :---: | :---: | :---: |
| `mri_embeddings.csv` | 195 | 1025 | 4096 | Verified |
| `clinical_prompts.csv` | 195 | 2 | 0 | Verified |
| `clinical_data_tabular.csv` | 195 | 9 | 0 | Verified |
| `clinical_reasoning.csv` | 195 | 13 | 1250 | Verified |
| `biopsy_decision.csv` | 195 | 2 | 104 | Verified |

## Imputation & Modeling Considerations
*   **MRI Embeddings:** 4 cases missing MRI data are completely padded with `'NONE'`. This will require either dropping these 4 cases during visual modeling or implementing fallback layers.
*   **Biopsy Decisions:** 104 test cases are correctly labeled as `'NONE'` in `biopsy_decision.csv`, establishing a clean train-test partition boundary.
*   **Clinical Reasoning:** 92 cases lack rationale data (test split) and are padded with `'NONE'` values.
