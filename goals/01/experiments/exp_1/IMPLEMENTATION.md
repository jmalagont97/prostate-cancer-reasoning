# Implementation Plan: Canonical Master Data Extraction & Structuring (exp_1)
**Experiment**: experiments/exp_1/ · **Project**: pathology-reasoning · **Date**: 2026-08-07 · **Status**: Approved

---

## 1. Overview & Objective

This implementation plan details the Python extraction module `src/data/build_master_dataset.py` and execution runner `experiments/exp_1/scripts/run_preprocessing.py`. It parses raw JSON files from `data/chimera26/raw/task1/` across all 195 cases and builds two master CSV datasets in `data/chimera26/preprocessed/task1/`:

1. `inputs.csv`: Flattened matrix containing all clinical, vital, longitudinal PSA, laboratory, pathology, text narrative (13 text columns), and 1024D MRI foundation embedding features.
2. `ground_truth.csv`: Master targets matrix containing decision (`yes`/`no`), confidence (`clear`, `borderline`, `uncertain`), 10 relevance weights, free-text reasoning, and reveal sequences.

---

## 2. File & Script Structure

```
pathology-reasoning/
├── data/
│   └── chimera26/
│       └── preprocessed/
│           └── task1/
│               ├── inputs.csv        ← Generated feature matrix
│               └── ground_truth.csv  ← Generated target matrix
├── src/
│   └── data/
│       └── build_master_dataset.py   ← Core parser & feature extraction library
└── experiments/
    └── exp_1/
        ├── DESIGN.md                 ← Approved research design
        ├── IMPLEMENTATION.md         ← This implementation plan
        ├── scripts/
        │   └── run_preprocessing.py  ← Main execution entry point & validator
        ├── results/
        │   └── validation_report.json ← Automated data validation audit report
        └── reports/
            └── summary.md            ← Final summary writeup
```

---

## 3. Detailed Data Extraction Logic

### Module 1: Parsing `structured-prompt.json` (`inputs.csv`)
1. **Scalar Clinicals (`cli_`)**:
   - `cli_age`: `float(d['age'])`
   - `cli_psa`: `float(d['psa'])`
   - `cli_psap`: `float(d['psap'])`
   - `cli_psav`: `float(d['psav'])`
   - `cli_psad`: `float(d['psad'])`
   - `cli_vol`: `float(d['vol'])`
   - `cli_months`: `float(d['months'])`
   - `cli_pirads`: `int(d['pirads'])`
   - `cli_dre`: `str(d['dre'])`
   - `cli_bx`: `str(d['bx'])`
   - `cli_cspca`: `float(d['cspca'])`
   - `cli_comorbidity_count`: `len(d.get('pmhx', []))`
   - `cli_allergies_count`: `len(d.get('allergies', []))`
   - `cli_ipss_score`: Extract numeric integer from `d.get('ipss', '')` (e.g., `18`).

2. **Vital Signs (`vit_`)**:
   - `vit_weight_kg`: Parse numeric float from `vitals.weight` (`"68 kg"` $\to 68.0$).
   - `vit_height_cm`: Parse numeric float from `vitals.height` (`"175 cm"` $\to 175.0$).
   - `vit_bmi`: Parse numeric float from `vitals.bmi` (`"22.2"` $\to 22.2$).
   - `vit_bp_systolic`: Parse systolic int from `vitals.bp` (`"147/77 mmHg"` $\to 147.0$).
   - `vit_bp_diastolic`: Parse diastolic int from `vitals.bp` (`"147/77 mmHg"` $\to 77.0$).
   - `vit_heart_rate_bpm`: Parse numeric float from `vitals.hr` (`"79 bpm"` $\to 79.0$).
   - `vit_smoking_status`: Parse status (`"Ex-smoker"`, `"Never"`, `"Current"`).
   - `vit_smoking_pack_years`: Parse pack-years float (e.g. `22.0`).

3. **Narrative Sections & Comorbidity Text (`txt_`)**:
   - `txt_chief_complaint`: Extract text from `note_sections` where `s == 'Chief complaint'`.
   - `txt_history`: Extract text from `note_sections` where `s == 'History'`.
   - `txt_physical_examination`: Extract text from `note_sections` where `s == 'Physical examination'`.
   - `txt_prompt_summary_notes`: Extract `d.get('notes', '')`.
   - `txt_full_prompt_narrative`: Concatenate all prompt sections.
   - `txt_comorbidities`: Join `d.get('pmhx', [])` with `", "` (e.g., `"Hypertension, COPD, CKD"`).
   - `txt_allergies`: Join `d.get('allergies', [])` with `", "`.

---

### Module 2: Parsing `prostate-biopsy-decision-clinical-data.json` (`inputs.csv`)
1. **Family History (`cli_fh_binary` & `txt_family_history_narrative`)**:
   - `cli_fh_binary`: $1.0$ if `family_history == 'Yes'`, $0.0$ if `'No'`, `np.nan` if missing.
   - `txt_family_history_narrative`: `str(d.get('family_history', ''))`.

2. **Longitudinal PSA Trend (`psa_tr_` & `txt_psa_trend_summary`)**:
   - Parse array `d.get('psa_trend', [])`.
   - `psa_tr_count`: `len(trend)`
   - `psa_tr_first_val`, `psa_tr_last_val`, `psa_tr_min`, `psa_tr_max`, `psa_tr_mean`, `psa_tr_delta`, `psa_tr_slope`.
   - `txt_psa_trend_summary`: Format trend into readable narrative (e.g., `"Feb 2022: 4.2 | Feb 2023: 4.4 | Dec 2024: 4.7"`).

3. **Laboratory Results (`lab_` & `txt_laboratory_results_summary`)**:
   - Parse array `d.get('laboratory_results', [])`.
   - Extract `lab_creatinine_mg_dl`, `lab_hemoglobin_g_dl`, `lab_free_psa_ng_ml`, `lab_free_total_ratio`.
   - `txt_laboratory_results_summary`: Concatenate all lab tests into narrative text.

4. **Radiology & Evolution Notes (`txt_`)**:
   - `txt_radiology_report`: `d.get('radiology_report', '')`
   - `txt_previous_notes`: Concatenate all `previous_notes` texts.
   - `txt_consolidated_ehr_narrative`: Master concatenation of all prompt sections + radiology report + previous notes + lab summary + family history.

---

### Module 3: Parsing `neural-representations.json` (`mri_emb_`)
- Parse `"MRI image"` array:
  - If array is present ($1 \times 1024$ floats): populate `mri_emb_0` through `mri_emb_1023`.
  - If array is empty or file missing: populate `mri_emb_0` through `mri_emb_1023` with `np.nan`.

---

### Module 4: Parsing Targets (`ground_truth.csv`)
- Parse `prostate-biopsy-decision.json` and `prostate-biopsy-decision-reasoning.json`:
  - `target_biopsy_decision`: `"yes"` / `"no"` ($1.0 / 0.0$).
  - `target_confidence`: `"clear"`, `"borderline"`, `"uncertain"` ($2.0, 1.0, 0.0$).
  - 10 relevance weights (`target_weight_age`... `target_weight_bx`) + ordinal codes ($0, 1, 2, 3$).
  - `target_reasoning_free_text`: Ground-truth explanation string.
  - `target_reveal_sequence_json`: JSON string of reveal entries.

---

## 4. Step-by-Step Execution Workflow

1. Create `src/data/build_master_dataset.py`.
2. Create `experiments/exp_1/scripts/run_preprocessing.py`.
3. Execute `run_preprocessing.py`.
4. Validate outputs in `data/chimera26/preprocessed/task1/`:
   - Check row count = $195$.
   - Check zero `list` or `dict` columns.
   - Verify all 13 `txt_*` columns populated.
   - Verify all 10 target variables present in `inputs.csv` and `ground_truth.csv`.
5. Write `experiments/exp_1/results/validation_report.json`.
6. Write `experiments/exp_1/reports/summary.md`.

---

## 5. Decision Rules & Acceptance Criteria

- **Acceptance Rule**: 100% of 195 cases parsed into `inputs.csv` and `ground_truth.csv` with zero errors, zero unparsed collections, and exact column schema compliance.
