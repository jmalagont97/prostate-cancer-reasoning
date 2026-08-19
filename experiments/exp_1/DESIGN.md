# Experiment Design: Canonical Master Data Structuring & Feature Extraction Pipeline (Task 1)
**Experiment**: experiments/exp_1/ · **Project**: pathology-reasoning · **Date**: 2026-08-07 · **Status**: Draft

---

## 1. Hypothesis

Unpacking and flattening all raw nested JSON structures, dictionary fields (`vitals`), longitudinal series (`psa_trend`), laboratory panels (`laboratory_results`), narrative sections (`note_sections`), comorbidity text arrays (`pmhx`), radiology reports (`radiology_report`), clinical evolution notes (`previous_notes`), and 1024D MRI foundation vectors into a unified scalar tabular matrix (`inputs.csv`) with explicit variable type prefixes (`cli_`, `vit_`, `psa_tr_`, `lab_`, `path_hist_`, `txt_`, `mri_emb_`), paired with a consolidated targets matrix (`ground_truth.csv`), preserves 100% of available clinical information across all 195 cases while eliminating collection types (lists/dicts) and standardizing missing value representation to `np.nan`.

---

## 2. Experimental Setup

- **Raw Data Source**: `data/chimera26/raw/task1/` ($195$ case subdirectories `PT-pseudo_XXXX`).
- **Target Output Directory**: `data/chimera26/preprocessed/task1/`
- **Output Files to Generate**:
  1. `data/chimera26/preprocessed/task1/inputs.csv`: All flattened input features across all 195 cases.
  2. `data/chimera26/preprocessed/task1/ground_truth.csv`: All decision and reasoning targets across 91 labeled cases ($104$ test cases padded with `np.nan`).
- **Data Integrity Constraints**:
  - **No Collections or Lists**: Every list, array, or nested dict must be flattened into independent scalar columns or explicit text strings.
  - **Explicit Variable Prefixes**: Every column name must carry a functional prefix identifying its source and type.
  - **Strict Scalar & Text Data Types**: Only `numeric` (`float64`, `int64`), `binary` ($0/1$), `categorical` (`string`/`code`), or `text` (`string`) columns are permitted.
  - **Clean Missing Value Handling**: Missing or non-applicable values are represented exclusively as `np.nan` (zero dummy string `"NONE"` artifacts).

---

## 3. Detailed Schema Specification for `inputs.csv`

Every row represents 1 patient case, indexed by `case_id`.

### A. Scalar Clinical & Demographic Features (Prefix: `cli_`)
| Column Name | Raw Source | Data Type | Description / Missing Value Rule |
|:---|:---|:---:|:---|
| `case_id` | `structured-prompt.json` $\rightarrow$ `case_id` | `str` | Primary key identifier (e.g. `PT-pseudo_0020cfca66c8`). |
| `cli_age` 🌟 | `structured-prompt.json` $\rightarrow$ `age` | `float64` | **Target Variable 1:** Patient age in years. |
| `cli_psa` 🌟 | `structured-prompt.json` $\rightarrow$ `psa` | `float64` | **Target Variable 2:** Current serum PSA (ng/mL). |
| `cli_psap` | `structured-prompt.json` $\rightarrow$ `psap` | `float64` | Prior serum PSA (ng/mL). |
| `cli_psav` | `structured-prompt.json` $\rightarrow$ `psav` | `float64` | PSA velocity (ng/mL/yr). |
| `cli_psad` 🌟 | `structured-prompt.json` $\rightarrow$ `psad` | `float64` | **Target Variable 3:** PSA density (ng/mL/mL). |
| `cli_vol` 🌟 | `structured-prompt.json` $\rightarrow$ `vol` | `float64` | **Target Variable 4:** Prostate volume on imaging (mL). |
| `cli_months` | `structured-prompt.json` $\rightarrow$ `months` | `float64` | Months since last PSA measurement. |
| `cli_pirads` 🌟 | `structured-prompt.json` $\rightarrow$ `pirads` | `int64` | **Target Variable 5:** PI-RADS category ($1\text{--}5$). |
| `cli_dre` 🌟 | `structured-prompt.json` $\rightarrow$ `dre` | `str` | **Target Variable 6:** DRE finding (`Normal`, `Nodus`, `Abnormal`). |
| `cli_bx` 🌟 | `structured-prompt.json` $\rightarrow$ `bx` | `str` | **Target Variable 7:** Prior biopsy status (`None`, `Negative`, `Positive`). |
| `cli_cspca` 🌟 | `structured-prompt.json` $\rightarrow$ `cspca` | `float64` | **Target Variable 8:** Deep learning csPCa probability ($0.0\text{--}1.0$). |
| `cli_comorbidity_count` 🌟 | `structured-prompt.json` $\rightarrow$ `pmhx` | `int64` | **Target Variable 9:** Integer count of comorbidities (`len(pmhx)`). |
| `cli_fh_binary` 🌟 | `clinical-data.json` $\rightarrow$ `family_history` | `float64` | **Target Variable 10:** Family history flag ($1.0$ if `Yes`, $0.0$ if `No`, `np.nan` if missing). |
| `cli_allergies_count` | `structured-prompt.json` $\rightarrow$ `allergies` | `int64` | Count of recorded patient allergies. |
| `cli_ipss_score` | `structured-prompt.json` $\rightarrow$ `ipss` | `float64` | Parsed numeric IPSS symptom score ($0\text{--}35$). |

### B. Vital Signs Features (Flattened from `vitals`, Prefix: `vit_`)
| Column Name | Raw Source | Data Type | Description |
|:---|:---|:---:|:---|
| `vit_weight_kg` | `vitals.weight` | `float64` | Patient weight in kg. |
| `vit_height_cm` | `vitals.height` | `float64` | Patient height in cm. |
| `vit_bmi` | `vitals.bmi` | `float64` | Body Mass Index ($\text{kg/m}^2$). |
| `vit_bp_systolic` | `vitals.bp` | `float64` | Systolic blood pressure (mmHg). |
| `vit_bp_diastolic` | `vitals.bp` | `float64` | Diastolic blood pressure (mmHg). |
| `vit_heart_rate_bpm` | `vitals.hr` | `float64` | Heart rate (bpm). |
| `vit_smoking_status` | `vitals.smoking` | `str` | Category (`Never`, `Ex-smoker`, `Current`). |
| `vit_smoking_pack_years` | `vitals.smoking` | `float64` | Cumulative smoking pack-years. |

### C. Longitudinal PSA Trend Features (Flattened from `psa_trend`, Prefix: `psa_tr_`)
| Column Name | Raw Source | Data Type | Description |
|:---|:---|:---:|:---|
| `psa_tr_count` | `clinical-data.json` $\rightarrow$ `psa_trend` | `int64` | Number of historical PSA measurements. |
| `psa_tr_first_val` | `clinical-data.json` $\rightarrow$ `psa_trend` | `float64` | Earliest recorded historical PSA value. |
| `psa_tr_last_val` | `clinical-data.json` $\rightarrow$ `psa_trend` | `float64` | Most recent recorded historical PSA value. |
| `psa_tr_min` | `clinical-data.json` $\rightarrow$ `psa_trend` | `float64` | Minimum historical PSA value. |
| `psa_tr_max` | `clinical-data.json` $\rightarrow$ `psa_trend` | `float64` | Maximum historical PSA value. |
| `psa_tr_mean` | `clinical-data.json` $\rightarrow$ `psa_trend` | `float64` | Mean historical PSA value. |
| `psa_tr_delta` | `clinical-data.json` $\rightarrow$ `psa_trend` | `float64` | Net PSA change ($\text{Last} - \text{First}$). |
| `psa_tr_slope` | `clinical-data.json` $\rightarrow$ `psa_trend` | `float64` | Calculated PSA trajectory slope ($\Delta \text{PSA} / \Delta t$). |

### D. Laboratory Results Features (Flattened from `laboratory_results`, Prefix: `lab_`)
| Column Name | Raw Source | Data Type | Description |
|:---|:---|:---:|:---|
| `lab_creatinine_mg_dl` | `laboratory_results` | `float64` | Serum creatinine level. |
| `lab_hemoglobin_g_dl` | `laboratory_results` | `float64` | Blood hemoglobin level. |
| `lab_free_psa_ng_ml` | `laboratory_results` | `float64` | Free PSA fraction level. |
| `lab_free_total_ratio` | `laboratory_results` | `float64` | Ratio of Free PSA / Total PSA. |

### E. Prior Pathology History Features (Prefix: `path_hist_`)
| Column Name | Raw Source | Data Type | Description |
|:---|:---|:---:|:---|
| `path_hist_bx_isup` | `structured-prompt.json` $\rightarrow$ `bx_isup` | `float64` | Prior biopsy ISUP grade ($1\text{--}5$). |
| `path_hist_bx_gl_prim` | `structured-prompt.json` $\rightarrow$ `bx_gl_prim` | `float64` | Prior primary Gleason score. |
| `path_hist_bx_gl_sec` | `structured-prompt.json` $\rightarrow$ `bx_gl_sec` | `float64` | Prior secondary Gleason score. |
| `path_hist_bx_gl_tert` | `structured-prompt.json` $\rightarrow$ `bx_gl_tert` | `float64` | Prior tertiary Gleason score. |

### F. Text & Narrative Section Features (Extensive Clinical Reports, Prefix: `txt_`)
| Column Name | Raw Source | Data Type | Description |
|:---|:---|:---:|:---|
| `txt_chief_complaint` | `note_sections` $\rightarrow$ `Chief complaint` | `str` | Text of chief complaint section. |
| `txt_history` | `note_sections` $\rightarrow$ `History` | `str` | Text of medical history, LUTS, lifestyle section. |
| `txt_physical_examination` | `note_sections` $\rightarrow$ `Physical examination` | `str` | Text of physical exam, DRE, vitals section. |
| `txt_prompt_summary_notes` | `structured-prompt.json` $\rightarrow$ `notes` | `str` | Headline prompt summary notes text. |
| `txt_full_prompt_narrative` | `note_sections` + `notes` | `str` | Complete rendered prompt narrative text. |
| `txt_radiology_report` | `clinical-data.json` $\rightarrow$ `radiology_report` | `str` | Full multiparametric MRI radiology report text. |
| `txt_previous_notes` | `clinical-data.json` $\rightarrow$ `previous_notes` | `str` | Full concatenated prior progress notes text. |
| `txt_laboratory_results_summary` | `clinical-data.json` $\rightarrow$ `laboratory_results` | `str` | Full text narrative summary of all lab results. |
| `txt_psa_trend_summary` | `clinical-data.json` $\rightarrow$ `psa_trend` | `str` | Full text narrative trajectory summary of PSA history. |
| `txt_family_history_narrative` | `clinical-data.json` $\rightarrow$ `family_history` | `str` | Full narrative text of family history. |
| `txt_comorbidities` 🌟 | `structured-prompt.json` $\rightarrow$ `pmhx` | `str` | Comma-separated comorbidity text (ej. `"Hypertension, COPD, CKD"`). |
| `txt_allergies` | `structured-prompt.json` $\rightarrow$ `allergies` | `str` | Comma-separated allergies text (ej. `"NSAIDs, Penicillin"`). |
| `txt_consolidated_ehr_narrative` | All EHR text fields combined | `str` | **Master Narrative:** Complete concatenated narrative of all prompt sections, MRI radiology report, progress notes, labs, and family history. |

### G. MRI Neural Representations (Prefix: `mri_emb_`)
| Column Name | Raw Source | Data Type | Description |
|:---|:---|:---:|:---|
| `mri_emb_0` ... `mri_emb_1023` | `neural-representations.json` $\rightarrow$ `MRI image` | `float64` | 1024 scalar float columns representing frozen MRI foundation embedding vector (`np.nan` if array empty). |

---

## 4. Detailed Schema Specification for `ground_truth.csv`

Every row represents 1 patient case, indexed by `case_id`. Unlabeled test cases ($104$ cases) contain `np.nan` across all target columns.

### A. Task 1A: Biopsy Recommendation Targets
| Column Name | Data Type | Description / Valid Values |
|:---|:---:|:---|
| `case_id` | `str` | Primary key identifier. |
| `target_biopsy_decision` | `str` | Text decision token (`"yes"` or `"no"`). |
| `target_biopsy_decision_binary` | `float64` | Binary code ($1.0$ for `"yes"`, $0.0$ for `"no"`). |

### B. Task 1B: Diagnostic Confidence Targets
| Column Name | Data Type | Description / Valid Values |
|:---|:---:|:---|
| `target_confidence` | `str` | Confidence string (`"clear"`, `"borderline"`, `"uncertain"`). |
| `target_confidence_code` | `float64` | Ordinal code ($2.0$ for `"clear"`, $1.0$ for `"borderline"`, $0.0$ for `"uncertain"`). |

### C. Task 1C: Clinical Relevance Weights Targets (The 10 Official Variables)
| Target Column Name | Discrete String Values | Ordinal Numeric Code Column | Code Values |
|:---|:---:|:---|:---:|
| `target_weight_age` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_age` | $0, 1, 2, 3$ |
| `target_weight_fh` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_fh` | $0, 1, 2, 3$ |
| `target_weight_cspca` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_cspca` | $0, 1, 2, 3$ |
| `target_weight_pirads` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_pirads` | $0, 1, 2, 3$ |
| `target_weight_vol` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_vol` | $0, 1, 2, 3$ |
| `target_weight_psa` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_psa` | $0, 1, 2, 3$ |
| `target_weight_comorbidity` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_comorbidity` | $0, 1, 2, 3$ |
| `target_weight_psad` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_psad` | $0, 1, 2, 3$ |
| `target_weight_dre` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_dre` | $0, 1, 2, 3$ |
| `target_weight_bx` | `"not_used"`, `"noted"`, `"important"`, `"decisive"` | `target_code_weight_bx` | $0, 1, 2, 3$ |

### D. Task 1C: Reasoning Narrative & Tool Sequence Targets
| Column Name | Data Type | Description |
|:---|:---:|:---|
| `target_reasoning_free_text` | `str` | Ground-truth reference rationale text written by expert urologist. |
| `target_reveal_sequence_json` | `str` | Serialized JSON string of tool reveal events in ground truth. |

---

## 5. File Layout for `exp_1`

```
experiments/exp_1/
├── DESIGN.md                  ← this experiment design file
├── scripts/                   ← data extraction & validation scripts (to be defined in IMPLEMENTATION.md)
├── results/                   ← output execution logs & validation reports
└── reports/                   ← data distribution summaries & figure plots
```

---

## 6. Verification Criteria & Success Decision Rules

1. **Completeness Verification**: `inputs.csv` must contain exactly $195$ rows corresponding to all 195 raw cases.
2. **Target Alignment Verification**: `ground_truth.csv` must contain exactly $195$ rows, with non-null values for $91$ training cases and `np.nan` for $104$ test cases.
3. **No Collections Guarantee**: Automated pandas data-type check asserting zero columns of type `list`, `dict`, or object collections.
4. **Complete Text Field Coverage**: All 12 text and narrative columns (`txt_*`) must be populated with clean string representations.
5. **10 Target Variables Parity**: All 10 official relevance variables (`age`, `fh`, `cspca`, `pirads`, `vol`, `psa`, `comorbidity`, `psad`, `dre`, `bx`) must have 1-to-1 matching feature columns in `inputs.csv` and target weight columns in `ground_truth.csv`.

---

## 7. Next Steps

1. Review and approve this complete experiment design plan (`experiments/exp_1/DESIGN.md`).
2. Upon approval, create the corresponding **Implementation Plan** (`experiments/exp_1/IMPLEMENTATION.md`) in plan mode to specify the Python data extraction script `src/data/build_master_dataset.py`.
