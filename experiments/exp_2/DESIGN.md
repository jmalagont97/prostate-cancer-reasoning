# Experiment Design: Multimodal Preprocessing & Canonical Ingestion for Task 1 (`exp_2`)

**Experiment**: experiments/exp_2/  
**Project**: prostate-cancer-reasoning  
**Date**: 2026-08-29  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Updated — BX integration pending execution validation

---

## 1. Hypothesis & Research Objective
Extracting, unrolling, and canonicalizing the baseline-accessible multimodal data of Task 1 (`data/raw/train_release/task1/`) into four standardized CSV representations (Clinical Tabular Features, Medical Narrative Texts, 1024-dimensional MRI Neural Embeddings, and Supervised Multi-Subtask Ground Truth) will provide a machine-learning-ready foundation for all 195 cohort patients (including the 91 supervised instances). Prior-biopsy status `bx` is read from the structured prompt and normalized to `1` (Positive), `0` (Negative), or IEEE 754 `NaN` (raw `None`).

---

## 2. Scope & Exclusion Constraints
- **Cohort**: `data/raw/train_release/task1/` (195 patient directories).
- **Target Output Directory**: `data/processed/task1/`
- **Global File Exclusion**: All `prostate-biopsy-decision-clinical-data.json` files are **strictly excluded** across all 195 cases (preventing leakage of tool-gated EHR sections: `radiology_report`, `previous_notes`, `laboratory_results`, `psa_trend`, `family_history`).
- **Subject-Specific Exclusion**: The file `PT-pseudo_0bd72f429a8b/prostate-modality-level-neural-representations.json` is explicitly excluded from extraction; embedding columns for `PT-pseudo_0bd72f429a8b` are imputed with `NaN`.

---

## 3. Mapping of the 10 Essential Clinical Evaluation Variables

The official evaluation protocol for Task 1 evaluates reasoning alignment across exactly **10 essential clinical variables**. The extraction pipeline guarantees their presence across tabular, text, and ground-truth artifacts:

| # | Essential Clinical Variable | Clinical Meaning | Feature Extraction Source (`structured-prompt.json`) | Ground Truth Target (`ground_truth_task1.csv`) |
|---|---|---|---|---|
| 1 | **`age`** | Patient age in years | `sp['age']` (numeric float/int) | `weight_age` + factor status |
| 2 | **`psa`** | Serum Prostate-Specific Antigen (ng/mL) | `sp['psa']` (numeric float) | `weight_psa` + factor status |
| 3 | **`vol`** | Prostate gland volume (mL) | `sp['vol']` (numeric float) | `weight_vol` + factor status |
| 4 | **`psad`** | PSA Density (PSA / Volume, ng/mL/mL) | `sp['psad']` (numeric float) | `weight_psad` + factor status |
| 5 | **`dre`** | Digital Rectal Examination findings | `sp['dre']` (categorical/text: Normal, Nodus, etc.) | `weight_dre` + factor status |
| 6 | **`pirads`** | mpMRI PI-RADS v2.1 score (1 to 5) | `sp['pirads']` (categorical string/int: 1-5) | `weight_pirads` + factor status |
| 7 | **`fh`** | Family history of prostate cancer | Parsed from `sp['history']`, `sp['notes']`, or `NaN` | `weight_fh` + factor status |
| 8 | **`comorbidity`** | Clinical comorbidities & systemic status | `sp['medhx']`, `sp['pmhx']`, `sp['meds']`, `sp['vitals']` | `weight_comorbidity` + factor status |
| 9 | **`bx`** | Prior biopsy history / status | `sp['bx']`: Positive → 1, Negative → 0, None → `NaN` | `weight_bx` + factor status |
| 10 | **`cspca`** | Risk score for clinically significant PCa | `sp['cspca']` (numeric float: 0.0 to 1.0) | `weight_cspca` + factor status |

---

## 4. Extraction Specifications for Target CSV Artifacts

| Target Artifact | Primary Source | Extracted Features / Columns | Missing Value Handling |
|---|---|---|---|
| **`clinical_tabular.csv`** | `structured-prompt.json` | `pid`, `case_id`, `age`, `psa`, `months`, `pirads`, `psad`, `psav`, `psap`, `vol`, `dre`, `bx`, `cspca`, `ipss`, `active_treatment_flag`, `enc_dept`, `enc_date`, `enc_ref`, `enc_type`, `mri_hospital`, `ct`, `occupation`, `marital`, `living`, `next_of_kin`, `alcohol`, `exercise`, `vitals_weight`, `vitals_height`, `vitals_bmi`, `vitals_bp_sys`, `vitals_bp_dia`, `vitals_hr`, `vitals_smoking` | `NaN` for missing BX and other unavailable values |
| **`clinical_texts.csv`** | `structured-prompt.json` | `pid`, `case_id`, `summary_notes`, `chief_complaint`, `history`, `physical_exam`, `medhx`, `meds`, `pmhx`, `allergies`, `recent_other`, `admin_notes`, `full_encounter_narrative` | `NaN` / Empty String |
| **`mri_embeddings.csv`** | `prostate-modality-level-neural-representations.json` | `pid`, `case_id`, `emb_0` ... `emb_1023` (1024 float columns) | `NaN` for `PT-pseudo_0bd72f429a8b` |
| **`ground_truth_task1.csv`** | `prostate-biopsy-decision.json` & `prostate-biopsy-decision-reasoning.json` | `pid`, `case_id`, `decision`, `confidence`, `weight_age`, `weight_psa`, `weight_vol`, `weight_psad`, `weight_dre`, `weight_pirads`, `weight_fh`, `weight_comorbidity`, `weight_bx`, `weight_cspca`, `factors_decisive`, `factors_important`, `reveal_sequence`, `free_text` | Filtered to 91 labeled cases (`NaN` for unlabelled) |

---

## 5. Validation Protocol & Quality Gates
- **Cohort Completeness**: Exactly 195 rows for `clinical_tabular.csv`, `clinical_texts.csv`, and `mri_embeddings.csv`. Exactly 91 labeled rows in `ground_truth_task1.csv`.
- **Essential Variables Audit**: 100% representation and coverage of the 10 evaluation variables across the tabular and ground truth outputs.
- **Integrity Check**: 100% bijective match on `pid` across all 4 CSV files.
- **Embedding Dimensionality**: Exact $(195, 1024)$ matrix; five cases have all embedding values as `NaN` (one explicitly excluded case plus four missing raw representation files).
- **BX Integrity**: All 195 structured prompts contain `bx`; canonical non-missing values are restricted to `{0, 1}`, and raw `None` is preserved as `NaN`.
- **Leakage Compliance**: Tabular BX is sourced from `structured-prompt.json`; tool-gated clinical-data files are not read by the preprocessor.

---

## 6. Next Steps
1. Review summary report (`experiments/exp_2/reports/summary.md`).
2. Proceed to downstream supervised modeling and agentic prompt engineering experiments (`exp_3`).
