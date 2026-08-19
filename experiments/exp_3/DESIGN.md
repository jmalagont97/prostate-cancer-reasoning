# Experiment Design: Multimodal Input Reorganisation for the Three-Model Pipeline (Task 1)
**Experiment**: experiments/exp_3/ · **Project**: pathology-reasoning · **Date**: 2026-08-16 · **Status**: Approved

---

## 1. Hypothesis

Reorganising the canonical `inputs.csv` (195 × 1077) into three minimal, purpose-built
input views for the three-model pipeline — tabular (`main_tabular`), text
(`full_prompt_narrative`) and image (`images`) — is sufficient to feed the models
that predict biopsy decision, clinical confidence and the 10 clinical relevance
weights, while keeping all non-essential information out of scope. This simplifies
the data contract and guarantees that the 10 official relevance variables are
present as tabular inputs from the start.

## 2. Experimental Setup

- **Source**: `data/chimera26/preprocessed/task1/inputs.csv` (canonical matrix, 195 × 1077).
- **Protected artifacts** (must remain byte-identical): `ground_truth.csv`, `mccv_loocv_splits.csv`.
- **Output directory**: `data/chimera26/preprocessed/task1/`.

### Output Files

| File | Content | Expected shape |
|:---|:---|:---:|
| `main_tabular.csv` | `case_id` + all tabular `main` variables: 15 `cli_*` (incl. `cli_fh_binary`, `cli_comorbidity_count`, `cli_allergies_count`, `cli_ipss_score`), 8 `vit_*`, 4 `path_hist_*`. No `txt_*`. | 195 × 28 |
| `full_prompt_narrative.csv` | `case_id` + `txt_full_prompt_narrative` (visible text from `structured-prompt.json`). | 195 × 2 |
| `images.csv` | `case_id` + `mri_emb_0` … `mri_emb_1023`. | 195 × 1025 |

All other `inputs.csv` columns are intentionally ignored for this experiment
(`psa_tr_*`, `lab_*`, remaining `txt_*`, `txt_consolidated_ehr_narrative`).

## 3. Main Tabular Composition

The 10 official relevance variables are all present in `main_tabular.csv`:

| Relevance variable | Column |
|:---|:---|
| `age` | `cli_age` |
| `fh` | `cli_fh_binary` |
| `cspca` | `cli_cspca` |
| `pirads` | `cli_pirads` |
| `vol` | `cli_vol` |
| `psa` | `cli_psa` |
| `comorbidity` | `cli_comorbidity_count` |
| `psad` | `cli_psad` |
| `dre` | `cli_dre` |
| `bx` | `cli_bx` |

Note: this is a **practical** input organisation, not a strict reconstruction of
pre-tool visibility. `cli_fh_binary` (family history) is included because the
experiment reuses all `main` tabular variables; downstream pre-reveal models must
decide explicitly whether to use it as input or hold it for the family-history
stage.

## 4. Verification Criteria & Decision Rules

1. Every view holds exactly 195 rows with unique `case_id`; all views share the same case set.
2. The 10 official relevance variables are present in `main_tabular.csv`.
3. No feature is duplicated across views; no `txt_*` column inside `main_tabular`.
4. No `ground_truth.csv` column appears in any view.
5. Values are byte-faithful to the source (CSV round-trip equality).
6. The 4 MRI-missing cases keep all-`NaN` rows in `images.csv`.
7. Generation is deterministic (double-run identical).
8. `ground_truth.csv` and `mccv_loocv_splits.csv` are byte-unchanged after generation.

## 5. Next Steps

1. This replaces the previous exp_3 (7 reveal-group views + feature manifest), whose
   artifacts are removed.
2. Train the three models (decision, confidence, relevance) on these three views with
   the established nested-CV protocol, and convert predicted relevance to
   `reveal_sequence` via the dictionary rule.
