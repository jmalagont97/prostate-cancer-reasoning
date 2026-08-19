# Implementation Plan: Multimodal Input Reorganisation (exp_3)
**Experiment**: experiments/exp_3/ · **Project**: pathology-reasoning · **Date**: 2026-08-16 · **Status**: Approved

---

## 1. Overview & Objective

Generate, deterministically, the three input-view CSVs from the canonical
`inputs.csv`, following `experiments/exp_3/DESIGN.md`. This replaces the previous
exp_3 contract (7 reveal-group views + feature manifest), whose artifacts are
removed. `inputs.csv`, `ground_truth.csv` and `mccv_loocv_splits.csv` are untouched.

## 2. File & Script Structure

```
experiments/exp_3/
├── DESIGN.md                   ← Approved research design
├── IMPLEMENTATION.md           ← This implementation plan
├── scripts/
│   └── build_multimodal_views.py ← Main execution entry point & validator
└── results/
    ├── validation_report.json    ← Automated validation audit report
    ├── data_manifest.csv         ← SHA-256 registry of all data artifacts
    └── git_commit.txt            ← Git commit at execution time
```

## 3. Detailed Logic

### 3.1 View construction
- `main_tabular` = `["case_id"] + cli_* + vit_* + path_hist_*` (in source order).
  Prefix counts are asserted: 15 `cli_*`, 8 `vit_*`, 4 `path_hist_*`, 1024 `mri_emb_*`.
- `full_prompt_narrative` = `["case_id", "txt_full_prompt_narrative"]`.
- `images` = `["case_id"] + mri_emb_*`.
- Values are copied without transformation.

## 4. Validation & Decision Rules

Hard assertions (fail → non-zero exit):

1. Each view: 195 rows, unique `case_id`; same case set across views.
2. The 10 official relevance variables present in `main_tabular`.
3. No feature duplicated across views; no `txt_*` in `main_tabular`.
4. No `ground_truth.csv` column in any view.
5. CSV round-trip faithfulness (`assert_frame_equal`).
6. `images.csv`: exactly the 4 expected all-`NaN` cases.
7. Expected shapes: `main_tabular` (195×28), `full_prompt_narrative` (195×2), `images` (195×1025).
8. Determinism: two in-process builds produce identical frames.
9. SHA-256 of `ground_truth.csv` and `mccv_loocv_splits.csv` unchanged after generation.

## 5. Execution

```bash
python3 experiments/exp_3/scripts/build_multimodal_views.py
```

Exit code 0 ⇒ all validations passed. Reports written to `experiments/exp_3/results/`.

## 6. Acceptance Criteria

- `main_tabular.csv`, `full_prompt_narrative.csv` and `images.csv` exist with the
  documented shapes.
- `validation_report.json` records `all_passed: true` for every check.
- `data_manifest.csv` records SHA-256 for all artifacts (source + derived).
- `ground_truth.csv` and `mccv_loocv_splits.csv` byte-unchanged.
