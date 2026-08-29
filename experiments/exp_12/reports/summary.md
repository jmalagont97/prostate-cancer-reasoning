# Experiment Report: Replication of the Three Task-2 Subtasks under MCCV-100 (exp_12)

**Experiment**: `experiments/exp_12/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Report Date**: 2026-08-29
**Plan Date**: 2026-08-26
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Complete (Phase A MCCV-100; Phase B LOOCV not executed)
**Based on**: exp_10 (fuzzy-KNN champion) / exp_11 (fuzzy-SVR) and Task-1 reasoning benchmarks (exp_6, exp_8, exp_9)

---

## 1. Summary
This experiment replicated the three Task-2 subtasks under the standardized MCCV-100 protocol on the complete trimodal cohort (N=70): Subtask 2.1 (treatment recommendation, 4-class), Subtask 2.2 (clinical confidence, 3-class ordinal), and Subtask 2.3 (variable weights, 11 variables x 4 ordinal levels). Subtask 2.1 re-evaluated the frozen exp_10 fuzzy/hard KNN configs and reproduced the published champion exactly (fuzzy Macro-F1 = 0.5971; hard = 0.5863), validating implementation consistency. Subtask 2.2 trained fuzzy-KNN ordinal models and fused them with best weights w=(0.00, 0.55, 0.45), reaching Macro-F1 = 0.3736 -- below the Task-1 benchmark (exp_6: 0.4836), as pre-registered because the Task-2 GT contains zero `uncertain` cases, structurally limiting 3-class learning. Subtask 2.3 trained 11 balanced RF/ExtraTrees meta-models on a 32+16+50-dimensional joint space; `bx_isup` was the best-learned variable (Macro-F1 = 0.4229), while clinical anchors such as `pirads` remained low (0.1812), suggesting the joint feature space alone -- without SHAP-style attribution -- is insufficient for clinical relevance modeling. All best models were persisted as three loadable pickle bundles, and the inference script reproduced the exact ground truth on the T2-001 smoke case (treatment, confidence, and all 11 variable weights).

---

## 2. Hypothesis & Verdict

**H1 (2.1 Treatment):** The frozen exp_10 fuzzy-KNN configs reproduce Macro-F1 >= 0.5971 and Brier <= 0.5176 under shared MCCV-100 splits.

**H2 (2.2 Confidence):** Fuzzy-KNN ordinal fusion achieves S_conf / Macro-F1 on par with the Task-1 equivalent despite the structural absence of `uncertain` in Task-2 GT.

**H3 (2.3 Variable Weights):** Balanced RF/ExtraTrees meta-models achieve S_weights comparable to Task-1 (exp_9: 0.8278) and significant positive Spearman rho on primary drivers.

**Verdict:**

| Hypothesis | Verdict | Evidence |
| :--- | :---: | :--- |
| H1 | ✅ **Supported** | Fuzzy fusion = 0.5971 (C1) and hard = 0.5863 (C2), matching exp_10 exactly |
| H2 | ⚠️ **Partially supported** | Macro-F1 = 0.3736 < exp_6 (0.4836); structural `uncertain` absence confirmed as limiting factor |
| H3 | ⚠️ **Not met** | Variable-level Macro-F1 low-to-moderate (0.18-0.42); no official primary-driver rho thresholds reached |

---

## 3. Experimental Setup (as run)

- **Dataset**: CHIMERA Task 2 trimodal complete cohort (N=70) -- `data/processed/task2/` (identical to exp_10/11).
- **Splits**: exact reuse of `data/processed/task2/mccv_splits_100.csv` (100 stratified 4-class splits, seed=42).
- **Phase B LOOCV**: **NOT executed** in the final runner (Phase A MCCV-100 grids and re-evaluation only). Pre-registered in DESIGN/IMPLEMENTATION; logged here as a deviation for future completion.
- **GT sources**:
  - 2.1: `prostate-treatment-decision.json` (4-class).
  - 2.2: `prostate-treatment-decision-reasoning.json` -> `confidence` (clear=56, borderline=14, uncertain=0).
  - 2.3: same reasoning file -> `variable_weights` (11 vars; 18/70 omissions for bx_* vars treated as implicit `not_used`).
- **Preprocessing**: identical to exp_10 (Tabular MinMaxScaler 32 cols; MRI 1024D; Text TF-IDF). `active_treatment_flag` excluded; scalers/TF-IDF fit only on train folds.

---

## 4. Conditions

| Subtask | Condition | Description |
| :--- | :--- | :--- |
| 2.1 | C1 | Fuzzy-KNN frozen exp_10-H2 (k_tab=1, k_mri=5, k_txt=5; w=(0.35,0.45,0.20)) |
| 2.1 | C2 | Hard-KNN frozen exp_10-H1 (k_tab=1, k_mri=5, k_txt=3; w=(0.35,0.45,0.20)) |
| 2.2 | C3-C5 | Fuzzy-KNN grids per modality (k, weights, max_features) |
| 2.2 | C6 | Fusion soft-voting over simplex Delta^2 (231 weights) |
| 2.3 | C7 | RandomForest balanced (11 independent models) |
| 2.3 | C8 | ExtraTrees balanced (11 independent models) |
| 2.3 | C9 | Best per-variable algorithm by mean val Macro-F1 |

---

## 5. Results

### 5.1 Subtask 2.1 -- Treatment (4-class, re-evaluation)

| Condition | Macro-F1 (MCCV-100) | Std |
| :--- | :---: | :---: |
| C1 (fuzzy) | **0.5971** | 0.1252 |
| C2 (hard) | 0.5863 | 0.1285 |

### 5.2 Subtask 2.2 -- Confidence (3-class ordinal)

| Modality | Best config |
| :--- | :--- |
| Tabular | k=1, weights=uniform |
| MRI | k=1, weights=uniform |
| Text | k=3, max_features=500 |
| **Fusion** | **w=(0.00, 0.55, 0.45)** | Macro-F1=**0.3736** |

### 5.3 Subtask 2.3 -- Variable Weights (11 vars, per-variable Macro-F1)

| Variable | Best algorithm | Macro-F1 |
| :--- | :--- | :---: |
| bx_isup | RandomForest | **0.4229** |
| bx_gl_sec | ExtraTrees | 0.3235 |
| bx_gl_prim | ExtraTrees | 0.2895 |
| psa | ExtraTrees | 0.2750 |
| psad | ExtraTrees | 0.2257 |
| cspca | ExtraTrees | 0.2190 |
| fh | ExtraTrees | 0.2168 |
| ct | RandomForest | 0.2112 |
| comorbidity | ExtraTrees | 0.2073 |
| age | ExtraTrees | 0.1889 |
| pirads | ExtraTrees | 0.1812 |

---

## 6. Model Persistence (Pickles)

Three pickle bundles were persisted under `experiments/exp_12/models/` and verified loadable:

| Bundle | Content |
| :--- | :--- |
| `subtask1_treatment_fuzzy_knn.pkl` | 3 MulticlassKNN + 3 preprocessors + fusion weights |
| `subtask2_confidence_fuzzy_knn.pkl` | 3 MulticlassKNN (3-class ordinal) + preprocessors + fusion weights |
| `subtask3_weights_ensemble.pkl` | 11 meta-models (9 ExtraTrees + 2 RandomForest) + preprocessors + metadata |

**Smoke test T2-001** (`predict_case_task2.py`):

| Field | GT | Prediction | Match |
| :--- | :--- | :--- | :---: |
| Treatment | active_surveillance | active_surveillance | ✅ |
| Confidence | clear | clear | ✅ |
| Variable weights | 11/11 | 11/11 | ✅ |

---

## 7. Missing Data & Caveats

- **Phase B LOOCV not executed** -- only Phase A MCCV-100 completed (deviation from plan).
- **Subtask 2.2 `uncertain` absent** (n=0) -- 3-class F1 structurally limited; per-class F1 for that class is undefined.
- **Subtask 2.3 feature space insufficient** -- raw multimodal features without SHAP-style attribution (exp_8/9 lesson) yielded low per-variable F1; the pre-registered exp_9-style 23D fallback was not triggered.
- **No split-level Wilcoxon tests persisted** -- `wilcoxon_tests.json` was not generated; statistical comparisons remain future work.
- Runtime: 2168.3s (~36.5 min) for the full Phase-A benchmark.

---

## 8. Conclusions & Next Steps

- **Conclusions**:
  1. Subtask 2.1 reproduces exp_10 exactly, confirming the frozen-config re-evaluation is consistent.
  2. Subtask 2.2 is limited by the absent `uncertain` class; the ordinal fuzzy-KNN aligns with Task-1 lessons but cannot reach parity with exp_6 under this GT.
  3. Subtask 2.3 requires attribution-augmented features (SHAP-style, as in exp_8/9) rather than raw multimodal concatenation alone.
- **Next Steps**:
  - Execute Phase B LOOCV for all three subtasks (complete the protocol).
  - Implement exp_9-style attribution augmentation for Subtask 2.3 (or SHAP over frozen 2.1 models).
  - Persist per-split metrics and Wilcoxon tests.
  - Run official LLM-as-a-judge reasoning evaluation when available.

---

## 9. Reproducibility Record

| Element | Status | Detail |
| :--- | :---: | :--- |
| MCCV-100 splits reused | ✅ | `data/processed/task2/mccv_splits_100.csv` |
| Zero leakage | ✅ | `active_treatment_flag` excluded; fits train-only |
| Soft-target formulas | ✅ | Treatment / confidence / weights documented |
| GT omissions as not_used | ✅ | Audited (18/70 for bx_* vars) |
| Pickles persisted + loadable | ✅ | 3 bundles verified |
| Inference smoke test | ✅ | T2-001 exact match |
| Storage minimal | ✅ | JSON + pickles only; no prediction CSVs |

---