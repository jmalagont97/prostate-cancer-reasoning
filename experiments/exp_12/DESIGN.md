# Experiment Design: Replication of the Three Task-2 Subtasks under MCCV-100 + LOOCV (exp_12)

**Experiment**: `experiments/exp_12/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Date**: 2026-08-26
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Draft — pending PI approval
**Based on**: exp_10 (fuzzy-KNN champion, Macro-F1 0.5971) and exp_11 (fuzzy-SVR, Macro-F1 0.6166)

---

## 1. Hypotheses

**H1 (Subtask 2.1 — Treatment Recommendation, 4-class):** The fuzzy-KNN late fusion champion of exp_10 (hard/fuzzy comparison, weights w=(0.35,0.45,0.20)), re-evaluated under the same MCCV-100 + LOOCV protocol, reproduces Macro-F1 >= 0.5971 and Brier <= 0.5176 (non-inferiority, Wilcoxon paired p >= 0.05).

**H2 (Subtask 2.2 — Clinical Confidence, 3-class):** A fuzzy-KNN late fusion over the same multimodal features, with ordinal soft-targets (clear=1.0, borderline=0.5, uncertain=0.25), achieves S_conf (ordinal distance) on par with the Task-1 equivalent (exp_6: Macro-F1 0.4836, S_conf 0.7157) despite the structural absence of the `uncertain` class in Task-2 GT (n=0).

**H3 (Subtask 2.3 — Variable Weights, 11 variables x 4 ordinal levels):** Multivariate meta-models (RandomForest / ExtraTrees with class_weight='balanced') over the same multimodal features, treating GT omissions as implicit `not_used` (audited: 18/70 omissions for bx_gl_prim, bx_gl_sec, bx_isup), achieve S_weights comparable to the Task-1 benchmark (exp_9: S_weights 0.8278) and significant positive Spearman rho on primary clinical drivers (psa, pirads, bx_isup, ct).

---

## 2. Experimental Setup

- **Dataset**: CHIMERA Task 2 trimodal complete cohort (N=70) — `data/processed/task2/` (identical to exp_10/11).
- **Splits**: EXACT REUSE of `data/processed/task2/mccv_splits_100.csv` (100 stratified 4-class splits, seed=42) — enables paired Wilcoxon vs exp_10.
- **Phase B**: Frozen LOOCV (N=70) with Phase-A hyperparameters.
- **GT files used**:
  - 2.1: `prostate-treatment-decision.json` (4-class string).
  - 2.2: `prostate-treatment-decision-reasoning.json` -> "confidence" (clear=56, borderline=14, uncertain=0).
  - 2.3: same reasoning file -> "variable_weights" (11 vars; omissions = 18 cases for bx_* vars).
- **Preprocessing**: IDENTICAL to exp_10 for all three subtasks (Tabular MinMaxScaler 32 cols; MRI 1024D no PCA; Text TF-IDF-500 or full vocab). Zero leakage: active_treatment_flag excluded; scalers/TF-IDF fit only on train fold.

---

## 3. Soft-Target Formulations per Subtask

| Subtask | Hard target | Soft target (fuzzy) |
| :--- | :--- | :--- |
| **2.1 Treatment** | one-hot 4-class | y_k = 0.25 + 0.75c; y_j!=k = (1-y_k)/3 (exp_10 formula) |
| **2.2 Confidence** | one-hot 3-class | ordinal: clear=1.0, borderline=0.5, uncertain=0.25 (exp_6-style) |
| **2.3 Weights** | one-hot 4-nivel per variable | ordinal per variable: not_used=0.0, noted=1/3, important=2/3, decisive=1.0 |

---

## 4. Conditions per Subtask

### 4.1 Subtask 2.1 (Treatment)
Re-evaluates the exp_10 champion methodology under the same protocol (NO re-training on new configs):

| Condition | Model | Config |
| :--- | :--- | :--- |
| C1 | Trimodal fuzzy-KNN (H2 exp_10) | k_tab=1, k_mri=5, k_txt=5; weights uniform; w=(0.35,0.45,0.20) |
| C2 | Trimodal hard-KNN (H1 exp_10) | k_tab=1, k_mri=5, k_txt=3; weights uniform; w=(0.35,0.45,0.20) |

### 4.2 Subtask 2.2 (Confidence)
Fuzzy-KNN grids per modality + fusion:

| Condition | Modality | Grid |
| :--- | :--- | :--- |
| C3 | Tabular fuzzy | k in {1,3,5,7} x weights {uniform,distance} |
| C4 | MRI fuzzy | k in {1,3,5} x weights {uniform,distance} |
| C5 | Text fuzzy | k in {1,3,5} x max_features {500,None} |
| C6 | **Fusion 2.2** | soft-voting w in Delta^2 (step 0.05, 231 weights) |

### 4.3 Subtask 2.3 (Variable Weights)
11 independent meta-models per variable, one per algorithm:

| Condition | Meta-model | Space |
| :--- | :--- | :--- |
| C7 | RandomForest (balanced) | Per-variable on multimodal feature union (32 tab + MRI-PCA-16 + TF-IDF top-k) |
| C8 | ExtraTrees (balanced) | Same as C7 |
| C9 | **Best per-variable algorithm selection** | Chosen by mean val Macro-F1 over 100 splits |

---

## 5. Evaluation Protocol

| Subtask | Primary metrics | Secondary metrics | Official gate |
| :--- | :--- | :--- | :--- |
| 2.1 | Macro-F1, Brier | Micro-F1, Weighted-F1, AUROC OVR, per-class F1 | compute_treatment_decision_gate |
| 2.2 | S_conf (1 - normalized ordinal distance), Macro-F1 | Brier, per-class F1 | compute_confidence_score |
| 2.3 | S_weights, Decisive Factors Set-F1 | per-variable Spearman rho, Macro-F1 per variable | compute_variable_weights_score, compute_important_decisive_f1 |

**Homologation**: sklearn.metrics for Macro/Micro/Weighted-F1/AUROC; multiclass Brier implemented explicitly (same as exp_10).

---

## 6. Statistical Tests

1. Wilcoxon paired (two-sided, alpha=0.05) 2.1 C1 vs C2 (internal).
2. Wilcoxon paired exp_12-2.1 vs exp_10-H2 (cross, 100 shared splits).
3. Wilcoxon paired exp_12-2.2 vs exp_6 (Task-1 cross, if split alignment allows; otherwise descriptive).
4. Per-variable Spearman rho with p-values (2.3) vs synthetic majority baseline.

---

## 7. Storage Footprint (Minimal) & Model Persistence (Pickles)

### 7.1 JSON artifacts (minimal, same as exp_10)

Per condition: `aggregate_metrics.json`, `split_metrics.json`, `wilcoxon_tests.json`, `best_hyperparams.json`, `version.txt`. NO prediction CSVs.

### 7.2 Persisted best models (`.pickle`) — REQUIRED

Following the established pattern from exp_10/exp_11, the best model per subtask is retrained on ALL N=70 labeled cases (Phase-A frozen hyperparameters) and persisted under `experiments/exp_12/models/` as pickle bundles `{preprocessor, model}`:

| Subtask | Pickle artifact(s) | Content |
| :--- | :--- | :--- |
| **2.1 Treatment (4-class)** | `subtask1_treatment_fuzzy_knn.pkl` | Best fuzzy-KNN fusion (exp_10 H2: k_tab=1, k_mri=5, k_txt=5; fuzzy; weights uniform) + 3 preprocessors + `subtask1_fusion_weights.json` |
| **2.2 Confidence (3-class)** | `subtask2_confidence_fuzzy_knn.pkl` | Best fuzzy-KNN for ordinal soft-targets (clear=1.0, borderline=0.5, uncertain=0.25) + 3 preprocessors + `subtask2_fusion_weights.json` |
| **2.3 Variable Weights (11 vars x 4 levels)** | `subtask3_weights_ensemble.pkl` (single bundle with 11 meta-models in dict) | Best meta-model per variable (RF or ExtraTrees, balanced) selected by mean val Macro-F1 + shared preprocessors + `variable_weights_metadata.json` (per-variable config, thresholds, class mapping) |

Each pickle is loadable with the standard `sys.modules['__main__']` resolution trick (validated in exp_10/exp_11). Inference script `predict_case_task2.py` (exp_12) uses these bundles to produce the two official Task-2 JSON outputs per case (decision + reasoning).

---

## 8. Risks & Mitigations (updated after audit)

| Risk | Prob./Sev. | Mitigation |
| :--- | :---: | :--- |
| **2.2: `uncertain` class absent (n=0)** | Alta / Alta | Pre-registered: Macro-F1 for that class is undefined; report S_conf + Brier as primary. No oversampling (would be artificial). |
| **2.3: GT omissions of bx_* vars (18/70)** | Media / Media | Treat omission as implicit `not_used` (class 0) — confirmed that `compute_variable_weights_score` defaults to `not_used` for missing pred vars. |
| **2.3: 11 vs 17 variable docstring mismatch** | Baja / Baja | AUDITED: function iterates over actual GT dict keys, not fixed list — no functional misalignment. |
| **2.1: Re-evaluation overfits to exp_10 configs** | Media / Media | Configs frozen from exp_10; no Phase-A re-selection for 2.1. |
| **Small N=70** | Alta / Media | MCCV-100 + frozen LOOCV double confirmation. |

---

## 9. Expected Results & Decision Rules

- **2.1 success**: Macro-F1 >= 0.5971 (non-inferiority, p >= 0.05 vs exp_10-H2).
- **2.2 success**: S_conf >= 0.70 on validation (descriptive vs Task-1 exp_6 0.7157).
- **2.3 success**: S_weights >= 0.70 and Decisive Factors Set-F1 >= 0.60; rho(psa) and rho(pirads) >= +0.30 (p < 0.05).
- **Failure/adjustment**: if any subtask significantly underperforms, propose targeted feature engineering (exp_13).

---

## 10. Reproducibility Checklist

- [ ] REUSE mccv_splits_100.csv (paired comparisons valid)
- [ ] Zero leakage (active_treatment_flag excluded; fits train-only)
- [ ] Soft-target formulas documented per subtask
- [ ] GT omissions treated as not_used (audited)
- [ ] Metrics via chimera_metrics + sklearn homologation
- [ ] Storage minimal (JSON only)
- [ ] Version + fallbacks logged in version.txt

---

## 11. Next Steps

1. PI approval of this DESIGN.md.
2. Generate IMPLEMENTATION.md (single runner run_exp12_mccv.py handling 3 subtasks).
3. Execute in background (nohup) and persist results/.
4. experiment-reporter -> reports/summary.md.
5. Update .logbook.md and .discussion.md after consensus.

---