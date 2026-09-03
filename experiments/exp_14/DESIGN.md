# Experiment Design: Tabular Biopsy-Field Ablation for the Three Task-2 Subtasks under MCCV-100 + LOOCV (exp_14)

**Experiment**: `experiments/exp_14/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Date**: 2026-08-31
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Draft — pending PI approval
**Based on**: exp_12 (three-subtask KNN pipeline), exp_10 (fuzzy-KNN champion)

---

## 1. Context & Hypothesis

The audit of the Task-2 cohort (70 cases) confirmed that exactly three biopsy- / pathology-derived fields exist as explicit tabular features in the current training space: `bx_isup`, `bx_gl_prim`, `bx_gl_sec` (present 70/70). The remaining 11 fields requested for exclusion (`bx_gl_tert`, `bx_isup_pred`, `cores_positive`, `cores_total`, `max_core_pct`, `growth_pattern`, `high_risk_patterns`, `pni`, `lvi`, `tumor_location`) do NOT exist as tabular columns; however, they are present inside the free-text `pathology_report` and `radiology_report` of the TEXT modality (e.g., "isup"=216 hits, "gleason"=149, "growth pattern"=44, "perineural"=63, "peripheral zone"=194).

**Option A (chosen by PI)**: remove ONLY the three tabular fields from `T2_NUMERIC_COLS`; keep the text modality unchanged.

**H1 (Redundancy, expected)**: Removing `bx_isup`, `bx_gl_prim`, `bx_gl_sec` from the tabular space does NOT significantly degrade Subtask 2.1 Macro-F1 vs exp_12 (tolerance |Δ| ≤ 0.02, Wilcoxon paired p ≥ 0.05), because the text modality already encodes the pathology information that these fields carry (Gleason / ISUP appear abundantly in pathology narratives).

**H2 (Structural limitation of ablations)**: Subtask 2.2 confidence and Subtask 2.3 variable weights will also remain statistically unchanged (within tolerance), since their exp_12 baselines (0.3736 and per-var best 0.4229) are limited by the `uncertain` n=0 and raw-feature insufficiency respectively — not by the three removed fields.

---

## 2. Experimental Setup

- **Dataset**: Task 2 trimodal cohort (N=70), `data/processed/task2/` — identical to exp_10/11/12/13.
- **Feature ablation**: from `T2_NUMERIC_COLS`, REMOVE `bx_isup`, `bx_gl_prim`, `bx_gl_sec` → **29 remaining tabular features** (was 32). Categorical (`dre`, `bx`, `ct`) unchanged.
- **Splits**: EXACT REUSE of `mccv_splits_100.csv` (100 stratified 4-class splits, seed=42) — paired Wilcoxon vs exp_12 valid.
- **Phase B LOOCV**: included, as in exp_13 (not as in exp_12), to close the protocol and give honest generalization.
- **Preprocessing**: identical to exp_12 (MinMaxScaler for tabular 29 cols; MRI 1024D; TF-IDF text). Zero leakage: `active_treatment_flag` excluded; scalers fit train-only.

---

## 3. Conditions

### 3.1 Subtask 2.1 (Treatment 4-class)
| Cond | Family | Config | Justification |
| :--- | :--- | :--- | :--- |
| C1 | Fuzzy-KNN frozen | exp_10 H2 config (k_tab=1, k_mri=5, k_txt=5; w=0.35/0.45/0.20) | Isolates pure feature effect (same hyperparameters as exp_12) |
| C2 | Fuzzy-KNN re-grid | k in {1,3,5,7} per modality on 29-col space | Measures best achievable without the fields |

Baseline reference: exp_12-2.1 C1 (fuzzy frozen, full features) = Macro-F1 0.5971.

### 3.2 Subtask 2.2 (Confidence 3-class)
| Cond | Modality | Grid |
| :--- | :--- | :--- |
| C3 | Tabular fuzzy | k in {1,3,5,7} x weights {uniform,distance} (29 cols) |
| C4 | MRI fuzzy | k in {1,3,5} x weights {uniform,distance} |
| C5 | Text fuzzy | k in {1,3,5} x max_features {500,None} |
| C6 | Fusion | soft-voting w in Delta^2 (step 0.05, 231) |

Baseline reference: exp_12-2.2 = Macro-F1 0.3736.

### 3.3 Subtask 2.3 (Variable weights, 11 vars x 4 ordinal)
| Cond | Algorithm | Space |
| :--- | :--- | :--- |
| C7 | RandomForest balanced | 29-dim tabular + MRI-PCA-16 + TF-IDF-50 (95D joint, recomputed) |
| C8 | ExtraTrees balanced | same |
| C9 | Best per-variable | by mean val Macro-F1 |

Baseline reference: exp_12-2.3 best per variable = bx_isup 0.4229.

---

## 4. Evaluation Protocol

| Subtask | Primary metrics | Secondary | Official gate |
| :--- | :--- | :--- | :--- |
| 2.1 | Macro-F1, Brier | Micro-F1, Weighted-F1, AUROC OVR, per-class F1 | compute_treatment_decision_gate |
| 2.2 | S_conf (ordinal), Macro-F1 | Brier, per-class F1 | compute_confidence_score |
| 2.3 | S_weights, Decisive Factors Set-F1 | per-variable Spearman rho, per-variable Macro-F1 | compute_variable_weights_score, compute_important_decisive_f1 |

---

## 5. Statistical Tests

1. Wilcoxon paired (100 shared splits) exp_14-2.1 C1 vs exp_12-2.1 C1 (same frozen config, only features differ): tests ablation effect directly.
2. Wilcoxon paired exp_14-2.1 C2 (re-grid) vs exp_12-2.1 C1: measures best achievable without fields vs frozen-with-fields.
3. Wilcoxon paired exp_14-2.2 fusion vs exp_12-2.2 fusion (Brier / Macro-F1).
4. Descriptive 2.3 per-variable deltas vs exp_12.
5. McNemar exact if any Wilcoxon marginal (p in [0.05,0.10]).

---

## 6. Storage Footprint & Model Persistence

### 6.1 JSON artifacts (minimal)
`aggregate_metrics.json`, `split_metrics.json`, `wilcoxon_tests.json`, `best_hyperparams.json`, `version.txt`. NO prediction CSVs.

### 6.2 Persisted best models (REQUIRED, exp_12 pattern)
| Subtask | Pickle artifact | Content |
| :--- | :--- | :--- |
| 2.1 | `subtask1_treatment_ablated_fuzzy_knn.pkl` | Best frozen/re-grid fuzzy-KNN + 3 preprocessors + fusion weights |
| 2.2 | `subtask2_confidence_ablated_fuzzy_knn.pkl` | Best fuzzy-KNN + preprocessors + fusion weights |
| 2.3 | `subtask3_weights_ablated_ensemble.pkl` | 11 meta-models dict + preprocessors + metadata |

Each pickle loadable with `sys.modules['__main__']` trick.

---

## 7. Risks & Mitigations

| Risk | Prob./Sev. | Mitigation |
| :--- | :---: | :--- |
| **Text still leaks pathology** (by design, Option A) | Alta / Media | Registered: H1 expects no degradation precisely because text carries the info. If Macro-F1 DROPS despite text, that proves the explicit tabular fields carry non-redundant signal — a meaningful finding. |
| **Re-grid C2 overfits 29-col space** | Media / Media | Frozen C1 is the isolated primary; C2 is exploratory secondary. |
| **2.2/2.3 ablations confounded by structural limits** | Media / Media | Pre-registered: thresholds relative to exp_12 baselines hold regardless of ablation. |
| **watchful_waiting n=2** | Alta / Alta | Already documented; per-class F1 unstable. |

---

## 8. Expected Results & Decision Rules

- **H1 supported** if C1 vs exp_12-2.1: Δ ∈ [-0.02, +0.02] and Wilcoxon p ≥ 0.05.
- **H1 rejected (meaningful ablation)** if Δ < -0.02 with p < 0.05 → the 3 tabular fields carry non-redundant discriminative signal.
- **2.2/2.3** unchanged within tolerance ±0.02 vs exp_12.
- **Failure/adjustment**: if C2 re-grid recovers performance vs C1, the degradation is hyperparameter-dependent, not feature-dependent (document separately).

---

## 9. Reproducibility Checklist

- [ ] REUSE mccv_splits_100.csv
- [ ] FEATURE DELTA verified: 32 → 29 tabular cols (bx_isup, bx_gl_prim, bx_gl_sec removed; categorical unchanged)
- [ ] Zero leakage (active_treatment_flag excluded)
- [ ] Phase B LOOCV included
- [ ] Metrics via chimera_metrics + sklearn homologation
- [ ] 3 pickle bundles persisted + loadable
- [ ] Storage minimal (JSON + pickles only)

---

## 10. Next Steps

1. PI approval of this DESIGN.md.
2. Generate IMPLEMENTATION.md (adapt exp_13 runner: remove 3 cols, add frozen-vs-regrid contrast).
3. Execute in background; verify; commit to jtarquino_task2.
4. Update .logbook.md and .discussion.md after consensus.

---