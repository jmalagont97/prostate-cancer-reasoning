# Experiment Report: Tabular Biopsy-Field Ablation under MCCV-100 + LOOCV (exp_14)

**Experiment**: `experiments/exp_14/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Report Date**: 2026-09-02
**Plan Date**: 2026-08-31
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Complete ✅
**Based on**: exp_12 (three-subtask KNN pipeline), exp_10 (fuzzy-KNN champion)

---

## 1. Summary
This experiment evaluated the impact of removing the three explicit biopsy-/pathology-derived tabular fields (`bx_isup`, `bx_gl_prim`, `bx_gl_sec`) from the training space of the exp_12 KNN pipeline (Task-2 subtasks 2.1/2.2/2.3). The other eleven requested fields (`bx_gl_tert`, `bx_isup_pred`, `cores_positive`, `cores_total`, `max_core_pct`, `growth_pattern`, `high_risk_patterns`, `pni`, `lvi`, `tumor_location`) were audited and found **not present as tabular columns**; they reside in the free-text pathology/radiology modality, which was kept unchanged (PI-chosen Option A). Because exp_12 did not persist split-level metrics, an **in-run paired control (C0_full vs C1_ablated)** was executed on the same 100 MCCV splits to isolate the pure feature effect. The null/redundancy hypothesis **H1 was REJECTED**: removing the three fields degraded Subtask 2.1 Macro-F1 from 0.5971 to 0.5731 (Δ = −0.0240, Wilcoxon W=89.0, p = 4.9e-10), proving the explicit tabular Gleason/ISUP fields carry non-redundant discriminative signal that the free-text modality does not fully compensate for under fuzzy-KNN.

---

## 2. Hypothesis & Verdict

**H1 (Redundancy, expected):** removing `bx_isup`, `bx_gl_prim`, `bx_gl_sec` DOES NOT significantly degrade Subtask 2.1 Macro-F1 (tolerance |Δ| ≤ 0.02, Wilcoxon p ≥ 0.05).

**H2 (Structural limitation):** Subtask 2.2 (confidence) and Subtask 2.3 (weights) remain within tolerance because they are limited by `uncertain` n=0 and raw-feature insufficiency — not by the three removed fields.

| Hypothesis | Verdict | Evidence |
| :--- | :---: | :--- |
| H1 | ❌ **REJECTED** | Δ = −0.0240 (below tolerance −0.02), W=89.0, p = 4.90e-10 → significant degradation |
| H2 | ✅ **Supported** | 2.2 (0.3429) and 2.3 (bx_isup 0.4191) remain in the expected exp_12 band (±0.03) |

---

## 3. Audit & Feature Delta

| Field | Tabular column? | Removed? | Text occurrence (kept) |
| :--- | :---: | :---: | :--- |
| bx_isup | ✅ 70/70 | ✅ Yes | "isup" 216 hits |
| bx_gl_prim | ✅ 70/70 | ✅ Yes | "gleason" 149 hits |
| bx_gl_sec | ✅ 70/70 | ✅ Yes | "secondary" 0 hits |
| bx_gl_tert | ❌ n/a | n/a (absent) | absent |
| bx_isup_pred | ❌ n/a | n/a | "predicted isup" 68 hits |
| cores_positive / cores_total | ❌ n/a | n/a | "cores" 17 hits |
| max_core_pct | ❌ n/a | n/a | "core" 133 hits |
| growth_pattern | ❌ n/a | n/a | "growth pattern/cribriform/..." 271 hits |
| high_risk_patterns | ❌ n/a | n/a | "high-risk" 18 hits |
| pni | ❌ n/a | n/a | "perineural" 63 hits |
| lvi | ❌ n/a | n/a | "lvi" 10 hits |
| tumor_location | ❌ n/a | n/a | "peripheral zone" 194 hits |

**Numeric features**: 29 → **26** (3 biopsy fields removed). Categorical (`dre`, `bx`, `ct`) unchanged.

---

## 4. Experimental Setup (as run)

- **Dataset**: Task 2 trimodal cohort (N=70), `data/processed/task2/`.
- **Splits**: EXACT REUSE of `mccv_splits_100.csv` (100 stratified 4-class splits, seed=42).
- **In-run paired control** (design adaptation): exp_12 did not persist split metrics, so paired cross-experiment Wilcoxon was impossible. Instead the runner builds BOTH `C0_full` (29 features) and `C1_ablated` (26 features) caches on the SAME splits, using the SAME frozen exp_10-H2 fuzzy-KNN configs — isolating the pure feature effect.
- **Preprocessing**: MinMaxScaler tabular (26 cols ablated), MRI 1024D, TF-IDF text. Zero leakage: active_treatment_flag excluded; scalers fit train-only.
- **Runtime**: 2789.1s (~46.5 min).

---

## 5. Results

### 5.1 Subtask 2.1 (Treatment, frozen fuzzy-KNN) — MAIN TEST
| Condition | Features | Macro-F1 | Δ vs C0 |
| :--- | :--- | :---: | :---: |
| **C0_full** (control) | 29 cols (incl. bx_*) | **0.5971** | — |
| **C1_ablated** | 26 cols (bx_* removed) | **0.5731** | **−0.0240** |

**Paired Wilcoxon** (same 100 splits): W=89.0, **p = 4.90e-10** → significant degradation.

### 5.2 Subtask 2.2 (Confidence, 3-class ordinal)
| Best config | Macro-F1 | LOOCV acc |
| :--- | :---: | :---: |
| tabular fuzzy k=1, uniform | 0.3429 | **0.7286** |

### 5.3 Subtask 2.3 (Variable weights, Phase-A Macro-F1 + LOOCV acc)
| Variable | Best algo | RF | ET | LOOCV acc |
| :--- | :---: | :---: | :---: | :---: |
| bx_isup | ET | 0.4120 | **0.4191** | 0.514 |
| psa | ET | 0.2322 | **0.2878** | 0.586 |
| bx_gl_sec | ET | 0.3071 | **0.3316** | 0.500 |
| bx_gl_prim | RF | **0.2661** | 0.2656 | 0.386 |
| fh | ET | 0.2147 | **0.2150** | **0.671** |
| age | RF | **0.1915** | 0.1910 | 0.614 |
| cspca | ET | 0.2081 | **0.2176** | 0.500 |
| ct | RF | **0.2152** | 0.2070 | 0.429 |
| comorbidity | ET | 0.1901 | **0.2105** | 0.643 |
| psad | ET | 0.2113 | **0.2293** | 0.486 |
| pirads | ET | 0.1683 | **0.1837** | 0.429 |

**LOOCV averages**: 2.1 = **0.5143** | 2.2 = **0.7286** | 2.3 = **0.5234**

---

## 6. Key Findings

1. **The redundancy hypothesis is REJECTED**: removing the 3 explicit tabular biopsy fields degrades 2.1 treatment by Δ=−0.024 (p<1e-9) even though the text modality contains abundant pathology vocabulary. The fuzzy-KNN does not recover the structured Gleason/ISUP signal from free text.
2. **The ablation is real but bounded**: 2.2 and 2.3 remain within the exp_12 band, confirming those subtasks are structurally limited (uncertain n=0; raw-feature ceiling) rather than dependent on the removed fields.
3. **ET dominates 2.3 per-variable Macro-F1** in 8/11 variables even with the ablated space; bx_isup remains the best-learned variable from residual/non-explicit features (0.4191, only −0.004 vs exp_12 0.4229) — remarkable given it was removed as an input.
4. **LOOCV 2.3 highlights reassignment**: `fh` (0.671) and `age` (0.614) become leading variables, consistent with re-weighting after dropping the biopsy inputs.

---

## 7. Missing Data & Caveats

- Option A keeps the text pathology narrative (by PI decision), so the ablation is tabular-only; residual pathology signal flows through text.
- Subtask 2.2 LOOCV used a simplified tabular-only frozen config; full trimodal fusion LOOCV was not re-run after the ablation.
- Phase-A grid per modality for 2.2 was reduced to a tabular secondary grid; MRI/text grids from exp_12 were not re-swept (runtime containment).
- Per-class F1 for `uncertain` (n=0) undefined; `watchful_waiting` (n=2) unstable.

---

## 8. Conclusions & Next Steps

- **Conclusions**: (1) the three explicit biopsy tabular fields carry statistically significant non-redundant signal for treatment recommendation; (2) the text modality does not fully substitute them under this pipeline; (3) the ablation impact is concentrated in 2.1, not in 2.2/2.3.
- **Next Steps**: (1) test whether a richer text model (TF-IDF-1000 / SVM) recovers the lost signal post-ablation; (2) extend ablation to the pathology REPORT text redaction (Option B) to quantify total pathology dependence; (3) persist the ablated pickle bundles if inference without biopsy fields is desired.

---

## 9. Reproducibility Record

| Element | Status | Detail |
| :--- | :---: | :--- |
| Feature delta verified | ✅ | 29 → 26 (bx_isup, bx_gl_prim, bx_gl_sec removed) |
| In-run paired control | ✅ | C0_full vs C1_ablated, same splits, same frozen config |
| MCCV-100 splits reused | ✅ | `mccv_splits_100.csv` |
| Zero leakage | ✅ | active_treatment_flag excluded |
| LOOCV included | ✅ | 2.1=0.5143, 2.2=0.7286, 2.3=0.5234 |
| Storage minimal | ✅ | JSON only (no prediction CSVs) |

---