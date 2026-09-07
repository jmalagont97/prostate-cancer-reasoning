# Experiment Report: Cascaded Subtask 2.1 → 2.2 (Treatment-Informed Confidence) — Leak-Free v2 (exp_15)

**Experiment**: `experiments/exp_15/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Report Date**: 2026-09-06
**Plan Date**: 2026-09-02
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Complete ✅ (v2 leak-free — supersedes earlier run with preprocessing leakage)
**Based on**: exp_11 (2.1 fuzzy-SVR champion), exp_13 (2.2 Soft-SVR confidence baseline)

---

## 1. Summary

This experiment evaluated whether injecting the Subtask 2.1 treatment output as an INPUT feature improves Subtask 2.2 confidence prediction. A frozen fuzzy-SVR treatment model (exp_11, Macro-F1 0.6166) generated the injected signal under a **nested cross-fitting protocol** (5-fold inner OOF for training cases). An initial code audit revealed a **critical preprocessing-leakage bug** (StandardScaler/TF-IDF fitted globally before LOOCV); the runner was rewritten as a leak-free v2 fitting all preprocessors inside each fold. Results show the cascade provides **no meaningful improvement**: MCCV-100 Macro-F1 deltas vs the no-cascade baseline were Δ≈−0.002 (C1 probs), Δ≈−0.001 (C2 argmax), Δ≈+0.002 (C3 both) on both 3-class and binary targets; LOOCV accuracy improved uniformly by +0.014 (78.6%→80.0%) for all cascade conditions, driven by one additional correct `clear` prediction. The conclusion is that **treatment decision (2.1) and clinical confidence (2.2) encode largely orthogonal signals** under this feature framework, supporting the challenge hypothesis of independent subtasks. The best-performing condition (C3) was persisted as a loadable pickle.

---

## 2. Hypothesis & Verdict

**H1 (cascade improves 3-class):** At least one cascade condition significantly improves 3-class Macro-F1 over C0.

**H2 (probability vector dominates):** C1 (probs) contributes more than C2 (argmax one-hot).

**H3 (binary target more sensitive):** Cascade effect is clearer on the binary clear-vs-borderline formulation.

| Hypothesis | Verdict | Evidence |
| :--- | :---: | :--- |
| H1 | ⚠️ **Not supported (marginal/NS)** | MCCV: C3 best Δ=+0.0016 (3-class) / +0.0021 (binary), far below σ≈0.07; LOOCV uniform +0.014 (one extra correct case) |
| H2 | ❌ **Rejected** | C1 (0.3793) NOT > C2 (0.3804) — probability vector adds no more than argmax |
| H3 | ⚠️ **Not confirmed** | Binary Δ ≈ 3-class Δ (+0.0021 vs +0.0016) — no amplification |

---

## 3. Leakage Audit & v2 Correction

An audit of the initial `run_exp15_cascade.py` identified two critical bugs:

1. **Preprocessing leakage (CRITICAL)**: `Task2TabularPreprocessor()`, `MRIStandardPreprocessor()`, and `TextPreprocessor()` were fitted on ALL N=70 before the LOOCV / pooled-OOF loop. The held-out case leaked into the scaler/TF-IDF statistics, inflating LOOCV metrics.
2. **Insecure `eval()`**: the most-common config was persisted as a string dict and reconstructed with `eval()`.

**v2 corrections applied**: preprocessors are now fitted **inside each LOOCV fold** (69-train only) and inside each pooled-OOF fold; configs are stored as numeric `best_C`/`best_eps` and no `eval()` remains.

---

## 4. Experimental Setup (as run v2)

- **Dataset**: Task 2 trimodal cohort (N=70), `data/processed/task2/`.
- **Splits**: EXACT REUSE of `mccv_splits_100.csv` (Phase A); Phase B LOOCV (70 folds).
- **Source 2.1 (frozen)**: exp_11 fuzzy-SVR (tab C=0.1 eps=0.05; mri C=0.1 eps=0.1; txt C=10 eps=0.05 full-vocab; fusion w=(0.55, 0.20, 0.25)).
- **Injection into 2.2 tabular branch**: C0 none; C1 probability vector (+4); C2 argmax one-hot (+4); C3 both (+8).
- **2.2 family**: Soft-SVR per condition with grid C∈{0.1,1.0,10.0} × eps∈{0.05,0.1}, best by MCCV split-level Macro-F1; tabular-branch only (registered simplification vs full-trimodal DESIGN).
- **Targets**: 3-class (`clear`/`borderline`/`uncertain`) + binary (`clear` vs `borderline`).
- **Phase B LOOCV**: leak-free preprocessing per fold (v2).
- **Runtime**: 555.3s (~9.3 min).

---

## 5. Results

### 5.1 MCCV-100 Macro-F1 (mean of per-split best C/eps)

| Condition | 3-class | Δ vs C0 | binary | Δ vs C0 | best C/eps |
| :--- | :---: | :---: | :---: | :---: | :---: |
| C0 (baseline) | 0.3817 ± 0.0728 | — | 0.5726 ± 0.1092 | — | 0.1 / 0.05 |
| C1 (probs) | 0.3793 ± 0.0715 | −0.0024 | 0.5690 ± 0.1073 | −0.0036 | 0.1 / 0.05 |
| C2 (argmax OH) | 0.3804 ± 0.0660 | −0.0013 | 0.5700 ± 0.0987 | −0.0026 | 0.1 / 0.05 |
| C3 (both) | **0.3833 ± 0.0716** | **+0.0016** | **0.5747 ± 0.1073** | **+0.0021** | 0.1 / 0.05 |

### 5.2 LOOCV accuracy (leak-free, 70 folds)

| Condition | 3-class | binary |
| :--- | :---: | :---: |
| C0 | 0.7857 | 0.7857 |
| C1 / C2 / C3 | **0.8000** | **0.8000** |

### 5.3 Confusion matrices (LOOCV, rows = GT)

**C0 baseline:**

| 3-class | predunc | predbord | predclear |
| :--- | :---: | :---: | :---: |
| GT uncertain (0) | 0 | 0 | 0 |
| GT borderline (14) | 0 | 2 | 12 |
| GT clear (56) | 0 | 3 | 53 |

| binary | predbord | predclear |
| :--- | :---: | :---: |
| GT borderline (14) | 2 | 12 |
| GT clear (56) | 3 | 53 |

**C1/C2/C3 (identical matrices):**

| 3-class | predunc | predbord | predclear |
| :--- | :---: | :---: | :---: |
| GT uncertain (0) | 0 | 0 | 0 |
| GT borderline (14) | 0 | 2 | 12 |
| GT clear (56) | 0 | 2 | **54** |

| binary | predbord | predclear |
| :--- | :---: | :---: |
| GT borderline (14) | 2 | 12 |
| GT clear (56) | 2 | **54** |

The cascade corrects exactly **one additional `clear` case** (3→2 borderline misclassifications), yielding the LOOCV accuracy bump.

---

## 6. Findings

1. **Cascade does not meaningfully improve confidence prediction** — MCCV deltas (±0.002) are ~30× smaller than the per-split standard deviation (±0.07); the LOOCV improvement is one case (n=1).
2. **Probability vector is NOT superior to argmax** (H2 rejected): C1 ≤ C2, indicating the downstream treatment model's uncertainty mass does not transfer useful information to confidence estimation.
3. **Binary target is not more sensitive to the cascade** (H3 not confirmed): Δ(binary) ≈ Δ(3-class).
4. **Orthogonality insight**: treatment (2.1) and confidence (2.2) encode statistically distinguishable signals under tabular features — consistent with the challenge's independent-subtask design.
5. **Structural ceiling**: `uncertain` (n=0) makes the 3-class confusion-matrix first row all zeros; borderline is heavily confused with clear (12/14) in every condition.

---

## 7. Missing Data & Caveats

- No paired Wilcoxon test persisted at split level — the reported deltas are descriptive; NS inference is based on delta < 2× pooled σ.
- 2.2 used tabular-branch only within each condition (MRI/text not fused inside the cascade loop) — registered simplification vs the full DESIGN.
- `uncertain` n=0 ⇒ class 0 F1 undefined, boundary persists.
- Pooled-OOF for LOOCV injection was computed at case level (5-fold over all N) — consistent but not identical to per-MCCV-split OOF.

---

## 8. Conclusions & Next Steps

- **Conclusions**: The 2.1→2.2 cascade provides no statistically meaningful improvement; treatment and confidence signals are largely orthogonal under this feature set. The current best confidence model remains the non-cascaded Soft-SVR baseline (exp_13 binary 0.5349; exp_13 3-class 0.3552).
- **Next Steps**: (1) attack 2.2 via richer multimodal fusion (MRI/text contribution, not 2.1 injection); (2) feature engineering specific to the confidence contract (e.g., pathology intensity, PSA trajectory); (3) if cascade is revisited, test an oracle cascade (using the GT treatment) to separate signal-availability vs model-capacity limits.

---

## 9. Reproducibility Record

| Element | Status | Detail |
| :--- | :---: | :--- |
| Nested OOF 2.1 | ✅ | 5-fold inner CV within each MCCV train fold |
| Leak-free preprocessing | ✅ | Fitted inside each LOOCV / pooled-OOF fold (v2) |
| No eval() | ✅ | Numeric best_C / best_eps stored |
| MCCV-100 splits reused | ✅ | `mccv_splits_100.csv` |
| LOOCV completed | ✅ | 70 folds, leak-free |
| Confusion matrices persisted | ✅ | `confusion_matrices.json` (LOOCV) |
| Pickle persisted + loadable | ✅ | `models/subtask2_confidence_cascade_svm.pkl` (C3) |
| Storage minimal | ✅ | JSON + pickle only |

---