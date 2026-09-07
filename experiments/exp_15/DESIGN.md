# Experiment Design: Cascaded Subtask 2.1 → 2.2 (Treatment-Informed Confidence) under MCCV-100 + LOOCV (exp_15)

**Experiment**: `experiments/exp_15/`
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)
**Date**: 2026-09-02
**Author**: Jonathan Tarquino (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)
**Status**: Draft — pending PI approval
**Based on**: exp_13 (2.2 confidence SVM baseline), exp_11 (2.1 fuzzy-SVR champion)

---

## 1. Context & Hypotheses

Subtask 2.1 (treatment recommendation) and Subtask 2.2 (clinical confidence) are independently modeled in the current pipeline. This experiment tests a **cascade**: inject the downstream treatment prediction produced by the 2.1 model as an INPUT feature for the 2.2 confidence model. The clinical rationale is that the treatment probability mass distribution carries confidence-relevant information: a crisp active-treatment decision vs. a near-uniform probability simplex should induce different confidence levels.

**Source of 2.1 signal (Choice 2 = B)**: the fuzzy-SVR treatment model from exp_11/exp_13 (frozen, Macro-F1 0.6166).

**Injected features from 2.1 (Choice 1 = A, B, C — tested as separate conditions)**:

| Condition | Injected representation | Extra dimensionality |
| :--- | :--- | :---: |
| C0 (baseline) | none — pure 2.2 | 0 |
| C1 (Option A) | 2.1 probability vector (4-dim simplex) | +4 |
| C2 (Option B) | 2.1 argmax class → one-hot (4-class) | +4 |
| C3 (Option C) | both: 4 probs + 4 one-hot | +8 |

**H1 (cascade improves 3-class)**: At least one cascade condition (C1/C2/C3) significantly improves Subtask 2.2 3-class Macro-F1 over the no-cascade baseline C0 (Wilcoxon paired, p < 0.05).

**H2 (probability vector dominates)**: C1 (probability simplex) contributes more than C2 (argmax one-hot) because the probability mass distribution encodes uncertainty — whereas the argmax discards it.

**H3 (binary target more sensitive)**: The cascade effect is clearer on the binary (clear-vs-borderline) formulation than on the 3-class (uncertain n=0 dilution), following the exp_13 finding.

---

## 2. Leakage Protocol (NESTED CROSS-FITTING — mandatory)

Naively injecting 2.1's prediction would leak, because the 2.1 model has seen the training cases it predicts. The cascade therefore uses **nested cross-fitting inside every MCCV split**:

| Step | Train fold | Val fold |
| :--- | :--- | :--- |
| 1. Train 2.1 fuzzy-SVR | on train fold | — |
| 2. 2.1 output for TRAIN cases | **5-fold inner-CV OOF predictions** within the train fold (no self-prediction leakage) | — |
| 3. 2.1 output for VAL cases | — | predict with the 2.1 model from step 1 |
| 4. Train 2.2 with [original features + 2.1-injected] | on train fold (using OOF 2.1 outputs) | — |
| 5. Evaluate 2.2 | — | on val fold (using val 2.1 outputs from step 3) |

This guarantees the injected 2.1 features are always out-of-fold with respect to the 2.2 training.

---

## 3. Experimental Setup

- **Dataset**: Task 2 trimodal cohort (N=70), `data/processed/task2/`.
- **Splits**: EXACT REUSE of `mccv_splits_100.csv` (100 stratified 4-class splits).
- **Phase B LOOCV**: included (70 folds), following the exp_13 standard.
- **2.1 source model (frozen)**: fuzzy-SVR from exp_11 `best_hyperparams.json` (tab C=0.1 eps=0.05; mri C=0.1 eps=0.1; txt C=10 eps=0.05 full-vocab; fusion w=(0.55, 0.20, 0.25)).
- **2.2 model family**: Soft-SVR (the best 2.2 family in exp_13) per modality + late fusion. Preprocessing: StandardScaler tabular/MRI (SVM-required), TF-IDF text {500, None}.
- **Target 2.2 (Choice 3 = C)**: 3-class primary (`clear`/`borderline`/`uncertain`) + binary secondary (`clear` vs `borderline`).

---

## 4. Conditions

| Cond | 2.1 injected | 2.2 model | Target |
| :--- | :--- | :--- | :--- |
| C0 | none | Soft-SVR fusion (re-grid) | 3-class + binary |
| C1 | probability vector (4) | Soft-SVR fusion on [tab + 4] | 3-class + binary |
| C2 | argmax one-hot (4) | Soft-SVR fusion on [tab + 4] | 3-class + binary |
| C3 | both (8) | Soft-SVR fusion on [tab + 8] | 3-class + binary |

For each condition, per-modality feature spaces are:
- Tabular: 41 original + injected (0/4/4/8).
- MRI: 1024D (unchanged).
- Text: TF-IDF {500, None} (unchanged).
- Injection is applied ONLY to the tabular branch (the injection is tabular-shaped and naturally joins the structured features).

---

## 5. Evaluation Protocol

| Target | Primary metrics | Secondary | Official gate |
| :--- | :--- | :--- | :--- |
| 3-class | Macro-F1, Brier | S_conf (ordinal distance), per-class F1 | compute_confidence_score |
| binary | Macro-F1, Brier | precision/recall per class | (binary confidence score) |

### 5.1 Confusion Matrices (COMPLETE, per model — REQUIRED)

A full confusion matrix is computed and persisted for EACH condition (C0, C1, C2, C3) and for EACH target formulation:

| Target | Matrix shape | Labels |
| :--- | :--- | :--- |
| 3-class | 3x3 | uncertain (idx0), borderline (idx1), clear (idx2) |
| binary | 2x2 | borderline (idx0), clear (idx1) |

Matrix convention: rows = GROUND TRUTH, columns = PREDICTED. Each cell stores integer counts.

Two aggregation levels are persisted for every condition/target:
1. **MCCV-100 aggregated matrix**: sum of all 100 val-fold confusion matrices (total N ≈ 2100 for 3-class; N ≈ 2100 restricted-to-clear/borderline for binary).
2. **LOOCV matrix**: sum over the 70 leave-one-out predictions.

**Artifact** `confusion_matrices.json` structure:

```
{
  "C0": {
    "3class": {"mccv_aggregated": [[..3x..]], "loocv": [[..3x..]]},
    "binary": {"mccv_aggregated": [[..2x..]], "loocv": [[..2x..]]}
  },
  "C1": { ... },
  "C2": { ... },
  "C3": { ... }
}
```

---

## 6. Statistical Tests

1. Wilcoxon paired (100 shared splits): each of C1, C2, C3 vs C0, on both 3-class Macro-F1 and binary Macro-F1.
2. Wilcoxon paired C1 vs C2 (isolates probability-vector vs argmax contribution) — tests H2.
3. McNemar exact on the aggregated binary 2x2 matrix for C_i vs C0 (per condition) if any Wilcoxon marginal (p in [0.05, 0.10]).
4. LOOCV accuracy reported per condition.

---

## 7. Storage Footprint & Model Persistence

### 7.1 JSON artifacts (minimal)
`aggregate_metrics.json`, `split_metrics.json`, `wilcoxon_tests.json`, `best_hyperparams.json`, **`confusion_matrices.json`**, `version.txt`. NO prediction CSVs (the confusion matrices carry the necessary count-level detail compactly).

### 7.2 Persisted best models (REQUIRED)
| Subtask | Pickle artifact | Content |
| :--- | :--- | :--- |
| 2.2 cascade | `subtask2_confidence_cascade_svm.pkl` | Best 2.2 (C1/C2/C3) + preprocessors + fusion weights + the frozen 2.1 source model |

---

## 8. Risks & Mitigations

| Risk | Prob./Sev. | Mitigation |
| :--- | :---: | :--- |
| Leakage via naive 2.1 injection | Alta / Critical | Nested 5-fold inner-CV OOF for all train 2.1 outputs (Section 2). |
| 2.1 OOF cost | Media / Media | 2.1 fuzzy-SVR is fast; 5-fold inner CV x 100 splits = 500 extra fits (cheap). |
| Injection only on tabular suppresses multimodal complementarity | Media / Media | Pre-registered: future work adds text-branch injection ablation. |
| uncertain n=0 dilutes 3-class | Alta / Alta | Binary target reported in parallel (H3); confusion matrices expose the class-level degradation. |

---

## 9. Expected Results & Decision Rules

- **H1 supported**: if any cascade condition beats C0 in 3-class Macro-F1 with p<0.05.
- **H2 supported**: if C1 (probs) > C2 (argmax) in paired Wilcoxon.
- **H3 supported**: if cascade Δ(binary) > Δ(3-class).
- **Failure/adjustment**: if no cascade improvement, the 2.1 output is orthogonal to confidence — document and examine confusion matrices to identify which class transitions are responsible.

---

## 10. Reproducibility Checklist

- [ ] REUSE mccv_splits_100.csv
- [ ] Nested cross-fitting implemented (no 2.1 self-prediction)
- [ ] 2.1 frozen fuzzy-SVR configs from exp_11
- [ ] StandardScaler tabular/MRI; TF-IDF text
- [ ] Phase B LOOCV included
- [ ] 3-class + binary targets reported
- [ ] **Complete confusion matrices persisted (MCCV-aggregated + LOOCV) per condition/target**
- [ ] Best cascade pickle persisted + loadable
- [ ] Storage minimal

---

## 11. Next Steps

1. PI approval of this DESIGN.md.
2. Generate IMPLEMENTATION.md (run_exp15_cascade.py with nested OOF + 4 conditions + LOOCV + confusion matrices).
3. Build save_best_models.py (cascade pickle).
4. Execute in background; verify; commit to jtarquino_task2.
5. Update .logbook.md and .discussion.md after consensus.

---