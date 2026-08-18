# Consolidated Results Summary — All Experiments

**Generated**: 2026-08-18 · Source: `experiments/exp_*/results/*.json` (read directly, not re-derived from `.discussion.md` prose).

This file consolidates the quantitative results of every modeling experiment (`exp_5`–`exp_22`). `exp_1`–`exp_4` are infrastructure (EDA, preprocessing, MCCV/LOOCV split design) and are excluded — they produce no model metrics.

## Comparability check (read before comparing numbers across rows)

The 18 modeling experiments split into **three tasks that are not mutually comparable** — a Macro-F1 from Table 1 cannot be compared to one in Table 2 or 3, they're different label spaces:

| Task | Experiments | Label space |
|---|---|---|
| Biopsy decision | `exp_5,6,7,8,13,14,15,16,18` | binary (biopsy yes/no) |
| Diagnostic confidence | `exp_9,10,11,12,17,19` | 3-class ordinal (uncertain/borderline/clear) |
| Clinical feature relevance | `exp_20,21,22` | 4-class ordinal per tabular feature (not_used/noted/important/decisive) |

**Within each table, the setups check out as comparable:**
- All 18 experiments evaluate on the **same N=88 complete-case labeled cohort** under **LOOCV** (verified: every confusion matrix / TP+TN+FP+FN sums to exactly 88).
- Table 1's binary target is identical across all 9 rows — verified via `TP+FN` (actual positives): every row recovers **54 positive / 34 negative**, confirming the same patients and the same label alignment are used throughout, not just the same count.
- All experiments source their Phase A hyperparameter/threshold search from the same `experiments/exp_4` 100-split MCCV design, with scalers/encoders refit inside each split/fold (no global fit).

**Caveats that *do* break naive comparison — read these before drawing conclusions:**

1. **Table 2, `exp_9` vs `exp_10` are numerically identical**, not just similar: same confusion matrix `[[6,3,5],[7,8,3],[17,18,21]]`, same Macro-F1 (0.3691), same Spearman ρ. `exp_10`'s `class_weight='balanced'` shifted the Phase A meta-thresholds (τ₁: 0.0743→0.0669, τ₂: 0.3321→0.2960) but not enough to flip any of the 88 LOOCV classifications. `.discussion.md`'s `exp_10` entry frames this as a successful threshold adjustment — that's true of the thresholds, but the claim should not be read as an LOOCV performance improvement over `exp_9`, since none occurred.
2. **Table 2, `exp_9`/`exp_10`/`exp_11`/`exp_12` share one Spearman ρ (0.1228) by construction**, not coincidence — verified in each `train.py`: all four call `spearmanr(oof_ici, y_confidence)` on the *same* `oof_ici`, computed from the same three hard-KNN unimodal models (`exp_5`/`exp_6`/`exp_7`). Only the discretization method differs (`exp_9` unweighted tree, `exp_10` balanced tree, `exp_11` dynamic per-fold tree, `exp_12` a 3D-vector tree that still happens to reduce to the same rank order here). Ranking/correlation is not a useful axis to compare these four on — only Macro-F1/Accuracy differ meaningfully. `exp_17`/`exp_19` are the only rows in Table 2 with a genuinely different underlying continuous score (Fuzzy-KNN-based ICI), and are the only ones where ρ moves.
3. **Table 3, `exp_20` covers 8 tabular features; `exp_21`/`exp_22` cover only 6`** (`psav`, `psap` dropped). Compare only the 6 common features (`age`, `psa`, `vol`, `pirads`, `psad`, `dre`) across all three rows; `psav`/`psap` numbers exist only for `exp_20`.
4. Table 1's `exp_8`/`exp_16`/`exp_18` are late-fusion *sweeps* (one row per weight combination); only the headline `Optimal-Weighted-*` and unimodal rows are reproduced below — see each experiment's own `results/*.json` for the full weight grid (bimodal combinations, equal-weight fusion, etc.).

---

## Table 1 — Biopsy Decision (binary, LOOCV N=88, 54 positive / 34 negative)

| Exp | Model | Setup (frozen from Phase A) | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Brier |
|---|---|---|---|---|---|---|---|---|
| exp_5 | Tabular — Hard KNN | k=3, uniform, euclidean | 0.6333 | 68.18% | 85.19% | 41.18% | — | — |
| exp_6 | MRI — Hard KNN | EmbedKit-supervised 384D, k=3, uniform, euclidean | 0.5335 | 56.82% | 68.52% | 38.24% | — | — |
| exp_7 | Text — Hard KNN | TF-IDF (max_features=500) + PCA, k=1, uniform, cosine | 0.6988 | 71.59% | 77.78% | 61.76% | — | — |
| exp_8 | Trimodal Late Fusion — Hard (optimal weights: 0.25 tab / 0.41 mri / 0.34 text) | soft-voting over exp_5/6/7 | **0.7171** | 75.00% | 88.89% | 52.94% | 0.7715 | — |
| exp_13 | Tabular — Fuzzy KNN (soft targets) | k=1, uniform, euclidean | 0.6364 | 65.91% | 74.07% | 52.94% | 0.6304 | 0.2908 |
| exp_14 | MRI — Fuzzy KNN (soft targets) | EmbedKit-unsupervised 384D, k=3, uniform, euclidean | 0.5335 | 56.82% | 68.52% | 38.24% | 0.5387 | 0.2623 |
| exp_15 | Text — Fuzzy KNN (soft targets) | TF-IDF (full vocab) + PCA, k=3, uniform, cosine | 0.6558 | 69.32% | 83.33% | 47.06% | 0.6868 | 0.2195 |
| exp_16 | Trimodal Late Fusion — Fuzzy (optimal weights: 0.15 tab / 0.55 mri / 0.30 text) | soft-voting over exp_13/14/15 | 0.6813 | 71.59% | 85.19% | 50.00% | 0.7053 | **0.2093** |
| exp_18 | Hybrid Late Fusion (optimal weights: 0.05 tab-fuzzy / 0.50 mri-hard / 0.45 text-hard) | soft-voting over exp_13 + exp_6 + exp_7 | 0.6713 | 70.45% | 83.33% | 50.00% | **0.7334** | 0.2151 |

**Best Macro-F1**: `exp_8` (all-hard trimodal fusion, 0.7171). **Best calibration (Brier)**: `exp_16` (all-fuzzy trimodal fusion, 0.2093). Hard-vote fusion wins on discrete F1 because hard 0/⅓/⅔/1 probability steps produce sharper decision boundaries; fuzzy fusion wins on calibration because soft targets avoid those steps.

## Table 2 — Diagnostic Confidence (3-class, LOOCV N=88)

| Exp | Underlying continuous score | Threshold method | Macro-F1 | Accuracy | Spearman ρ | p-value |
|---|---|---|---|---|---|---|
| exp_9 | ICI on hard-KNN probs (exp_5/6/7) | MCCV-mean 1D tree thresholds | 0.3691 | 39.77% | 0.1228 | 0.254 |
| exp_10 | same as exp_9 | MCCV-mean 1D tree thresholds, `class_weight='balanced'` | 0.3691 | 39.77% | 0.1228 | 0.254 |
| exp_11 | same as exp_9 | **Dynamic** per-LOOCV-fold tree thresholds (balanced) | 0.3388 | 36.36% | 0.1228 | 0.254 |
| exp_12 | 3D probability vector `[p_tab,p_mri,p_text]` (hard-KNN) | 3D decision tree (balanced) | 0.3331 | 37.50% | 0.1228 | 0.254 |
| exp_17 | Composite ICI on **Fuzzy**-KNN probs (exp_13/14/15) | MCCV-mean 1D tree thresholds (balanced) | **0.4470** | **57.95%** | **0.2790** | **0.0085** ✓ significant |
| exp_19 | Hybrid Composite ICI (exp_13 fuzzy + exp_6/7 hard) | MCCV-mean 1D tree thresholds (balanced) | 0.3885 | 42.05% | 0.1681 | 0.117 |

**Best**: `exp_17` — the only row with a statistically significant rank correlation to urologist-annotated confidence (p<0.01), and the best Macro-F1/Accuracy by a wide margin. See caveats 1–2 above before reading anything into the exp_9/10/11/12 spread.

## Table 3 — Clinical Feature Relevance Attribution (4-class per feature, LOOCV N=88)

Spearman ρ (attribution vs. urologist reasoning weight) per tabular feature; only the 6 features common to all three experiments are shown (see caveat 3):

| Feature | exp_20 (perturbation) ρ (p) | exp_21 (1D SHAP) ρ (p) | exp_22 (6D multivariate SHAP) ρ (p) |
|---|---|---|---|
| age | 0.196 (0.067) | −0.025 (0.816) | 0.074 (0.493) |
| psa | 0.184 (0.086) | **0.359 (0.00059)** ✓ | **0.374 (0.00033)** ✓✓ |
| vol | 0.169 (0.115) | 0.080 (0.459) | −0.014 (0.896) |
| pirads | −0.166 (0.122) | 0.073 (0.498) | **0.302 (0.00425)** ✓ |
| psad | −0.003 (0.976) | −0.100 (0.356) | **−0.292 (0.0058)** ✓ |
| dre | **0.429 (0.0000304)** ✓✓ | 0.205 (0.056) | 0.002 (0.983) |

4-class Macro-F1 (classification view of the same task) for the same 6 features:

| Feature | exp_20 | exp_21 | exp_22 |
|---|---|---|---|
| age | 0.126 | 0.141 | **0.261** |
| psa | 0.051 | 0.173 | **0.345** |
| vol | 0.102 | 0.191 | **0.221** |
| pirads | 0.173 | 0.199 | **0.399** |
| psad | 0.058 | **0.218** | 0.023 |
| dre | 0.127 | 0.203 | **0.246** |

**Pattern**: `exp_22` (multivariate 6D SHAP-vector decision trees) improves Macro-F1 over `exp_21` (1D SHAP) on 5 of 6 features and is the only method giving `pirads` and `psa` statistically significant rank correlations simultaneously — but it *loses* rank correlation entirely on `dre` and `vol`, and its `psad` correlation flips sign and becomes significant in the opposite direction from what `dre`'s perturbation-based signal (`exp_20`) suggested was the strongest driver. No single method dominates on both metrics across all 6 features; `dre` (exp_20) and `psa`/`pirads` (exp_22) are the standout significant findings, not one model beating the others everywhere.

---

*Regenerate this file by re-reading `experiments/exp_{5..22}/results/*.json` — it is not itself produced by a script.*
