# Experiment Design: TF-IDF Text Modality for Biopsy Decision (Task 1.1)
**Experiment**: experiments/exp_10/  
**Project**: pathology-reasoning  
**Date**: 2026-08-17  
**Status**: Draft

---

## 1. Hypothesis
TF-IDF representation of clinical narrative text provides predictive signal for biopsy decision beyond the majority baseline (macro-F1 > 0.430, balanced accuracy > 0.550, MCC > 0).

## 2. Experimental Setup
- **Dataset**: `full_prompt_narrative.csv` (195 × 2, case_id + txt_full_prompt_narrative), filtered to 88 `usable_labeled` cases from `mccv_loocv_splits.csv`
- **Cohort**: same 88 cases as exp_4–exp_9 (54 yes / 34 no)
- **Text preprocessing**: lowercase, remove special characters (keep hyphens), remove numeric tokens (digits + written numbers), remove stopwords (protect negation words), lemmatize with spaCy `en_core_web_sm` v3.8.0
- **Vectorization**: TF-IDF with `max_features` ∈ {500, 1000, 2000, None}
- **Classifier**: KNN (72 configurations × 4 max_features = 288 total)
- **Hardware**: CPU only

## 3. Baselines
| Baseline | Description | Expected |
|----------|------------|----------|
| Majority (no text) | Always predict yes | macro-F1 = 0.380, balanced_acc = 0.500, MCC = 0 |
| exp_5 best (tabular) | tau_0.60 cosine uniform fuzzy | macro-F1 = 0.689 (LOO, external reference) |

## 4. Proposed Conditions
- **max_features**: 500, 1000, 2000, None (vocabulary size)
- **KNN grid**: k ∈ {1,3,5,7,9,11,15,21,31}, metric ∈ {euclidean, cosine}, weights ∈ {uniform, distance}, variant ∈ {standard, confidence_weighted(fuzzy v2)}
- **Total**: 288 configurations, 14,400 MCCV evaluations

## 5. Evaluation Protocol
- **MCCV**: 50 stratified splits (70/18), search over 288 configs
- **LOO**: 88 folds, single selected config only
- **Primary metric**: F1_macro (MCCV selection)
- **Secondary**: brier_score (tie-break), F1_yes, balanced_accuracy, MCC
- **Selection**: F1_macro → brier_score ↓ → F1_yes → balanced_accuracy → MCC → deterministic by name

## 6. Expected Results & Decision Rules
- **Success**: macro-F1 > 0.430, balanced accuracy > 0.550, MCC > 0 → text is informative
- **Partial**: macro-F1 > 0.380 but < 0.430 → weak signal, needs investigation
- **Failure**: macro-F1 ≤ 0.380 → text not useful without preprocessing revision
- **Competitive**: macro-F1 approaching 0.689 (exp_5) → candidate for fusion

## 7. Risks & Mitigations
- **Information loss from tokenization**: preserve numeric tokens (PSA, IPSS, age, BMI, BP values)
- **Vocabulary explosion with small N**: controlled via max_features grid
- **spaCy version drift**: log exact model version (en_core_web_sm 3.8.0, spaCy 3.8.14)
- **Leakage from text containing targets**: validated by exp_3 (no ground truth columns in narrative CSV)

## 8. Reproducibility Checklist
- [ ] spaCy model + version recorded
- [ ] Frozen splits (mccv_loocv_splits.csv)
- [ ] Git commit hash saved per run
- [ ] Config YAML/log saved
- [ ] All 288 configs logged with per-split metrics
