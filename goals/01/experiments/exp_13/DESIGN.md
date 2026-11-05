# Experiment Design: Late Multimodal Fusion with Learnable Modality Weights (exp_13)
**Experiment**: experiments/exp_13/ · **Project**: pathology-reasoning · **Date**: 2026-08-17 · **Status**: Draft

---

## 1. Hypothesis

A vector of fixed, nonnegative, sum-to-one modality weights outperforms or matches the equal-weight fusion baseline while allowing zero-weight modality deactivation, given the fixed 21-variable tabular feature set inherited from `exp_5`.

## 2. Experimental Setup

- **Cohort**: 88 `usable_labeled` cases (54 yes / 34 no) with frozen splits from `data/chimera26/preprocessed/task1/mccv_loocv_splits.csv`.
- **Tabular feature set**: the fixed intersection of 21 variables selected in `exp_5`, without rerunning Spearman correlation pruning, association-matrix computation, or `tau` search:

```text
cli_age
cli_allergies_count
cli_bx
cli_comorbidity_count
cli_cspca
cli_dre
cli_fh_binary
cli_ipss_score
cli_months
cli_pirads
cli_psa
cli_psad
cli_psav
cli_vol
vit_bp_diastolic
vit_bp_systolic
vit_heart_rate_bpm
vit_height_cm
vit_smoking_pack_years
vit_smoking_status
vit_weight_kg
```

- **Tabular model**: KNN fuzzy v2 with `k=1`, cosine, uniform, confidence-weighted. Fold-local preprocessing only: missing indicators, zero-fill, one-hot encoding, and MinMax scaling on training data only.
- **MRI model**: PCA `n_components=1` plus KNN fuzzy v2 with `k=1`, euclidean, distance, confidence-weighted.
- **Text model**: TF-IDF `max_features=2000` plus KNN fuzzy v2 with `k=3`, cosine, distance, confidence-weighted.
- **Fusion rule**: `p_fusion = w_T p_T + w_M p_M + w_X p_X`, with threshold 0.5.
- **Hyperparameter search space**: simplex grid of nonnegative weights summing to 1, step 0.05, yielding 231 configurations.
- **Automatic modality deactivation**: any weight equal to 0 disables that modality, so the 231 candidates include all meaningful modality subsets and recover the equal-weight fusion baseline.

## 3. Evaluation Protocol

- **MCCV**: 50 stratified splits, 70/18 train/validation. The three component models are trained per split; fusion probabilities are computed for all 231 candidate weight vectors.
- **LOO**: 88 folds executed only for the single best configuration selected by MCCV.
- **Primary selection metric**: F1 macro.
- **Tie-break / guardrails**: `brier_score` (lower is better) → F1 yes → balanced accuracy → MCC.
- **Threshold**: fixed at 0.5. No threshold search is allowed.

## 4. Why not rerun correlation pruning

`exp_5` already fixed the tabular feature set used for later stages. The objective of `exp_13` is to isolate the effect of modality weighting and modality deactivation, not to revisit the upstream variable-selection procedure. Consequently:

- no Spearman association matrix is recomputed;
- no clustering is performed;
- no `tau` value is optimized;
- `apply_pruning` is not executed.

The tabular MCCV predictions are regenerated from scratch with the fixed 21-variable contract to preserve the no-reuse policy of this pipeline stage.

## 5. Baselines to report

The report must include at least the following rows:

1. Best weighted fusion from the 231 candidates.
2. Equal-weight fusion of T, M, and X as the reference to beat.
3. The best combination per active modality subset (best T-only, best M-only, best X-only, best T+M, best T+X, best M+X, best T+M+X).
4. All seven fusion conditions from `exp_12`.

Because the fixed-feature tabular contract in `exp_13` differs from the fold-local pruning in `exp_12`, the reported uniform fusion baseline in `exp_13` should be treated as the direct experimental comparator, not `exp_12` itself.

## 6. Next Step

Accept this design, then produce an implementation plan for `experiments/exp_13/scripts/run_weighted_fusion_experiment.py` and for saving `experiments/exp_13/IMPLEMENTATION.md` before execution.
