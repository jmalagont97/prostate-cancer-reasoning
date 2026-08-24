# Experiment Summary: Multidimensional Stratified Monte Carlo Protocol & Governance (`exp_3`)

**Experiment**: experiments/exp_3/  
**Project**: prostate-cancer-reasoning  
**Author**: Juan Sebastian Malagon Torres  
**Date**: 2026-08-22  
**Status**: Complete  
**Verdict**: ACCEPTED (✅)

---

## 1. Executive Summary
This experiment established the project's permanent, standardized cross-validation protocol and evaluation governance framework:
1. Generated a fixed, deterministic 30-fold multidimensional stratified Monte Carlo Cross-Validation matrix ($B=30$, 80/20 train/val ratio) saved at `data/processed/task1/mccv_splits_30.csv`.
2. Stratified strictly on the core supervised target dimensions (exact $11\text{ yes} / 7\text{ no}$ decision balance, Wasserstein-1 matching on 3-level ordinal confidence, and active critical factors discrepancy minimization).
3. Ratified the project's inviolable evaluation manifesto in `utils/pkg/chimera_metrics/PROTOCOL.md`, mandating centralized evaluation via `chimera_metrics` and prohibiting ad-hoc scoring scripts within `experiments/`.

---

## 2. Statistical Characterization of the 30 Partitions

Across all $B=30$ splits in `data/processed/task1/mccv_splits_30.csv`:

```
+-----------------------------------------------------------------------------------------------+
| Dimension / Metric                     | Validation Set (per Split) | Training Set (per Split)|
+----------------------------------------+----------------------------+-------------------------+
| Supervised Case Count                  | 18                         | 73                      |
| Unlabeled Case Count                   | 0 (Tagged 'unlabeled')     | 0 (Tagged 'unlabeled')  |
| Positive Decision (Biopsy 'yes')       | 11 (61.11% exact)          | 45 (61.64% exact)       |
| Negative Decision (Biopsy 'no')        | 7 (38.89% exact)           | 28 (38.36% exact)       |
| Mean Confidence Wasserstein Distance   | EMD_mean = 0.0412          | Reference               |
| Mean Critical Factors Discrepancy      | Delta_factors = 0.0385     | Reference               |
| Total Cohort Matrix Shape              | (195 patients, 31 columns: pid + split_0..split_29)   |
+-----------------------------------------------------------------------------------------------+
```

---

## 3. Evaluation Governance Directives (`PROTOCOL.md`)

The governance protocol formally ratified in `utils/pkg/chimera_metrics/PROTOCOL.md` establishes:
1. **Strict Prohibition of Local Scoring Code:** No script under `experiments/<exp_id>/` may define or execute ad-hoc metrics or scoring functions.
2. **Centralized Evaluation:** All future model evaluations must import and execute functions from `chimera_metrics` (`load_mccv_splits`, `evaluate_task1_case`, `BiopsyDecisionGateMetric`, etc.).
3. **Split Invariance:** Every future modeling experiment must report mean and standard deviation across all 30 splits, enabling paired Wilcoxon signed-rank significance testing ($p < 0.05$).

---

## 4. Deliverables & Artifact Index
- Splits Matrix: `data/processed/task1/mccv_splits_30.csv`
- Protocol Manifesto: `utils/pkg/chimera_metrics/PROTOCOL.md`
- Split Utilities: `chimera_metrics.splits` (`load_mccv_splits`, `get_split_indices`)
- Diagnostics: `experiments/exp_3/results/split_diagnostics.json`
