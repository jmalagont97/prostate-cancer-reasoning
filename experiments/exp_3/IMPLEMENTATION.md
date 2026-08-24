# Implementation Plan: Multidimensional Stratified Monte Carlo Protocol & Governance (`exp_3`)

**Experiment**: experiments/exp_3/  
**Project**: prostate-cancer-reasoning  
**Author**: Juan Sebastian Malagon Torres  
**Date**: 2026-08-22  
**Status**: In Progress

---

## 1. Implementation Overview
This plan implements:
1. A reproducible, multi-subtask stratified split generator (`experiments/exp_3/scripts/generate_mccv_splits.py`) that constructs 30 Monte Carlo Cross-Validation partitions ($B=30$, 80/20 train/val ratio) on `data/processed/task1/ground_truth_task1.csv`.
2. The persistent split artifact `data/processed/task1/mccv_splits_30.csv`.
3. The centralized evaluation protocol manifesto `utils/pkg/chimera_metrics/PROTOCOL.md` and a split loader utility in `chimera_metrics`.

---

## 2. Multi-Objective Stratification Algorithm
For each iteration $b \in \{0, \dots, 29\}$ with pseudo-random seed $\text{seed}_b = 42 + b$:
1. **Decision Gate Constraint:** Sample exactly $11$ cases from the positive class (`decision == 'yes'`, $N=56$) and $7$ cases from the negative class (`decision == 'no'`, $N=35$), yielding an exact validation size $n_{\text{val}} = 18$ ($n_{\text{train}} = 73$).
2. **Confidence Calibration Matching:** Evaluate Earth Mover's Distance (EMD / Wasserstein-1) between train and validation for ordinal confidence levels $\text{rank}(\text{uncertain})=0, \text{rank}(\text{borderline})=1, \text{rank}(\text{clear})=2$.
3. **Critical Factors Matching:** Evaluate mean absolute frequency discrepancy $\Delta_{\text{factors}}$ across the 10 binary indicators of decisive/important variables.
4. **Iterative Rejection Sampling:** Accept the candidate partition if $\mathcal{L}(I_{\text{val}}^{(b)}) = \text{EMD}_{\text{conf}} + 0.5 \cdot \Delta_{\text{factors}} < \tau$, otherwise resample with deterministic seed evolution.
5. **Cohort Invariance:** Assign all 104 unlabeled cases to role `'unlabeled'`.

---

## 3. Deliverables & Validation Protocol
- Run script: `python3 experiments/exp_3/scripts/generate_mccv_splits.py`
- Validate outputs:
  * Check shape: $(195, 31)$ where column 0 is `pid` and columns 1..30 are `split_0` to `split_29`.
  * Validate per-split counts: exactly 73 `train`, 18 `val`, 104 `unlabeled` per column.
  * Check decision balance: exactly 11 positive, 7 negative in `val` for every single split.
- Install `utils/pkg/chimera_metrics/PROTOCOL.md` forbidding ad-hoc evaluation in `experiments/`.
