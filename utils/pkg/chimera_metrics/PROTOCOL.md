# PROTOCOL.md — Official Evaluation Protocol & Governance Directives

**Package**: `chimera_metrics`  
**Project**: prostate-cancer-reasoning (CHIMERA-Agent Challenge / MICCAI 2026)  
**Author**: Juan Sebastian Malagon Torres  
**Ratification Date**: 2026-08-22  
**Status**: NORMATIVE & INVIOLABLE

---

## 1. Principle of Centralized and Decoupled Evaluation

To guarantee uncompromising scientific rigor, statistical validity, and zero evaluator bias across all computational experiments in this project:

1. **Strict Prohibition of Ad-Hoc Scoring Code:**
   * Under **no circumstances** shall any experiment script within `experiments/<exp_id>/` implement its own scoring functions, loss approximations as evaluation metrics, custom ranking algorithms, or ad-hoc metrics calculation.
   * Any experiment report referencing metrics computed via local, non-centralized code is considered **scientifically invalid** and rejected.

2. **Mandatory Exclusive Use of `chimera_metrics`:**
   * Every model (including tabular classifiers, deep neural networks, multimodal fusion architectures, LLM reasoning harnesses, and agentic ReAct workflows) **MUST** compute all case-level and cohort-level scores exclusively by importing and invoking `chimera_metrics`.

3. **Parity with Grand-Challenge Container Contracts:**
   * All metrics in `chimera_metrics` strictly mirror the official mathematical equations and container contracts of `docs/chimera-agent-evalaution/evaluation/evaluate.py`.

---

## 2. Standardized 30-Fold Multidimensional Stratified Monte Carlo Protocol

1. **Inviolable Split Invariance:**
   * All predictive models and reasoning agents for Task 1 must be trained and evaluated across the exact 30 deterministic partitions recorded in:
     ```
     data/processed/task1/mccv_splits_30.csv
     ```
   * Ad-hoc random splits, arbitrary single train/test splits, or unstratified folds are strictly disallowed.

2. **Stratification Contract (Per Split $b \in \{0, \dots, 29\}$):**
   * **Cohort Distribution:** Exactly 73 `train` cases, 18 `val` cases, and 104 `unlabeled` cases.
   * **Decision Stratification:** Exactly 11 `yes` (positive biopsy) and 7 `no` (negative biopsy) in each validation split ($N_{\text{val}} = 18$).
   * **Confidence & Factor Invariance:** Distributional matching across ordinal pathological confidence (`clear`, `borderline`, `uncertain`) and active critical variables (`important` / `decisive` factors).

3. **Statistical Hypothesis Testing:**
   * Any claim of architectural or methodological superiority between Model $A$ and Model $B$ must report:
     * Mean metric across 30 folds: $\overline{M} = \frac{1}{30} \sum_{b=0}^{29} M^{(b)}$
     * Empirical 95% Confidence Interval: $\overline{M} \pm 1.96 \cdot \frac{\sigma}{\sqrt{30}}$
     * Two-sided Wilcoxon signed-rank paired test: $p\text{-value} < 0.05$.

---

## 3. Governance Checklist for Experiment Approval

Before any experiment is marked as `Complete` in `experiments/INDEX.md`:
- [ ] Evaluation was executed via `from chimera_metrics import ...`
- [ ] Evaluation was run over all 30 splits from `data/processed/task1/mccv_splits_30.csv`
- [ ] No metric calculation code exists inside the experiment directory
- [ ] Results and artifacts are committed under `experiments/<exp_id>/results/`
