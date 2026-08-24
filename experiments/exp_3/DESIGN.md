# Experiment Design: Multidimensional Stratified Monte Carlo Protocol & Governance (`exp_3`)

**Experiment**: experiments/exp_3/  
**Project**: prostate-cancer-reasoning  
**Date**: 2026-08-22  
**Author**: Juan Sebastian Malagon Torres  
**Status**: Complete

---

## 1. Hypothesis & Research Objective
Falsifiable claim:
> A fixed, deterministic 30-fold multidimensional stratified Monte Carlo Cross-Validation (MCCV, B=30, 80/20 split) focusing strictly on the core target tasks (binary biopsy decision, pathological confidence calibration, and active critical factors) coupled with centralized evaluation governance (`chimera_metrics`) eliminates multi-subtask sampling bias and variance artifacts, providing a statistically sound ($p < 0.05$) foundation for all subsequent modeling experiments.

---

## 2. Target Variables for Stratification (Supervised Subtasks Only)
The stratification process operates strictly over the supervised multi-target space of Task 1 on the 91 labeled cases:

```
+---------------------------------------------------------------------------------------------------------------+
| Dimensión Subtarea           | Tipo Matemático         | Clases / Niveles Evaluados                          |
+------------------------------+-------------------------+-----------------------------------------------------+
| 1. Decisión Primaria         | Binaria                 | {yes, no} (Exact 11 yes / 7 no per val split)       |
| 2. Confianza Patológica      | Ordinal (3 niveles)     | {clear, borderline, uncertain}                      |
| 3. Factores Críticos         | Vector Multietiqueta    | Indicador binario f_v in {0, 1} de si la variable   |
|    (Decisivos / Importantes) | (10 variables)          | v in {age, psa, vol, psad, dre, pirads, fh,         |
|                              |                         | comorbidity, bx, cspca} es 'decisive' o 'important' |
+---------------------------------------------------------------------------------------------------------------+
```

*Nota Metodológica:* Se excluyen covariadas de entrada (PI-RADS) y el histograma completo de 4 niveles de pesos de variables para evitar sobre-restricción combinatoria debido al severo desbalance de niveles pasivos (`not_used` vs `noted`), el cual queda cubierto de forma parsimoniosa y representativa por el vector de factores críticos.

---

## 3. Experimental Setup & Governance Directives
- **Cohort**: 195 total patients (`data/processed/task1/ground_truth_task1.csv`), containing 91 supervised reference cases and 104 unlabeled cases.
- **Split Ratio**: 80% Training ($n_{\text{train}} = 73$) / 20% Validation ($n_{\text{val}} = 18$) across all $B=30$ iterations.
- **Target Split Artifact**: `data/processed/task1/mccv_splits_30.csv`
  - 195 rows (one per `pid`).
  - Columns: `pid`, `split_0` through `split_29`, with values in `{'train', 'val', 'unlabeled'}`.

### Strict Governance & Inviolable Evaluation Policy:
1. **Zero Ad-Hoc Evaluation Code in `experiments/`:** Under no circumstances shall an experiment script define its own scoring, metric calculation, or ranking logic.
2. **Mandatory Centralized Evaluation Package:** Every experiment MUST import and execute its evaluation exclusively via `chimera_metrics`.
3. **Universal Split Invariance:** All future models (tabular baselines, deep neural networks, multimodal fusion, and LLM/agentic architectures) MUST evaluate on these exact 30 splits to allow paired statistical testing (Wilcoxon signed-rank test).
4. **Package Protocol Documentation:** A normative protocol document (`PROTOCOL.md`) and enforcement validator will be installed directly into `utils/pkg/chimera_metrics/`.

---

## 4. File Layout for This Experiment
```
experiments/exp_3/
├── DESIGN.md                  <- this experiment design document
├── IMPLEMENTATION.md          <- implementation plan (written after approval)
├── scripts/
│   └── generate_mccv_splits.py <- multidimensional stratified split generator
├── results/
│   ├── split_diagnostics.json  <- statistical balance across 30 folds
│   └── version.txt
└── reports/
    └── summary.md             <- report on split uniformity and protocol enforcement
```

---

## 5. Evaluation Protocol & Statistical Criteria
- **Quality Metric of Partitions**:
  - Exact preservation of binary decision ratio in every fold: $P_{\text{val}}(\text{yes}) = 11/18 \approx 61.1\%$.
  - Wasserstein-1 (EMD) distance on confidence distribution: $\text{EMD}_{\text{conf}} < 0.05$.
  - Mean absolute difference in critical factor frequency between train and val: $\Delta_{\text{factors}} < 0.05$.
- **Validation Check**: Verify that all 104 unlabeled cases are tagged as `'unlabeled'`.

---

## 6. Reproducibility Checklist
- [x] Seeds deterministically fixed ($\text{seed}_b = 42000 + b \cdot 1000$).
- [x] Protocol ratified in `utils/pkg/chimera_metrics/PROTOCOL.md`.
- [x] Partitions committed and saved to `data/processed/task1/mccv_splits_30.csv`.
- [x] Version and audit trail recorded in `experiments/exp_3/results/version.txt`.

---

## 7. Status & Verdict
- **Status**: Complete
- **Verdict**: ACCEPTED (✅)
