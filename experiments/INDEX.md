# Experiments Index — Pathology Reasoning (CHIMERA Task 1)

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan)
→ `results/<condition>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Canonical Master Data Structuring | Complete | Unpacking all nested JSON structures, dicts, series, and 1024D vectors into a flattened prefixed tabular matrix (`inputs.csv` & `ground_truth.csv`) with `np.nan` handling preserves 100% of information. | ✓ PASS | 2026-08-07 |
| [exp_2](exp_2/DESIGN.md) | Clean Cohort Selection & Rigorous Validation Protocol (MCCV + LOOCV) | Complete | Excluding missing MRI ($N=4$) and PI-RADS ($N=1$) yields a clean $N=88$ labeled cohort; 50-repeat Monte Carlo CV (search) + 88-fold LOOCV (eval) guarantees zero-leakage model selection. | ✓ PASS | 2026-08-09 |
| [exp_3](exp_3/DESIGN.md) | Tabular KNN Baseline (Task 1.1) | Complete | KNN sobre las 39 variables tabulares sin imputación, seleccionado por Macro-F1 sobre 50 splits MCCV, alcanza Macro-F1=0.6234 y supera a las mayoritarias en ~49/50 folds; LOO pooled=0.6207 (sin gap vs MCCV). | ✓ PASS | 2026-08-15 |
| [exp_4](exp_4/DESIGN.md) | Feature-Subset Ablation (>50% missingness) sobre KNN | Complete | Eliminar variables con >50% de ausencia (esenciales siempre retenidas) mejora el Macro-F1 del KNN: ganador `manhattan_fuzzy_confidence_inverse_distance_k3` con 0.6430 MCCV (LOO pooled 0.6474) vs 0.6234 de exp_3; mejora pareada +0.0916 (p=3.96e-08). | ✓ PASS | 2026-08-15 |
