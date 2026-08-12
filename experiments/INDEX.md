# Experiments Index — Pathology Reasoning (CHIMERA Task 1)

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan)
→ `results/<condition>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Canonical Master Data Structuring | Complete | Unpacking all nested JSON structures, dicts, series, and 1024D vectors into a flattened prefixed tabular matrix (`inputs.csv` & `ground_truth.csv`) with `np.nan` handling preserves 100% of information. | ✓ PASS | 2026-08-07 |
| [exp_2](exp_2/DESIGN.md) | Clean Cohort Selection & Rigorous Validation Protocol (MCCV + LOOCV) | Complete | Excluding missing MRI ($N=4$) and PI-RADS ($N=1$) yields a clean $N=88$ labeled cohort; 50-repeat Monte Carlo CV (search) + 88-fold LOOCV (eval) guarantees zero-leakage model selection. | ✓ PASS | 2026-08-09 |
