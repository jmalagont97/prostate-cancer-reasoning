# Experiments Index — Pathology Reasoning (CHIMERA Task 1)

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan)
→ `results/<condition>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Canonical Master Data Structuring | Draft | Unpacking all nested JSON structures, dicts, series, and 1024D vectors into a flattened prefixed tabular matrix (`inputs.csv` & `ground_truth.csv`) with `np.nan` handling preserves 100% of information. | — | 2026-08-07 |
