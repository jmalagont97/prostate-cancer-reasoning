# Experiments Index — prostate-cancer-reasoning

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan) → `results/<condition>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Task 1 Baseline Benchmark | Complete | Official baseline establishes a baseline ranking score of R_1 = 0.6103 across 91 cases | ✅ | 2026-08-22 |
| [exp_2](exp_2/DESIGN.md) | Multimodal Preprocessing Task 1 | Complete | Canonical 4-CSV extraction structures 195 cohort cases with zero data leakage | ✅ | 2026-08-22 |
| [exp_3](exp_3/DESIGN.md) | Multidimensional MCCV-30 & Protocol Governance | Complete | Deterministic 30-fold multidimensional stratified MCCV and centralized evaluation governance eliminates subtask bias | ✅ | 2026-08-22 |
| [exp_4](exp_4/DESIGN.md) | Standardized MCCV-30 Unimodal Sweep and Late Fusion Benchmark | Complete | Decoupled unimodal sweeps and calibrated late fusion over simplex Δ² and threshold θ outperform ReAct baseline on MCCV-30 | ✅ | 2026-08-23 |
| [exp_5](exp_5/DESIGN.md) | Replication of BrentMemKDM Multimodal (continuous σ alignment) | Draft | Continuous Brent-optimized bandwidths in a multimodal KDM memory reproduce/beat source Macro-F1 0.7068 under leak-free MCCV-30 | — | 2026-08-23 |
| [exp_6](exp_6/DESIGN.md) | exp_12 Scalar Direct-KDM replication for Subtask 1.2 (leak-free) | Complete | Leak-free scalar direct-KDM (20 clean features, MCCV-30) reaches Macro-F1 0.4836 and MOE_abs 0.3436 without class collapse | ✅ | 2026-08-24 |
| [exp_7](exp_7/DESIGN.md) | Composite Fuzzy ICI on MCCV-30 (Frozen exp_4 Modalities) | Complete | Post-hoc Composite Fuzzy ICI with frozen exp_4 models yields Macro-F1 0.2892 and reveals degradation from k=1 MRI quantization | ⚠️ | 2026-08-24 |
| [exp_8](exp_8/DESIGN.md) | Multivariate Balanced Decision Trees over 10D Full SHAP and Coalesced Feature Spaces | Complete | Coalesced Balanced Forest (C4) outperforms raw trees (p=0.00061) and baseline B0, yielding Macro-F1 0.2523, S_weights 0.8050, and eliminating DRE/PSAD pathologies | ✅ | 2026-08-24 |
| [exp_9](exp_9/DESIGN.md) | Joint Feature-Attribution Fusion (Raw Clinical Features + Coalesced SHAP 23D) | Complete | Joint ExtraTrees 23D (C4) resolves polarity ambiguity, achieving Macro-F1 0.2934, S_weights 0.8278, and record rho(PIRADS)=+0.4486 (p=1.86e-9 vs SHAP-only) | ✅ | 2026-08-24 |
