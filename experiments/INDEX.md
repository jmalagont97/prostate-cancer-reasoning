# Experiments Index — prostate-cancer-reasoning

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan) → `results/<condition>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Task 1 Baseline Benchmark | Complete | Official baseline establishes a baseline ranking score of R_1 = 0.6103 across 91 cases | ✅ | 2026-08-22 |
| [exp_2](exp_2/DESIGN.md) | Multimodal Preprocessing Task 1 | Complete | Canonical 4-CSV extraction structures 195 cohort cases with zero data leakage | ✅ | 2026-08-22 |
| [exp_3](exp_3/DESIGN.md) | Multidimensional MCCV-30 & Protocol Governance | Complete | Deterministic 30-fold multidimensional stratified MCCV and centralized evaluation governance eliminates subtask bias | ✅ | 2026-08-22 |
| [exp_4](exp_4/DESIGN.md) | Corrected Standardized MCCV-30 Unimodal Sweep and Late Fusion Benchmark | Complete | Corrected late fusion with canonical `bx` outperforms the ReAct baseline on MCCV-30, reaching Macro-F1 0.7051 and S_gate 0.7185 | ✅ | 2026-08-29 |
| [exp_5](exp_5/DESIGN.md) | Replication of BrentMemKDM Multimodal (continuous σ alignment) | Complete | Corrected multimodal KDM benchmark with canonical `bx` reaches its best performance with soft fusion under leak-free MCCV-30 | ✅ | 2026-08-29 |
| [exp_6](exp_6/DESIGN.md) | exp_12 Scalar Direct-KDM replication for Subtask 1.2 (corrected, leak-free) | Complete | Corrected scalar direct-KDM with canonical `bx` and MCCV-30 reaches Macro-F1 0.5259 and MOE_abs 0.3060 | ✅ | 2026-08-29 |
| [exp_7](exp_7/DESIGN.md) | Composite Fuzzy ICI on MCCV-30 (corrected, complementary) | Excluded | Complementary ICI rerun is not part of the canonical deployment suite; `exp_7` is explicitly excluded from the selected references | ⚠️ | 2026-08-29 |
| [exp_8](exp_8/DESIGN.md) | Multivariate Balanced Decision Trees over corrected SHAP and Coalesced Feature Spaces | Complete | Corrected Coalesced Balanced Forest (C4) provides the SHAP-space benchmark with Macro-F1 0.2444 and S_weights 0.7978 | ✅ | 2026-08-29 |
| [exp_9](exp_9/DESIGN.md) | Corrected Joint Feature-Attribution Fusion (Raw Clinical Features + Coalesced SHAP 23D) | Complete | Corrected Joint ExtraTrees 23D (C4) is the best exp_9 condition, with Macro-F1 0.2779 and S_weights 0.8276 | ✅ | 2026-08-29 |
| [exp_10](exp_10/DESIGN.md) | Corrected exploratory full-cohort evaluation of `deploy/agent` on Task 1 | Complete (exploratory) | Corrected `deploy/agent` produces valid artifacts for all 195 cases and improves exp_1-compatible `R_1` in-sample | ⚠️ R_1=0.7600; optimistic in-sample | 2026-08-29 |

**Canonical corrected deployment suite:** `exp_4` (Subtask 1.1 decision), `exp_6` (Subtask 1.2 confidence), and `exp_9` (Subtask 1.3 variable relevance). `exp_7` is excluded from the canonical suite. `exp_10` is an exploratory in-sample agent evaluation: predictions were generated for 195 cases, but metrics were evaluated only on the 91 annotated cases and must not be presented as held-out validation.
