# Experiments Index — challenge_chimera_2

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan)
→ `results/<condition>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

📋 **[PROJECT_REPORT.md](PROJECT_REPORT.md)** — consolidated results across all three
experiments + the bugfix + held-out test verification, in one place.

⚠️ **2026-08-10 data-integrity correction**: all **decision**-target results across all three
experiments below were originally computed on a corrupted target vector
(`target_biopsy_decision` is `NaN`, not "no", for 104/195 cases; the old code silently coded
those as `y=0`). Fixed via `train_decision.load_labeled_data()`; every affected
`results/decision_*/metrics.json` and `reports/summary.md` has been corrected in place, each
carrying a `bugfix_corrected` field with the old (wrong) value for audit. Confidence, weights,
and reveal-sequence were unaffected. See `exp_1/reports/summary.md`'s erratum for full detail.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Hybrid ML Baseline for CHIMERA-Agent Task 1 | Complete | Small supervised models on structured features beat naive per-target baselines closely enough to justify submission infra | ❌ Refuted (decision/confidence/weights) — ✅ Supported (reveal-sequence only) ⚠️ decision numbers corrected 2026-08-10, verdict direction unchanged | 2026-08-08 |
| [exp_2](exp_2/DESIGN.md) | Official-Schema Feature Scope + Comorbidity Grouping + KDM Decision Model | Complete | Restricting to the 11 officially-documented Task-1 input variables + grouped comorbidity flags + a KDM decision classifier beats exp_1's baselines | ⚠️ Decision: modest schema-restriction lift but no longer beats baseline; **KDM now best decision model** (reversed from ❌) — ❌ Confidence — ✅ Reveal ⚠️ corrected 2026-08-10 | 2026-08-08 |
| [exp_3](exp_3/DESIGN.md) | Broader Model Family Comparison + MRI-PCA + Decorrelated PSA Family | Complete | An 8-model search (SVM/RF/XGBoost/ExtraTrees/MLP/NaiveBayes/kNN/KDM) + 2-component MRI-PCA + psa/psad-only PSA family clearly beats exp_1/exp_2's best per target | ⚠️ Decision: kNN/SVM/KDM lead (not Extra Trees), still below baseline (reversed from ✅) — ✅ Confidence (SVM, first-ever baseline beat, unaffected) — ❌ Weights — ⚠️ Reveal (flat) ⚠️ corrected 2026-08-10 | 2026-08-09 |
| [exp_4](exp_4/DESIGN.md) | Clinical-Only Features (No MRI-PCA) — exp_3 Ablation | Complete | Removing MRI-PCA from exp_3's frame doesn't flip any headline verdict, but has real per-model effects (target-dependent) | ⚠️ MRI not decisive for verdicts, but decisive (positively) for MLP/NB on decision and decisive (negatively) for confidence overall | 2026-08-10 |
| [exp_5](exp_5/DESIGN.md) | 8-Model Search for Variable-Weight Prediction | Complete | Extending exp_3's 8-model search to weights (official + restricted scope) beats the incumbent best and ideally the naive baseline | ✅ Weights (SVM, first-ever baseline beat — 0.382/0.392 vs. 0.413) | 2026-08-10 |
| [exp_6](exp_6/DESIGN.md) | KDM as a Unified Probabilistic Backbone for Decision + Confidence + Weights | Complete | One trained KDM's own predictive uncertainty (output entropy + prototype-neighborhood dispersion, via `kdm.utils.dm_rbf_variance`) and local per-factor sensitivity can match or beat confidence_svm (0.468) and weights_svm (0.382/0.392) with no separate model family per target | ❌ Confidence (all 5 signal variants worse than baseline) — ⚠️ Weights (occlusion narrowly beats baseline, 0.405 vs 0.413, but not the incumbent; replicates exp_5's 4/9-factor solvable split via an independent mechanism) — ✅ decision backbone reproduces exp_3 (0.593 vs 0.588) | 2026-08-10 |
| [exp_7](exp_7/DESIGN.md) | Improving exp_6's KDM Backbone (Hyperparameter Tuning + Skew-Aware Preprocessing) | Complete | A bounded, low-risk grid search over epochs/lr/sigma_mult plus log1p-transforming exp_6's most-skewed columns (psa skew=4.28, psad skew=4.24, vol skew=1.29) closes some of the gap between KDM's own decision macro-F1 (0.593) and Extra Trees (0.650), and possibly improves exp_6's derived confidence/weights signals too, without touching KDM's architecture | ❌ Refuted — 144-way search found a config that beat baseline under CV (0.622 vs 0.593) but scored worse on genuine held-out data (0.490 vs 0.593, held-out check caught the spurious CV win as designed); confidence/weights unchanged | 2026-08-11 |
