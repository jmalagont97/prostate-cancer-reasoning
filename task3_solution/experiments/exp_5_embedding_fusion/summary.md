# exp_5 Report: Task3 Embedding Fusion (T3.2/T3.3)

**Status**: Complete. **Verdict: confirmed, decisively — prostatectomy-slide embeddings promoted.**

## Headline result

| Condition | LOOCV ranking_score (C-index) |
|---|---|
| baseline (tabular + pathology text, no embeddings) | 0.7655 |
| + mri_pca5 | 0.7522 (worse) |
| + biopsy_pca5 | 0.7327 (worse) |
| **+ prostatectomy_pca5** | **0.8575** (+0.0920) |
| + all_three_pca5 | 0.8133 (+0.0478, but worse than prostatectomy alone) |

**Bootstrap significance** (2000 resamples, paired): point estimate +0.0920, **95% CI [+0.028,
+0.176]**, 99.6% of resamples favor the embedding-fused model. This is the first task3 gain in this
project's history that clears a formal significance bar comfortably — `exp_4`'s Cox-penalizer gain
(+0.017) did not.

**k-robustness** (Stage B): 0.845 (k=2), 0.858 (k=5), 0.847 (k=10) — consistent across the swept
range, not a k=5-specific fluke.

**T3.7 cross-check** (IPCW time-dependent AUC) also improved across every horizon: 12m 0.854→0.922,
24m 0.760→0.859, 36m 0.753→0.853, 60m 0.758→0.875. Confirms the C-index gain isn't an artifact of
how that one metric weights pairs — the model is genuinely better calibrated at every clinically
relevant horizon, not just overall concordance.

## The result that surprised the stated hypothesis (and how it was checked before trusting it)

`DESIGN.md`'s hypothesis predicted prostatectomy-slide embeddings were the **weakest** candidate of
the three (most redundant with the pathology-text features already extracted from the same tissue
in `exp_3`) and MRI the strongest (different modality, different time point, no existing structured
field). The result was the opposite: prostatectomy embeddings won by a wide margin; MRI and biopsy
embeddings both made things worse.

Given the size of the effect (+0.092, the largest of any task3 experiment) and that it contradicted
the stated prior, this was treated as a "too good — check for leakage" moment before being written
up, the same discipline task1's `exp_1` used on `clinician_shortlist`'s suspiciously strong 5-fold
result (which turned out to be leakage). Checks performed:
1. **Case-ID alignment**: pooled prostatectomy vectors are 75/75 distinct (no collapse to a shared
   fallback value that would indicate a silent ID-matching failure).
2. **PCA leakage**: fit strictly inside each LOOCV training fold, confirmed by code review and by
   the fact `predict_months()` only ever calls the *stored*, already-fitted PCA transform, never
   refits it on data being predicted.
3. **Raw correlation sanity check**: the top 2 principal components (before any Cox fitting) show
   moderate, not suspiciously-perfect, correlation with event/time (r≈0.37–0.43, p≈0.001) —
   consistent with genuine partial signal, not a leak-to-1.0 pattern.

Conclusion: the win looks real, not a bug. Plausible explanation — WSI foundation-model embeddings
of the actual prostatectomy tissue can encode richer morphological detail (texture, growth pattern
nuance) than a single discretized ISUP grade number captures; this is consistent with published
digital-pathology literature on deep-learning-derived features outperforming discrete grading alone
for BCR prediction.

## A second real bug caught mid-experiment

First Stage A run crashed with the same "delta contains nan" Cox convergence error seen in `exp_3`
Issue #2 — same root cause, different feature: `prostatectomy_slides` has 100% coverage (every case
has ≥1 valid slide, confirmed in `DESIGN.md` §2), so a `has_prostatectomy_slides` missingness flag
column would be **constant across all 75 cases**, singularizing the Cox design matrix in every
single fold. Fixed by only adding a bag source's missingness flag when that source actually has
missing cases in the full dataset (`biopsy_slides` does, 24/75; `prostatectomy_slides` doesn't).

## Promoted

`agent/src/task3_config.py`:
- New `_CoxEmbeddingAdapter` class (stateful PCA/imputer across `fit()`/`predict_months()`, never
  refit on prediction data).
- New registry entries: `cox_p03_prostatectomy_pca{2,5,10}`, `cox_p03_all_embeddings_pca5`.
- `DEFAULT_MODEL_KEY = "cox_p03_prostatectomy_pca5"`.
- New `agent/src/task3_embeddings.py` (production `EmbeddingSource`, promoted from `exp_5`'s
  exploration code).

Verified end-to-end: `models_bridge_task3.py` → `run_task3_agent_loocv.py` → `local_eval_task3.py`
reproduces 0.8575 exactly, matching both the exploratory sweep and the in-memory
`task3_variant_tester.py` check.

## Follow-up (2026-08-25): fixed a real submission-readiness gap

`task3_embeddings.py`'s `EmbeddingSource` originally only read from the pre-compiled
`data/processed/task3/embeddings/*.npz` (built once from the 75 training cases by
`scripts/build_dataset.py`). A genuinely new case at real submission time has no entry there at
all — the agent would have silently treated every new case as having zero embedding data, quietly
defeating the entire point of this experiment's promotion. Fixed by adding a fallback that reads a
case's raw `prostate-modality-level-neural-representations.json` directly (same file, same parsing
convention `build_dataset.py` itself uses — "MRI image"'s first element, "Biopsy slide"/
"Prostatectomy slide" as unpadded variable-length lists). Validated three ways: (1) the raw-file
path reproduces the compiled-npz path's pooled vectors exactly (max abs diff ~3e-7, float32/64
rounding only) for all 75 known cases; (2) forcing the dispatcher down the fallback path for a
known case (by removing it from the compiled maps) still returns the correct value; (3) a truly
nonexistent case_id returns `None` gracefully rather than crashing. Full LOOCV regression re-run
confirms `ranking_score` unchanged at 0.8575 (expected — the fix only adds a new code path, doesn't
touch the one all 75 training cases already use).

## Not done / next candidates

- Stage C (re-sweep Cox penalizer on the new feature set) — not run; the k-sweep already showed a
  large, robust margin over baseline at every k tried, and penalizer=0.3 was already the winner on
  the tabular+text set, so a re-sweep is a lower-priority refinement, not a likely source of another
  large gain.
- `all_three_pca5` (all embedding sources combined) underperforms `prostatectomy_pca5` alone —
  mri/biopsy appear to add noise rather than orthogonal signal on top of prostatectomy's strong
  signal; not investigated further (diminishing-returns territory, per this project's usual
  practice of not chasing marginal combinations once a clear single winner exists).
- Task2's biopsy-slide embedding fusion (a parallel item on the plan) not attempted this session.
