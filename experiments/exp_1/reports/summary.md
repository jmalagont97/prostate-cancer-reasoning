# Experiment Report: Hybrid ML Baseline for CHIMERA-Agent Task 1
**Experiment**: experiments/exp_1/
**Project**: challenge_chimera_2
**Report date**: 2026-08-08
**Plan date**: 2026-08-08
**Author**: TBD
**Status**: Complete

---

## ⚠️ Erratum (2026-08-10)

All **decision** numbers in this report were originally computed on a corrupted target vector.
`target_biopsy_decision` is `NaN` for 104 of the 195 cases (the same 91 cases carry every other
ground-truth label — decision, confidence, weights, and reveal-sequence all share the identical
91-case labeled subset, not 195 for decision and 91 for the rest as originally assumed). The
original code computed `y = (df["target_biopsy_decision"] == "yes")` on the full 195-row merge;
since pandas' `NaN == "yes"` evaluates to `False` rather than raising an error, the 104 unlabeled
cases were silently coded as `y=0` ("no") instead of being excluded. This was found and fixed on
2026-08-10 (`train_decision.load_labeled_data()`, filtering to the 91 actually-labeled cases,
matching how confidence/weights/reveal were already handled correctly). **All decision numbers
below have been corrected** (each `results/decision_*/metrics.json` now carries both the
corrected value and the old wrong one in a `bugfix_corrected` field for audit). Confidence,
weights, and reveal-sequence were unaffected — they were already correctly filtered to the
91-case subset from the start. The qualitative conclusion for decision (models don't clearly beat
naive baseline) still holds post-correction, but the actual numbers, baseline, and N are all
substantially different — see §5.1.

---

## 1. Summary

The hypothesis was that small supervised models (logistic regression, gradient boosting, and a
Kernel Density Matrix classifier) trained on Task 1's structured clinical features could beat
naive per-target baselines closely enough to justify building the submission pipeline. **The
hypothesis is refuted on 3 of 4 target groups and only weakly supported on 1.** *(Corrected
2026-08-10 — see erratum above)* The decision model's best result (HGB+MRI-PCA, F1 0.705) doesn't
clearly beat "always predict yes" (F1 0.762, N=91 — not the originally-reported N=195/F1=0.446);
confidence and per-factor variable weights are both worse than naive majority-class guessing in
aggregate; only the reveal-sequence model clearly beat its baseline (0.840 vs. 0.783 set
precision). Classifier choice mattered little for decision post-correction either — all four
conditions land within 0.635–0.705, still below the (now much higher, 0.762) baseline — pointing
to the same feature/data-problem conclusion as before, just with corrected numbers.

---

## 2. Hypothesis & Verdict

**Hypothesis (from plan):** Small supervised models trained on Task 1's structured features can
beat trivial per-target naive baselines on decision F1, confidence ordinal distance, variable-
weight ordinal error/decisive-set F1, and reveal-sequence set precision, closely enough in
aggregate to justify building the downstream submission pipeline.

**Verdict:** ❌ Refuted (decision, confidence, weights) / ✅ Supported (reveal-sequence only)

**Evidence:** Decision F1 0.705 (best of 4, HGB+MRI-PCA) vs. 0.762 naive baseline (worse,
corrected 2026-08-10 — see erratum above); confidence ordinal distance
0.564 (best, via KDM) vs. 0.527 naive baseline (worse); variable-weights mean ordinal error 0.574
vs. 0.401 naive baseline (worse), though mean decisive-set F1 0.451 vs. 0.341 (better); reveal-
sequence set precision 0.840 vs. 0.783 (better). 3 of 4 target groups fail to clear their
baseline on the primary metric named in the plan.

---

## 3. Experimental Setup (as run)

As described in `DESIGN.md`, with one addition made mid-experiment: the KDM confidence model
(not in the original code, added after a mid-session detour to try `kdm-torch` for confidence
specifically, given its native uncertainty calibration). No `IMPLEMENTATION.md` was written for
this experiment (it was built and run directly in an interactive session, then documented
retroactively) — deviations below are inferred from the code and this session's transcript, not
from a separate implementation-plan record.

- **Dataset**: `data/inputs.csv` + `data/ground_truth.csv`, 195 cases (91 with full annotation).
- **Model**: see DESIGN.md §2 — no single "model", a set of small classical/shallow classifiers.
- **Training**: CPU only; KDM used CPU-only PyTorch 2.13 (`torch==2.13.0+cpu`).
- **Hardware**: local machine, RTX 4050 laptop GPU present but unused (not needed at this N).
- **Deviations from plan**: HistGradientBoosting's first attempt used default (unregularized)
  hyperparameters and was unstable (one CV fold hit F1=0.0); regularization
  (`max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=1.0`) was added mid-experiment
  before the numbers in §5 were finalized. The confidence-model comparison (KDM) was not in the
  original decision/reasoning scripts — added as a targeted follow-up once the plain logistic
  confidence model underperformed baseline.

---

## 4. Code Version

| Condition | Git commit | Commit message |
|-----------|-----------|-----------------|
| all | _N/A_ | This project is not a git repository — no commit hashes exist for any run. |

⚠️ No `git_commit.txt` exists for any condition; results are traceable only via the scripts in
`src/chimera_task1/` as they exist on disk at report time, not a pinned historical version.

---

## 5. Results

### 5.1 Primary Metric

| Condition | Metric | Value | vs. baseline (Δ) |
|-----------|--------|-------|-------------------|
| decision_baseline (always "yes", N=91) ⚠️ *corrected* | F1 | 0.762 | — |
| decision_logistic_clinical ⚠️ *corrected* | F1 | 0.658 ± 0.085 | **−0.104** ❌ |
| decision_hgb_clinical ⚠️ *corrected* | F1 | 0.653 ± 0.115 | −0.109 ❌ |
| decision_logistic_mri_pca ⚠️ *corrected* | F1 | 0.635 ± 0.092 | −0.127 ❌ |
| decision_hgb_mri_pca ⚠️ *corrected* | F1 | 0.705 ± 0.114 | −0.057 ❌ (closest to baseline) |
| confidence_baseline (always "clear") | ordinal distance | 0.527 | — |
| confidence_logistic | ordinal distance | 0.576 | +0.049 (worse) ❌ |
| confidence_kdm | ordinal distance | 0.564 ± 0.054 | +0.037 (worse) ❌ |
| weights_baseline (per-factor majority) | mean ordinal error | 0.401 | — |
| weights_logistic | mean ordinal error | 0.574 | +0.173 (worse) ❌ |
| reveal_baseline (mode pattern) | set precision | 0.783 | — |
| reveal_logistic | set precision | 0.840 ± 0.010 | **+0.057** ✅ |

> No single success threshold was set numerically in the plan beyond "clearly beats baseline";
> by that qualitative bar, only reveal-sequence clears it.

### 5.2 Secondary Metrics

- **Decision, threshold-independent** ⚠️ *corrected*: ROC-AUC 0.673 / PR-AUC 0.774 (clinical
  only) vs. chance 0.5 / 0.615 (N=91) — above chance, though PR-AUC's margin over chance is much
  smaller than it looked pre-correction (chance PR-AUC scales with the positive rate, which is
  now 61.5% not 28.7%). A diagnostic post-hoc-threshold-tuned F1 reached 0.746 (optimistic —
  threshold chosen on the same out-of-fold scores it's evaluated on).
- **Confidence, KDM predictive entropy**: 0.716 of a max possible 1.099 — not degenerate
  (didn't collapse to a single always-confident or always-uncertain prediction).
- **Variable weights, decisive-set F1**: 0.451 vs. 0.341 baseline — the model *is* better at
  this specific sub-metric even though it's worse on mean ordinal error, because several
  factors (fh, cspca, vol, comorbidity, psad, dre) have a majority class that is never
  "important"/"decisive", so the trivial baseline scores 0 there while the model picks up some.

### 5.3 Ablation Results

| Ablation | Primary metric | Δ vs. clinical-only | Interpretation |
|----------|----------------|----------------------|-----------------|
| + MRI-PCA(10) [logistic] ⚠️ *corrected* | F1 0.635 | −0.023 | MRI embedding PCA components still add noise for logistic |
| + MRI-PCA(10) [HGB] ⚠️ *corrected* | F1 0.705 | **+0.052** | Reverses pre-correction finding — MRI now *helps* HGB |

⚠️ **This ablation's conclusion changed with the correction.** Pre-correction, both models got
worse with MRI added, suggesting a clean "MRI adds noise" story. Post-correction, logistic still
gets worse (−0.023) but HGB gets *better* (+0.052) — the two models now disagree on whether the
MRI-PCA signal helps. This is a smaller, noisier N=91 (down from the originally-assumed 195), so
this reversal may itself partly be a small-N CV artifact rather than a robust HGB-specific
finding; not investigated further here.

### 5.4 Learning Curves

Not applicable — these are cross-validated classical models, not iterative training with
learning curves worth plotting. No figures were generated for this experiment
(`reports/figures/` is empty).

---

## 6. Statistical Analysis

- **Test used**: none. Repeated k-fold CV (5-fold × 8-20 repeats) gives a mean ± std across
  folds, but per-seed/per-repeat values were not persisted to `results/`, only the aggregate.
- **p-value / CI**: not computed. Per-seed values not available; cannot compute a formal
  significance test from the aggregate stats alone.
- **Conclusion**: the mean ± std spread reported per condition is the only uncertainty estimate
  available. Given N=91 for every target now (post-correction, decision included) and
  fold-to-fold std comparable in magnitude to the Δ vs. baseline in most rows above, none of the
  "worse than baseline" findings should be read as a precise point estimate — but the consistent
  direction across every classifier tried on the same target (logistic, HGB, KDM all landing in
  the same weak band) is a stronger signal than any single point estimate's precision.

---

## 7. Comparison to Expected Results

| Expected | Observed | Match? |
|----------|----------|--------|
| Models clearly beat naive baselines across decision + confidence + weights → proceed to steps 5-8 | Models did not beat baselines on decision, confidence, or weights (mean ordinal error) | ❌ |
| If not → stop and reassess feature set/architecture before submission infrastructure | Decision made: **paused**, per DESIGN.md §8's own decision rule | ✅ (the decision rule itself was followed) |

---

## 8. Missing Data & Caveats

- **See the erratum at the top of this report** — all decision numbers were corrected on
  2026-08-10 after discovering `target_biopsy_decision` was NaN (not "no") for 104/195 cases,
  silently mis-coded as `y=0` by the original code. Confidence/weights/reveal were unaffected.
- No `IMPLEMENTATION.md` exists for this experiment (see §3) — build details are reconstructed
  from the scripts and this session's record, not a separately-approved implementation plan.
- Per-seed/per-repeat metric values were not persisted, only aggregates (blocks §6's
  significance testing).
- Steps 5-8 of the broader Task-1 plan (rubric scorer, MCP+LLM wiring, Docker packaging,
  platform submission) were never started, by design — this experiment's negative-leaning
  result is exactly what stopped that work from beginning.
- The "add text features" and "prototype the full LLM-agent baseline" alternatives discussed
  during this session were not tried — they're open follow-ups, not completed conditions.

---

## 9. Conclusions & Next Steps

- **What this experiment established**: the bottleneck for Task 1 with the current approach is
  the structured-feature set, not the classifier — three different model families (linear,
  tree-based, kernel-density) landed in the same weak range on the same targets. The one
  target with a real win (reveal-sequence) is also the one where the near-canonical fixed
  ordering pattern this session found made it more of a pattern-matching problem than a hard
  prediction problem.
- **What remains uncertain**: whether adding the excluded `txt_*` free-text narrative features
  would close the gap (plausible — clinician confidence/reasoning may be recorded in prose, not
  in the numeric fields tried here), and whether more training data (the 75-case validation set,
  released 2026-08-10) would help given N=91-195 is small for 10-way ordinal weight targets.
- **Recommended follow-up experiments** (would be `exp_2` via ml-experiment-planner):
  1. Add simple text features (keyword flags or TF-IDF) from `txt_*` columns to the existing
     feature set and re-run the same decision/confidence/weights comparison.
  2. Prototype the organizers' full LLM-agent baseline (`vendor/chimera-agent-baseline/`) on a
     handful of cases, since it reasons over the narrative text directly — a genuine
     architectural alternative to pure tabular ML, not just a feature addition.
  3. Revisit once the validation set is available, purely for more training signal (not for
     tuning against hidden labels).

---

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Seeds logged | ✅ (`RANDOM_STATE = 0` per script) |
| Configs versioned | ⚠️ inline constants, not separate config files |
| Git commits recorded | ❌ not a git repository |
| Checkpoints saved | ❌ N/A, no persisted model artifacts |
| Environment frozen | ⚠️ recorded in DESIGN.md §10 prose, no `requirements.txt`/`environment.yml` committed |
| Experiment tracker linked | ❌ not used |
