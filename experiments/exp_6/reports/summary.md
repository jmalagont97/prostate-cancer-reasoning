# exp_6 Report: KDM as a Unified Probabilistic Backbone for Decision + Confidence + Weights

**Experiment**: experiments/exp_6/ · **Project**: challenge_chimera_2 · **Date**: 2026-08-10 · **Status**: Complete

---

## 1. Summary

**Mostly refuted, with one narrow, genuine partial win.** A single KDM trained on the decision
label — with confidence and variable-weights derived from that same model's own predictive
uncertainty and local sensitivity, no separate model family per target — does **not** beat the
confidence incumbent (`confidence_svm`, 0.468) under any of five signal/recalibration variants
tried, and only narrowly beats the weights baseline (0.413) under one of three mechanisms
(`weights_kdm_occlusion`, 0.405), still well short of the weights incumbent (`weights_svm`,
0.382/0.392). The decision backbone itself reproduces `exp_3`'s `decision_kdm` macro-F1 almost
exactly (0.593 vs. 0.588), confirming the fit/predict refactor is correct — this experiment's
result is a genuine negative/mixed finding, not an implementation artifact.

| Target | This experiment's best | Baseline | Incumbent | Beats baseline? | Beats incumbent? |
|---|---|---|---|---|---|
| Decision (re-verification) | 0.593 macro-F1 | 0.381 | 0.650 (Extra Trees) | ✅ (as expected — same as `decision_kdm`) | ❌ |
| Confidence | 0.731 ord. dist. (`entropy_isotonic`) | 0.527 | 0.468 (`confidence_svm`) | ❌ | ❌ |
| Variable weights | 0.405 ord. error (`occlusion`) | 0.413 | 0.382/0.392 (`weights_svm`) | ✅ (narrowly) | ❌ |

## 2. What Was Tested

9 conditions, all derived from one KDM trained per-fold on the biopsy-decision target
(`exp_3`'s 19-column with-MRI frame), per `DESIGN.md`:

- `decision_kdm_backbone` — the backbone's own decision output, re-verified against `exp_3`.
- 5 confidence conditions: raw output entropy (Signal A), prototype-neighborhood dispersion via
  `kdm.utils.dm_rbf_variance` (Signal B), participation ratio (Signal C), each isotonic-
  recalibrated to the 3-level confidence scale, plus a zero-training tercile-binned entropy
  condition and a small ordinal-logistic blend of A+B+C.
- 3 weights conditions, one per factor: local occlusion (Signal D — perturb a factor's columns
  to that fold's training median, measure the shift in `p(yes)`), kernel-distance contribution
  (Signal E — a per-case attribution read directly off the forward pass, no re-inference), and a
  blend of D+E.

All 9 conditions share the exact same per-fold-per-repeat KDM fit (5-fold × 10-repeat CV,
`RANDOM_STATE=0`, matching `exp_3`–`exp_5`'s convention) — the backbone is trained once, not once
per condition, so the confidence and weights comparisons below are genuinely reading different
signals off one model, not retraining eight times.

**Verification passed**: `probs_check_ok=True` across every one of the 50 folds — the
hand-replicated posterior-weight normalization (needed to compute Signals B–E) matches the
model's own `forward()` output exactly, so Signals B–E can be trusted as faithful reads of the
model's internals, not an artifact of a normalization bug.

**One implementation bug found and fixed during this experiment**: `sklearn.isotonic.
IsotonicRegression`'s default is `increasing=True` (not `"auto"`) — every isotonic-calibrated
condition was silently forcing a monotonically-*increasing* fit regardless of each signal's true
relationship to the ordinal rank. Fixed by passing `increasing="auto"` (direction picked via
Spearman correlation on that fold's training rows only, no leakage). The numbers below are all
post-fix.

## 3. Results

### 3.1 Decision (re-verification only)

`decision_kdm_backbone`: macro-F1 = **0.593** (std 0.045, 50 folds) vs. `exp_3`'s `decision_kdm`
= 0.588. The 0.005 difference is well within repeated-CV noise for the same hyperparameters and
fold seed — the fit/predict split refactor did not change the model's behavior. As already
established (`exp_4`), KDM's own decision accuracy trails Extra Trees (0.650) by a real margin;
that gap is the cost side of this experiment's central trade, and it did not shrink or grow here.

### 3.2 Confidence — all 5 conditions worse than baseline

| Condition | Ordinal distance | vs. baseline (0.527) | vs. incumbent (0.468) |
|---|---|---|---|
| `confidence_kdm_entropy_isotonic` | 0.731 | ❌ worse | ❌ worse |
| `confidence_kdm_blend` | 0.754 | ❌ worse | ❌ worse |
| `confidence_kdm_dispersion_isotonic` | 0.776 | ❌ worse | ❌ worse |
| `confidence_kdm_participation_isotonic` | 0.844 | ❌ worse | ❌ worse |
| `confidence_kdm_entropy_zeroshot` | 1.232 | ❌ much worse | ❌ much worse |

None of Signals A (entropy), B (dispersion), or C (participation ratio) — alone or blended —
recover anything close to baseline, let alone `confidence_svm`. Two findings worth separating:

1. **The zero-shot condition's failure (1.232, more than double baseline) is itself informative,
   not just a weak result.** It tested a specific, literal version of the hypothesis raised in
   the exp_6 planning discussion — "is the schema's confidence label just decision-uncertainty,
   in the direction you'd expect (low model entropy = clinician felt clear)." The zero-shot
   condition assumed that direction outright, with no fitting at all, and did far worse than a
   majority-class guess. That's a direct answer: no, not in the naively-assumed direction, and
   not strongly even once the direction is corrected for.
2. **Even with the direction learned from data** (the isotonic conditions, `increasing="auto"`),
   performance stays well below baseline. During implementation, `sklearn` repeatedly warned that
   "the confidence interval of the Spearman correlation coefficient spans zero" for these signals
   against confidence rank on ~73-row training folds — i.e., the signal-to-confidence-rank
   correlation is frequently not statistically distinguishable from noise at this fold size. This
   is consistent with decision-uncertainty (however it's measured) and the schema's
   human-annotated confidence label being more distinct constructs than the original hypothesis
   assumed, at least at N≈73/fold.
3. **Signal B (dispersion) does not outperform Signal A (entropy)** — 0.776 vs. 0.731 — so the
   richer, previously-unused `dm_rbf_variance`-based signal did not add information beyond plain
   output entropy in this setting, despite being conceptually distinct (representation-space
   spread of supporting prototypes vs. output-class balance). Worth recording as a negative
   result specific to this project's frozen-prototype KDM configuration, per `DESIGN.md`'s own
   framing of this as a diagnostic question, not just a leaderboard entry.

### 3.3 Variable weights — one mechanism narrowly beats baseline

| Condition | Mean ordinal error | Mean decisive-set F1 | vs. baseline (0.413) | vs. incumbent (0.382/0.392) |
|---|---|---|---|---|
| `weights_kdm_occlusion` | **0.405** | 0.442 | ✅ narrowly beats | ❌ |
| `weights_kdm_kernel_distance` | 0.526 | 0.454 | ❌ | ❌ |
| `weights_kdm_blend` | 0.742 | 0.428 | ❌ | ❌ |

Only local occlusion (Signal D) beats baseline, and only narrowly (0.405 vs. 0.413 — about a
fifth the margin `weights_svm` achieves over the same baseline). Signal E (kernel-distance
contribution) and the D+E blend are both worse than baseline, despite Signal E being computationally
free (no re-inference needed) — the extra cost of occlusion's re-inference pass appears to buy
real signal, not just overhead.

**Per-factor breakdown for `weights_kdm_occlusion`** (per `DESIGN.md`'s explicit instruction not
to report only the aggregate):

| Factor | Baseline | `weights_kdm_occlusion` | Beats baseline? | `exp_5` incumbent |
|---|---|---|---|---|
| dre | 0.308 | **0.263** | ✅ beats baseline **and** exp_5's own best (0.284) | 0.284 (SVM) |
| psa | 0.451 | **0.426** | ✅ beats baseline, ties exp_5's best exactly | 0.426 (SVM) |
| bx | 0.527 | **0.451** | ✅ beats baseline | 0.420 (SVM) |
| pirads | 0.527 | **0.490** | ✅ beats baseline | 0.336 (kNN) |
| comorbidity | 0.308 | 0.316 | ❌ narrowly worse | 0.321 (not beaten) |
| cspca | 0.451 | 0.470 | ❌ worse | 0.455 (not beaten) |
| vol | 0.264 | 0.288 | ❌ worse | 0.264 (tied) |
| age | 0.396 | 0.414 | ❌ worse | 0.360 (SVM, beats baseline) |
| psad | 0.484 | 0.526 | ❌ worse | 0.486 (not beaten) |

**4 of 9 factors beat baseline** — `dre`, `psa`, `bx`, `pirads` — closely reproducing (via a
completely different mechanism: local sensitivity off a shared decision backbone, not a
per-factor-trained classifier) 4 of the 5 factors `exp_5`'s per-factor model search found
tractable. The one factor that flips is `age`: a win for `exp_5`'s SVM, a loss here. The 4
factors that remain unsolved (`cspca`, `comorbidity`, `psad`, `vol`) match `exp_5`'s own ceiling
exactly, and their `decisive_set_f1` is 0.000 for 3 of the 4 here too (`cspca`, `comorbidity`,
`psad` all 0.000; `vol` = 0.121, weak but nonzero — slightly better than `exp_5`'s flat tie on
that factor). This is a meaningful partial replication: an independent mechanism converges on
almost the same solvable/unsolvable split, which is evidence the split reflects a real property
of the training data's label distribution rather than an artifact of any one modeling approach.

## 4. Interpretation

1. **The central trade this experiment was designed to test did not pay off.** Confidence and
   weights derived "for free" from one decision-trained KDM do not match the accuracy of
   independently-trained SVM models, and confidence in particular is not close. The one exception
   — occlusion-based weights narrowly beating baseline — is real but too small a margin to
   justify preferring it over `weights_svm` for anything beyond its diagnostic value.
2. **The experiment still answers real questions it was designed to answer**, independent of
   whether the headline trade succeeded:
   - Confidence is *not* well-explained by this KDM's own decision-uncertainty, in either the
     naively-assumed direction or a data-fit one — a genuine, useful negative result about the
     relationship between model uncertainty and the schema's human-annotated confidence label.
   - The novel `dm_rbf_variance`-based dispersion signal did not outperform plain output entropy
     — the added library-native machinery didn't earn its complexity here.
   - Local occlusion, not the cheaper kernel-distance decomposition, is where whatever real
     per-case attribution signal exists in this model actually lives.
   - The solvable/unsolvable split among the 9 weight factors replicates closely across two
     structurally different mechanisms (exp_5's per-factor model search; exp_6's shared-backbone
     attribution), reinforcing that it's a property of the data, not of either method.
3. **This closes the specific hypothesis tested here** — a shared KDM backbone as a drop-in
   replacement for independent per-target models — without closing the broader idea of
   model-derived uncertainty/attribution generally. A future variant conditioning the backbone
   differently (e.g. fit jointly on decision+confidence rather than deriving confidence purely
   post-hoc from a decision-only fit) is a distinct experiment, not a retry of this one.

## 5. Recommendation

- **Do not replace `confidence_svm` or `weights_svm` with any exp_6 mechanism** for the paused
  submission-pipeline steps — the incumbents remain the best result for both targets.
- **`weights_kdm_occlusion`'s per-factor breakdown is worth keeping on record** as independent
  corroboration of `exp_5`'s 4–5-factor solvable/unsolvable split, not as a candidate for
  production use.
- No further KDM-backbone variants are recommended without a different core mechanism (e.g.
  jointly-trained rather than decision-only-derived signals) — incrementally tuning this
  experiment's recalibration layer is unlikely to close a gap this size.
