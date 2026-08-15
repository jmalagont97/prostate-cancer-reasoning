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

📊 **2026-08-12 macro-F1 reporting initiative**: `exp_6`/`exp_7`/`exp_8`'s confidence/weights/
reveal results were backfilled with macro-F1 (confidence: 3-class; weights: per-factor 4-class
averaged across 9 factors; reveal: per-section binary F1 macro-averaged across 4 modeled
sections), reported alongside each subtask's official rubric metric, which remains primary. See
each experiment's `results/*/metrics.json` (`macro_f1_mean`/`mean_macro_f1` fields) and updated
`reports/summary.md`. `exp_9` onward reports macro-F1 natively from the start.

📈 **2026-08-13 full metric-suite reporting initiative** (started as decision-only AUROC + Brier,
expanded same day to a project-wide standard, confirmed by the user): **every experiment from
`exp_11` onward reports this full metric set per subtask**, each subtask using the natural
definition of each metric for its own target type — no subtask skips a metric that applies to it,
and none force-fits one that doesn't:

| Subtask (target type) | Accuracy | Macro-F1 | AUROC | Ordinal distance/error | Brier score |
|---|---|---|---|---|---|
| Decision (binary) | `accuracy_score` | `f1_score(average="macro")` | `roc_auc_score(y_true, p[:,1])` | *(N/A — equals 1−accuracy for binary, redundant)* | `brier_score_loss(y_true, p[:,1])` |
| Confidence (3-class ordinal) | `accuracy_score` | `f1_score(average="macro", labels=[0,1,2])` | one-vs-rest macro: `roc_auc_score(y_true, proba, multi_class="ovr", average="macro")` | `ordinal_distance()` (existing, official metric, unchanged) | multiclass: mean over samples of `Σ_k (p_k − 1[y=k])²` |
| Weights (4-class ordinal, per factor) | `accuracy_score` | `f1_score(average="macro", labels=[0,1,2,3])` | one-vs-rest macro (same formula as confidence) | `ordinal_distance()` → reported as `ordinal_error` (existing, official metric, unchanged) | multiclass (same formula as confidence) |
| Reveal (multi-label, per section) | *(N/A — set-precision is this target's natural summary)* | per-section binary F1, macro-averaged (existing) | *(no natural single-target fit — a multi-label target, not single-label classification)* | *(N/A)* | *(N/A)* |

CV/repeated-holdout aggregation for every new metric matches macro-F1's own established convention
(per-repeat-pooled, then mean±std across repeats — never a per-fold average, which is noisy/
ill-defined on ~14–18-row folds); one-shot held-out/LOO checks report a single value. Multi-class
AUROC needs every class present in the scored set (or an explicit `labels=` argument) — per-factor
weights folds with a rare class (already a known issue, see `exp_5`'s `ValueError`-catch precedent)
skip and log rather than crash, same discipline as every other per-factor metric in this project.

This originated as a decision-only AUROC + Brier addition (`exp_9`/`exp_10`'s decision results were
backfilled — `backfill_decision_auroc_brier.py` in each experiment's `scripts/`, reusable templates)
after they surfaced a concrete failure mode macro-F1 alone missed: on `exp_10`'s lucky held-out
split, AUROC/Brier were *already* worse than `exp_9`'s reference on that exact split, while macro-F1
(argmax-threshold-dependent) briefly looked like a new best result — see
`experiments/exp_9/reports/summary.md` §3 and `experiments/exp_10/reports/summary.md` §3c. The
confidence/weights columns above are **not yet backfilled into `exp_6`–`exp_10`** — this note
documents the confirmed going-forward standard for `exp_11`+; backfilling accuracy/AUROC/Brier into
confidence and weights for `exp_9`/`exp_10` specifically is a separate, not-yet-requested task.

📐 **2026-08-13 leave-one-out (LOO) evaluation protocol, added not substituted**: **every
experiment from `exp_11` onward reports LOO (91-fold, deterministic, single pass) *in addition to*
the existing 5-fold × 10-repeat CV and the mandatory held-out check — LOO does not replace either.**
The three protocols answer different questions and all three stay: CV compares many candidate
conditions cheaply during development; the held-out split is the one number never touched by any
model-selection decision; LOO adds a third, maximal-training-data (90/91 per fold), deterministic
estimate immune to a single split's sampling luck. Motivated by `exp_10`'s finding that a single
19-case held-out split can land ±0.13+ from a repeated-split mean (§3b of that report) — LOO gave
the same warning without needing 10 repeated splits to surface it.

- **Aggregation**: pool every subtask's out-of-fold predictions across all 91 LOO folds into one
  confusion matrix / prediction set, then score once — never a per-fold average (a single test row
  has no meaningful macro-F1, AUROC, etc. on its own). Same full metric suite as the note above
  applies (accuracy, macro-F1, AUROC, ordinal distance/error, Brier score, per subtask's natural
  subset) — a single value per metric, no repeat variance to report (LOO has none).
- **Cost, stated plainly**: LOO is 91 folds vs. CV's 50 (5×10) — for weights specifically (9 factors
  × several conditions × up to 2 frames) this is a real, non-trivial addition, not free. Decision
  already has LOO from `exp_10`'s verification
  (`experiments/exp_10/scripts/verify_decision_loo_repeated_holdout.py`, reusable template — its
  `run_loo()` function generalizes directly). Confidence/weights/reveal do not have LOO yet for
  any experiment — this is a confirmed going-forward standard for `exp_11`+, not yet backfilled.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | Hybrid ML Baseline for CHIMERA-Agent Task 1 | Complete | Small supervised models on structured features beat naive per-target baselines closely enough to justify submission infra | ❌ Refuted (decision/confidence/weights) — ✅ Supported (reveal-sequence only) ⚠️ decision numbers corrected 2026-08-10, verdict direction unchanged | 2026-08-08 |
| [exp_2](exp_2/DESIGN.md) | Official-Schema Feature Scope + Comorbidity Grouping + KDM Decision Model | Complete | Restricting to the 11 officially-documented Task-1 input variables + grouped comorbidity flags + a KDM decision classifier beats exp_1's baselines | ⚠️ Decision: modest schema-restriction lift but no longer beats baseline; **KDM now best decision model** (reversed from ❌) — ❌ Confidence — ✅ Reveal ⚠️ corrected 2026-08-10 | 2026-08-08 |
| [exp_3](exp_3/DESIGN.md) | Broader Model Family Comparison + MRI-PCA + Decorrelated PSA Family | Complete | An 8-model search (SVM/RF/XGBoost/ExtraTrees/MLP/NaiveBayes/kNN/KDM) + 2-component MRI-PCA + psa/psad-only PSA family clearly beats exp_1/exp_2's best per target | ⚠️ Decision: kNN/SVM/KDM lead (not Extra Trees), still below baseline (reversed from ✅) — ✅ Confidence (SVM, first-ever baseline beat, unaffected) — ❌ Weights — ⚠️ Reveal (flat) ⚠️ corrected 2026-08-10 | 2026-08-09 |
| [exp_4](exp_4/DESIGN.md) | Clinical-Only Features (No MRI-PCA) — exp_3 Ablation | Complete | Removing MRI-PCA from exp_3's frame doesn't flip any headline verdict, but has real per-model effects (target-dependent) | ⚠️ MRI not decisive for verdicts, but decisive (positively) for MLP/NB on decision and decisive (negatively) for confidence overall | 2026-08-10 |
| [exp_5](exp_5/DESIGN.md) | 8-Model Search for Variable-Weight Prediction | Complete | Extending exp_3's 8-model search to weights (official + restricted scope) beats the incumbent best and ideally the naive baseline | ✅ Weights (SVM, first-ever baseline beat — 0.382/0.392 vs. 0.413) | 2026-08-10 |
| [exp_6](exp_6/DESIGN.md) | KDM as a Unified Probabilistic Backbone for Decision + Confidence + Weights | Complete | One trained KDM's own predictive uncertainty (output entropy + prototype-neighborhood dispersion, via `kdm.utils.dm_rbf_variance`) and local per-factor sensitivity can match or beat confidence_svm (0.468) and weights_svm (0.382/0.392) with no separate model family per target | ❌ Confidence (all 5 signal variants worse than baseline) — ⚠️ Weights (occlusion narrowly beats baseline, 0.405 vs 0.413, but not the incumbent; replicates exp_5's 4/9-factor solvable split via an independent mechanism) — ✅ decision backbone reproduces exp_3 (0.593 vs 0.588) | 2026-08-10 |
| [exp_7](exp_7/DESIGN.md) | Improving exp_6's KDM Backbone (Hyperparameter Tuning + Skew-Aware Preprocessing) | Complete | A bounded, low-risk grid search over epochs/lr/sigma_mult plus log1p-transforming exp_6's most-skewed columns (psa skew=4.28, psad skew=4.24, vol skew=1.29) closes some of the gap between KDM's own decision macro-F1 (0.593) and Extra Trees (0.650), and possibly improves exp_6's derived confidence/weights signals too, without touching KDM's architecture | ❌ Refuted — 144-way search found a config that beat baseline under CV (0.622 vs 0.593) but scored worse on genuine held-out data (0.490 vs 0.593, held-out check caught the spurious CV win as designed); confidence/weights unchanged | 2026-08-11 |
| [exp_8](exp_8/DESIGN.md) | Expanded Variables + Hyperparameter Tuning + Reveal-Sequence via Uncertainty Reduction | Complete | Restoring cli_psav/cli_psap (external CHIMERA-challenge reference), cli_isup and vit_bmi (own EDA, strongest decision/confidence correlates checked) expands exp_6's frame to 23 columns; re-attempts exp_7's 144-combo hyperparameter search on this new frame (mandatory held-out check + std-relative margin this time, correcting exp_7's stated lessons) with a 3-way features-vs-tuning ablation; and derives reveal-sequence from the same backbone via per-section entropy-increase-from-occlusion, a 4th target for the first time | ❌ Decision inconclusive (CV says no improvement, 0.585-0.591 vs 0.593; held-out says +0.042 — the two disagree, a finding in itself) — ❌ Confidence slightly worse (4/5 conditions) — ⚠️ Weights flat (0.412 vs 0.405, margin near noise) but psad's decisive-F1 moved 0.000→0.254 — ✅ **Reveal-sequence beats baseline on first attempt** (0.799 vs 0.783), a new target for the shared backbone | 2026-08-12 |
| [exp_13](exp_13/DESIGN.md) | Direct KDM for Weights — Backbone and Scope Ablation | Complete | Reviving `exp_5`'s direct-training precedent for **weights** (never beat baseline: 0.478/0.454 official/restricted vs. 0.413) the way `exp_11`/`exp_12` revived it for confidence — now with the 23-column frame, testing both backbones (scalar, ARD) × both scopes (official, restricted) as 4 parallel conditions in one experiment, since weights has no prior pointing to which combination might work. Staged LOO (only for conditions that clear CV/held-out first) given the scale (4 conditions × 9 factors) | ❌ Clean negative, unlike confidence's revival — all 4 conditions converge tightly on almost exactly `exp_5`'s original numbers (restricted CV reproduces 0.454 to 3 decimals; official CV within 0.006 of 0.478); scalar vs. ARD backbone makes no meaningful difference here (opposite of confidence's story); none beat baseline (0.413) or `weights_svm` incumbent (0.382/0.392) — 🔍 a suspiciously clean held-out tie to baseline (restricted scope) was repeated-holdout-checked (10 seeds) and confirmed as noise (9/10 seeds above baseline, mean 0.438–0.441); full LOO skipped per staged criterion since no condition approached the bar — `weights_kdm_occlusion` (`exp_6`) remains the best KDM weights approach; direct training's revival helps confidence specifically, not weights | 2026-08-15 |
| [exp_12](exp_12/DESIGN.md) | Direct Scalar-KDM for Confidence (ARD Ablation of exp_11) | Complete | Reran `exp_11`'s exact direct-training protocol with the scalar backbone (`exp_6`'s `fit_kdm_backbone`) instead of ARD, to isolate whether the 23-column frame or ARD explained `exp_11`'s improvement over `exp_3`'s original result. Small, mechanical ablation of just-validated code — no separate plan-mode cycle | ✅ Scalar clearly beats ARD on the 23-col frame (ordinal distance 0.44–0.49 across CV/LOO/repeated-holdout vs. `exp_11` ARD's 0.47–0.55) — 🏆 LOO (0.440) and repeated-holdout mean (0.447) both numerically beat the `confidence_svm` incumbent (0.468) for the first time in this project, though CV alone (0.491) doesn't quite — new project-best macro-F1 (0.589, LOO) — 🔍 a single held-out split again showed an identical, suspicious value across both frames (0.368); this time the repeated-holdout follow-up *confirmed* the promising result rather than refuting it — ARD helps decision but hurts confidence on the same data, a genuinely useful negative result for ARD specifically here | 2026-08-14 |
| [exp_11](exp_11/DESIGN.md) | Direct ARD-KDM for Confidence | Complete | Reviving `exp_2`/`exp_3`'s pre-`exp_6` approach — a KDM trained *directly* on the confidence label (not a signal derived from the decision-trained backbone) — now combined with the two real backbone improvements confirmed since (`exp_9`'s ARD, the 23-column frame). `exp_3`'s original directly-trained `confidence_kdm` already holds this project's best-ever confidence macro-F1 (0.508) despite using the worst available backbone; nothing has retried it with ARD. First experiment under both new reporting conventions (full metric suite + mandatory LOO) | ✅ Decisive, 4-method-confirmed macro-F1 win over every derived-signal condition since `exp_6` (0.47–0.51 vs. prior best 0.283) — ❌ ordinal distance converges to baseline (0.527), not below incumbent (0.468) — direct **training** was the active ingredient, not ARD (a wash vs. `exp_3`'s original scalar direct-trained result, 0.530/0.508) — 🔍 a single held-out split showed an identical, suspicious 0.316 for both frames; an unplanned repeated-holdout follow-up confirmed it was a lucky outlier (10-seed mean 0.47–0.52, matching CV/LOO) — second confirmation of `exp_10`'s lucky-split lesson, this time caught proactively | 2026-08-14 |
| [exp_10](exp_10/DESIGN.md) | External Full-Schema Replication (Frame.md's 37 Variables + MRI-PCA(2)) under ARD-KDM | Complete | Adopting `jmalagont97/prostate-cancer-reasoning`'s complete 37-variable tabular schema (pasted into `Frame.md` this session) + MRI-PCA(2) = 48 encoded columns (after dropping 2 exact-duplicate PSA-trend columns found during EDA; verified programmatically after two manual counting errors), using that repo's own preprocessing convention (MinMax + one-hot + missing-flags, not this project's median-imputation) and `exp_9`'s ARD-KDM backbone — tests whether the external full schema beats `exp_9`'s already-curated 23-column reference (0.680 held-out decision macro-F1) | ❌ CV: uniform regression across all 9 conditions (decision -0.060 vs. `exp_9`); held-out initially looked like the **best decision result ever** (0.708) but a same-session leave-one-out + 10-repeated-holdout follow-up (`verify_decision_loo_repeated_holdout.py`) confirmed it was a lucky single split (repeated mean 0.509±0.134, LOO 0.530 — both agree with CV) — ✅/❌ `dispersion_isotonic`'s dilution regression, closed by ARD at 23 columns, **re-opens at 48** (0.804→1.410 ord.dist.) — a real ARD capacity ceiling found — ❌ Weights and reveal both regressed uniformly; reveal lost to naive baseline for the first time — ❌ Importance-vs-`exp_5` agreement stayed 2/5 (3rd frame in a row) — not adopted; `exp_9`'s 23-col frame remains best-validated, now further confirmed by the same LOO/repeated-holdout check | 2026-08-13 |
| [exp_9](exp_9/DESIGN.md) | ARD (Per-Dimension) Kernel Bandwidth for the KDM Backbone | Complete | Replacing KDM's single shared kernel bandwidth with one trained σ per input dimension, tested on both the 19-column (exp_6) and 23-column (exp_8) frames, directly motivated by exp_8's measured +0.309 confidence degradation when the frame expanded under a scalar bandwidth. Macro-F1 reported natively across all four subtasks from the start. No hyperparameter search this round (deliberate scope cut); mandatory held-out check carried forward from exp_7/exp_8 | ✅ Central mechanism confirmed — `dispersion_isotonic`'s regression collapses (0.776→1.085 scalar, 0.797→0.804 ARD) — ✅ Decision, 23-col, held-out-confirmed (+0.087 vs exp_6); ❌ Decision, 19-col, CV win reverses on held-out (exp_7's failure mode repeated); cleanest frame-only comparison run yet shows 23-col beating 19-col under ARD (+0.128 held-out) but not under CV, a 3rd straight CV/held-out disagreement — ⚠️ Confidence mixed (`dispersion_isotonic` rescued, `entropy_isotonic` worsened) — ❌ Weights: loses the one narrow baseline win (`occlusion`) on both frames — ✅ Reveal improves on both metrics, 23-col (0.799→0.823 set-precision, 0.531→0.599 macro-F1); first 19-col result — ❌ Importance-vs-exp_5 secondary hypothesis not corroborated (2/5 agreement, both frames) | 2026-08-12 |
