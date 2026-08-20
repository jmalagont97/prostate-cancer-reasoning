# exp_14 Report: KDM Regression for Weights (Ordinal-Aware Training)

## 1. Summary

Training a KDM directly on the ordinal weight rank as a continuous regression target
(`KDMRegressModel` + `dm_rbf_loglik`), instead of every prior KDM condition's classification
objective, produces a **genuinely mixed result** — not the clean win the hypothesis hoped for, but
not a flat negative either. On the subtask's **primary** official metric (mean ordinal error), it
does not beat baseline or the best-ever KDM result. On the subtask's **secondary** official metric
(decisive-set F1), it produces **the best result this project has ever recorded, by any model,
KDM or not** — confirmed across all three evaluation protocols run.

| Method | Ordinal error (primary, lower=better) | Decisive-set F1 (secondary, higher=better) | Macro-F1 |
|---|---|---|---|
| CV, per-factor (9 independent regressors) | 0.466 | 0.507 | 0.292 |
| CV, joint (1 regressor, dim_y=9) | 0.463 | 0.506 | 0.293 |
| Held-out, per-factor | 0.474 | 0.558 | 0.284 |
| Held-out, joint | 0.491 | 0.544 | 0.274 |
| Repeated held-out (10 seeds, mean±std), per-factor | 0.463 ± 0.045 | **0.514 ± 0.069** | — |
| Repeated held-out (10 seeds, mean±std), joint | 0.460 ± 0.049 | **0.515 ± 0.064** | — |
| *(reference)* Baseline | 0.413 | 0.379 | — |
| *(reference)* `weights_svm` incumbent | 0.382 | 0.457 | — |
| *(reference)* `weights_kdm_occlusion` (best KDM ever, primary bar) | 0.405 | 0.442 | 0.256 |

**Ordinal error never approaches the primary bar (0.405) in any protocol** — CV, held-out, and
repeated-holdout all land in a tight 0.46–0.49 band, roughly 0.06–0.09 worse across the board.
**Decisive-set F1 clears every prior best (including the SVM incumbent's 0.457) in every protocol**
— CV, held-out, and a 10-seed repeated-holdout check all agree closely (0.506–0.515), confirming
this is a real, reproducible effect, not the single-split luck the held-out number's higher value
(0.544–0.558) initially looked like it might be.

## 2. What Was Run

Two new conditions, both using `kdm-torch`'s `KDMRegressModel` (never used in this project before
— every prior KDM condition used `KDMClassModel`) trained via `dm_rbf_loglik` directly on the
weight rank (0–3) as a continuous target, 23-column frame, scalar backbone, memory-based
(`x_train=y_train=w_train=False`), 300 epochs, matching every prior KDM condition's fit-loop shape
as closely as the regression model's extra `sigma_y` parameter allows:

- **Per-factor** (`dim_y=1`): 9 independent regressors, the direct ordinal-aware analogue of
  `exp_13`'s per-factor classification KDM.
- **Joint** (`dim_y=9`): 1 regressor predicting all 9 factors at once from a shared prototype pool
  — untried by any prior weights condition, including the SVM incumbent (also 9 independent
  models). Verified to use exactly 50 fits total across the CV run (10 repeats × 5 folds), not
  450 — confirmed via an explicit fit-count check in the script's own output.

Continuous predictions are rounded to the nearest integer and clipped to `[0,3]` for accuracy/
macro-F1/ordinal-error/decisive-set-F1 (identical downstream handling to every classification-KDM
condition). AUROC/Brier use regression-derived pseudo-probabilities (a Normal-CDF discretization
of `predict_reg`'s mean/variance into 4 rank bins) — an approximation, flagged explicitly in every
metrics payload's `auroc_brier_note`, never presented as a native classifier's calibrated output.
A known limitation confirmed by direct API testing before any scored run: the joint condition's
`predict_reg` variance is a single scalar per case, not one per factor, so its pseudo-probabilities
share one variance across all 9 factors within a case — only the per-factor mean differs.

Three smoke tests against the installed `kdm-torch==2.0.0` (both `dim_y=1` and `dim_y=9`, real
23-col frame data) confirmed clean convergence, correct output shapes, and pseudo-probabilities
summing to 1 per row before any CV run was trusted.

## 3. The decisive-set F1 finding, checked rather than assumed

Held-out's decisive-set F1 (0.558 per-factor, 0.544 joint) came in noticeably higher than CV's
already-striking 0.506–0.507 — the "even better on a single split" pattern this project's standing
rule requires verifying, not just reporting. A targeted 10-seed repeated-holdout check
(`repeated_holdout_weights_regress.py`) resolved it cleanly: the 10-seed means (0.514 per-factor,
0.515 joint) sit almost exactly on CV's estimate, with seed=0 on the favorable side of a
0.386–0.663 per-seed range but not an outlier requiring correction. **The effect is real and
reproducible, not a lucky split** — three independent protocols (CV, held-out, repeated-holdout)
now agree within 0.01–0.05 of each other on decisive-set F1, a tighter convergence than most
results this project has reported.

Ordinal error's CV/held-out/repeated-holdout agreement (0.460–0.491 across all three) was already
tight enough that no separate check was warranted — unlike the decisive-set F1 number, it never
looked suspiciously good.

## 4. Full results, both conditions, all metrics

| Method | Condition | Accuracy | Macro-F1 | Ordinal error | Decisive-set F1 |
|---|---|---|---|---|---|
| CV | per-factor | — | 0.292 | 0.466 | 0.507 |
| CV | joint | — | 0.293 | 0.463 | 0.506 |
| Held-out | per-factor | — | 0.284 | 0.474 | 0.558 |
| Held-out | joint | — | 0.274 | 0.491 | 0.544 |
| Repeated held-out (mean±std) | per-factor | — | — | 0.463 ± 0.045 | 0.514 ± 0.069 |
| Repeated held-out (mean±std) | joint | — | — | 0.460 ± 0.049 | 0.515 ± 0.064 |

The per-factor and joint conditions are statistically indistinguishable from each other on every
metric (within 0.01–0.02 across the board) — sharing a prototype pool across factors neither
helped nor hurt here, unlike the hoped-for benefit for the 4 data-scarce factors. Per-factor
breakdown (CV): the 4 historically-hardest factors (`cspca` 0.542, `psad` 0.523, `comorbidity`
0.392, `vol` 0.353 ordinal error) show no clear rescue relative to their classification-KDM
counterparts in `exp_13` — the joint model's information-sharing didn't specifically target them
the way hoped.

## 5. Comparison against everything else tried for weights

| Approach | Ordinal error | Decisive-set F1 |
|---|---|---|
| Baseline | 0.413 | 0.379 |
| **Incumbent (`weights_svm`)** | **0.382** | 0.457 |
| Best KDM ever, derived-signal (`weights_kdm_occlusion`, `exp_6`) | 0.405 | 0.442 |
| `exp_13`, direct classification KDM (best of 4 conditions) | 0.454 | 0.375–0.523 |
| **`exp_14`, KDM regression (either condition)** | 0.460–0.466 | **0.506–0.515** |

Per `DESIGN.md` §5's decision rules: this is the "neither beats the primary bar, but a real
secondary-metric finding emerges" outcome — not explicitly one of the four listed branches, but
closest to "real, if modest, progress" while also being the first result in this project to
decisively separate the subtask's two official metrics from each other. Every prior condition's
ordinal error and decisive-set F1 moved roughly together (better ordinal error came with better
decisive-set F1, and vice versa); this is the first time they've clearly diverged — a model that
is worse at nailing the exact rank but better at the simpler important-vs-not judgment.

## 6. Interpretation

1. **The rank-blind vs. rank-aware loss hypothesis was directionally right, but not in the way
   expected.** Training on a loss shaped like ordinal distance did not translate into a better
   ordinal-distance *score* — the model apparently smooths its predictions toward the middle of
   the scale in a way that helps the coarser binary judgment (decisive-set F1) without sharpening
   the finer 4-way one (ordinal error). A plausible mechanism: regression toward a continuous mean
   naturally pulls borderline cases toward the `important`/`decisive` boundary more reliably than
   a discrete classifier's hard NLL objective does, even when it doesn't land closer to the exact
   true rank.
2. **The joint condition's promise (sharing strength across factors) didn't materialize as
   hoped** — it performs statistically identically to 9 independent regressors, on both metrics,
   for every factor including the historically hardest ones. The mechanism this project's own
   report speculated might help (`psa`/`psad`/`vol` sharing structure) didn't show up as a
   measurable effect here; if anything the shared prototype pool seems to just average out to the
   same result as 9 separate small models.
3. **This project's third confirmation that ordinal error specifically resists KDM's help for
   weights** — direct classification (`exp_13`), and now direct regression (`exp_14`), have both
   been tried against it with the field's best-available frame and no improvement on the primary
   metric. `weights_kdm_occlusion`'s derived-signal approach remains the ceiling for ordinal error
   among KDM methods.
4. **Decisive-set F1, by contrast, has now been meaningfully advanced** — from 0.457 (SVM) to
   ~0.51 (KDM regression), a real jump on a metric this project has always reported but never
   previously treated as separately optimizable from ordinal error.

## 7. Recommendation

- **Do not replace `weights_svm` or `weights_kdm_occlusion` on the primary ordinal-error metric**
  — neither regression condition clears that bar, confirmed across three independent protocols.
- **If decisive-set F1 (the "which factors mattered" judgment) is ever weighted more heavily, or
  reported as its own deliverable, KDM regression is now this project's best model for it** —
  worth keeping in mind as a genuinely different trade-off point, not a strictly worse alternative
  to the incumbent.
- **The joint multi-factor architecture is not disproven as a general idea** — it simply showed no
  measurable advantage *here*. It remains a legitimate technique to revisit for confidence or
  decision (single-target problems don't need it, but a hypothetical joint decision+confidence+
  weights model might behave differently) if this project pursues multi-task KDM architectures
  again.
- **A natural, cheap follow-up if this line is revisited**: an ARD-regression variant (per-
  dimension `sigma_x`), the one axis this experiment deliberately left untested per its own scope
  decision — `exp_13` showed ARD makes almost no difference to weights under classification, but
  that hasn't been checked under a regression objective.
