# exp_12 Report: Direct Scalar-KDM for Confidence (ARD Ablation of exp_11)

## 1. Summary

Rerunning `exp_11`'s exact direct-training protocol with the **scalar** backbone instead of ARD
answers `DESIGN.md`'s question decisively, and produces the best confidence result this project has
ever recorded: **scalar is not just equal to ARD, it's clearly better** on the 23-column frame, and
— for the first time in this project's history — **plausibly closes the gap to the `confidence_svm`
incumbent on the official metric**, not just macro-F1.

| Method | 23-col ordinal distance | 23-col macro-F1 |
|---|---|---|
| CV | 0.491 | 0.544 |
| LOO (91-fold) | **0.440** | **0.589** |
| Repeated held-out (10-seed mean) | **0.447 ± 0.175** | 0.555 ± 0.129 |
| *(reference)* Incumbent (`confidence_svm`) | 0.468 | 0.404 |
| *(reference)* Baseline | 0.527 | 0.260 |
| *(reference)* `exp_11`, direct ARD-KDM (converged) | 0.47–0.55 | 0.47–0.51 |

**All three methods clearly beat baseline. Two of three (LOO, repeated-holdout mean) numerically
beat the incumbent** — the first time any KDM-based confidence approach, direct or derived, has
gotten there on the official rubric metric. CV alone (0.491) doesn't quite. This should be reported
as a genuine, multi-method-confirmed improvement, with the honest caveat that it is not a clean
sweep across all three methods the way `exp_9`'s decision result was.

19-column is a different, noisier story: CV=0.543, LOO=0.560, repeated-holdout mean=0.479±0.142 —
these don't converge as tightly, and center closer to baseline than 23-column's do. The frame
matters, not just the architecture-vs-target-supervision question `exp_11`/`exp_12` set out to
answer.

## 2. What Was Run

Direct structural copies of `exp_11`'s three scripts, swapping `experiments/exp_9/scripts/
ard_kernel.py`'s `fit_kdm_backbone_ard`/`compute_signals_ard` for `experiments/exp_6/scripts/
kdm_backbone.py`'s `fit_kdm_backbone`/`compute_signals` (scalar `σ`) — confirmed generic over
`n_classes` by the same smoke test discipline before any scored run. No plan-mode cycle this round
(small, mechanical ablation of just-completed, just-validated code — `DESIGN.md` §1). Same four
methods as `exp_11`: CV, held-out, LOO, and — triggered again by a suspicious identical-across-both-
frames held-out result (§3) — a repeated-holdout follow-up, run proactively per the standing
practice `exp_11` established rather than waited for.

## 3. The held-out split repeated its exp_11 pattern — checked again, resolved differently this time

Seed=0's held-out check again produced an identical ordinal distance for both frames — **0.368**
this time (`exp_11`'s scalar-adjacent check had shown 0.316). Two experiments in a row now showing
this exact "both frames agree exactly" signature on the same fixed split. Per the standing rule,
a 10-seed repeated-holdout check was run immediately rather than treating 0.368 as a finding.

| Frame | CV | Held-out, seed=0 | Held-out, 10-seed mean±std | LOO |
|---|---|---|---|---|
| 19-col | 0.543 | 0.368 | 0.479 ± 0.142 | 0.560 |
| 23-col | 0.491 | 0.368 | 0.447 ± 0.175 | 0.440 |

Unlike `exp_11` (where the repeated-holdout mean settled *back toward* CV/LOO, revealing 0.316 as
purely a lucky outlier), this time the picture is more nuanced: **23-column's repeated-holdout mean
(0.447) lands close to LOO (0.440), not close to CV (0.491)** — both now below baseline and close
to incumbent. Seed=0's 0.368 was still on the favorable side (best or near-best of the 10 seeds
for both frames, per-seed detail in `results/repeated_holdout_confidence_direct_scalar/
metrics.json`), but this time the *other* two independent methods (LOO, repeated-holdout mean)
corroborate a genuinely strong result rather than debunking it outright. The lesson from `exp_10`/
`exp_11` — check before trusting a suspiciously clean single-split number — held again; this time
the check *confirmed* rather than *refuted* the promising signal, which is exactly why the check
has to run every time, not just when a result looks bad.

## 4. Full results, all four methods, both frames

| Method | Frame | Accuracy | Macro-F1 | Ordinal distance | AUROC (OvR macro) | Brier |
|---|---|---|---|---|---|---|
| CV | 19-col | 0.618 | 0.500 | 0.543 | 0.696 | 0.628 |
| CV | 23-col | 0.642 | 0.544 | 0.491 | 0.723 | 0.561 |
| Held-out (seed=0) | 19-col | 0.737 | 0.504 | 0.368 | 0.664 | 0.498 |
| Held-out (seed=0) | 23-col | 0.737 | 0.504 | 0.368 | 0.744 | 0.433 |
| Held-out (10-seed mean) | 19-col | — | 0.518 ± 0.105 | 0.479 ± 0.142 | 0.741 | 0.577 |
| Held-out (10-seed mean) | 23-col | — | 0.555 ± 0.129 | 0.447 ± 0.175 | 0.771 | 0.533 |
| LOO (91-fold) | 19-col | 0.626 | 0.514 | 0.560 | 0.700 | 0.633 |
| LOO (91-fold) | 23-col | **0.681** | **0.589** | **0.440** | **0.731** | 0.538 |

`exp_12`'s LOO on the 23-column frame (macro-F1 0.589) is the **best macro-F1 confidence result
this project has ever produced** — beating `exp_11`'s own LOO best (0.509) by a wide margin, and far
beyond every derived-signal condition (`exp_9`'s best, 0.283).

## 5. Comparison against everything else tried for confidence

| Approach | Ordinal distance | Macro-F1 |
|---|---|---|
| Baseline | 0.527 | 0.260 |
| **Incumbent (`confidence_svm`)** | **0.468** | 0.404 |
| `exp_3`, direct scalar-KDM (worse frame) | 0.530 | 0.508 |
| Best derived-signal, any experiment (`exp_9` ARD `blend`) | 0.836 | 0.283 |
| `exp_11`, direct ARD-KDM, converged (both frames) | 0.47–0.55 | 0.47–0.51 |
| **`exp_12`, direct scalar-KDM, 23-col, converged** | **0.44–0.49** | **0.54–0.59** |

Per `DESIGN.md` §5's decision rules: this is the **third branch** — `exp_12` is clearly *better*
than both `exp_11`'s ARD result and `exp_3`'s original, not merely tied with either. ARD's extra
per-dimension parameters appear to have been actively unhelpful for this target at this N=91,
21–23-column scale, not just neutral — the plain scalar backbone, freed of ARD's larger parameter
count, generalizes better here. This is a clean, useful negative result for ARD specifically on
confidence, in contrast to `exp_9`'s clear positive result for ARD on decision — the same
architecture change helps one target and appears to hurt another, on the same data.

## 6. Interpretation

Three findings, layered:
1. **Direct supervision beats deriving signals from the decision backbone** — confirmed again,
   even more strongly than in `exp_11`.
2. **Scalar beats ARD for this specific target** — the opposite of decision's story. A plausible
   read: confidence's 3-class target, or its interaction with the 23-column frame, doesn't reward
   ARD's ~19–23 extra trainable `σⱼ` parameters at N=91 the way decision's binary target did;
   more parameters without a correspondingly larger gradient signal (91 cases, 3-way split vs.
   binary) may simply add variance here rather than resolving it.
3. **The 23-column frame specifically, not 19-column, carries this result.** Both `exp_11` and
   `exp_12` show 23-column converging more tightly and scoring better than 19-column — echoing
   `exp_9`'s decision finding that the 23-column frame is genuinely richer, not just wider.
4. **This is the closest any KDM approach — direct or derived, scalar or ARD — has ever come to
   beating `confidence_svm` on its own metric.** Not a clean sweep (CV alone doesn't beat
   incumbent), but two of three independent methods do, which is meaningfully stronger evidence
   than any single number this project has produced for confidence before.

## 7. Recommendation

- **Update confidence's best-known KDM approach**: direct scalar-KDM training on the 23-column
  frame, not ARD — reverse of decision's recommendation, and worth stating explicitly so a future
  session doesn't assume ARD is a universal upgrade.
- **This result is strong enough to be worth a leaderboard-facing comparison against
  `confidence_svm` head-to-head on genuinely new data**, if this project reaches that stage —
  it's the first KDM confidence result credible enough to warrant that comparison.
- **Do not yet claim a definitive win over the incumbent** — CV alone doesn't clear that bar, and
  three methods giving two different verdicts (beats / doesn't-quite-beat) is itself information,
  not noise to average away, per this project's standing discipline.
- A natural, cheap follow-up: rerun `exp_5`'s equivalent test (direct KDM for **weights**, which
  scored below baseline) on the current 23-column frame — `exp_5` predates every backbone
  improvement since, the same way `exp_3`'s confidence number did before `exp_11`/`exp_12` revisited
  it.
