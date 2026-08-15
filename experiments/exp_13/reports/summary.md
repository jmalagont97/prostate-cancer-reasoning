# exp_13 Report: Direct KDM for Weights — Backbone and Scope Ablation

## 1. Summary

Unlike `exp_11`/`exp_12`'s revival of direct training for confidence — which produced this
project's best-ever confidence result — **reviving direct KDM training for weights on the modern
23-column frame does not move the needle at all**. All 4 conditions (2 backbones × 2 scopes)
converge tightly on almost exactly `exp_5`'s original pre-modernization numbers, and none of them
approach baseline, let alone the `weights_svm` incumbent. This is a clean negative result, confirmed
independently by CV, held-out, and a repeated-holdout follow-up.

| Condition | CV ordinal error | Held-out (seed=0) | Held-out 10-seed mean | *(reference)* `exp_5` original |
|---|---|---|---|---|
| scalar × official | 0.483 | 0.462 | — | 0.478 |
| scalar × restricted | 0.454 | 0.413 | **0.438 ± 0.049** | 0.454 |
| ARD × official | 0.484 | 0.456 | — | 0.478 (backbone didn't exist yet) |
| ARD × restricted | 0.454 | 0.421 | **0.441 ± 0.049** | 0.454 (backbone didn't exist yet) |
| *(reference)* Baseline | **0.413** | **0.413** | **0.413** | 0.413 |
| *(reference)* Incumbent (`weights_svm`), scope-matched | 0.382 / 0.392 | — | — | 0.382 / 0.392 |

**Backbone choice (scalar vs. ARD) makes almost no difference** — the two backbones land within
0.001–0.005 of each other on every metric, in sharp contrast to confidence's story (`exp_11`/
`exp_12`), where scalar vs. ARD was the entire finding. **Frame modernization (19→23 columns) also
doesn't help** — restricted-scope's CV (0.454) exactly reproduces `exp_5`'s original restricted
number to three decimal places, and official-scope's CV (0.483–0.484) sits within 0.006 of `exp_5`'s
original 0.478. Direct KDM training simply does not benefit weights the way it benefited confidence.

## 2. What Was Run

Four parallel conditions (`run_weights_direct_scalar.py`, `run_weights_direct_ard.py` — CV, both
scopes, all 9 in-scope factors, 23-column frame only), each with a mandatory held-out check
(`holdout_eval_weights_direct_{scalar,ard}.py`, same fixed decision-stratified split used since
`exp_3`). Per `DESIGN.md` §2d's staged-execution plan, full LOO (91-fold × 4 conditions × 9 factors
= 3,276 fits) was to run only for conditions that beat or approached baseline on CV/held-out — see
§3 below for why that bar was never cleared, so LOO was not run for any condition. Two factors
(`pirads`, `bx`) degenerate-fit under restricted scope in every condition (near-zero-variance
single-column groups, same failure mode `exp_5` first documented) and are excluded from
restricted-scope aggregates (7/9 factors).

## 3. A suspicious held-out tie, checked and resolved as noise

Restricted scope's seed=0 held-out split showed ordinal error tying baseline exactly for scalar
(0.413) and coming close for ARD (0.421) — while both backbones' own CV numbers sat at 0.454,
clearly worse than baseline. This is precisely the "suspiciously clean single-split result" pattern
this project's standing rule requires checking (`exp_10`'s original lesson, reused by `exp_11`/
`exp_12`). Rather than commit to the much more expensive full-LOO staging path for a result that
might just be one lucky split, a cheap targeted `repeated_holdout_weights_direct_restricted.py`
(10 seeds, both backbones, restricted scope only — ~140 fits total, not 3,276) ran instead.

**Result: the tie was noise.** Nine of ten seeds land above baseline (range 0.383–0.556); seed=0 was
the most favorable draw of the ten for both backbones. The 10-seed means (scalar 0.438 ± 0.049, ARD
0.441 ± 0.049) both sit clearly above baseline and land close to CV's independent 0.454 estimate —
CV, held-out mean, and CV converge; only the single unlucky-favorable seed=0 split disagreed. This
confirms the negative verdict rather than complicating it, and justifies skipping full LOO per
§2d's staged criterion: no condition, at any evaluation granularity, approaches the bar that would
have warranted the 91-fold commitment.

## 4. Full results, all 4 conditions

| Condition | Method | Ordinal error | Decisive-set F1 | Macro-F1 | Factors incl. |
|---|---|---|---|---|---|
| scalar × official | CV | 0.483 | 0.509 | 0.300 | 9/9 |
| scalar × official | Held-out | 0.462 | 0.537 | 0.281 | 9/9 |
| scalar × restricted | CV | 0.454 | 0.375 | 0.277 | 7/9 |
| scalar × restricted | Held-out | 0.413 | 0.398 | 0.271 | 7/9 |
| scalar × restricted | Held-out (10-seed mean) | **0.438 ± 0.049** | — | — | 7/9 |
| ARD × official | CV | 0.484 | 0.523 | 0.307 | 9/9 |
| ARD × official | Held-out | 0.456 | 0.549 | 0.290 | 9/9 |
| ARD × restricted | CV | 0.454 | 0.374 | 0.275 | 7/9 |
| ARD × restricted | Held-out | 0.421 | 0.398 | 0.270 | 7/9 |
| ARD × restricted | Held-out (10-seed mean) | **0.441 ± 0.049** | — | — | 7/9 |

### Per-factor CV breakdown (scalar × official, the complete 9/9 condition)

| Factor | Ordinal error | Macro-F1 |
|---|---|---|
| dre | **0.331** | 0.313 |
| vol | **0.355** | 0.235 |
| pirads | 0.362 | 0.337 |
| comorbidity | 0.410 | 0.285 |
| age | 0.531 | 0.296 |
| cspca | 0.531 | 0.307 |
| psad | 0.575 | 0.302 |
| psa | 0.608 | 0.310 |
| bx | 0.640 | 0.313 |

`dre` remains the easiest factor (consistent with every prior experiment). `age` — historically one
of the two easiest factors under derived-signal approaches (`exp_2`–`exp_10`) — is comparatively
hard here (0.531), while `vol` — historically one of the hardest — does relatively well (0.355).
Direct supervision reshuffles which factors are easy/hard versus the derived-signal approach; it
doesn't uniformly help or hurt across the 9-factor set.

## 5. Comparison against everything else tried for weights

| Approach | Ordinal error (official / restricted) | Macro-F1 |
|---|---|---|
| Baseline | 0.413 | — |
| **Incumbent (`weights_svm`)** | **0.382 / 0.392** | 0.235 |
| Best derived-signal ever (`weights_kdm_occlusion`, `exp_6`, scalar) | 0.405 | 0.256 |
| `exp_5`, direct scalar-KDM (19-col, pre-modernization) | 0.478 / 0.454 | 0.295 / 0.275 |
| `exp_13`, direct scalar-KDM (23-col) | 0.483 / 0.454 | 0.300 / 0.277 |
| `exp_13`, direct ARD-KDM (23-col) | 0.484 / 0.454 | 0.307 / 0.275 |

Per `DESIGN.md` §6's decision rules: this is the **third branch** — direct training does not beat
`exp_5`'s own original numbers at all (it essentially reproduces them, occasionally by a
thousandth of a decimal). This narrows where this project's "revisit direct training" idea
actually applies: it was the right call for confidence (`exp_11`/`exp_12`), but weights' 4-class,
9-factor structure gets no benefit from it, on either the old or new frame, with either backbone.
`weights_kdm_occlusion` (`exp_6`, deriving an occlusion-delta signal from the decision-trained
backbone) remains the best KDM-based weights result this project has produced — modestly beating
baseline (0.405 vs. 0.413) — and `weights_svm` remains the real, unmatched incumbent.

## 6. Interpretation

1. **Direct KDM training is not a universal fix** — it depends on the target's structure, not just
   on being "the pre-`exp_6` approach nobody retried." Confidence (a single 3-class target) responded
   strongly; weights (nine separate 4-class targets, most with far fewer usable training examples
   per class once split into per-factor sub-problems) did not respond at all.
2. **ARD adds nothing here, positively or negatively** — the two backbones are statistically
   indistinguishable on every metric, a third data point (alongside decision's positive ARD verdict
   and confidence's negative one) that ARD's benefit is target-specific, not a property of the KDM
   architecture in general.
3. **A plausible explanation for why weights doesn't respond**: each of the 9 per-factor
   sub-problems trains on ~72–91 rows split across 4 ordinal classes (often with 1–2 examples in the
   rarest class), a much smaller effective sample than confidence's single 91-row 3-class problem.
   Direct KDM training may simply need more per-class examples than weights' per-factor split can
   supply, regardless of backbone or frame.
4. **The repeated-holdout discipline paid for itself again** — a single split's exact tie with
   baseline would have been easy to over-interpret as "restricted scope might be viable" without the
   10-seed check; the check converted a genuinely ambiguous single number into a confident negative
   verdict, and did so at a fraction of full LOO's cost.

## 7. Recommendation

- **`weights_kdm_occlusion` (`exp_6`) remains this project's best KDM-based weights approach.**
  Direct training, revisited here with every backbone/frame improvement available, does not unseat
  it — a genuine negative result, not an oversight to revisit again without a new idea.
- **Do not spend further budget on backbone or frame variations of direct-training weights** — this
  experiment already covered the two backbones this project has (scalar, ARD) and its
  best-performing frame (23-col); the ablation space is exhausted for this particular approach.
- **`weights_svm` remains the unmatched incumbent** (0.382/0.392) — no KDM approach across `exp_2`–
  `exp_13` has beaten it. Closing that gap would need a genuinely new idea (e.g., a joint multi-factor
  model instead of 9 independent per-factor fits, sharing statistical strength across factors) rather
  than another backbone/frame/training-target permutation of the existing per-factor approach.
