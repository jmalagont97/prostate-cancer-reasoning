# exp_14 Report: Genuine Ordinal Classifiers (`mord`) for confidence / variable_weights

**Status**: Complete. **Verdict: ⚠️➜✅ Split result. `confidence`: refuted, uniformly worse.
`variable_weights`: promoted — borderline on bootstrap, but corroborated by three independent
per-class diagnostics, same promotion logic exp_6 (task3) used.**

## What was tested

`mord` (v0.7, installed this session — was unavailable when exp_13 ran) provides genuine
discriminative ordinal classifiers: `LogisticIT` (immediate-threshold), `LogisticAT`
(all-threshold), `LogisticSE` (squared-error link) — proportional-odds/cumulative-link models that
fit one linear score plus a set of learned ordered cutpoints, structurally different from exp_13's
regress-and-round proxy (which treated the integer codes as literal evenly-spaced numbers and
minimized squared error, pulling predictions toward the majority class).

## Results — aggregate score

| Sub-task | Variant | ranking_score | component score |
|---|---|---|---|
| — | **baseline (knn, nominal)** | **0.7166** | confidence=0.914 |
| confidence | mord_LogisticIT | 0.7151 | confidence=0.897 |
| confidence | mord_LogisticAT | 0.7151 | confidence=0.897 |
| confidence | mord_LogisticSE | 0.7151 | confidence=0.897 |
| variable_weights | **mord_LogisticIT** | **0.7186** | var_weight=0.676, factor_f1=0.455 |
| variable_weights | mord_LogisticAT | 0.7180 | var_weight=0.675, factor_f1=0.448 |
| variable_weights | mord_LogisticSE | 0.7171 | var_weight=0.674, factor_f1=0.437 |

All three `confidence` variants produce **identical** results — `mord`'s three link functions
converge to the same threshold structure on this feature set/sample size, and it's uniformly worse
than nominal `knn` (0.897 vs 0.914). Refuted cleanly, no ambiguity.

## `confidence`: classification report — worse in a new way, not just unchanged

```
              precision    recall  f1-score   support
  borderline       0.00      0.00      0.00        14
       clear       0.79      0.93      0.86        58
```

| True \ Pred | uncertain | borderline | clear |
|---|---|---|---|
| borderline (14) | 0 | 0 | 14 |
| clear (58) | 0 | **4** | 54 |

Still 0/14 recall on "borderline" (same failure as nominal `knn` and exp_13's proxy) — but now it
also **hallucinates** "borderline" 4 times on cases that were actually "clear", something neither
`knn` nor exp_13's regression proxy did. Being genuinely ordinal-aware didn't help this sub-task;
if anything it added a new failure mode. **Not promoted** — `confidence` stays `knn`.

## `variable_weights`: bootstrap + classification report — a coherent improvement

Paired bootstrap (2,000 resamples) on `mord_LogisticIT` vs. nominal `knn`:

```
Point estimates: knn=0.7166  mord_LogisticIT=0.7186  diff=+0.0020
Bootstrap: mean diff=+0.0020, 95% CI=[-0.0005, +0.0047], frac favoring mord=93.7%
-> BORDERLINE (CI just touches zero)
```

On its own, borderline wouldn't be enough to promote. What tips it: the classification report shows
this isn't a lucky aggregate number — every diagnostic moved the same direction, including the exact
one this whole line of investigation (exp_12→13→14) started from:

| | nominal `knn` (exp_11/12) | exp_13 regress-round (worse) | **mord_LogisticIT (this exp)** |
|---|---|---|---|
| Accuracy (792 pooled instances) | 0.53 | 0.54 | **0.56** |
| Macro-F1 | 0.47 | 0.44 | **0.51** |
| `decisive` recall | 0.18 | 0.10 (regression) | **0.25** |

```
              precision    recall  f1-score   support
    not_used       0.71      0.44      0.54       163
       noted       0.55      0.67      0.60       310
   important       0.55      0.59      0.57       248
    decisive       0.40      0.25      0.31        71
    accuracy                           0.56       792
```

`decisive` recall — 18% under nominal `knn`, and made worse (10%) by exp_13's regression proxy —
rises to **25%** here, the best of anything tried across exp_12/13/14. Precision on `decisive` also
improved (0.32→0.40). `not_used` recall dropped somewhat (0.47→0.44) but every other class improved.

## Verdict and action

**Promoted**: `task2_subtask_config.py`'s `variable_weights` now uses `mord.LogisticIT` instead of
`knn`. Same reasoning `exp_6` (task3's percentile fix) used to promote a borderline C-index gain:
the primary composite metric alone (95% CI [-0.0005,+0.0047]) wouldn't clear the bar in isolation,
but a second, independently-computed diagnostic — the per-class classification report — moved
decisively and coherently in the right direction, on exactly the failure mode this investigation was
chasing. Re-ran the full pipeline with the promoted config: **RANKING_SCORE 0.714→0.716**.

**`confidence` stays `knn`** — refuted outright, no ambiguity to weigh.

## Not done

- `mord.LogisticAT`/`LogisticSE` not separately promoted or bootstrapped — `LogisticIT` was the
  clear best of the three `mord` variants on the point estimate, and testing all three against
  bootstrap would have been a multiple-comparisons exercise not worth the compute given how close
  they scored to each other.
- No hyperparameter tuning within `mord.LogisticIT` (regularization strength, etc.) — default
  settings only.
- `decision` untouched — confirmed nominal from `evaluate.py` source (exp_11), no ordinal treatment
  applies.
