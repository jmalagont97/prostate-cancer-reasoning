# exp_18 Report: Radiology + Pathology Free-Text Staging Features

**Status**: Complete. **Verdict: ❌ Refuted everywhere. A well-motivated hypothesis (direct
precedent from task3's biggest early win) that didn't transfer. No change to any promoted config.**

## What was tested

Extracted 5 new features from `radiology_report`/`pathology_report` free text, closed-vocabulary-
confirmed via direct corpus scan (several regex iterations caught real false positives along the
way — see `scripts/extract_features_task2.py`'s docstrings and `experiments/exp_18/DESIGN.md`):

| Feature | Source | Coverage | Positive rate |
|---|---|---|---|
| `epe_present` | radiology | 134/153 | 5/134 |
| `svi_present` | radiology | 130/153 | 2/130 |
| `ai_isup_discordance` | pathology | 147/153 | mean +1.73 (AI systematically grades higher) |
| `cribriform_present` | pathology | 147/153 | 51/147 |
| `perineural_invasion_present` | pathology | 95/153 | 25/95 |

Tested as two separate groups (radiology, pathology — not combined), against each sub-task's own
currently-promoted feature set, mirroring exp_16/17's isolation discipline.

## Results

| Sub-task | +radiology | +pathology | +both |
|---|---|---|---|
| decision (base=12 feat) | +0.0000 | **−0.0717** | −0.0717 |
| confidence (base=10 feat) | +0.0000 | −0.0008 | −0.0008 |
| variable_weights (base=10 feat) | −0.0001 | −0.0002 | −0.0006 |

No bootstrap needed anywhere — nothing beat baseline on the point estimate, and `decision`'s
pathology-group loss is large and unambiguous on its own.

## Why, diagnosed rather than left a mystery

**`epe_present`/`svi_present` had literally zero effect**, everywhere, to 4 decimal places. This
traces directly to their rarity: 5 and 2 positive cases out of the whole labeled cohort. With that
few positives, median imputation and the sheer sparsity mean the feature is nearly constant across
almost every training fold — it barely perturbs kNN's standardized distance metric at all. This is
a different failure mode from task3's own encounter with an equally-rare feature (`on_5ari`, exp_10)
— there it crashed Cox's Newton-Raphson step outright (a parametric-model problem); here kNN just
quietly ignores it (a non-parametric model's more graceful, but equally unhelpful, response to
near-zero variance).

**`ai_isup_discordance`/`cribriform_present`/`perineural_invasion_present` hurt `decision` badly**
(−0.0717) — the same class of problem `labs` caused in exp_16: real missingness (`perineural_
invasion_present` is only 62% covered) compounding with `decision`'s already-tuned 12-feature space,
adding noise faster than signal at n=72. `confidence`/`variable_weights` barely moved either
direction — small enough that the feature is closer to inert than actively harmful there.

## Verdict and action

**No change to any promoted config.** `decision` stays on its exp_16 feature set (10 base +
`psa_kinetics`); `confidence` stays 10-feature `knn`; `variable_weights` stays 10-feature
`mord.LogisticIT`. The direct task3 precedent (EPE/SVI anchored that project's biggest early win)
did not transfer — a useful negative result on its own: not every clinically-plausible free-text
feature category generalizes across tasks, sample sizes, and model families, and this project's own
prior success with one feature category isn't evidence for the next one. All five features are kept
as documented, unused engineered columns.

## Not done

- No attempt to salvage a smaller subset (e.g. `ai_isup_discordance` alone, without the sparser
  `cribriform`/`PNI`) — the full pathology group's loss was large enough on `decision` that a
  narrower retry wasn't prioritized here; a natural follow-up if this line is revisited.
- No re-extraction of EPE/SVI with a broader/less strict definition to raise their positive rate
  (e.g. treating "cannot be excluded"/equivocal language as a third category rather than folding it
  into negative) — the corpus scan didn't surface enough of that phrasing to make it worth building.
