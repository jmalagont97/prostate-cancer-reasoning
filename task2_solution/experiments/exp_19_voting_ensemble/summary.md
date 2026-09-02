# exp_19 Report: 3-Model Voting Ensemble (kNN + Logistic + Random Forest) for `decision`

**Status**: Complete. **Verdict: ❌ Not significant, cleanly. No change to the promoted config —
`decision` stays on plain `knn`.**

## What was tested

`sklearn.ensemble.VotingClassifier` over the three named base models (`knn`, `logistic`,
`random_forest` — `experiments/exp_2/scripts/models.py`'s existing factories, unchanged), both hard
(majority vote) and soft (averaged probability) voting, on `decision`'s already-promoted 12-feature
set (10 base + `psa_kinetics`, exp_16).

## Results

| Variant | ranking_score | vs. baseline (0.7429) | decision_f1 |
|---|---|---|---|
| **baseline (`knn` alone)** | **0.7429** | — | 0.7960 (this run's decision_f1 reflects only decision's own config)|
| voting_hard | 0.7452 | +0.0024 | 0.8284 |
| voting_soft | 0.7445 | +0.0016 | 0.8280 |

Both point estimates are small gains — an order of magnitude smaller than exp_12's already-refuted
`random_forest` swap (+0.0149) and far smaller than exp_16's promoted `psa_kinetics` gain (+0.0243).

## Bootstrap: voting_hard (the larger of the two) vs. knn

```
Point estimates: knn=0.7429  voting_hard=0.7452  diff=+0.0024
Bootstrap (2000 resamples): mean diff=+0.0024, 95% CI=[-0.0463, +0.0521], frac favoring=56.0%
-> NOT SIGNIFICANT
```

56% is barely above a coin flip — the widest, least-convincing CI of any decision-model comparison
tried in this project so far (wider even than exp_12's rejected `random_forest` swap).

## Classification report — confirms there's no hidden story either

```
                        precision    recall  f1-score   support
      watchful_waiting       0.00      0.00      0.00         2
   active_surveillance       0.76      0.88      0.81        25
continued_surveillance       0.93      0.93      0.93        14
      active_treatment       0.89      0.81      0.85        31
              accuracy                           0.83        72
             macro avg       0.65      0.65      0.65        72
          weighted avg       0.83      0.83      0.83        72
```

Essentially identical to `knn` alone (accuracy 0.83 either way, macro-F1 0.64→0.65, same confusion
pattern — `watchful_waiting` still 0/2). No corroborating diagnostic to tip a borderline call the
way exp_14/16 had one; the ensemble just isn't meaningfully different from its strongest single
member here.

## Verdict and action

**No change to the promoted config.** `decision` stays plain `knn`. `logistic` (this project's
weakest performer on `decision` throughout, see exp_12: 0.6704) likely diluted the ensemble's votes
enough to cancel out most of whatever `random_forest` might have contributed on its own — a
plausible reason three-way voting undershoots even `random_forest`'s own (already non-significant)
solo point estimate from exp_12.

## Not done

- No 2-model ensemble (`knn` + `random_forest` only, dropping the weaker `logistic`) — a reasonable
  follow-up if this line is revisited, given `logistic` is this project's consistently weakest
  `decision` model.
- No weighted voting (giving `random_forest` more influence than `logistic`) — only the default
  equal-weight `VotingClassifier` was tried.
