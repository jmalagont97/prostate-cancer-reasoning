# exp_19 Design: 3-Model Voting Ensemble (kNN + Logistic + Random Forest) for `decision`

## Background

Requested directly, scoped to exactly 3 models (`knn`, `logistic`, `random_forest`) and one
sub-task (`decision`). Motivated by exp_12's finding that `knn` and `random_forest` show genuinely
different strengths on `decision` (`random_forest`'s point estimate beat `knn` by the sweep's
largest margin, though not bootstrap-significant) — averaging across models with different biases
is a standard way to reduce variance, plausible here given `decision`'s severe 4-class imbalance.

## Method

`sklearn.ensemble.VotingClassifier` with the three named base estimators (`models.py`'s existing
factories, unchanged), tested both ways:

- `voting="hard"` — majority vote across the 3 predicted labels
- `voting="soft"` — average the 3 models' predicted-class probabilities, pick the argmax

Tested via `task2_variant_tester.py` on `decision`'s already-promoted 12-feature set (10 base +
`psa_kinetics`, exp_16), `confidence`/`variable_weights` held at their own promoted defaults.

## What would change the promoted config

Only a voting variant that beats the current single-model `knn` default on `ranking_score`,
bootstrap-confirmed before promotion — same bar as every other experiment.
