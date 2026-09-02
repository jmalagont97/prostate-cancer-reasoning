# exp_12 Report: Task2 Model-Family Sweep

**Status**: Complete. **Verdict: ⚠️ One promising point estimate, refuted by bootstrap — no change
to the promoted config. `knn` stays the default for all three sub-tasks.**

## What was swept

The same 8-model roster task1's exp_2 validated (`experiments/exp_2/scripts/models.py`, unchanged):
logistic, elastic_net, svm_rbf, knn, random_forest, extra_trees, naive_bayes, mlp. One sub-task's
`model_factory` varied at a time, the other two held at their exp_11-promoted default (`knn`, zero
tools), full `ranking_score` measured under LOOCV each time (24 runs total, 72 folds each).

## Results

| Model | decision | confidence | variable_weights |
|---|---|---|---|
| logistic | 0.6704 | 0.7073 | 0.7078 |
| elastic_net | 0.6680 | 0.7073 | 0.7088 |
| svm_rbf | 0.7115 | 0.7026 | 0.7075 |
| **knn (current default)** | 0.7166 | **0.7166** | **0.7166** |
| **random_forest** | **0.7315** | 0.7104 | 0.7052 |
| extra_trees | 0.7128 | 0.7080 | 0.7037 |
| naive_bayes | 0.6612 | 0.7088 | 0.7108 |
| mlp | 0.7143 | 0.7159 | 0.7164 |

(cells are `ranking_score`, the one sub-task varying per column, other two fixed at `knn`)

`knn` remains the best choice for `confidence` and `variable_weights` — no candidate beat it on
either axis. `random_forest` beat `knn` on `decision` by the largest single margin in the sweep:
**0.7315 vs 0.7166 (+0.0149)**. Verified the combined config (decision=random_forest,
confidence/variable_weights=knn) reproduces exactly (0.7315130...), confirming no interaction
effect between sub-task model choices — each was validly optimized in isolation.

## Bootstrap: is the decision=random_forest gain real?

Paired bootstrap (2,000 resamples, both models' full row sets recomputed, not just the mean):

```
Point estimates: knn=0.7166  random_forest=0.7315  diff=+0.0149
Bootstrap: mean diff=+0.0147, 95% CI=[-0.0725, +0.1091], frac favoring random_forest=63.7%
-> NOT SIGNIFICANT
```

The 95% CI comfortably straddles zero, and only 63.7% of resamples favor `random_forest` — barely
better than a coin flip at n=72 with 4 imbalanced classes. **Not promoted.** This is the same
pattern exp_4 (task3's Cox-penalizer sweep) and exp_9 (task3's event-classifier horizon/threshold)
already found: a real point-estimate winner that doesn't clear the bar once resampling noise is
accounted for.

## Classification report + confusion matrix — `knn` (promoted) vs `random_forest` (not promoted)

**knn (promoted, unchanged from exp_11):**

```
                        precision    recall  f1-score   support
      watchful_waiting       0.00      0.00      0.00         2
   active_surveillance       0.70      0.84      0.76        25
continued_surveillance       0.93      0.93      0.93        14
      active_treatment       0.86      0.77      0.81        31
              accuracy                           0.81        72
             macro avg       0.62      0.64      0.63        72
          weighted avg       0.79      0.81      0.80        72
```

**random_forest (not promoted, shown for the record):**

```
                        precision    recall  f1-score   support
      watchful_waiting       0.00      0.00      0.00         2
   active_surveillance       0.83      0.76      0.79        25
continued_surveillance       0.87      0.93      0.90        14
      active_treatment       0.84      0.87      0.86        31
              accuracy                           0.82        72
             macro avg       0.63      0.64      0.64        72
          weighted avg       0.82      0.82      0.82        72
```

| Model | True \ Pred | ww | as | cs | at |
|---|---|---|---|---|---|
| knn | watchful_waiting (2) | 0 | 1 | 0 | 1 |
| random_forest | watchful_waiting (2) | 0 | 0 | 0 | 2 |

Both models still miss `watchful_waiting` completely (0/2, unsurprising at n=2 per class). The real
difference is `random_forest` trading recall for precision on `active_surveillance` (0.84→0.76
recall, 0.70→0.83 precision) while gaining on `active_treatment` (0.77→0.87 recall) — a real,
visible behavioral difference, just not one the bootstrap can distinguish from noise at this sample
size.

## Verdict and action

**No change to `task2_subtask_config.py`.** `knn` stays the default for `decision`, `confidence`,
and `variable_weights` alike. The sweep was worth running — it confirmed rather than assumed that
task1's model choice transfers to task2, and it surfaced a plausible-looking but statistically
unsupported alternative before it could get promoted on a point estimate alone.

## Not done

- No hyperparameter tuning within any model family (e.g. `n_neighbors` for kNN, `max_depth` for the
  tree ensembles) — only the 8 fixed configs from `models.py` were compared.
- No joint sweep across sub-tasks (e.g. decision=random_forest *and* confidence=mlp together) —
  each sub-task's best was found independently; the isolation discipline means a genuine
  interaction effect (if one exists) wouldn't be caught by this design.
