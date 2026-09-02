# exp_12 Design: Task2 Model-Family Sweep

## Background

exp_11 built task2's agent and got a first honest baseline (RANKING_SCORE=0.714) using `knn`
(task1's proven winner) for all three sub-tasks, unmeasured for task2 specifically. Task1's own
exp_2 found the best model varies by sub-task (kNN won broadly, but task-specific structure
mattered), so carrying `knn` over to a different task/variable-set/decision-domain without
re-checking is exactly the kind of assumption this project's discipline pushes back against.

## Hypothesis

Some sub-task, for some model in the 8-model roster (`experiments/exp_2/scripts/models.py`:
logistic, elastic_net, svm_rbf, knn, random_forest, extra_trees, naive_bayes, mlp), beats `knn`'s
current `ranking_score`. Stated without a directional prediction — task1's own exp_2 found gains
were sub-task-dependent and not always in the "more capacity" direction, so no single model is
assumed to win across all three sub-tasks here either.

## Method

One axis at a time, mirroring exp_5's/task1's isolation discipline: for each sub-task
(`decision`, `confidence`, `variable_weights`), swap only that sub-task's `model_factory` across
the full 8-model roster, holding the other two sub-tasks at their current promoted default (`knn`,
zero tools everywhere per exp_11) and the tool policy fixed. Measure the **full** `ranking_score`
each time (not an isolated component score) via `task2_variant_tester.py`, under LOOCV on the 72
labeled cases. `variable_weights`' model_factory applies to all 11 of its per-variable classifiers
simultaneously (one model family per sub-task, not per variable).

## What would change the promoted config

Only a per-sub-task model swap that beats the current `knn` baseline on `ranking_score`, checked
with a paired bootstrap before being called significant. `task2_subtask_config.py`'s
`default_configs()` would be updated to the winning model per sub-task that clears the bar;
anything that doesn't clear it stays reported, not promoted.
