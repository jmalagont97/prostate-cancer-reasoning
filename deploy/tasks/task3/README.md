# Task 3 deployment

## Training

```bash
PYTHONPATH=. python deploy/tasks/task3/training/train.py
```

Trains `CoxPHFitter(penalizer=0.3)` on all annotated cases, with the canonical 19 clinical/pathology features and PCA-5 pooled prostatectomy embeddings. The serialized artifact contains preprocessing, PCA, Cox model, feature order, and event policy.

## Inference

Input is a raw case directory containing:

```text
structured-prompt.json
prostate-time-to-recurrence-or-last-follow-up-clinical-data.json
prostate-modality-level-neural-representations.json
```

The label JSON is ignored at inference. Run:

```bash
PYTHONPATH=. python deploy/tasks/task3/inference.py <case-directory>
```

Output contract:

```json
{
  "months_to_recurrence": 24.0,
  "event": 1
}
```

`months_to_recurrence` is the Cox survival-time percentile `p=0.90`. `event` is derived independently from the same survival curve: `1` when `S(60 months | x) < 0.7`, otherwise `0`.
