# Experiment Design: Hybrid ML Baseline for CHIMERA-Agent Task 1
**Experiment**: experiments/exp_1/
**Project**: challenge_chimera_2
**Date**: 2026-08-08
**Author**: TBD
**Status**: Complete

> Written retroactively — this documents work already run in this session, following the
> ml-experiment-planner/-reporter skill conventions adopted partway through. See
> `experiments/exp_1/results/` for the raw numbers this is grounded in.

---

## 1. Hypothesis

Small supervised models (logistic regression, gradient boosting, and — for the confidence
target specifically — a Kernel Density Matrix classifier) trained on the CHIMERA-Agent Task 1
training set's hand-engineered structured features can beat trivial per-target naive baselines
(majority-class / always-positive guessing) on: the biopsy yes/no decision (F1), the confidence
label (ordinal distance), the 10 per-factor variable weights (mean ordinal error, decisive-set
F1), and the reveal-sequence section selection (set precision) — closely enough, in aggregate,
to justify building the downstream submission pipeline (steps 5-8 of the broader Task-1 plan:
rubric scorer, MCP+LLM wiring, Docker packaging, platform submission).

## 2. Experimental Setup

- **Dataset**: `data/inputs.csv` (195 cases × 53 clinical/text columns + 1024-dim MRI
  embedding) merged with `data/ground_truth.csv` (195 cases; only 91 carry the full
  confidence/variable-weight/reveal-sequence annotation) on `case_id`. This is the official
  CHIMERA-Agent Task 1 training set (Radboudumc, 195 cases).
- **Features**: `src/chimera_task1/features.py` — ordinal-encoded discrete clinical columns,
  one-hot categorical columns (explicit "missing" category), median-imputed continuous columns
  with missingness indicators for the highest-missing ones, all free-text `txt_*` columns
  dropped. Optional 10-component PCA of the MRI embedding, tested as an additional feature block.
- **Models compared**:
  - Decision (yes/no): `LogisticRegression(class_weight="balanced")` vs.
    `HistGradientBoostingClassifier` (regularized: `max_leaf_nodes=7, min_samples_leaf=20,
    l2_regularization=1.0`), each with and without the MRI-PCA block.
  - Confidence: one-vs-rest `LogisticRegression` vs. a memory-based `KDMClassModel`
    (`kdm-torch`; prototypes frozen at the training data, only the RBF kernel bandwidth
    trained — see `src/chimera_task1/train_confidence_kdm.py`).
  - Variable weights (10 factors) and reveal-sequence: one-vs-rest `LogisticRegression` only.
- **Evaluation**: repeated k-fold cross-validation (5 folds, 8-20 repeats depending on script)
  with out-of-fold predictions, since N=195 (decision) / N=91 (confidence/weights/reveal) makes
  any single split noisy. Every target is scored against a feature-blind naive baseline
  (majority class / always-positive / mode pattern) computed the same way.
- **Hardware**: local CPU (decision/weights/reveal models) + CPU-only PyTorch (KDM). No GPU
  needed at this N.

## 3. File Layout for This Experiment

```
experiments/exp_1/
├── DESIGN.md                  ← this file
├── results/
│   ├── decision_logistic_clinical/metrics.json
│   ├── decision_hgb_clinical/metrics.json
│   ├── decision_logistic_mri_pca/metrics.json
│   ├── decision_hgb_mri_pca/metrics.json
│   ├── decision_baseline/metrics.json
│   ├── confidence_logistic/metrics.json
│   ├── confidence_kdm/metrics.json
│   ├── confidence_baseline/metrics.json
│   ├── weights_logistic/metrics.json
│   ├── weights_baseline/metrics.json
│   ├── reveal_logistic/metrics.json
│   └── reveal_baseline/metrics.json
└── reports/
    └── summary.md
```
Reusable code lives in `../../src/chimera_task1/` (shared, not experiment-specific):
`features.py`, `reasoning_labels.py`, `train_decision.py`, `train_reasoning.py`,
`train_confidence_kdm.py`. These double as both shared library code and the run scripts for
this experiment (a minor deviation from the skill's `scripts/` convention, noted since they
aren't duplicated under `experiments/exp_1/scripts/`).

## 4. Baselines

| Baseline | Description | Source |
|----------|-------------|--------|
| decision_baseline | Always predict "yes" | `train_decision.py::naive_baselines` |
| confidence_baseline | Always predict majority class ("clear") | `train_reasoning.py::eval_confidence` |
| weights_baseline | Per-factor majority-class guess | `train_reasoning.py::eval_weights` |
| reveal_baseline | Always predict the single most common section-subset | `train_reasoning.py::eval_reveal` |

## 5. Proposed Conditions

| Condition | Model | Features |
|---|---|---|
| decision_logistic_clinical | LogisticRegression | clinical only |
| decision_hgb_clinical | HistGradientBoosting | clinical only |
| decision_logistic_mri_pca | LogisticRegression | clinical + MRI-PCA(10) |
| decision_hgb_mri_pca | HistGradientBoosting | clinical + MRI-PCA(10) |
| confidence_logistic | OvR LogisticRegression | clinical |
| confidence_kdm | KDMClassModel (memory-based) | clinical (standardized) |
| weights_logistic | OvR LogisticRegression × 10 factors | clinical |
| reveal_logistic | MultiOutputClassifier(LogisticRegression) | clinical |

## 6. Ablation Studies

MRI-PCA vs. clinical-only is the one ablation run (decision model only) — isolates whether the
1024-dim MRI embedding adds signal beyond the already-MRI-derived `cli_pirads`/`cli_cspca`
columns.

## 7. Evaluation Protocol

- **Decision**: F1 (positive = "yes"), matches the official Task-1 leaderboard's `Biopsy F1`
  term. Also checked ROC-AUC / PR-AUC (threshold-independent) to distinguish "no signal" from
  "signal exists, wrong default threshold."
- **Confidence**: mean ordinal distance over `uncertain < borderline < clear` (matches the
  official rubric's confidence-scoring component), plus mean predictive entropy as a sanity
  check against degenerate (collapsed) predictions.
- **Variable weights**: mean ordinal error over `not_used < noted < important < decisive`
  (matches the official rubric) and decisive-set F1 (matches the official rubric's
  important/decisive factor-set component).
- **Reveal-sequence**: set precision of predicted vs. actual revealed sections (matches the
  official rubric's "tool efficiency" component).
- **Decision rule**: a condition is judged as "worth carrying forward" only if it beats its
  matched naive baseline by a clear margin — not just nominally higher.
- Results written to `results/<condition>/metrics.json`.

## 8. Expected Results & Decision Rules

- If models clearly beat their naive baselines across decision + confidence + weights →
  proceed to steps 5-8 (rubric scorer, MCP+LLM wiring, Docker, submission).
- If models are at or below naive baselines on multiple targets → **stop and reassess** the
  feature set / architecture before investing in submission infrastructure, since that
  infrastructure doesn't improve prediction quality.

## 9. Risks & Mitigations

- **Small N (91-195)**: mitigated by repeated k-fold CV with out-of-fold aggregation rather
  than a single train/test split; kept models deliberately simple (linear/shallow-tree, and
  for KDM, frozen prototypes) to limit overfitting.
- **Class imbalance (29% positive decision rate)**: `class_weight="balanced"` throughout;
  explicitly compared against the "always predict yes" baseline that trivially maximizes
  recall, since plain accuracy/F1 at default threshold can be misleading under imbalance.
- **Feature-importance artifacts**: permutation importance on the HGB decision model surfaced
  clinically implausible top features (heart rate, systolic BP) — flagged as likely noise from
  fitting importance on the full training set, not trusted for interpretation.

## 10. Reproducibility Checklist

- [x] Random seeds fixed and logged (`RANDOM_STATE = 0` in each script)
- [ ] Config YAML saved in `scripts/configs/` — N/A, configs are inline constants in each script
- [x] Dataset version noted: official Task-1 training set, 195 cases, as of 2026-08-08
- [ ] Model checkpoints saved — N/A, no persisted model artifacts (CV-only, no final fit saved)
- [ ] Experiment tracker run linked — not used
- [x] Environment: project `.venv` (Python 3.14, pandas 3.0.5, scikit-learn 1.9.0, torch 2.13 CPU,
  `kdm-torch` from `github.com/fagonzalezo/kdm`)
- [ ] Git commit hash recorded — **N/A: this project is not a git repository**
- [ ] Working tree clean at run time — N/A, no git repository to check

## 11. Next Steps

1. ✅ Plan reviewed and accepted (retroactively, alongside the results).
2. See `reports/summary.md` for the verdict and recommended follow-ups — the broader Task-1
   plan (steps 5-8) is **paused** pending one of: adding text features from the `txt_*`
   narrative columns, more data from the validation set (released 2026-08-10), or prototyping
   the organizers' full LLM-agent baseline instead of pure tabular ML.
