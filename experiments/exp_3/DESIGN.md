# Experiment Design: Broader Model Family Comparison + MRI-PCA + Decorrelated PSA Family for Task 1
**Experiment**: experiments/exp_3/
**Project**: challenge_chimera_2
**Date**: 2026-08-09
**Author**: TBD
**Status**: Complete

---

## 1. Hypothesis

A broader search over model families (SVM, Random Forest, XGBoost, Extra Trees, MLP, Naive Bayes,
kNN, KDM) combined with a refined 19-column feature set — `exp_2`'s official clinical variables
with the collinear PSA family reduced to `psa` + `psad` (dropping `psap`/`psav`), comorbidity
fixed to the grouped flags (no longer an ablation axis), and a 2-component MRI-embedding PCA
added (84.25% cumulative variance) — finds at least one model/target combination that clearly
beats both the naive baseline and the best result so far from `exp_1`/`exp_2` for that target,
particularly for confidence and variable-weights, which have not yet beaten baseline in either
prior experiment.

**Explicit caution built into this hypothesis**: comparing 8 model families on the same
~91–195 cases via CV substantially increases the risk that whichever one looks best is partly
CV noise rather than genuine superiority. Any "winner" here should be read as a candidate for
further scrutiny, not a settled conclusion — see §9.

## 2. Experimental Setup

- **Dataset**: same as `exp_1`/`exp_2` — `data/inputs.csv` + `data/ground_truth.csv`, 91
  annotated cases for confidence/weights/reveal, 195 for decision.
- **Feature set (19 columns, fixed — no comorbidity-treatment or feature-scope ablation this
  time except where noted for weights)**:

  | Group | Columns |
  |---|---|
  | Clinical (10) | `cli_psa`, `cli_psad`, `cli_vol`, `cli_age`, `cli_cspca`, `cli_pirads`, `cli_dre_ordinal`, `cli_dre_not_done`, `cli_bx_positive`, `cli_bx_missing` |
  | Comorbidity (6) | `comorb_cardiometabolic`, `comorb_renal`, `comorb_bleeding_risk`, `comorb_respiratory`, `comorb_bph`, `comorb_other_unmatched` |
  | MRI (3) | `mri_pca_0`, `mri_pca_1`, `mri_missing` |

  Changes from `exp_2`'s 18-column official-flags frame: **dropped** `cli_psap`, `cli_psav`
  (correlation with `cli_psa`: 0.99 and 0.95 respectively — near-redundant, confirmed this
  session: `cli_psad` ≈ `cli_psa`/`cli_vol` almost exactly, r=0.9999); **added** the 3 MRI
  columns. `fh` and `ct` remain excluded, same reasoning as `exp_2`.
- **MRI PCA**: fit on the 191/195 cases with a non-missing embedding (4 missing, imputed to the
  PCA-space origin + `mri_missing` flag — same pattern as `exp_1`'s 10-component version).
  2 components chosen over `exp_1`'s original 10 specifically to keep total dimensionality sane
  relative to N (see §9) — confirmed this session at 79.60% + 4.65% = 84.25% cumulative variance.
- **Models, decision + confidence only** (8 each): `SVC(probability=True)`, `RandomForestClassifier`,
  `XGBClassifier`, `ExtraTreesClassifier`, `MLPClassifier`, `GaussianNB`, `KNeighborsClassifier`,
  and KDM (memory-based, sigma-only trained — same design as `exp_1`/`exp_2`). All given
  `class_weight="balanced"` where supported; regularization chosen per-model to suit N=91–195
  (see IMPLEMENTATION.md for exact hyperparameters — a design decision deferred to that phase
  per this project's convention, not decided here).
- **Models, weights**: `OneVsRestClassifier(LogisticRegression(solver="liblinear"))`, unchanged
  from `exp_1`/`exp_2` — two conditions, "official" (full 19-column frame) and "restricted"
  (per-factor groups, MRI excluded from every group — see table below).
- **Models, reveal-sequence**: `MultiOutputClassifier(...)`, unchanged from `exp_1`/`exp_2` — one
  condition on the full 19-column frame.
- **Per-factor restricted feature groups (weights_restricted only)** — MRI deliberately excluded
  from every group, resolved this session after checking that `cli_cspca` (r=0.456) correlates
  with the MRI PCA signal *slightly more* than `cli_pirads` (r=0.412) does, meaning there's no
  principled basis to attribute MRI to one factor over the other, and doing so risks a false
  importance signal from double-counting shared MRI/pirads/cspca correlation (pirads↔cspca r=0.475):

  | Factor | Restricted group |
  |---|---|
  | `psa` | `cli_psa` |
  | `age` | `cli_age` |
  | `dre` | `cli_dre_ordinal`, `cli_dre_not_done` |
  | `bx` | `cli_bx_positive`, `cli_bx_missing` |
  | `pirads` | `cli_pirads` |
  | `psad` | `cli_psad` |
  | `vol` | `cli_vol` |
  | `cspca` | `cli_cspca` |
  | `comorbidity` | 6 `comorb_*` flags |

- **Evaluation**: same repeated out-of-fold CV and rubric-matching metrics as `exp_1`/`exp_2`
  (`reasoning_labels.py`), so all results remain directly comparable across all three experiments.

## 3. File Layout for This Experiment

```
experiments/exp_3/
├── DESIGN.md
├── IMPLEMENTATION.md        ← written after this design is accepted, in plan mode
├── scripts/
├── results/
│   ├── decision_{svm,rf,xgb,extratrees,mlp,nb,knn,kdm}/   (8)
│   ├── confidence_{svm,rf,xgb,extratrees,mlp,nb,knn,kdm}/ (8)
│   ├── weights_official/, weights_restricted/               (2)
│   └── reveal/                                               (1)
└── reports/
    └── summary.md
```
(19 condition folders total.)

## 4. Baselines

Reuses `exp_1`'s naive baselines directly (feature-blind, unchanged) — no need to recompute.
`exp_2`'s model results (`decision_logistic_count` F1=0.473, etc.) serve as the incumbent
best-so-far comparison point per target, alongside `exp_1`'s original numbers.

## 5. Proposed Conditions

| Condition | Target | Model |
|---|---|---|
| `decision_svm` | decision | SVC (RBF, probability=True) |
| `decision_rf` | decision | RandomForestClassifier |
| `decision_xgb` | decision | XGBClassifier |
| `decision_extratrees` | decision | ExtraTreesClassifier |
| `decision_mlp` | decision | MLPClassifier |
| `decision_nb` | decision | GaussianNB |
| `decision_knn` | decision | KNeighborsClassifier |
| `decision_kdm` | decision | KDM (memory-based) |
| `confidence_svm` … `confidence_kdm` | confidence | same 8 models |
| `weights_official` | variable-weights | OvR logistic, full 19-col frame |
| `weights_restricted` | variable-weights | OvR logistic, per-factor groups (table above) |
| `reveal` | reveal-sequence | MultiOutput OvR logistic, full 19-col frame |

## 6. Ablation Studies

- **Model family** (8-way) × **decision, confidence** — the primary comparison this experiment
  adds. Not crossed with any feature-scope variant (feature set is fixed for these two targets).
- **Feature scope** (official vs. restricted), **weights only** — re-testing `exp_2`'s finding
  (restricted was worse) with the updated feature set, per this session's explicit request rather
  than assuming the prior result still holds unchanged.

## 7. Evaluation Protocol

- **Decision**: F1 (primary), ROC-AUC/PR-AUC (secondary) — compare all 8 conditions against each
  other, against `exp_2`'s `decision_logistic_count` (0.473, current best), and the naive
  baseline (0.446).
- **Confidence**: ordinal distance (primary) — compare all 8 against `exp_1`'s `confidence_kdm`
  (0.564, current best) and the naive baseline (0.527).
- **Weights**: mean ordinal error + mean decisive-set F1 across the 9 in-scope factors — compare
  both conditions against `exp_2`'s `weights_official_flags` (0.585 / 0.546, current best) and
  the naive baseline (0.401–0.413 depending on factor count).
- **Reveal**: set precision — compare against `exp_2`'s `reveal_flags` (0.853, current best) and
  the naive baseline (0.783).
- **Statistical rigor**: same limitation as `exp_1`/`exp_2` — repeated CV aggregates, no formal
  significance test. Given 8-way comparisons specifically increase false-winner risk (see
  Hypothesis caution above), report the **top 2-3** models per target rather than only the single
  best, so a close second isn't silently discarded.
- Results written to `results/<condition>/metrics.json`, same schema as `exp_1`/`exp_2`.

## 8. Expected Results & Decision Rules

- If any decision/confidence model clearly beats both its naive baseline and the current
  cross-experiment best → strong candidate to carry into the paused steps 5-8 (rubric scorer,
  MCP+LLM wiring, Docker, submission) for that target specifically.
- If `weights_restricted` again underperforms `weights_official` (replicating `exp_2`) → treat
  per-factor restriction as settled negative, not worth re-testing again in future experiments.
  If it reverses (now that MRI/PSA-family changes are in play) → worth understanding why before
  generalizing either direction.
- If confidence and weights remain below baseline across all 8+2 conditions respectively →
  stronger evidence that no amount of classical-ML model search fixes those two targets with the
  current feature set, and the free-text/LLM-agent alternative (on record since `exp_1`) becomes
  the more clearly justified next move for those two targets specifically — while decision (which
  already has a working result) continues on the tabular path independently.

## 9. Risks & Mitigations

- **8-way model comparison at N=91–195 risks a false "winner" from CV noise alone** — mitigated
  by reporting top 2-3 per target (§7) rather than just the argmax, and by requiring a clear
  margin over both baseline *and* the incumbent best before calling a result a genuine
  improvement (§8), not just a nominal one.
- **XGBoost is a new dependency** — not yet installed in the project `.venv`; needs
  `pip install xgboost` before implementation.
- **Feature scaling requirements differ by model** — SVM, kNN, MLP, and KDM need standardized
  features (as KDM already required in `exp_1`/`exp_2`); tree-based models (RF, XGBoost, Extra
  Trees) don't need it but aren't hurt by it either. Deferred to IMPLEMENTATION.md whether to
  scale uniformly for all 8 (simpler pipeline) or conditionally (matches each model's native
  assumptions) — a genuine implementation choice, not a design-level one.
- **MRI-to-factor attribution ambiguity (resolved this session)**: `cli_cspca` correlates with
  the MRI PCA signal (r=0.456) slightly more than `cli_pirads` does (r=0.412) — no principled
  basis to assign MRI to one factor's restricted group over the other, so it's excluded from all
  per-factor groups (§2) rather than arbitrarily assigned.
- **PSA-family reduction (`psa`+`psad` only) may lose real signal from `psap`/`psav`** —
  `exp_2`'s decision model didn't have this issue (it kept all 4), so this experiment can't
  cleanly isolate whether dropping `psap`/`psav` specifically helps or hurts, since MRI and model
  family are changing simultaneously. A dedicated ablation isolating just the PSA-family question
  would need to be a separate follow-up if this matters later.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, same as `exp_1`/`exp_2`)
- [ ] Config YAML — N/A, inline constants as in prior experiments
- [x] Dataset version: same as `exp_1`/`exp_2`
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — **N/A: project is not a git repository** (same caveat as prior experiments)

## 11. Next Steps

1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (Claude Code plan mode) covering: per-model
   hyperparameter choices appropriate to N=91–195, the feature-scaling decision (§9), the
   `xgboost` install, and which script(s) under `experiments/exp_3/scripts/` house the 8-model
   decision/confidence runners vs. the weights/reveal runners. Save as
   `experiments/exp_3/IMPLEMENTATION.md` before editing any files.
