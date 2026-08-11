# Experiment Design: Official-Schema Feature Scope + Comorbidity Grouping + KDM Decision Model for Task 1
**Experiment**: experiments/exp_2/
**Project**: challenge_chimera_2
**Date**: 2026-08-08
**Author**: TBD
**Status**: Complete

---

## 1. Hypothesis

Restricting the entire candidate feature set to the **11 officially documented Task-1 input
variables** (rather than all 47 engineered columns in `exp_1`, many of which — `vit_*`, `lab_*`,
`psa_tr_*`, `path_hist_*` — aren't part of the documented `structured-prompt.json` schema and may
not reliably exist at real inference time), combined with grouped comorbidity flags for `pmhx`,
improves the variable-weight models enough to beat `exp_1`'s naive per-factor baseline (mean
ordinal error 0.401), which `exp_1`'s full-47-column, count-only model (0.574) did not — and does
so with a feature set that's actually faithful to what the real submission will receive.

**Secondary hypothesis (schema-fidelity risk)**: `exp_1`'s HistGradientBoosting decision model
ranked `vit_heart_rate_bpm` / `vit_bp_systolic` as its top features — columns outside the official
schema. If those were noise specific to this CSV export rather than real signal, removing them
should not hurt (and may help) the decision model's F1/AUC.

**Third hypothesis (KDM for decision)**: a memory-based KDM classifier (same design as `exp_1`'s
`confidence_kdm` — frozen prototypes, only the kernel bandwidth trained) gives a better-calibrated
yes/no decision boundary than logistic regression / HistGradientBoosting on the 11-variable
feature set, following the same direction of improvement KDM showed for the confidence target in
`exp_1` (worse than baseline, but closer than plain logistic regression).

## 2. Experimental Setup

- **Dataset**: same as `exp_1` — `data/inputs.csv` + `data/ground_truth.csv`, 91 annotated cases
  for the confidence/weights/reveal-sequence targets, 195 for the decision model.
- **Feature universe, restricted to the 11 officially documented Task-1 input variables**
  (per the challenge's variable table), mapped to our columns:

  | Official variable | Column |
  |---|---|
  | psa | `cli_psa` |
  | psap | `cli_psap` |
  | psav | `cli_psav` |
  | psad | `cli_psad` |
  | vol | `cli_vol` |
  | age | `cli_age` |
  | pirads | `cli_pirads` |
  | dre | `cli_dre` |
  | cspca | `cli_cspca` |
  | bx | `cli_bx` |
  | pmhx | `txt_comorbidities` (via `cli_comorbidity_count` and/or 6 binary `comorb_*` flags) |

  **Encoding choices confirmed against actual value counts** (not assumed):
  - `pirads`: ordinal (1–5), as in `exp_1`.
  - `dre`: real categories are `Normal` (136), `Nodus` (49), `Not done` (5), `Abnormal` (4),
    `Suspicious` (1). Encoded **ordinal by clinical severity**: `Normal`(0) < `Abnormal`(1) <
    `Nodus`(2) < `Suspicious`(3). `Not done` is pulled out as its own `dre_not_done` missing-style
    flag rather than treated as a severity rung — it represents absence of an exam, not a finding.
  - `bx`: binary, `Positive`/`Negative` (`NaN` = not previously biopsied, imputed as its own
    category, same treatment as `exp_1`).
  - `pmhx`: 6 binary flags (`comorb_cardiometabolic`, `comorb_renal`, `comorb_bleeding_risk`,
    `comorb_respiratory`, `comorb_bph`, `comorb_other_unmatched`) in the "flags" comorbidity
    treatment, or the single `cli_comorbidity_count` ordinal in the "count" treatment — these
    are the two conditions of the comorbidity-treatment ablation axis (§6), not both used at once.
  - `psa`, `psap`, `psav`, `psad`, `vol`, `age`, `cspca`: continuous, 7 variables total.

  `ct` (clinical T-stage) is excluded — confirmed absent from both `data/inputs.csv` and
  `data/ground_truth.csv` in this training release (see prior investigation: only 2/195 cases
  have any T-stage mention at all, buried in free text, not a usable structured signal).

  **`fh` (family history) is handled separately, outside this restriction** — it's not a
  `structured-prompt.json` base field in the official schema (retrieved via the
  `get_family_history` tool call instead), so it keeps using its existing source
  (`cli_fh_binary` / `txt_family_history_narrative`) rather than being forced into the
  11-variable table it was never part of.
- **New feature (already implemented)**: `comorbidity_flags()` in `../../src/chimera_task1/features.py`
  — `comorb_cardiometabolic`, `comorb_renal`, `comorb_bleeding_risk`, `comorb_respiratory`,
  `comorb_bph`, `comorb_other_unmatched`, derived from `txt_comorbidities` (i.e. from `pmhx`).
  The "count-only" conditions below deliberately exclude these to isolate their effect from the
  broader schema-restriction change.
- **New code needed** (implementation plan, not yet written): a new, smaller feature-selection
  function (e.g. `select_official_feature_frame()`) restricted to the 11-variable table above,
  separate from `exp_1`'s `select_feature_frame()` (untouched — `exp_1`'s own results stay the
  47-column comparison point, no rerun needed); a `comorbidity_treatment` switch (`count` vs.
  `flags`) usable by every training script; and an extended `TASK1_VARIABLE_TO_FEATURE` for the
  per-factor restricted weight conditions now that the mapping is closer to 1:1 within this
  smaller table.
- **Models**: unchanged from `exp_1` — `OneVsRestClassifier(LogisticRegression(solver=liblinear,
  class_weight="balanced", C=0.5))` for weights/confidence/reveal; logistic regression +
  HistGradientBoosting for the decision model, same regularization as `exp_1`; **plus a new KDM
  decision model** — `KDMClassModel(dim_y=2)`, same memory-based configuration as `exp_1`'s
  `confidence_kdm` (prototypes frozen at the training data, `x_train=y_train=w_train=False`,
  only the RBF kernel bandwidth `sigma` trained via Adam), features `StandardScaler`-normalized
  first since the kernel needs comparable scales. Kept deliberately at the same conservative
  capacity as the confidence-KDM design for a fair comparison, even though N=195 (decision) is
  larger than N=91 (confidence) — revisit only if this condition's results suggest the frozen-
  prototype constraint is the bottleneck, not the feature set.
- **Evaluation**: same repeated out-of-fold CV (`KFold`/`RepeatedStratifiedKFold`, `n_splits=5`)
  and rubric-matching metrics as `exp_1` (`reasoning_labels.py`), so results are directly
  comparable to `exp_1`'s numbers without re-deriving a new baseline.

## 3. File Layout for This Experiment

```
experiments/exp_2/
├── DESIGN.md
├── scripts/                 ← 11-variable feature selector, per-factor feature-group definitions
├── results/
│   ├── decision_logistic_count/    ├── decision_logistic_flags/
│   ├── decision_hgb_count/         ├── decision_hgb_flags/
│   ├── decision_kdm_count/         ├── decision_kdm_flags/
│   ├── confidence_logistic_count/  ├── confidence_logistic_flags/
│   ├── confidence_kdm_count/       ├── confidence_kdm_flags/
│   ├── reveal_count/                ├── reveal_flags/
│   ├── weights_official_count/     ├── weights_official_flags/
│   └── weights_restricted_count/   └── weights_restricted_flags/
└── reports/
    └── summary.md
```
(16 condition folders total, listed in full in §5.)

## 4. Baselines

Reuses `exp_1`'s naive baselines directly (`weights_baseline`, `confidence_baseline`,
`reveal_baseline` — see `experiments/exp_1/results/`) rather than recomputing them, since they
are feature-blind and don't depend on anything changed here.

## 5. Proposed Conditions

Expanded to a fully-crossed ablation (16 conditions) rather than assuming a comorbidity-treatment
winner from the weights models carries over to the others — **comorbidity treatment** (count-only
vs. grouped-flags) is tested against every target, and the decision model's three classifier
families are kept as separate conditions rather than bundled.

"Official" feature scope means *all 11 official variables* (not `exp_1`'s 47-column frame);
"restricted" (weights only) means each factor sees only its own corresponding official
variable(s).

| Condition | Target | Comorbidity treatment | Classifier / scope |
|---|---|---|---|
| `decision_logistic_count` | decision | count | logistic, 11 official variables |
| `decision_logistic_flags` | decision | grouped flags | logistic, 11 official variables |
| `decision_hgb_count` | decision | count | HistGradientBoosting, 11 official variables |
| `decision_hgb_flags` | decision | grouped flags | HistGradientBoosting, 11 official variables |
| `decision_kdm_count` | decision | count | KDM (memory-based, sigma-only), 11 official variables |
| `decision_kdm_flags` | decision | grouped flags | KDM (memory-based, sigma-only), 11 official variables |
| `confidence_logistic_count` | confidence | count | OvR logistic, 11 official variables |
| `confidence_logistic_flags` | confidence | grouped flags | OvR logistic, 11 official variables |
| `confidence_kdm_count` | confidence | count | KDM (memory-based, sigma-only), 11 official variables |
| `confidence_kdm_flags` | confidence | grouped flags | KDM (memory-based, sigma-only), 11 official variables |
| `reveal_count` | reveal-sequence | count | MultiOutput logistic, 11 official variables |
| `reveal_flags` | reveal-sequence | grouped flags | MultiOutput logistic, 11 official variables |
| `weights_official_count` | variable-weights | count | all 11 official variables (minus `fh`) |
| `weights_official_flags` | variable-weights | grouped flags | all 11 official variables |
| `weights_restricted_count` | variable-weights | count | per-factor: own official variable(s) only |
| `weights_restricted_flags` | variable-weights | grouped flags | per-factor: own official variable(s) only |

`exp_1`'s existing `results/` (47-column frame, and confidence's original single KDM run)
remain the comparison point for the "does the 11-variable schema restriction help or hurt"
question — no need to re-run a "full 47-column" condition here.

## 6. Ablation Studies

Two orthogonal ablations, both now fully crossed:

- **Comorbidity treatment** (count-only vs. grouped-flags) × **every target** (decision,
  confidence, reveal, weights) — 8 of the 16 pairs above isolate this cleanly, since each target
  gets both treatments under an otherwise-identical setup.
- **Classifier family** (logistic/OvR-logistic vs. KDM) × **comorbidity treatment**, for decision
  and confidence specifically (HGB included as a third decision classifier, but only compared
  against logistic, not run through KDM's memory-based init).
- **Feature scope** (all-11-official vs. per-factor-restricted), for the weights models only —
  the one place "restricted" is a meaningful sub-scope, since decision/confidence/reveal aren't
  naturally decomposable per-factor.

## 7. Evaluation Protocol

- **Primary metric (weights)**: mean ordinal error across the 9 in-scope variable-weight factors
  (`fh` excluded — evaluated separately, unchanged from its `exp_1` source, not a condition in
  this experiment) — matches the official rubric's variable-weights component. Success
  threshold: clearly beats `exp_1`'s `weights_logistic` (0.574), ideally also beats the naive
  baseline (0.401). Compare all four `weights_*` conditions against each other and against both.
- **Primary metric (decision)**: F1, compared across all six `decision_*` conditions against
  each other, against `exp_1`'s `decision_logistic_clinical` (0.435) / `decision_hgb_clinical`
  (0.404), and against the naive "always yes" baseline (0.446). Also report ROC-AUC/PR-AUC for
  every decision condition, and predictive entropy for the two `decision_kdm_*` conditions —
  same diagnostics as `exp_1`.
- **Primary metric (confidence)**: ordinal distance across all four `confidence_*` conditions,
  compared against `exp_1`'s `confidence_logistic` (0.576) / `confidence_kdm` (0.564) and the
  naive baseline (0.527).
- **Primary metric (reveal)**: set precision, `reveal_count` vs. `reveal_flags`, compared against
  `exp_1`'s `reveal_logistic` (0.840) and the naive baseline (0.783).
- **Statistical rigor**: same as `exp_1` — repeated out-of-fold CV aggregates, no formal
  significance test (no per-repeat values persisted); note this limitation again in the report
  rather than silently repeating it.
- Results written to `results/<condition>/metrics.json`, same schema as `exp_1`, one file per
  condition (16 total).

## 8. Expected Results & Decision Rules

- **Comorbidity treatment**: for each target, compare its `_count` vs. `_flags` condition. If
  `_flags` wins consistently across most/all targets → adopt grouped flags as the standard
  comorbidity encoding going forward. If it's a wash or inconsistent → the flags aren't worth the
  added complexity outside whichever specific target(s) they helped.
- **Feature scope (weights only)**: if `weights_restricted_*` clearly beats `weights_official_*`
  on mean ordinal error → factor-restriction is worth adopting broadly. If not → the extra
  features weren't the problem for weights specifically.
- **Schema restriction (decision)**: if `decision_logistic_flags`/`decision_hgb_flags` match or
  beat `exp_1`'s 47-column results → confirms the dropped non-official columns (`vit_*`/`lab_*`/
  `psa_tr_*`/`path_hist_*`) weren't adding real signal, and the model is now schema-faithful at
  no cost. If clearly worse → those columns *were* carrying some signal despite being outside the
  official schema — a genuine trade-off to flag, not resolve silently.
- **KDM for decision/confidence**: if any `_kdm_*` condition beats its same-target,
  same-comorbidity-treatment logistic/HGB counterpart *and* the naive baseline → real evidence
  KDM is worth adopting as the model of choice for that target. If it only narrows the gap to
  baseline (as `exp_1`'s `confidence_kdm` did) → same "real but modest" read as before.
- **Overall**: if every condition across all four targets remains at or below its naive baseline
  → strong evidence the structured-feature ceiling from `exp_1` is real regardless of feature
  engineering, schema scope, or classifier family — prioritize the free-text / LLM-agent
  alternatives (already on record as follow-ups) over further tabular feature work.

## 9. Risks & Mitigations

- **Defining each factor's restricted feature group is still a small judgment call** even within
  just the 11 official variables — e.g., does the `psa` weight's restricted group include `psap`/
  `psav` (previous value / velocity, official variables without their own weight key), or just
  `cli_psa` alone? Get this reviewed in the implementation plan before running.
- **N=91 unchanged** — same small-sample risk as `exp_1`; kept models identical in capacity
  (no upgrade to a higher-capacity classifier) so any improvement is attributable to features/
  scope, not a stronger model confounding the comparison. With only 11 candidate variables total,
  overfitting risk should be substantially lower than `exp_1`'s 47-column frame regardless.
- **`comorb_other_unmatched` is 0 for all 91 training cases** (by construction — the vocabulary
  was derived from this exact training set) — it contributes nothing to these results and exists
  purely for validation/test-set robustness; don't misread its all-zero training column as a bug.
- **Dropping `vit_*`/`lab_*`/`psa_tr_*`/`path_hist_*` is a real trade-off, not a free lunch** —
  if any of those columns were carrying real (not spurious) signal, `decision_official` could
  come out worse than `exp_1`. That result would itself be informative (see §8), not a failure of
  this experiment.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, same as `exp_1`)
- [ ] Config YAML — N/A, inline constants as in `exp_1`
- [x] Dataset version: same as `exp_1`, noted there
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — **N/A: project is not a git repository** (same caveat as `exp_1`)

## 11. Next Steps

1. Review and accept this experiment plan.
2. Once accepted, produce an **implementation plan** (Claude Code plan mode) covering: the
   per-factor feature-group definitions, the `feature_scope` switch in the training loop, and
   which script(s) under `experiments/exp_2/scripts/` vs. `../../src/chimera_task1/` house the
   changes. Save it as `experiments/exp_2/IMPLEMENTATION.md` before editing any files.
