# Experiment Design: Direct ARD-KDM for Confidence
**Experiment**: experiments/exp_11/
**Project**: challenge_chimera_2
**Date**: 2026-08-14
**Author**: TBD
**Status**: Complete — see `experiments/exp_11/reports/summary.md`. Verdict: direct training (not
ARD) was the active ingredient — confirmed by CV/LOO/repeated-holdout (4 independent methods) as a
decisive macro-F1 win over every derived-signal condition tried in `exp_6`–`exp_10` (0.47–0.51 vs.
the prior best 0.283), but ordinal distance converges to baseline (0.527), not below the incumbent
(0.468). ARD showed no clear improvement over `exp_3`'s original scalar-backbone direct training
(0.530/0.508) — a wash. A single held-out split initially showed an identical, suspicious 0.316 for
both frames; an unplanned repeated-holdout follow-up (triggered by that suspicion, per this
project's own standing rule from `exp_10`) confirmed it was a lucky outlier, converging with
CV/LOO's 0.47–0.55 once averaged over 10 seeds.

---

## 1. Hypothesis

Since `exp_6`, every KDM confidence result has come from the same architecture: fit one KDM on the
**decision** label, then read confidence off signals derived from that fitted model's internal
state (entropy, dispersion, participation — see `experiments/exp_9` and `exp_10`'s reports for the
full mechanism). Across 25 conditions and 5 experiments (`exp_6`–`exp_10`), **not one has beaten the
`confidence_svm` incumbent, or even the naive baseline**.

Before that architecture existed, `exp_2`/`exp_3` did something different: a KDM trained **directly**
on the confidence label itself (a genuine 3-class classifier, `y_train` = confidence rank, not
decision). `exp_3`'s `confidence_kdm` scored 0.530 ordinal distance / **0.508 macro-F1** — worse
than incumbent on the official metric, but the **best macro-F1 any confidence condition has ever
achieved in this project**, comfortably ahead of every derived-signal condition tried since
(`exp_6`'s best is 0.269; `exp_9`'s ARD-backbone best is 0.283). That result used the *original*,
worst-performing backbone available at the time — no ARD, a smaller/different frame, none of the
architectural improvements `exp_9`/`exp_10` later validated.

**This experiment asks the question nobody has asked since: does direct training + the two real
backbone improvements this project has since confirmed (ARD, the 23-column frame) close more of
the gap to the incumbent than five experiments of derived-signal refinement did?** It is a
different axis entirely from every prior confidence experiment — not a new signal, not a new
recalibration method, a different *target* for the same gradient. `exp_5`'s equivalent test for
**weights** (`weights_kdm`, directly trained) scored *worse* than baseline, so this is confidence-
specific — nothing here proposes retrying direct training for weights.

## 2. Experimental Setup

### 2a. Model: `exp_9`'s ARD-KDM, unchanged, retargeted

No new KDM code. `experiments/exp_9/scripts/ard_kernel.py`'s `fit_kdm_backbone_ard()` and
`compute_signals_ard()` are reused exactly as they exist — the only change is what gets passed as
`y_train`: the 3-level confidence rank (`CONFIDENCE_RANK`) instead of the binary decision label,
and `n_classes=3` instead of `2`. This is the same function signature `exp_1`'s
`train_confidence_kdm.fit_predict_kdm` and `exp_2`/`exp_3`'s KDM confidence conditions already used
for the scalar backbone — ARD's version needs zero new logic to support this, only a different
caller.

### 2b. Frame: 23-column primary, 19-column for comparison

Both are `exp_9`'s already-fitted, already-validated frames (`select_exp8_feature_frame`,
`select_exp3_feature_frame`) — no new frame work. 23-column is primary (this project's
best-validated frame for the decision-trained ARD backbone, per `exp_9`'s four-way-confirmed
result); 19-column is included for the same reason `exp_9` tested both — isolating whether a
direct-trained target responds to frame width the same way the decision target did.

### 2c. Hyperparameters: fixed, no search — same guardrail as `exp_9`/`exp_10`

`n_epochs=300, lr=1e-2, sigma_mult=1.0` — `exp_9`'s original ARD defaults, unchanged. This is a new
combination (direct training × ARD) with no stable baseline yet to search around; per the same
reasoning `exp_9`'s and `exp_10`'s designs both used, a hyperparameter search is deferred to a
follow-up once this round establishes whether the combination is worth searching around at all.

### 2d. First experiment under the new reporting standard

`exp_11` is the first experiment run under both conventions confirmed this session
(`experiments/INDEX.md`'s "📈 full metric-suite reporting initiative" and "📐 leave-one-out (LOO)
evaluation protocol" notes):
- **Full metric suite for confidence** (3-class ordinal): accuracy, macro-F1, one-vs-rest macro
  AUROC, ordinal distance (official metric, unchanged), multiclass Brier score — all reported from
  the start, not backfilled after the fact.
- **Three evaluation protocols, all mandatory, none replacing another**: the existing 5-fold ×
  10-repeat CV; a held-out check on the same fixed 19-case split used since `exp_3` (which already
  has precedent for scoring confidence, not just decision — `exp_3`'s original `holdout_eval.py`
  checked `confidence_svm`/`confidence_kdm` on this exact split); and LOO (91-fold, deterministic,
  pooled aggregation), new as of this experiment for any subtask other than decision.

## 3. File Layout for This Experiment

- `experiments/exp_11/scripts/run_confidence_direct_ard.py` — CV loop, both frames, full metric
  suite, mirroring `exp_9`'s `run_signals_*col.py` structure but confidence-only (no decision/
  weights loop needed — this experiment doesn't touch either).
- `experiments/exp_11/scripts/holdout_eval_confidence_direct_ard.py` — held-out check, both frames,
  same fixed split, reusing `mri_pca_train_only`/`fit_transform_features` (19-col) and
  `select_exp8_feature_frame`'s own preprocessing (23-col) unchanged from prior experiments.
- `experiments/exp_11/scripts/loo_confidence_direct_ard.py` — LOO check, both frames, following
  `experiments/exp_10/scripts/verify_decision_loo_repeated_holdout.py`'s `run_loo()` pattern
  (pooled prediction, single deterministic pass) generalized to the 3-class confidence target.
- No changes to `exp_6`–`exp_10`'s scripts, the `kdm` library, or `src/chimera_task1/*.py` — same
  rule as every prior experiment.

## 4. Baselines

| Comparison | Ordinal distance | Macro-F1 | Note |
|---|---|---|---|
| Naive baseline (majority class) | 0.527 | 0.260 | the floor |
| Incumbent (`confidence_svm`, `exp_3`) | **0.468** | 0.404 | the target to beat |
| Best derived-signal, any experiment (`exp_6` `entropy_isotonic`) | 0.731 | 0.223 | official-metric-best derived signal |
| Best derived-signal macro-F1 (`exp_9` ARD 19-col `blend`) | 0.836 | 0.283 | macro-F1-best derived signal |
| **`exp_3`'s directly-trained `confidence_kdm` (scalar backbone, pre-ARD)** | 0.530 | **0.508** | **the result this experiment is trying to improve on** |

## 5. Proposed Conditions

| Condition | Frame | Protocol |
|---|---|---|
| `confidence_kdm_direct_ard_19col` | 19-col | CV (5×10) |
| `confidence_kdm_direct_ard_23col` | 23-col | CV (5×10) |
| `holdout_confidence_direct_ard_19col` / `_23col` | both | held-out (n=19, fixed split) |
| `loo_confidence_direct_ard_19col` / `_23col` | both | LOO (91-fold, pooled) |

## 6. Ablation Studies

- **Direct training vs. derived signals, same backbone, same frame** — the core comparison:
  `confidence_kdm_direct_ard_23col` vs. `exp_9`'s `confidence_kdm_{5 signals}_ard_23col`. Isolates
  whether direct supervision helps, holding architecture and frame fixed.
- **Does direct training respond to frame width the way decision did?** `exp_9` found ARD decision
  improved 19→23 cols; `exp_10` found it broke down further out. Comparing
  `confidence_kdm_direct_ard_19col` vs. `_23col` checks whether a directly-trained confidence
  target shows the same frame sensitivity, or a different pattern — confidence's feature-relevance
  structure may not match decision's.
- **Does ARD recover what `exp_3`'s scalar backbone couldn't?** `confidence_kdm_direct_ard_23col`
  vs. `exp_3`'s original `confidence_kdm` (scalar, different frame) — the most direct test of this
  experiment's central hypothesis.

## 7. Evaluation Protocol

- CV: 5-fold × 10-repeat (`RANDOM_STATE=0`), same as every KDM condition since `exp_3`.
- **Held-out and LOO are both mandatory this round** — first-ever LOO check for a non-decision
  subtask, and confidence's first held-out check since `exp_3`'s original one (never repeated for
  any `exp_6`–`exp_10` derived-signal condition).
- Full metric suite (§2d) reported natively for every condition, per the confirmed reporting
  initiative — never only the official metric, never only macro-F1.
- If any single result looks like a standout (echoing `exp_10`'s lucky-split lesson), a repeated-
  holdout check (10 seeds) is available as an established follow-up tool
  (`verify_decision_loo_repeated_holdout.py`'s pattern) before it gets reported as a finding.

## 8. Expected Results & Decision Rules

- If direct-ARD clearly beats every derived-signal condition on **both** ordinal distance and
  macro-F1, confirmed by CV **and** held-out **and** LOO agreeing → strong evidence direct
  supervision was the missing piece, not the signal formulas — worth revisiting as confidence's new
  default approach, and worth asking whether the same logic applies anywhere else.
- If direct-ARD beats derived signals on macro-F1 but not ordinal distance (echoing `exp_3`'s own
  pattern) → a genuine, if partial, improvement — consistent with this project's recurring
  macro-F1-vs-ordinal-distance disagreement, not a new problem.
- If direct-ARD does **not** clearly beat `exp_3`'s original scalar-backbone `confidence_kdm` →
  evidence that ARD's benefit is decision-specific (tied to how decision's feature-relevance
  structure interacts with the frame), not a general KDM improvement — a real, useful negative
  result narrowing where ARD actually helps.
- Regardless of outcome vs. incumbent: if CV/held-out/LOO disagree with each other (as they have in
  four straight architecture experiments), treat that disagreement as the primary finding, not an
  inconvenience to average away — same discipline as `exp_7`–`exp_10`.

## 9. Risks & Mitigations

- **A second free-standing KDM (confidence-specific, not decision-derived) reintroduces the
  per-subtask-model variance risk `exp_6`'s original design deliberately avoided.** Mitigated the
  same way every KDM in this project already is: frozen prototypes (`x_train=False`), frozen labels
  (`y_train=False`), only `σⱼ` trained — the lowest-variance configuration available, same as every
  other KDM fit in this codebase.
- **N=91 with a 3-class target means smaller effective per-class counts than binary decision** —
  already a known constraint (`exp_3`'s original confidence work), not new to this experiment, but
  worth remembering when interpreting a single condition's result.
- **This is confidence-only, deliberately** — `exp_5`'s equivalent direct-training test for weights
  scored below baseline, so extending this same idea to weights is not proposed here; if this
  round's result is positive, weights remains a separate, lower-prior candidate for its own test.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, unchanged from `exp_1`–`exp_10`)
- [ ] Config YAML — N/A, inline constants as in prior experiments
- [x] Dataset version: same as `exp_3`–`exp_10`
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — record the commit this experiment builds from

## 11. Next Steps

1. Review this design — in particular §2c's no-search guardrail and §6's ablation framing — before
   implementation.
2. Once accepted, an implementation plan (Claude Code plan mode) covering the three new scripts
   (§3), confirming `fit_kdm_backbone_ard`/`compute_signals_ard` need zero changes to accept a
   3-class target (already true for the scalar-backbone equivalent, `fit_predict_kdm`, since
   `exp_1`), and the exact multiclass AUROC/Brier formulas from `experiments/INDEX.md`'s metric-suite
   note. Save as `experiments/exp_11/IMPLEMENTATION.md` before editing any files.
