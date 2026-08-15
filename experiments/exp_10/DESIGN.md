# Experiment Design: External Full-Schema Replication (Frame.md's 37 Variables + MRI-PCA(2)) under ARD-KDM
**Experiment**: experiments/exp_10/
**Project**: challenge_chimera_2
**Date**: 2026-08-13
**Author**: TBD
**Status**: Complete — see `experiments/exp_10/reports/summary.md`. Verdict: CV showed a uniform
regression across all 9 conditions vs. `exp_9`'s 23-col ARD reference (decision -0.060); the
mandatory held-out check initially showed what looked like the best decision result this project
has ever produced (0.708 macro-F1) — but a same-session follow-up (leave-one-out CV + 10 repeated
held-out splits, §3b of the report) confirmed this was a lucky single split (repeated-split mean
0.509 ± 0.134, LOO 0.530, both agreeing with CV). The full-schema frame underperforms `exp_9`'s
23-column reference on every subtask and every verification method now available.
`confidence_kdm_dispersion_isotonic`'s dilution regression, which ARD closed at 23 columns,
re-opens at 48 (0.804→1.410 ordinal distance) — ARD's rescue capacity has a ceiling between 23 and
48 columns at N=91. Weights and reveal both regressed uniformly; reveal lost to its naive baseline
for the first time. Not adopted; `exp_9`'s 23-column frame remains the best-validated configuration,
now further confirmed by the same LOO/repeated-holdout check (four independent estimates in a
0.072 range).

---

## 1. Hypothesis

`Frame.md` (pasted into the project root this session) documents a second external reference —
the same `jmalagont97/prostate-cancer-reasoning` schema `exp_8` partially adopted (`cli_psav`/
`cli_psap`), but here as that project's **complete** tabular-model variable list (37 columns),
plus its own stated preprocessing convention (MinMax-scale continuous, one-hot categoricals, a
binary missing-flag column instead of imputation) and its own per-subtask model choices (a fuzzy
kNN for decision; confidence/weights/reveal left unspecified or "unstable/in experimentation").

**This experiment tests whether adopting that complete external variable set — encoded with its
own stated preprocessing, not this project's median-imputation convention — improves this
project's own best-performing backbone (`exp_9`'s ARD-KDM) over its current best reference point**
(`exp_9`'s 23-column ARD-KDM: 0.608 CV / **0.680 held-out** decision macro-F1, the best decision
result this project has produced). It is a genuinely different question from `exp_8`'s (which
added 4 hand-picked columns after project-native EDA) and from `exp_9`'s (which changed the
backbone, not the frame): here the *entire* feature set and its encoding convention both come from
an external source, largely unfiltered, and only the model (ARD-KDM, not their fuzzy kNN) is this
project's own.

**Backbone**: ARD-KDM only (`exp_9`'s per-dimension kernel bandwidth), per this session's decision
— the frame below is substantially wider than anything tried so far, and `exp_9` already
established that a shared scalar bandwidth degrades under added dimensions while ARD does not (or
does so far less). Re-running the scalar backbone on top of this frame would very likely just
re-confirm that finding a third time rather than add new information.

**Preprocessing**: `Frame.md`'s own convention (MinMax scaler for continuous variables, one-hot for
categoricals, explicit missing-flag columns, no imputation), per this session's decision — this
makes `exp_10` a genuine external-methodology comparison, at the cost of not being directly
apples-to-apples with `exp_6`–`exp_9`'s `build_preprocessor` (median-impute + one-hot +
`StandardScaler`) numbers. Both backbone and preprocessing are held fixed for this experiment —
isolating "does the wider external frame help" is the whole question this round; a
preprocessing-only or backbone-only ablation is a natural `exp_11`, not in scope here.

## 2. EDA on `Frame.md`'s 37 variables (91 annotated cases) — required before implementation

All 37 variables are present in `data/inputs.csv` (checked directly — nothing in `Frame.md`'s list
is unavailable). Profiling on the 91 labeled cases surfaced four findings that materially affect
how the frame should be built. Per this project's standing discipline (`exp_8`'s
`cli_isup_missing`/`cli_bx_missing` catch), duplicated information should not silently enter the
model twice just because it appears twice in a source list.

### 2a. `psa_tr_last_val` and `psa_tr_max` are exact duplicates of `cli_psa` — drop both

Row-by-row equality check (not just matching aggregate stats): `cli_psa == psa_tr_last_val` and
`cli_psa == psa_tr_max` hold for **all 91 cases** (`r = 1.0` in both cases, exactly, not
approximately). This makes sense structurally — for a monotonically-rising PSA trend, "current
PSA," "last trend reading," and "trend maximum" are the same number by construction — but it means
including all three would enter the identical variable into the model three times under three
names, silently doubling that dimension's effective weight relative to every other feature no
matter what `ARD`'s own `σⱼ` later learns for it. **Recommendation: drop `psa_tr_last_val` and
`psa_tr_max`, keep `cli_psa`.**

### 2b. The rest of the PSA-trend family is highly but not perfectly correlated with `cli_psa`/`cli_psav` — keep, let ARD sort it out

`psa_tr_mean` (`r=0.988`), `psa_tr_delta` (`r=0.997`), and `psa_tr_slope` (`r=0.971`) with
`cli_psa`, and `psa_tr_slope` with `cli_psav` (`r=0.952`), are all very highly correlated but not
identical — real, if compressed, additional information (e.g. `psa_tr_delta` and `psa_tr_slope`
differ from `cli_psa` alone by incorporating the trend's *starting* point, not just its endpoint).
Unlike 2a, this isn't a literal duplicate, so it's exactly the kind of redundancy `exp_9`'s ARD
mechanism exists to handle (down-weighting a near-collinear dimension via a large trained `σⱼ`
rather than a person pruning it by hand). **Recommendation: keep `psa_tr_count`, `psa_tr_first_val`,
`psa_tr_min`, `psa_tr_mean`, `psa_tr_delta`, `psa_tr_slope`** (6 of the original 8 `psa_tr_*`
columns, after 2a's 2-column drop) — and treat `psa_tr`'s post-hoc `σⱼ` values in the eventual
`importance_comparison`-style analysis as a direct test of whether ARD actually does this.

### 2c. ISUP and Gleason primary/secondary are highly correlated but not deterministic — keep all three, flagged as a genuine judgment call

`path_hist_bx_isup` is clinically derived from the Gleason primary+secondary pattern by a known
grading table, so a priori this looked like 2a's situation. Checking directly: among the 67 cases
with all three present, grouping by `(gl_prim, gl_sec)` gives a single ISUP value for most
combinations but **not all** (one combination maps to 2 distinct observed ISUP values) —
`corr(isup, gl_prim) = 0.786`, `corr(isup, gl_prim + gl_sec) = 0.817`. Strongly related, but this
dataset's ISUP field is not a pure deterministic function of the two Gleason fields as recorded
(possibly reflecting tertiary-pattern adjustments or annotation noise upstream of this project).
**Decision (confirmed 2026-08-13): keep all three.** The alternative — dropping `gl_prim`/`gl_sec`
in favor of `isup` alone (already validated as `exp_8`'s single strongest confidence/decision
correlate), trading 2 fewer columns for a small amount of information given `r>0.79` — was
weighed explicitly and rejected in favor of staying faithful to `Frame.md`'s full schema beyond
the two literal-duplicate drops in §2a; ARD's own `σⱼ` is left to determine how much marginal
relevance `gl_prim`/`gl_sec` retain beyond `isup`, same philosophy as §2b's PSA-trend family.

### 2d. `vit_smoking_pack_years`'s missingness is 100% redundant with `vit_smoking_status == "Never"` — no separate flag

All 26 cases missing `vit_smoking_pack_years` are exactly the 26 cases with
`vit_smoking_status == "Never"` (verified by direct crosstab, 91/91 match) — never-smokers simply
don't have a pack-years value to record. This is the same pattern as `exp_8`'s
`cli_isup_missing`/`cli_bx_missing` redundancy: a separate `vit_smoking_pack_years_missing` flag
would duplicate information the `vit_smoking_status` one-hot encoding already carries.
**Recommendation: fill missing `vit_smoking_pack_years` with 0** (a defensible physiological
value — never-smokers truly have zero pack-years, not an unknown quantity) **and add no separate
missing-flag column.**

### 2e. `path_hist_bx_isup`/`gl_prim`/`gl_sec` all share `cli_bx`'s exact missingness pattern — one shared flag, not four

All three pathology fields are missing for exactly the same 24 cases as `cli_bx` (verified: 24/24
match on all three, i.e. these fields are only ever populated when a biopsy was actually
performed). A separate missing-flag per field would triple-count the same "no biopsy" signal.
**Recommendation: one shared `cli_bx_missing` flag** (as this project has used since `exp_3`)
covers all four fields' missingness; `cli_pirads` (1 case missing, overlapping but not identical
to `cli_bx`'s 24) and `cli_fh_binary` (3 cases missing, a distinct pattern) each still need their
own dedicated flag.

### 2f. Two low-information columns, decided (2026-08-13)

`cli_allergies_count` is 0 for 83/91 cases (91.2%) — very low variance, likely to end up with a
large trained `σⱼ` (ARD marking it low-relevance) rather than contributing much. **Decision: keep
it.** Dropping it would save exactly 1 of 48 columns — negligible dimensionality relief — for no
real benefit over staying faithful to `Frame.md`'s schema; a near-zero relevance score for it in
the eventual `importance_comparison` is an expected, useful confirmation, not a surprise to guard
against. `cli_months` takes only 3 distinct values (`{1, 2, 3}`, roughly even split, 32/30/29).
**Decision: encode as a single MinMax-scaled continuous column, not one-hot.** Nothing in the data
suggests a non-monotonic relationship that one-hot's extra 2 columns would be needed to capture;
continuous preserves the natural order at negligible cost either way.

## 3. Feature Frame

**37 `Frame.md` variables, minus the 2a duplicates (`psa_tr_last_val`, `psa_tr_max`) = 35 kept
fields, plus MRI-PCA(2) = 37 raw source fields before encoding.** After `Frame.md`'s own
preprocessing convention:

| Encoding | Columns | Count |
|---|---|---|
| MinMax-scaled continuous (complete, no missing) | `cli_age`, `cli_psa`, `cli_psap`, `cli_psav`, `cli_psad`, `cli_vol`, `cli_months`, `cli_cspca`, `cli_comorbidity_count`, `cli_allergies_count`, `cli_ipss_score`, `vit_weight_kg`, `vit_height_cm`, `vit_bmi`, `vit_bp_systolic`, `vit_bp_diastolic`, `vit_heart_rate_bpm`, `vit_smoking_pack_years`, `psa_tr_count`, `psa_tr_first_val`, `psa_tr_min`, `psa_tr_mean`, `psa_tr_delta`, `psa_tr_slope`, `lab_creatinine_mg_dl`, `lab_free_psa_ng_ml`, `lab_free_total_ratio`, `mri_pca_0`, `mri_pca_1` | 29 |
| MinMax-scaled continuous, per-fold median-imputed | `cli_pirads`, `path_hist_bx_isup`, `path_hist_bx_gl_prim`, `path_hist_bx_gl_sec`, `cli_fh_binary` | 5 |
| Missing-value flags | `cli_pirads_missing`, `cli_bx_missing` (shared by `cli_bx`/`isup`/`gl_prim`/`gl_sec`, per §2e), `cli_fh_missing`, `mri_missing` | 4 |
| One-hot categorical | `cli_dre` (5 levels: Normal/Nodus/Abnormal/Not done/Suspicious), `cli_bx` (2 levels: Positive/Negative — missing rows get `(0,0)`, covered by `cli_bx_missing` instead of a 3rd dummy level), `vit_smoking_status` (3 levels: Never/Ex-smoker/Current) | 10 |
| **Total** | | **48** |

For comparison: `exp_6`'s original frame was 19 columns, `exp_8`/`exp_9`'s expanded frame was 23.
**48 is roughly double `exp_9`'s already-flagged-as-risky frame** — the central risk this design
must be evaluated against (§7).

*(Correction, implementation planning 2026-08-13: this table originally stated 42 raw fields / 46
encoded columns, then a first-pass correction stated 47 — both wrong. Verified programmatically:
37 raw source fields, **48** encoded columns. No decisions changed, only the arithmetic.)*

`cli_pirads` itself, after its 1 missing value is filled (median of the 90 present values, the one
continuous-variable exception to "no imputation" that `Frame.md`'s own note implies is still
needed pre-scaling — its missing-flag is what makes this defensible, not imitating true
imputation-without-a-flag), is MinMax-scaled like the rest.

## 4. Baselines

| Subtask | Reference | Metric | Value |
|---|---|---|---|
| Decision | `exp_9` ARD-KDM, 23-col, held-out | macro-F1 | **0.680** (this project's best decision result to date) |
| Decision | `exp_9` ARD-KDM, 23-col, CV | macro-F1 | 0.608 |
| Decision | `exp_6` scalar-KDM, 19-col, held-out (original reference) | macro-F1 | 0.593 |
| Decision | Extra Trees incumbent | macro-F1 | 0.650 |
| Confidence | `exp_9` ARD-KDM 23-col `dispersion_isotonic` (best-validated ARD confidence result) | ord. dist. / macro-F1 | 0.804 / 0.132 |
| Confidence | `exp_6` `blend`, 19-col scalar (best macro-F1 seen for confidence pre-`exp_9`) | macro-F1 | 0.269 |
| Confidence | `confidence_svm` incumbent | ord. dist. | 0.468 |
| Weights | `exp_9` ARD-KDM 23-col `occlusion` | ord. err. / macro-F1 | 0.459 / 0.259 |
| Weights | `weights_svm` incumbent | ord. err. | 0.382 / 0.392 |
| Reveal | `exp_9` ARD-KDM 23-col `occlusion` (best reveal result to date) | set-precision / macro-F1 | **0.823** / **0.599** |
| Reveal | `reveal_flags` incumbent | set-precision | 0.853 |

## 5. Proposed Conditions

| Condition | Target | Notes |
|---|---|---|
| `decision_kdm_ard_fullschema` | decision | fixed config (`n_epochs=300, lr=1e-2, sigma_mult=1.0`), no search, per §1 |
| `confidence_kdm_{5 signals}_ard_fullschema` | confidence | same 5-signal family as `exp_6`/`exp_9` |
| `weights_kdm_{occlusion,kernel_distance,blend}_ard_fullschema` | variable-weights | |
| `reveal_kdm_ard_fullschema` | reveal-sequence | same 4 modeled sections (`family_history`/`pathology_report` still 0/91 positive) |
| `holdout_eval_fullschema` | decision | mandatory — same fixed 19-case held-out split used since `exp_3` |
| `importance_comparison_fullschema` | — | trained `σⱼ` vs. `exp_5`'s solved set, extending `exp_9`'s diagnostic to this frame |

## 6. Ablation Studies

- **Full-schema ARD vs. `exp_9`'s 23-col ARD** — the core comparison, both CV and (mandatory)
  held-out, isolating "does the wider external frame help beyond `exp_9`'s already-curated one."
- **Does `psa_tr`'s retained-but-collinear family (§2b) end up down-weighted by ARD?** Check the
  trained `σⱼ` for `psa_tr_mean`/`delta`/`slope` against `cli_psa`/`cli_psav`'s — if ARD assigns
  them large `σⱺ` (low relevance) despite their inclusion, that's direct evidence the mechanism
  behaves as intended on real near-duplicate data, not just the literal duplicates 2a already
  removed by hand.
- **`σⱼ` vs. `exp_5`'s solved set, again** — `exp_9`'s version of this ablation found only 2/5
  agreement on both frames tried so far; worth checking whether a much larger, externally-sourced
  frame changes that pattern at all, or whether it's a stable property of ARD's relevance signal
  vs. what a per-factor SVM search finds important.

## 7. Evaluation Protocol

- Same 5-fold × 10-repeat CV (`RANDOM_STATE=0`) as every KDM condition since `exp_3`.
- **Held-out check is mandatory for decision** — same discipline as every KDM experiment since
  `exp_7`, arguably more critical here than ever given 48 columns against N=91 (~73/fold): this is
  the widest frame this project has tried by a wide margin, and `exp_9` already showed even ARD's
  19-column CV gains can evaporate on held-out.
- Macro-F1 native for all four subtasks, alongside each subtask's official rubric metric, per the
  convention established starting `exp_9`.
- Weights and reveal: per-factor / per-section breakdowns required, not aggregate-only.

## 8. Expected Results & Decision Rules

- If full-schema ARD clearly beats `exp_9`'s 23-col ARD on held-out decision **and** the gap
  doesn't reverse under CV → a genuine case for adopting the wider external schema, and worth a
  follow-up hyperparameter search specifically for this frame (deferred from this round, §1).
- If full-schema ARD matches or only marginally beats `exp_9`'s 23-col result → the marginal
  variables (§2b's collinear PSA-trend family, the vitals, the labs) aren't pulling their weight
  at this N, and `exp_9`'s leaner, EDA-curated frame remains the better default — a genuinely
  useful negative result given how much wider this frame is.
- If full-schema ARD is clearly worse, especially if CV looks fine but held-out doesn't → direct
  evidence that even ARD's per-dimension bandwidth has a dimensionality ceiling somewhere between
  23 and 48 columns at N=91, worth stating as a concrete finding for any future frame-expansion
  attempt in this project.
- Regardless of decision's outcome, report confidence/weights/reveal on their own terms against
  §4's baselines — no subtask's result needs to "win" for this experiment to be informative.

## 9. Risks & Mitigations

- **48 columns is roughly double `exp_9`'s already-flagged-as-risky 23-column frame, against
  N=91 (~73/fold).** This is this experiment's central risk, not a side note. The mandatory
  held-out check exists specifically to catch CV-only overfitting, as it has twice before
  (`exp_7`, and partially `exp_9`'s 19-column ARD result).
  the frame this large — a legitimate `exp_11` follow-up if this round's held-out result looks
  promising but underpowered.
- **Preprocessing mismatch with the rest of the project** (§1) means `exp_10`'s absolute numbers
  aren't cleanly comparable to `exp_6`–`exp_9`'s on the metric alone — only the CV/held-out
  agreement *within* this experiment, and the relative comparison to `exp_9`'s own reference
  numbers (computed under a different preprocessing pipeline), are meaningful. This is a genuine
  limitation of combining "external frame" and "external preprocessing" in one experiment, not an
  oversight — flagged here so the eventual report doesn't overstate the comparison's cleanliness.
- **§2c's Gleason/ISUP call (keep all three) was made explicitly, not by default** — the
  drop-`gl_prim`/`gl_sec` alternative was weighed and rejected in favor of schema faithfulness.
  Worth revisiting if `gl_prim`/`gl_sec` show large trained `σⱼ` in the eventual
  `importance_comparison` (further evidence `isup` alone would have sufficed) — a natural,
  already-scoped `exp_11` ablation rather than a reason to reopen this design.
- **`family_history`/`pathology_report` reveal-sections remain unmodeled** (0/91 positive
  examples) — unaffected by this experiment's frame choice, same as `exp_8`/`exp_9`.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, unchanged from `exp_1`–`exp_9`)
- [ ] Config YAML — N/A, inline constants as in prior experiments
- [x] Dataset version: same as `exp_3`–`exp_9`
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — record the commit this experiment builds from

## 11. Next Steps

1. ~~Review this design~~ — done 2026-08-13: every §2 EDA judgment call (2a's 2-column drop, 2c's
   deliberate keep-all-three, 2f's `cli_months`/`cli_allergies_count` calls) was walked through
   point by point and confirmed; §1's ARD-only/`Frame.md`-preprocessing decisions and §9's
   dimensionality risk were reviewed alongside. §3's 48-column frame is final.
2. Next: an implementation plan (Claude Code plan mode) covering the new
   `Frame.md`-convention preprocessing function (MinMax + one-hot + explicit flags, replacing
   `build_preprocessor` for this experiment only, fit per-fold like every prior experiment's
   preprocessing), the frame-selection function applying §2/§3's exact column list, and reuse of
   `exp_9`'s `ard_kernel.py` unchanged (`fit_kdm_backbone_ard`, `compute_signals_ard`,
   `occlusion_delta_ard`, `dm_rbf_variance_ard` all remain valid — nothing about ARD itself depends
   on which preprocessing produced its input matrix). Save as `experiments/exp_10/IMPLEMENTATION.md`
   before editing any files.
