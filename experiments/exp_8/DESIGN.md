# Experiment Design: Expanded Variables + Hyperparameter Tuning + Reveal-Sequence via Uncertainty Reduction
**Experiment**: experiments/exp_8/
**Project**: challenge_chimera_2
**Date**: 2026-08-12
**Author**: TBD
**Status**: Complete — see `reports/summary.md`

---

## 1. Hypothesis

Three extensions to `exp_6`'s KDM backbone, bundled into one experiment because all three feed
the same shared model:

**(a) Restoring `psav`/`psap`/`cli_isup`/`vit_bmi` recovers real signal, not just noise `exp_3`
correctly pruned.** `exp_3` dropped `cli_psap`/`cli_psav` for collinearity with `cli_psa` (r=0.99,
r=0.95) — a defensible call for the 8-model sklearn search, where near-duplicate columns mostly
just add redundant dimensions. But KDM's single global kernel bandwidth treats every dimension as
equally scaled, not per-feature-regularized the way SVM/Extra Trees are — collinear columns may
matter differently to a kernel-distance method than to those. An external, independently-built
implementation of this same CHIMERA-Agent Task 1 challenge
(<a href="https://github.com/jmalagont97/prostate-cancer-reasoning">jmalagont97/prostate-cancer-reasoning</a>)
confirms `age, psa, vol, pirads, psad, psav, psap, dre` is a genuine, complete clinical variable
set another team settled on — restoring `psav`/`psap` closes that exact gap. `cli_isup` and
`vit_bmi` are this session's own EDA findings (§2a): the strongest decision-target correlate and
the strongest confidence-target correlate of any candidate checked, respectively, neither
collinear with what's already kept.

**(b) Hyperparameter tuning, re-attempted on the expanded frame — a real re-test, not a blind
retry.** `exp_7` searched 144 hyperparameter combinations on the *original* 19-column frame and
found nothing that survived held-out verification (CV showed +0.029 macro-F1; the held-out check
showed −0.102, worse). That result answered "does retuning help *this* feature set" — it did not
answer "does retuning help once genuinely new, non-redundant information is present," which is a
different question given more columns changes both the loss landscape and what a well-chosen
`sigma`/learning-rate combination needs to capture. This experiment reuses `exp_7`'s exact search
methodology (grid, protocol, held-out check) applied to the 23-column frame instead, with two
corrections from `exp_7`'s own stated lessons (§7): the "clear margin" bar is set relative to
`exp_6`'s *measured* CV std (0.045), not an arbitrary constant, and the held-out check is
mandatory before any result is reported as genuine, not an afterthought.

**(c) Reveal-sequence can be derived from the same shared backbone, extending `exp_6`'s "one
model, several readouts" pattern to a fourth target for the first time.** No prior experiment has
attempted this — `reveal` has always used a separate `MultiOutputClassifier`/OvR-logistic model
(`exp_1`–`exp_4`). The mechanism proposed here: for each modeled reveal section, measure how
much the backbone's own decision-entropy *increases* when that section's associated feature group
is occluded (reusing `exp_6`'s occlusion machinery, applied to entropy instead of `p(yes)`) — a
section whose absence spikes uncertainty the most is the one most "worth" revealing.

**Correction found during implementation planning (2026-08-12)**: checked how many of the 91
labeled cases actually have each of the 6 reveal sections marked as revealed —
`previous_notes` 77/91, `psa_trend` 79/91, `radiology_report` 88/91, `laboratory_results` 41/91,
**`family_history` 0/91, `pathology_report` 0/91**. Every prior reveal-sequence model in this
project (`exp_1`–`exp_5`) already handles this by dynamically restricting to sections with at
least one positive example (`experiments/exp_3/scripts/run_reveal.py:32`) — `exp_8`'s mechanism
follows that same, already-established convention, so only **4 of 6 sections are actually
modeled**, not 6. `family_history` staying unmapped was already a disclosed limitation (below);
`pathology_report`'s `cli_isup` mapping (§2b) is still well-motivated for the *decision* target
(§2a's own justification), but it doesn't get exercised by the reveal mechanism at all — a
section with zero positive examples can't be meaningfully modeled regardless of how good its
feature-group mapping is. This isn't worked around silently; the two sections are excluded
exactly the way every prior reveal model already excludes them.

**Explicit caution**: (b) carries a documented, specific failure mode from `exp_7` — do not treat
a CV improvement as real without the held-out check (§7). (c)'s mechanism is genuinely
exploratory even for the 4 sections it does model. (†`pathology_report`'s mapping to `cli_isup`,
§2b, matters for the frame's decision-target value, not for reveal — see correction above; only
`family_history` remains genuinely unmapped.)

## 2. Experimental Setup

### 2a. Expanded feature frame (23 columns)

Adds `cli_psav`, `cli_psap` (external-reference alignment, §1a), `cli_isup` and `vit_bmi` (own EDA
findings, below) to `exp_3`'s 19-column frame:

| Group | Columns | Change from exp_3/6/7 |
|---|---|---|
| Clinical (14, was 10) | `cli_psa`, `cli_psad`, **`cli_psav`, `cli_psap`**, `cli_vol`, `cli_age`, `cli_cspca`, `cli_pirads`, `cli_dre_ordinal`, `cli_dre_not_done`, `cli_bx_positive`, `cli_bx_missing`, **`cli_isup`, `vit_bmi`** | +4 |
| Comorbidity (6) | unchanged | — |
| MRI (3) | unchanged | — |

`cli_cspca`, `cli_bx_*`, and the 6 comorbidity flags are **kept**, not dropped to mirror the
external reference exactly — `bx` in particular is one of this project's own 5 established
"solvable" weight factors (`exp_5`–`exp_7`); dropping it to match a different team's narrower
schema would contradict evidence already on record in this project.

**`cli_isup`** (prior-biopsy ISUP grade group, 0–5) was found via a dedicated EDA pass over
`inputs.csv`'s 53 non-embedding columns before finalizing this design (session of 2026-08-12),
specifically to add real signal without duplicating what's already in the frame:
- **Strongest univariate correlation with the decision target of any candidate checked**
  (r=−0.195, vs. r ≤ 0.14 for every other candidate examined: `psa_tr_*`, `lab_free_psa_ng_ml`,
  `lab_free_total_ratio`, `lab_creatinine_mg_dl`, `cli_fh_binary`, vitals, `cli_ipss_score`).
- **Not collinear** with what's already kept — r=0.38 with `cli_cspca` (real but moderate
  overlap, not near-duplicate).
- **Missing exactly when `cli_bx` is missing** (confirmed via crosstab: 24/24 missing rows match,
  67/67 present rows match) — median-imputed for those 24 rows, **no separate missingness flag
  added**, since `cli_bx_missing` (already in the frame) carries that exact signal already; a
  second flag would be 100% redundant, precisely the kind of duplication this EDA pass was meant
  to catch.

**`vit_bmi`** was added after a second EDA pass (same session), specifically prompted by pushback
on screening candidates against the decision target alone. Checked against *all four* targets
(decision, confidence, and all 9 weight factors) rather than just decision:
- **Correlates with confidence specifically** (r=0.259) — not decision (r=0.100). Worth noting on
  its own terms: confidence is the target every KDM-derived signal in `exp_6`/`exp_7` has done
  worst against (best result still below its naive baseline), so a genuinely new correlate of it
  is a different kind of finding than another decision-correlated column would be.
- `vit_weight_kg` correlates even more strongly with confidence (r=0.299) but r=0.869 with
  `vit_bmi` — kept `vit_bmi` alone rather than both, since they'd otherwise be encoding almost the
  same body-composition signal twice.
- Not collinear with anything already in the frame.

**Explicitly rejected across both EDA passes** (see the session's analysis for full numbers):
`psa_tr_last_val`/`max`/`mean`/`delta`/`slope` (r ≥ 0.97 with `cli_psa` — near-exact duplicates;
their apparent weight-factor correlations, r≈0.41–0.46, turned out to be fully explained by
correlating with the `psa` factor specifically, not new information for any other factor),
`path_hist_bx_gl_prim`/`gl_sec` (redundant with `cli_isup`), `path_hist_bx_gl_tert` (98.9%
missing), `lab_hemoglobin_g_dl` (its single strongest correlation of anything checked, r=0.481
with the `bx` weight factor, is computed on only 23/91 non-missing values — too thin to trust),
`vit_height_cm`/`bp_systolic`/`bp_diastolic`/`heart_rate_bpm`/`smoking_pack_years`,
`cli_months`/`cli_ipss_score` (weak across all four targets, no compelling case),
`lab_free_psa_ng_ml`/`lab_free_total_ratio`/`lab_creatinine_mg_dl`/`psa_tr_count`/`cli_fh_binary`
(each individually defensible on a second look — `lab_free_psa_ng_ml`/`lab_free_total_ratio` in
particular are the literal correct fit for §2b's `laboratory_results` mapping below, and
`cli_fh_binary` would fill the `family_history` gap — but not selected this round; a candidate
list for a future pass if this round's additions prove worthwhile).

### 2b. Reveal-sequence: section → feature-group mapping (a judgment call, flagged as such)

**Only 4 of these 6 sections are actually modeled** (per the correction in §1c above —
`family_history`/`pathology_report` have zero positive examples among the 91 labeled cases, so
neither can be meaningfully fit regardless of feature-group mapping quality). The table below is
kept complete for the record; `run_reveal_kdm.py` only uses the first 4 rows.

| Reveal section | Mapped feature group | Rationale | Modeled? |
|---|---|---|---|
| `psa_trend` | `cli_psa`, `cli_psad`, `cli_psav`, `cli_psap` | `psav` = PSA *velocity*, a literal trend metric — direct fit | ✅ (79/91) |
| `radiology_report` | `cli_pirads`, `mri_pca_0`, `mri_pca_1`, `mri_missing` | PI-RADS + MRI embedding are both radiology-report-derived | ✅ (88/91) |
| `laboratory_results` | `cli_cspca`, `cli_vol` | csPCa risk score and volume are lab/imaging-derived summary values | ✅ (41/91) |
| `previous_notes` | `cli_bx_positive`, `cli_bx_missing`, `cli_age` | prior biopsy *status* and demographic facts are the kind of thing prior notes would carry | ✅ (77/91) |
| `pathology_report` | `cli_isup` | prior biopsy *grade* is a pathology-report-specific detail, distinct from merely knowing a biopsy happened | ❌ (0/91 — never modeled) |
| `family_history` | *(none)* | no structured proxy exists in the 23-column frame | ❌ (0/91 — never modeled) |

This mapping was this design's biggest judgment call before implementation surfaced the more
consequential fact: `family_history`/`pathology_report` aren't modeled at all, independent of how
good their mapping is. The `previous_notes`/`pathology_report` feature-group-sharing concern this
design originally flagged (and `cli_isup` was meant to resolve) turned out to be moot for the
same reason — neither issue matters once a section has zero positive examples to fit against.
`family_history` staying unmapped was already a deliberate, disclosed scope decision
(`cli_fh_binary` would have filled it, considered but not selected, §2a) — it would not have
changed the modeled-section count regardless, now confirmed.

### 2c. Reveal signal and recalibration

For section *S* with feature group *G<sub>S</sub>*, reusing `exp_6`'s `occlusion_delta`-style
mechanism but on entropy rather than `p(yes)`:

```
R_S(x) = H(x with G_S occluded) - H(x)        [entropy increase from hiding section S]
```

A separate small binary classifier per section (`make_classifier()`, this project's established
small-model pattern) predicts `P(section S actually revealed | R_S(x))`, trained on that fold's
training rows; the predicted reveal set is the union of sections whose predicted probability
exceeds 0.5. Scored via `reasoning_labels.reveal_set_precision()`.

### 2d. Hyperparameter search on the expanded frame

Reuses `exp_7`'s exact grid and methodology (`kdm_backbone_v2.fit_kdm_backbone`'s five exposed
parameters), applied to the 23-column frame instead of the 19-column one:

| Hyperparameter | Search values (unchanged from exp_7) |
|---|---|
| `N_EPOCHS` | {150, 300, 600} |
| learning rate | {3e-3, 1e-2, 3e-2} |
| `sigma_mult` | {0.5, 1.0, 1.5, 2.0} |
| optimizer / `weight_decay` | Adam (wd=0); AdamW (wd ∈ {0, 1e-4, 1e-3}) |

Same 144 combinations, same 5-fold × 3-repeat search-phase protocol, same
"re-evaluate only the winner at the full 10-repeat protocol" discipline. **Two corrections from
`exp_7`'s own report** (§7 below has the details): the clear-margin threshold is now relative to
`exp_6`'s measured CV std rather than a fixed constant, and the held-out check is mandatory, not
an optional final step.

**Three-way ablation** (mirrors `exp_7`'s `run_ablations.py` pattern, now with "expanded
features" as the analog of `exp_7`'s "log1p" lever):

| Condition | Features | Hyperparameters |
|---|---|---|
| `decision_kdm_v3` (combined) | 23-column (expanded) | winning config from this search |
| `decision_kdm_features_only` | 23-column (expanded) | `exp_6`'s original fixed config |
| `decision_kdm_tuned_only` | 19-column (`exp_3`'s original) | winning config from this search |

This isolates which lever (if either) is doing any work — exactly the question `exp_7`'s own
ablation answered for log1p vs. tuning, now asked for expanded-features vs. tuning.

## 3. File Layout for This Experiment

```
experiments/exp_8/
├── DESIGN.md
├── IMPLEMENTATION.md              ← written after this design is accepted
├── scripts/
│   ├── features_v3.py             (select_exp8_feature_frame: exp_3's frame + psav/psap/isup/bmi)
│   ├── search_hyperparameters_v3.py  (exp_7's grid/protocol, 23-col frame)
│   ├── run_ablations_v3.py        (the 3-way features/hyperparameters isolation, §2d)
│   ├── holdout_eval_v3.py         (mandatory held-out check, adapted from exp_7's)
│   ├── run_signals_v3.py          (exp_6/7's decision+confidence+weights readouts, combined config)
│   └── run_reveal_kdm.py          (new: the 6-section occlusion-on-entropy mechanism)
├── results/
│   ├── hyperparameter_search/     (grid.csv + winner.json, same shape as exp_7's)
│   ├── decision_kdm_v3/           (combined: expanded features + tuned hyperparameters)
│   ├── decision_kdm_features_only/
│   ├── decision_kdm_tuned_only/
│   ├── holdout_eval_v3/
│   ├── confidence_kdm_*_v3/       (5 conditions, same signals as exp_6, combined config)
│   ├── weights_kdm_*_v3/          (3 conditions, same signals as exp_6, combined config)
│   └── reveal_kdm_occlusion/      (new)
└── reports/
    └── summary.md
```

## 4. Baselines

- **Decision**: `exp_6`'s `decision_kdm_backbone` (0.593 macro-F1, both CV and held-out); Extra
  Trees incumbent (0.650). `exp_7`'s own cautionary numbers are also in scope for comparison:
  0.622 CV / **0.490 held-out** — the exact pattern this experiment must not silently repeat.
- **Confidence**: `exp_6`'s best (`confidence_kdm_entropy_isotonic`, 0.731); incumbent
  `confidence_svm` (0.468); naive baseline (0.527).
- **Weights**: `exp_6`'s best (`weights_kdm_occlusion`, 0.405); incumbent `weights_svm`
  (0.382/0.392); naive baseline (0.413).
- **Reveal**: incumbent `reveal_flags` (`exp_2`, 0.853 set precision); naive baseline (0.783).
  No prior KDM-based reveal result exists to compare against — this condition's own number is the
  first data point of its kind.

## 5. Proposed Conditions

| Condition | Target | Mechanism |
|---|---|---|
| `hyperparameter_search` (144 sub-conditions) | decision | §2d's grid, 23-column frame, search-phase only |
| `decision_kdm_v3` | decision | combined: 23-column frame + winning hyperparameters |
| `decision_kdm_features_only` | decision | 23-column frame + `exp_6`'s original hyperparameters |
| `decision_kdm_tuned_only` | decision | 19-column frame + winning hyperparameters |
| `holdout_eval_v3` | decision | `exp_6` plain vs. `decision_kdm_v3`, genuine held-out split |
| `confidence_kdm_{entropy_zeroshot,entropy_isotonic,dispersion_isotonic,participation_isotonic,blend}_v3` | confidence | exp_6's unchanged readout, combined config |
| `weights_kdm_{occlusion,kernel_distance,blend}_v3` | variable-weights | exp_6's unchanged readout, combined config |
| `reveal_kdm_occlusion` | reveal-sequence | new: per-section entropy-increase-from-occlusion + binary classifiers |

## 6. Ablation Studies

- **Features vs. hyperparameters, isolated** (§2d's three-way split) — the central ablation this
  experiment adds beyond `exp_6`/`exp_7`. Confidence/weights are only run against the *combined*
  config (`decision_kdm_v3`'s winner), not against all three decision variants — re-running the
  full confidence/weights readout for `features_only` and `tuned_only` separately would triple
  that workload for a question §2d's decision-only ablation already answers cleanly.
- **Reveal per-section breakdown**: report per-section precision/recall separately (not just the
  aggregate set-precision) across the 4 sections actually modeled, so a strong section's result
  doesn't mask a weak one or vice versa.

## 7. Evaluation Protocol

- Same 5-fold × 10-repeat CV (`RANDOM_STATE=0`) as every KDM condition since `exp_3`.
- Decision/confidence/weights: identical metrics and comparison points to `exp_6`/`exp_7`.
- Reveal: `reveal_set_precision()` per case, averaged, across the 4 modeled sections; **also
  report each section's individual precision/recall** — the aggregate alone would hide exactly
  the nuance this experiment is a first probe for.
- **Held-out verification is mandatory, not optional, per `exp_7`'s hard-won lesson**: no
  decision-macro-F1 improvement — from `decision_kdm_v3`, `decision_kdm_features_only`, or
  `decision_kdm_tuned_only` — gets reported as genuine without confirming it on the same held-out
  19-case split `exp_3`/`exp_7` used (`exp_7`'s own report: "treat a held-out check as mandatory
  before reporting any multi-way-search result as a genuine improvement — not optional
  verification"). This is `holdout_eval_v3.py`'s entire job.
- **Clear-margin threshold set relative to `exp_6`'s own measured CV std (0.045)**, not a fixed
  constant — `exp_7`'s report flagged its own fixed 0.02 threshold as a methodological gap
  (smaller than `exp_6`'s noise floor, so a "clear" margin by that bar could still be noise). This
  round's threshold: an improvement must exceed roughly **one measured std (≈0.045 macro-F1)**
  before the held-out check is even worth running as a confirmation step rather than an
  expected-to-fail formality.

## 8. Expected Results & Decision Rules

- If `decision_kdm_features_only` clearly beats `exp_6` **and survives held-out** while
  `decision_kdm_tuned_only` doesn't → the expanded variables are what matters, hyperparameter
  tuning still doesn't (consistent with, and extending, `exp_7`'s own tuned-only finding on the
  old frame — evidence tuning specifically isn't where KDM's remaining headroom is).
- If `decision_kdm_tuned_only` clearly beats `exp_6` and survives held-out while
  `decision_kdm_features_only` doesn't → the opposite: new variables aren't pulling weight, but
  the expanded feature space happened to open up a better hyperparameter region — worth
  understanding why before trusting it, not just reporting the number.
- If both isolated conditions beat `exp_6` and survive held-out → genuine, compounding
  improvements; `decision_kdm_v3` (combined) should be checked for whether the combination is
  better than either alone or just redundant with the stronger of the two.
- If any of the three shows a CV improvement that **doesn't** survive the held-out check → exact
  repeat of `exp_7`'s finding; report honestly as unverified for that specific condition, don't
  adopt it, and don't let a surviving sibling condition's success paper over the one that failed.
- If none of the three beats `exp_6` by a real margin → both levers exhausted on this backbone
  architecture across two experiments now; the case for `exp_6`'s originally-deferred
  architecture-level levers (ARD sigma, reduced-set prototypes, `y_train=True`) gets stronger with
  each additional low-risk lever that doesn't pan out.
- If `reveal_kdm_occlusion` beats the 0.783 baseline on the 4 modeled sections
  (`psa_trend`, `radiology_report`, `laboratory_results`, `previous_notes`) → confirms the
  mechanism has real merit for sections with a structured-feature proxy and enough positive
  examples to fit against. `family_history`/`pathology_report` are out of scope for this
  mechanism entirely (§1c) — a data-representation gap (e.g. mining
  `txt_family_history_narrative` directly) worth a dedicated future experiment, not something
  this one's result should be read as having failed to close.
- If `reveal_kdm_occlusion` doesn't beat baseline anywhere → first-probe mechanisms not working
  is a legitimate, reportable outcome; `reveal`'s existing OvR-logistic approach (0.85+) remains
  the practical choice regardless.

## 9. Risks & Mitigations

- **23-column frame vs. 19-column**: `psav`/`psap` are highly correlated with `cli_psa`
  (r=0.95/0.99) and could destabilize KDM's single shared kernel bandwidth if their scale
  dominates the Euclidean distance disproportionately post-`StandardScaler` — worth a quick
  sanity check on their post-scaling distribution during implementation, same discipline as
  `exp_7`'s skew check. `cli_isup` doesn't carry this specific risk (r=0.38 with `cli_cspca`,
  not near-collinear), but its median-imputed values for the 24 no-prior-biopsy cases are worth
  a quick sanity check too — confirm the imputed value doesn't sit implausibly close to a real
  observed grade in a way that misleads the kernel distance for those cases specifically.
- **144-way search at N=91 risks a false winner from CV noise — the exact failure mode `exp_7`
  already demonstrated on this same architecture.** Re-running the identical search on a different
  feature set is a legitimate new question (§1b), not a do-over, but it inherits the same risk
  profile: mitigated the same way `exp_7` eventually did (3-repeat provisional search, full
  10-repeat re-evaluation of only the winner, the corrected margin threshold, and — this time —
  a held-out check built into the protocol from the start rather than added after the fact).
- **This is a bundled, three-part experiment** (variable expansion, hyperparameter tuning, and a
  new sub-task) — if any part's implementation runs long, they can be reported independently
  rather than blocked on each other; nothing about (a)/(b) depends on (c) succeeding or vice
  versa, and (a)/(b) are cleanly separated from each other by §2d's three-way ablation.

## 10. Reproducibility Checklist

- [x] Random seeds fixed (`RANDOM_STATE = 0`, unchanged from `exp_1`–`exp_7`)
- [ ] Config YAML — N/A, inline constants as in prior experiments
- [x] Dataset version: same as `exp_3`–`exp_7`
- [ ] Checkpoints — N/A, no persisted model artifacts
- [ ] Experiment tracker — not used
- [ ] Git commit hash — this project now has a git history (as of this session) — record the
      commit this experiment builds from once implementation starts, unlike `exp_1`–`exp_7`
      which predate the repo's git init.

## 11. Next Steps

1. Review this plan — the section-mapping table (§2b), the `family_history` gap (§9), and the
   corrected margin-threshold logic (§7) are what's most worth pushback before implementation.
2. Once accepted, an implementation plan (Claude Code plan mode) covering
   `select_exp8_feature_frame()`, the hyperparameter-search + three-way-ablation machinery
   (largely `exp_7`'s own scripts adapted to a new feature frame, not built from scratch), the
   entropy-occlusion reveal mechanism, and the per-section reporting discipline (§7). Larger than
   this design's earlier draft estimated — the hyperparameter-tuning thrust (§1b/§2d) makes this
   closer to `exp_7`'s own scope than a lean, single-lever addition, even though most of its
   *code* is reused rather than novel. Save as `experiments/exp_8/IMPLEMENTATION.md` before
   editing any files.
