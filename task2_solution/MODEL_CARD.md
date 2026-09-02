# Task 2 Model Card — MRI + Biopsy Risk Stratification

## 1. Task

Recommend one of four post-biopsy management strategies — `watchful_waiting`,
`active_surveillance`, `continued_surveillance`, `active_treatment` — plus a reasoning trace
(confidence, per-variable importance weights, a free-text explanation). Ground truth derives
retrospectively from biopsy histopathology and PSA variables per EAU/NCCN guidelines.

Confirmed directly from source, not the challenge webpage's prose alone: `evaluate.py`'s
`process_interf1` reads the exact same two-file shape task1 writes (a bare decision string + a
reasoning object), and `compute_aggregate_metrics`'s `is_treatment` branch defines
**`ranking_score = (mean_case_score + decision_weighted_f1) / 2`** — support-weighted F1 across the
four classes, not the single positive-class F1 task1 uses. `mean_case_score`'s components (when no
LLM rationale judge is available, as here): `confidence_score` 22.5%, `variable_weight_score`
27.5%, `important_decisive_factor_score` 17.5%, `tool_score` 15%, `section_grounding_score` 17.5%.
A wrong `decision` zeroes the whole case's `case_score` — the gate is not a formality, it decides
whether the other 82.5 percentage points of weight are even reachable.

## 2. Inputs

| # | Field | Source | Always available? |
|---|---|---|---|
| 1–2 | `age`, `psa` | `structured-prompt.json` | Yes (`always_available_baseline` per `section_variable_mapping.json`) |
| 3–9 | `psad`, `pirads`, `bx_isup`, `bx_gl_prim`, `bx_gl_sec`, `cspca`, `ct` | `structured-prompt.json` | Yes — the numbers are already parsed, unlike task3's pathology grading which lived only in free text |
| 10 | `comorbidity` (`pmhx`, a list) | `structured-prompt.json` | Yes, but structurally ungroundable — its primary section (`section_s3-comorb`) has no entry in the reveal vocabulary at all, the same category as task1's `bx` |
| 11 | `fh` (family history) | `…clinical-data.json`, gated | No — needs `get_family_history` |

Six gated clinical-data sections exist (one more than task1's five: `pathology_report`, since a
biopsy now exists): `radiology_report`, `pathology_report`, `previous_notes`, `psa_trend`,
`laboratory_results`, `family_history`.

## 3. Data pre-processing

### 3.1 The always-available structured fields

`age, psa, psad, pirads, bx_isup, bx_gl_prim, bx_gl_sec, cspca` read directly.
`ct` (clinical T-stage, e.g. `"cT2a"`) parsed into `ct_ordinal`, reusing task3's exact
`[pc]T<digit><subletter?>` grammar (`cTx` → `None`, an indeterminate stage, not a severity rung).
`comorbidity_count` — the number of semicolon-separated conditions in `pmhx` — is always available
and requires no reveal, unlike `fh`.

### 3.2 PSA kinetics (§5.6, exp_16)

`structured.csv`'s `psa` is confirmed identical to `psa_trend`'s last reading and the lab panel's
own PSA entry (same value, three places) — only the *trajectory* is new signal:

```
psa_velocity_per_year = (last_reading − first_reading) / years_elapsed
psa_ratio_last_first  = last_reading / first_reading
```

### 3.3 Radiology/pathology staging text (§5.10, exp_18) — extracted, not promoted

`radiology_report`/`pathology_report` are real free prose, not task1/task3's closed templated
sentences — several regex iterations were needed, each caught by direct corpus scan before being
trusted: a naive first pass on extracapsular extension (EPE) matched "clinical suspicion for
extracapsular extension" inside the scan's *indication* line (why the MRI was ordered), not an
actual finding — fixed by restricting the search to the FINDINGS/IMPRESSION sections. Perineural
invasion's negation is a mix of prefix ("No perineural invasion") and suffix ("Perineural invasion
is absent") forms, unlike anything task3's pathology text needed. Final features: `epe_present`
(5/153 positive), `svi_present` (2/153), `ai_isup_discordance` (AI-predicted ISUP minus the
pathologist's own, mean +1.73 — the AI model systematically grades higher), `cribriform_present`
(51/153), `perineural_invasion_present` (25/153). `lymphadenopathy` was scanned and dropped before
extraction: 0 positive cases across the entire corpus, a zero-variance feature confirmed useless
before being coded.

## 4. Model architecture

Three independently-configured sub-tasks, mirroring task1's `subtask_config.py` architecture
exactly:

```
decision:          numeric features -> KNeighborsClassifier(n_neighbors=7, weights="distance")
confidence:         numeric features -> KNeighborsClassifier(n_neighbors=7, weights="distance")
variable_weights:   numeric features -> mord.LogisticIT()   [x11, one per variable]
```

`reveal_sequence` is one case-level list shared by all three sub-tasks' scoring — the real reveal
policy at inference time is the union across whichever configs are wired in
(`compute_called_tools`); every swap is measured against the real `ranking_score`, never a
sub-task's isolated component score.

## 5. Models actually tried (not just the one that shipped)

### 5.1 exp_11 — first baseline + tool-policy measurement

Ported task1's architecture with an 11-variable vocabulary, 4-class decision, 6-section reveal
policy. Base `knn` config: **ranking_score 0.714**. Ground-truth `reveal_sequence` is always empty
across all 72 labeled cases — `cost_aware_tool_score` gives a guaranteed `tool_score=1.0` for
calling nothing versus `0.0` for calling anything. Swept: grounding `variable_weights` with
`{get_mri_report, get_pathology_report}` raised `variable_weight_score` (0.673→0.803) and
`factor_f1` (0.431→0.633) but lost on `ranking_score` (0.7166→0.6848) — paired bootstrap on the
per-case gap: mean +0.0636, 95% CI **[+0.0525, +0.0748]**, 100% of resamples favor zero tools.
**Promoted**: all three sub-tasks call zero tools.

### 5.2 exp_12 — model-family sweep (8 models × 3 sub-tasks)

`knn` confirmed best for `confidence`/`variable_weights` (no challenger beat it). `random_forest`
beat `knn` on `decision` by the sweep's largest margin (0.7166→0.7315, +0.0149) — paired bootstrap:
mean +0.0147, 95% CI **[−0.0725, +0.1091]**, only 63.7% of resamples favoring. **Not promoted** — a
real point-estimate winner that doesn't clear the bar, the same pattern task3's exp_4 Cox-penalizer
sweep found.

### 5.3 exp_13 — ordinal targets treated as regression

`confidence_score`/`variable_weight_score` are confirmed L1 ordinal-distance formulas
(`1 − |true−pred|/max_dist`), yet every model in exp_12's roster is a nominal classifier. Tested
regression-then-round (`KNeighborsRegressor`/`Ridge`/`RandomForestRegressor`, rounded+clipped) —
gains were tiny (best +0.0019, an order of magnitude below exp_12's already-refuted +0.0149) and,
worse, the "best" variants regressed on exactly the rare classes the hypothesis targeted:
`confidence=rf_ordinal` predicted `clear` for 100% of cases (worse than nominal `knn`);
`variable_weights=knn_ordinal`'s `decisive`-class recall dropped 18%→10%. **Not promoted.**

### 5.4 exp_14 — genuine ordinal classifiers (`mord`)

`mord` (v0.7) provides real proportional-odds/threshold-based ordinal classifiers — structurally
different from exp_13's regress-and-round proxy (no assumption that classes are evenly spaced on a
number line). `confidence`: all three `mord` variants (`LogisticIT`/`LogisticAT`/`LogisticSE`)
score *identically* (0.897) and worse than nominal `knn` (0.914) — refuted, and it adds a new
failure mode (hallucinating `borderline` on true-`clear` cases, something nominal `knn` never did).
`variable_weights`: `mord.LogisticIT` — ranking_score 0.7166→0.7186, bootstrap borderline (95% CI
**[−0.0005, +0.0047]**, 93.7% favoring) but **promoted** on a corroborating-diagnostic basis: three
independent metrics moved together — accuracy 0.53→0.56, macro-F1 0.47→0.51, and critically
`decisive`-class recall 0.18→**0.25** (exp_13's proxy made this same recall *worse*, 0.18→0.10).
Same promotion logic task3's exp_6 used for a borderline C-index gain. Full pipeline re-run:
**ranking_score 0.716.**

### 5.5 exp_15 — is confidence's blind spot worth fixing?

`confidence`'s `knn` predicts `clear` for 71/72 cases (0/14 `borderline` recall, macro-F1 0.29).
`extra_trees` (`class_weight="balanced"`) catches 8/14 `borderline` cases — best macro-F1 (0.35) of
anything tried — at an apparent ~0.009 `ranking_score` cost. Bootstrapped: mean −0.0087, 95% CI
**[−0.0156, −0.0016]**, only 0.4% of resamples favoring the fix. **Real, confirmed loss — not
promoted.** Unlike exp_14's borderline-CI promotion, this CI is unambiguously negative; a
corroborating recall diagnostic doesn't override a confirmed loss on the metric that's actually
promotion-gated.

### 5.6 exp_16 — PSA kinetics for `decision`

The biggest single gain in this project's task2 work: `+psa_kinetics` alone,
0.7186→**0.7429** (+0.0243). Bootstrap on its own is borderline (95% CI
**[+0.0000, +0.0622]**, 85.8% favoring — short of the 90% bar used elsewhere) but promoted on the
same corroborating-diagnostic logic as exp_14: the classification report shows a coherent,
no-regression gain across every real class (accuracy 0.81→0.83, `active_surveillance` F1
0.76→0.80, `active_treatment` F1 0.81→0.85), plus genuine a priori clinical grounding — PSA
velocity is a guideline-recognized treatment-escalation factor. A same-size lab panel (7 features:
free PSA, %free PSA, testosterone, creatinine, alkaline phosphatase, LDH, eGFR) was also tried and
lost badly (0.7186→0.6548) — dimensionality noise, not signal, the same failure mode task3's exp_10
found for its own text-feature expansion. Full pipeline re-run: **ranking_score 0.740**, gate pass
rate 58/72→**60/72** (the "decision pays twice" mechanism — a correct decision both raises
`decision_weighted_f1` directly and lets the case through the `case_score` gate — confirmed).

### 5.7 exp_17 — feature-subset sweep, evaluator-vocabulary floor

With the 11-variable evaluator-scored floor fixed (completing it required extracting
`family_history_present`, never previously usable as a feature — 94.8% coverage, Yes/No/Unknown →
1.0/0.0/None), swept `psa_kinetics`/`labs` on top per sub-task. `family_history_present` is
decisively bad for `decision` — confirmed even in isolation (floor alone: −0.0362) and it actively
fights the already-promoted kinetics feature (combined: −0.0611, *worse* than either alone).
`confidence`/`variable_weights`: every variant landed within noise of baseline (largest magnitude
±0.0024) — the 11-variable floor is already sufficient there. **Not promoted anywhere.**

### 5.8 exp_18 — radiology + pathology free-text staging features

Direct precedent from task3's exp_3 (its biggest early win was exactly this feature category:
extracapsular extension / seminal vesicle invasion mined from free text). `epe_present`/
`svi_present`: exactly **zero** effect on any sub-task, to 4 decimal places — traced to extreme
rarity (5 and 2 positive cases in 153), too sparse to perturb `knn`'s distance metric at all (a
different, quieter failure mode than task3's `on_5ari`, which crashed Cox's fitter outright).
`ai_isup_discordance`/`cribriform_present`/`perineural_invasion_present`: hurt `decision` badly
(−0.0717), the same dimensionality/missingness problem `labs` caused in exp_16. **Not promoted** —
a useful negative result: this project's own precedent in one task isn't evidence for the next.

### 5.9 exp_19 — 3-model voting ensemble

`VotingClassifier(knn, logistic, random_forest)` for `decision`. Hard voting: 0.7429→0.7452
(+0.0024); soft voting: +0.0016 — both an order of magnitude smaller than exp_12's already-refuted
+0.0149. Bootstrap on the larger: mean +0.0024, 95% CI **[−0.0463, +0.0521]**, only 56.0% of
resamples favoring — the widest, least-convincing CI of any `decision`-model comparison tried.
Classification report confirms no hidden story (accuracy 0.83 either way, same confusion pattern).
Likely cause: `logistic` (this project's consistently weakest `decision` model, 0.6704 in exp_12)
dilutes the vote enough to cancel out `random_forest`'s own contribution. **Not promoted.**

### 5.10 exp_20–22 — `fagonzalezo/kdm` (Kernel Density Matrices), three tries

Deferred since task1's own model-family sweep, finally tried. Required a real environment fix:
`pip install torch` pulled the latest release (2.13.0), which fails to import on the source machine
(`WinError 1114`, `c10.dll` DLL-init error) — diagnosed properly, not assumed unfixable (VC++
redistributables present, CPU modern enough, standard OpenMP workaround didn't help);
`torch==2.2.0+cpu` imports cleanly, pinned as the working version. `agent/src/kdm_models.py` wraps
`KDMClassModel`/`KDMRegressModel` behind sklearn's `.fit`/`.predict`, `n_comp` set to the training
fold size (the standard "centers = training points" kernel-method scheme).

- **exp_20 (classifier)**: 0.7429→**0.6944** (−0.0485). Coherent, not catastrophic — worse on the
  two largest classes (`active_surveillance` recall 0.88→0.72), plausibly the neural encoder plus
  71 free components overfitting on ~70 rows/fold, echoing MLP's consistent mediocrity elsewhere in
  this project. ~170s per LOOCV pass vs. near-instant for every sklearn model.
- **exp_21 (regressor)**: encoding the 4 categories on a clinically-motivated 0–3 scale and
  rounding, 0.7429→**0.5865** (−0.1564) — worse still. `decision` is confirmed nominal;
  `continued_surveillance` (the *middle* of the imposed scale) absorbed an inflated recall (0.93)
  with collapsed precision (0.50) — the regression pulls ambiguous cases toward the middle of an
  arbitrary numeric encoding, not the correct answer.
- **exp_22 (generative co-training)**: `KDMClassModel`'s own documented recipe for exactly exp_20's
  overfitting regime — freeze the encoder, add a density-estimation loss term. All three weights
  tried (0.5/1.0/2.0) land on the **identical** score (0.2907, −0.4522) — confirmed a complete
  collapse to predicting `active_treatment` for **72/72 cases**, regardless of weight. Likely: the
  generative loss dominates once the encoder freezes, with no loss-scale balancing against the NLL
  term, destroying the class boundary rather than regularizing it.

**Not promoted, 0-for-3.** Wrapper infrastructure kept — real, tested, reusable for a better-tuned
attempt or task3's `months_to_recurrence` (a genuine regression target, never tried).

## 6. Output

Two files, matching task1's shape exactly. `prostate-treatment-decision.json`: a bare string, one
of the four categories. `prostate-treatment-decision-reasoning.json`:
`{confidence, variable_weights, reveal_sequence, free_text}`. `reveal_sequence` is always `[]`
(zero tools, exp_11); only `decision`/`decision_weighted_f1` and `mean_case_score` feed
`ranking_score` — `variable_weights`'s own component scores are part of `mean_case_score`.

## 7. Known limitations

- `watchful_waiting` (2/72) never predicted correctly by any model tried across all twelve
  experiments — a sample-size floor, not a fixable model weakness.
- `confidence`'s `borderline` class (14/72) never predicted correctly by the promoted model — a
  named, quantified, deliberately-**not**-paid trade-off (exp_15), not an unexamined gap.
- No model-family re-sweep on `decision`'s final 12-feature set — exp_12's sweep predates exp_16's
  PSA-kinetics addition.
- `fagonzalezo/kdm` is 0-for-3 on `decision` (exp_20–22) — the wrapper infrastructure is real and
  reusable, but no configuration tried here beat `knn`.
- No hyperparameter tuning within any model family beyond the fixed defaults each experiment used
  (e.g. `n_neighbors`, `mord`'s regularization, KDM's `encoded_size`/`n_comp`/`sigma`).
- Radiology/pathology free-text features (exp_18) were tried with real optimism given task3's
  precedent and refuted anyway — kept as documented, unused engineered columns.
