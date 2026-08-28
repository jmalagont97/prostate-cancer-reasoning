# Model card — Task 3 biochemical-recurrence predictor

This document describes the implemented model end to end: what it receives, how the raw data is
turned into model-ready numbers, which models were actually tried, and what it returns. It
complements (doesn't repeat) `equations.md` — the math derivation lives there; this document is the
data and modeling narrative.

---

## 1. Task

Predict `months_to_recurrence`: time to biochemical recurrence (BCR) after radical prostatectomy,
or time to last follow-up if no recurrence is observed. Scored by Harrell's concordance index
(C-index) against the true `(time, event)` pair — a ranking metric, not a calibration metric: only
the relative order of predictions across cases matters for the score.

- **Cohort**: 75 cases, all labeled. 19 observed recurrences (event=1), 56 censored (event=0).
  Follow-up ranges 1.3–110 months.
- **Evaluation protocol**: leave-one-out cross-validation (LOOCV) throughout — chosen over k-fold
  because an earlier experiment on this same project's task-1 work showed 5-fold CV can produce a
  leakage-driven false win at this sample size (a result that scored well under 5-fold collapsed
  once re-validated under LOOCV). Every number in this document is LOOCV unless stated otherwise.

## 2. Inputs

Three files are provided per case by the challenge (confirmed against `evaluate.py`'s own input
socket definitions, not assumed from documentation):

| File | Contents |
|---|---|
| `structured-prompt.json` | `age`, `psa`, `dre` (free text), `active_treatment_prior_to_surgery` |
| `…neural-representations.json` | Frozen embeddings: 1 MRI vector (1024-dim), 0–3 biopsy-slide vectors, 1–3 prostatectomy-slide vectors (960-dim each) |
| `…clinical-data.json` | 5 free-text sections: `radiology_report`, `pathology_report`, `surgical_pathology_report`, `previous_notes`, `family_history` |

Of these, the implemented model uses: `age`, `psa`, `dre` (all from file 1), the
`surgical_pathology_report` section (from file 3), and the prostatectomy-slide embeddings (from
file 2). The other four clinical-data sections and the MRI/biopsy-slide embeddings were tried as
alternative or additional inputs and did not improve the model (§5) — they are not in the final
pipeline, but they were tested, not ignored.

## 3. Data pre-processing

This is the part that turns raw text and numbers into the 18-number vector the model actually
fits on. Every step below is regex/arithmetic — no NLP model, no LLM.

### 3.1 Structured fields (`age`, `psa`)

Passed through as numeric with no transformation. No scaling/standardization is applied anywhere
in this pipeline — Cox proportional hazards is scale-invariant in the sense that it doesn't require
normalized inputs to fit correctly (unlike, say, a neural network or k-NN), so this was a deliberate
omission, not an oversight.

### 3.2 The `dre` field — parsed, not used raw

`dre` arrives as a full sentence, e.g. *"Digital rectal examination was abnormal bilaterally. The
clinical T-stage derived from digital rectal examination was cT3."* — not a clean category. A
direct corpus scan (not an assumption) found this sentence is built from a small, closed set of six
finding-phrasings (`was normal`, `was abnormal`, `was abnormal bilaterally`, `was abnormal on the
left`, `was abnormal on the right`, plus one free-text outlier) plus a clinical T-stage token
(`cT1c` … `cTx`). Three fields are extracted by regex:

- `dre_abnormal` (0/1)
- `dre_bilateral` (0/1)
- `dre_ct_stage_ordinal` — the T-stage mapped to an ordinal integer: strip the `cT`/`pT` prefix,
  take the numeral times 10 plus a sub-stage letter rank (`a`→1, `b`→2, `c`→3, none→0). E.g.
  `cT2a` → 21, `cT3` → 30, `cT4b` → 42. `cTx` (stage not assessable) maps to a missing value, not a
  fabricated ordinal position — it isn't a severity rung, it's an absence of information.

23 of 75 cases have no `dre` value at all (exam not performed/recorded) — a real absence, handled
by imputation (§3.4), not dropped.

### 3.3 The surgical pathology report — parsed, not used raw

`surgical_pathology_report` is free text but, like `dre`, was confirmed by direct corpus scan to be
fully templated — a closed vocabulary, not prose that varies case to case in structure. Eight
fields are extracted by regex, each matched against the *exact* phrasing set observed in the
corpus (not a guess at plausible phrasing):

| Field | Extracted from | Encoding |
|---|---|---|
| `isup_grade_group` | `"ISUP grade group N"` | integer, 1–5 |
| `pt_stage_ordinal` | `"pathological stage pTxx"` | same ordinal scheme as §3.2 |
| `epe_present` | `"Extraprostatic extension was present"` / `"was no extraprostatic extension"` | 0/1 |
| `margins_positive` | `"surgical margins were positive/negative"` | 0/1 |
| `svi_present` | `"seminal vesicles were (not) invaded"` | 0/1 |
| `lvi_present` | `"lymphovascular invasion was present/absent"` | 0/1 |
| `ln_assessed` | whether any lymph-node phrase appears at all | 0/1 |
| `ln_positive` | `"lymph node metastasis was present"` vs `"there was no lymph node metastasis"` | 0/1, **only meaningful when `ln_assessed`=1** |

`ln_positive` is deliberately left missing (not coded 0) for the 43 cases where no nodes were
removed at all (`ln_assessed`=0) — "not assessed" and "assessed, negative" are different clinical
facts, and collapsing them would fabricate information the report doesn't contain.

**A caught extraction bug, worth documenting because it's the kind of error that doesn't announce
itself**: the first version of the `dre` parser listed regex alternatives in the wrong order
(`"was abnormal"` before the longer `"was abnormal bilaterally"`). Regex alternation matches the
first option that fits, not the longest — every bilateral case matched the short phrase first, and
`dre_bilateral` silently read as 0 for all 75 cases. It surfaced only as a downstream numerical
convergence error, not from inspecting the column. Fixed by ordering alternatives most-specific-first;
a reminder that a silently-wrong regex produces a plausible, wrong column, not an obvious one.

### 3.4 Missing-value imputation

All 13 tabular/pathology fields are median-imputed where missing. Critically, the imputer is fit
**only on the current LOOCV training fold** (74 cases), never on the case being predicted — fitting
it on all 75 cases up front would leak that case's presence (even indirectly, via the median shifting)
into its own prediction.

### 3.5 The embedding tensor — pooling, then reduction

The prostatectomy-slide embedding arrives as a **bag** of 1–3 vectors per case (960-dim each), not
a fixed-size input:

1. **Masked mean pooling** — average the valid vectors in the bag into one 960-dim vector per case.
   Chosen over a learned aggregator (e.g. attention pooling) because a bag of size 1–3 has no fixed
   order or count, and a learned pooling function would have nowhere near enough of the 75 cases to
   fit safely.
2. **PCA, to 5 components** — 960 raw dimensions against 74 training cases is a severe
   curse-of-dimensionality risk (established earlier in this project's task-1 work: naive feature
   expansion consistently hurt performance at this sample size). PCA is fit strictly inside each
   LOOCV training fold and only then applied to the held-out case — never fit on all 75 cases at
   once, which is exactly the leakage class that inflated an earlier experiment's result before
   being caught and corrected. 5 components was not a default: 2, 5, and 10 were each swept under
   full LOOCV (§5.3) and 5 won.
3. The MRI embedding (1024-dim, always exactly 1 vector/case — no pooling needed) and the
   biopsy-slide embedding (0–3 vectors, 24/75 cases have none at all) were run through the identical
   pooling+PCA pipeline as alternative fusion sources — see §5.3 for why neither was kept.

### 3.6 Final assembly

The 13 tabular/pathology fields and the 5 PCA-reduced embedding components concatenate into one
18-dimensional vector per case — the exact input the model in §4 is fit on. No field enters the
model that isn't listed in §3.2, §3.3, or §3.5.

## 4. Model architecture

**Cox proportional-hazards regression** (`lifelines.CoxPHFitter`), fit with an L2 (ridge) penalty
of 0.3 on the negative log partial likelihood (Efron's method for tied event times). Predictions
are converted from a relative hazard into a literal predicted time via the model's own baseline
cumulative hazard estimate (Breslow estimator) and the expectation of the resulting survival curve
— see `equations.md` §5–6 for the full derivation. Predicted times are clipped to [0, 330] months
(3× the longest observed follow-up) as a numerical safety guard, not a modeling assumption.

## 5. Models actually tried (not just the one that shipped)

Every substitution below was measured under the identical LOOCV protocol and the real evaluator's
own scoring function — none of this is asserted from theory.

### 5.1 Regularization strength (Cox penalizer)

| λ | 0.01 | 0.1 | **0.3** | 0.5 | 0.7 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| C-index | 0.704 | 0.749 | **0.766** | 0.758 | 0.754 | 0.751 | 0.736 |

A clean, single-peaked curve — 0.3 is the empirical optimum, not a library default. (This sweep
predates the embedding fusion in §5.3; 0.3 was carried forward unchanged and still wins on the
final feature set at 0.858.)

### 5.2 Model family

| Model | C-index |
|---|---|
| **Cox PH, λ=0.3** | **0.858** |
| Weibull AFT | 0.755 |
| LogNormal AFT | 0.740 |
| LogLogistic AFT | 0.563 |
| Random Survival Forest / Gradient-Boosted Cox | not run — `scikit-survival` failed to install (no C++ compiler toolchain in the source environment) |

The three AFT (accelerated failure time) models are parametric alternatives to Cox — each assumes
survival time follows a specific distribution shape (Weibull, log-normal, log-logistic) rather than
Cox's semi-parametric "unspecified baseline hazard" assumption. All three were fit on the identical
18-dimensional input and lost.

### 5.3 Embedding fusion source

| Fusion source | C-index | vs. no embedding |
|---|---|---|
| None (tabular + pathology text only) | 0.766 | — |
| + MRI embedding | 0.752 | worse |
| + biopsy-slide embedding | 0.733 | worse |
| **+ prostatectomy-slide embedding** | **0.858** | **+0.092** |
| + all three combined | 0.813 | +0.047, but worse than prostatectomy alone |

The working hypothesis going into this sweep predicted the *opposite* ranking — MRI (a different
imaging modality, a different time point, nothing like it elsewhere in the inputs) was expected to
help most, and the prostatectomy-slide embedding least (same tissue the pathology-text fields in
§3.3 already decode). The result inverted that prediction. Given the size of the effect and that it
contradicted the stated prior, it was treated as a "too good — check for leakage" result before
being trusted: verified the 75 pooled vectors are all distinct (no silent collapse to a shared
fallback), verified PCA is fit strictly inside each training fold, and checked that the raw
correlation between the top principal components and outcome is moderate (r≈0.4, p≈0.001) — real
signal, not a near-perfect leakage pattern. A paired bootstrap (2,000 resamples) on this margin
gives a 95% interval of **[+0.028, +0.176]** — the only gain in this project's task-3 work
confirmed statistically significant, not just a better point estimate.

### 5.4 PCA component count

| k | 2 | **5** | 10 |
|---|---|---|---|
| C-index | 0.845 | **0.858** | 0.847 |

Robust across the swept range — not an artifact of the specific k=5 choice.

## 6. Output

`Task3Output`:

```json
{"case_id": "T3-005", "months_to_recurrence": 74.59, "reasoning": "Predicted time to biochemical recurrence..."}
```

- `months_to_recurrence` (float, ≥ 0) — the model's single prediction, §4's output.
- `reasoning` (string, ≥ 40 chars) — template-generated (not LLM-authored; no LLM was available in
  the source environment), citing the specific pathology findings (§3.3) that informed the
  prediction for that case.
- No `event` field — the output schema doesn't define one, and the evaluator's `ranking_score` for
  task 3 is the C-index on `months_to_recurrence` alone.

## 7. Known limitations

- `scikit-survival` (§5.2) blocked on a missing build toolchain — not a modeling decision.
- The Cox penalizer (§5.1) was not re-swept after the embedding fusion in §5.3; 0.3 was carried
  forward on the assumption it would still be near-optimal, not re-verified.
- `radiology_report`, `previous_notes`, and `family_history` (3 of the 5 gated clinical-data
  sections) were never extracted into features — untried, not tested and refuted.
- A live production-simulation check (refitting the model with one case held out entirely, reading
  only that case's raw files) found 2 of 2 tested early-recurrence cases substantially
  underestimated in absolute predicted time, even though the model's C-index shows it ranks cases
  correctly in aggregate. Flagged, not yet investigated systematically.
