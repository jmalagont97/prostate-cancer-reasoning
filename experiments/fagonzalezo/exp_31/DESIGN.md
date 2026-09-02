# Experiment Design: Bayesian Evidence-Combination Fusion — Log-Linear Pooling and a Non-Parametric KDM Reliability Gate (exp_31)
**Experiment**: experiments/exp_31/ · **Project**: pathology-reasoning · **Date**: 2026-08-20 · **Status**: Complete

---

## 1. Motivation — the fusion *mechanism*, not the per-modality experts, is untested

Every fusion condition in this lineage (exp_8, exp_16, exp_18, exp_28 `fusion_equal`, exp_29 §3 S1
`fusion_optimal_leakfree`) combines per-modality posteriors the same way: `soft_vote` (`src/methods/
mem_kdm.py`), a **weighted arithmetic mean** of class probabilities, `p_fused(y) = Σ_m w_m · p_m(y|x_m)`,
with `w` either fixed at `1/3` (`fusion_equal`) or grid-searched on a 3-simplex against Stage-1 MCCV
validation probabilities (`fusion_optimal_leakfree`, `simplex_grid(3, 0.05)` in `mem_kdm.py`). exp_29's
`fusion_optimal_leakfree__hard` reached **0.7068** LOOCV Macro-F1 (`tab=0.4, mri=0.1, txt=0.5`,
`results/fusion_weights_leakfree.json`) — the closest any KDM/MemKDM-lineage condition has come to
exp_8's 0.7171, and confirmed here to be a genuine three-way blend, not a collapse onto one modality.

This experiment holds the per-modality experts **exactly as exp_29 froze them** (§2) and varies only
*how their evidence is combined*, asking two separable questions:

- **Is arithmetic (linear) pooling the right Bayesian combination rule?** A weighted arithmetic mean of
  posteriors is a *linear opinion pool* — a reasonable, robust aggregation rule, but not the one implied
  by treating each modality as independent *evidence* under a shared prior. That reframing is
  *log-linear pooling* (a weighted geometric mean in probability space, equivalently a weighted sum of
  log-odds): `logit(p_fused) = Σ_m w_m · logit(p_m)`, the standard product-of-experts / naive-Bayes-style
  combination rule for conditionally-independent evidence sources. **C2** tests this swap, weights
  selected by the same simplex-grid protocol as exp_29's S1, holding the per-modality experts fixed.
- **Should the weight of each modality be a single global triple, or vary by patient?** exp_29's
  `(0.4, 0.1, 0.5)` is one fixed triple for the whole cohort. A **non-parametric, per-patient** weight —
  literally an estimate of "how reliable is each modality's evidence for a patient who looks like *this*"
  — is a genuinely different mechanism, not a finer grid search. **C3** builds this as a *kernel
  reliability gate*: three independent `BrentMemKDM` Nadaraya-Watson estimators, living not over the raw
  clinical/MRI/text feature spaces but over the 3-D **evidence space** `e(x) = [p_tab(x), p_mri(x),
  p_txt(x)]`, each regressing "was modality `m` individually correct" onto that evidence pattern. This is
  literally an additional KDM layer stacked on top of the three frozen per-modality KDMs, and its output
  — a normalized reliability vector `w(x)` on the 3-simplex — is a per-patient estimate of *how
  trustworthy each modality's evidence is in this region of evidence-space*, used directly as the fusion
  weight. **Precision needed here:** this is a reliability estimate (`P(modality m individually correct |
  evidence pattern)`, normalized), not itself a Bayesian log-odds evidence combination — C2, not C3, is
  the condition that implements evidence combination in the Bayesian sense (first bullet above). C3
  answers "how much should each modality count, per patient," C2 answers "how should modalities be
  combined, globally." They are complementary, not the same claim, and are evaluated as separate,
  independent hypotheses (§7) for exactly this reason.

Four straight prior experiments in this lineage (exp_23, exp_25–29) failed to beat exp_8's 0.7171 with
McNemar significance; exp_29 §9 explicitly recommends **not** continuing to search the same
sigma/representation/knn_k family blindly. This experiment is not a repeat of that search — it fixes the
per-modality architecture entirely and isolates the fusion mechanism as the only variable, which is a
different, previously-untested axis. Per that same recommendation, **0.7171 is not this experiment's
success bar**; see §8's decision rules for the actual (matched-input) comparison.

## 2. Background — what's reused verbatim, what's new

### 2.1 Reused verbatim from `experiments/fagonzalezo/exp_29`

The per-modality expert architecture is not re-searched. This experiment loads exp_29's own frozen
hard-arm Stage-1 winners directly from `experiments/fagonzalezo/exp_29/results/stage1_best_hparams.json`:

| Modality | `rep` | `knn_k` | `sigma` |
|---|---|---|---|
| `tab` | `None` (fixed representation) | `3` | `2.087259212329571` |
| `mri` | `None` (`raw_l2`) | `40` | `0.06012184192769902` |
| `txt` | `(500, 0.90)` | `3` | `0.3373623840977281` |

These reproduce exp_29's own hard-arm unimodal LOOCV Macro-F1 exactly (`tab=0.6333333333333333`,
`mri=0.5285714285714286`, `txt=0.7090909090909091` — `experiments/fagonzalezo/exp_29/results/
loocv_metrics.json`) — G0 (§6) re-verifies this before anything else runs, and was already run once as
a design-time diagnostic (a throwaway pilot script, not `scripts/train.py`) against the live cohort:
all three values matched to `1e-16`, and the same diagnostic reproduced exp_29's S1 simplex search
(`(0.4, 0.1, 0.5)`, `0.7067941474721136`) bit-for-bit at `label_smoothing=0`, so G0/G1 (§6) are
expected to pass, not merely hoped to. Also reused verbatim:
`Data/preprocessed_old/task1/` cohort loading, `src/evaluation/{data,protocol,metrics,reporting}.py`,
`src/methods/{base,mem_kdm,brent_mem_kdm}.py` — **unmodified**, same as exp_29's own scope discipline
(§9). `simplex_grid` and `soft_vote` (`src/methods/mem_kdm.py`) are reused directly for C1/C2.

**Hard arm only**, matching exp_29 S1's own restriction — this experiment is about the fusion mechanism,
not re-litigating the hard-vs-soft target question.

### 2.2 New: `label_smoothing` on the per-modality experts (why, and what it costs)

exp_29's frozen winners include `knn_k ∈ {3, 3, 40}` on **hard** targets. A `knn_k`-truncated
`BrentMemKDM` is a Nadaraya-Watson average of `knn_k` neighbors' hard 0/1 labels
(`src/methods/brent_mem_kdm.py` module docstring), so its output can land exactly on `{0, 1}`
(certain at `knn_k=1`, common at `knn_k=3`, possible even at `knn_k=40` if all 40 neighbors agree).
`logit(0)` and `logit(1)` are undefined — C2's log-linear pool cannot be computed on these probabilities
without smoothing. `BrentMemKDM(label_smoothing=eps)` (already a first-class constructor argument,
threaded through both whole-memory and `knn_k`-truncated paths — no code change needed) replaces the
Nadaraya-Watson target `y_soft` with `y_soft·(1−eps) + eps/2` before averaging, bounding every modality's
output to `[eps/2, 1−eps/2]`.

`eps = 0.05` is used for **all three conditions' per-modality experts** in this experiment (fixed a
priori, not grid-searched — searching it would open a third leaky Phase-A axis on top of the two already
accepted in this lineage, DESIGN.md §2.2 of exp_29). At `eps=0.05`, the extreme log-odds is
`logit(0.975) ≈ ±3.66` — bounded, but this is a structural choice, not a tuned one; it is documented here
so it can be revisited, not silently absorbed into "the model."

**Cost of this choice, stated plainly:** `eps=0.05` is applied uniformly to `tab`/`mri`/`txt`, but the
frozen `sigma` values (§2.1) were Brent-searched under `label_smoothing=0` in exp_29. Reusing those
sigmas unchanged under `eps=0.05` is an approximation — the smoothing-optimal sigma could differ
slightly. Not re-searched here (same reasoning as above: scope discipline over chasing every last
free parameter). This means **no condition in this experiment is numerically comparable to exp_29's
stored `0.7068`** on a bit-for-bit basis; §3's `fusion_arithmetic_smoothed` (C1) is the correct,
matched-input baseline for this experiment's own conditions, and exp_29's `0.7068` is reported only as
historical context (§5).

### 2.3 New: the two fusion mechanisms (C2, C3) — both implemented inline in `train.py`, no `src/methods` change

Neither log-linear pooling nor the reliability gate is added to `src/methods/mem_kdm.py` or
`brent_mem_kdm.py`. Both are small, experiment-local functions in `experiments/exp_31/scripts/train.py`,
matching how exp_28/29's own S1 simplex-grid search already lives in the experiment script rather than
the method module. `BrentMemKDM` itself is reused, unmodified, as the gate's estimator (§3.3) — the gate
*is* "an additional KDM layer," constructed by pointing the existing memory-based Nadaraya-Watson
machinery at a new, 3-D evidence feature space instead of raw clinical/MRI/text features.

### 2.4 Leakage posture — deliberately asymmetric across conditions, stated explicitly

- **C1/C2** select their fusion weights by the same construction as exp_29's S1: a `simplex_grid(3,
  0.05)` search scored by mean Macro-F1 over Stage-1 MCCV validation probabilities, aggregated across
  all 100 splits, then frozen and applied uniformly in Phase B. This carries the same accepted
  aggregate-selection bias exp_28 §2.3/exp_29 §2.3 already documented (the frozen weights were informed,
  in aggregate, by data including each Phase-B fold's own held-out patient) — **not claimed leak-free**,
  same standard as every weight/sigma selection in this lineage.
- **C3**'s gate kernel bandwidth is **not** Brent-searched or grid-searched at all — it uses the
  data-driven default heuristic (`kdm.init._sigma_from_knn`, `sigma_mult=1.0`, the same un-tuned default
  every `BrentMemKDM` search starts from), computed on the **deduplicated, one-row-per-patient** evidence
  table (§3.3 explains why: the full duplicated table, 88 patients × ~20 split-appearances each,
  produces a bandwidth dominated by near-identical same-patient rows). This removes aggregate
  hyperparameter selection from C3's fusion layer entirely. Combined with the per-LOOCV-fold leak-free
  memory construction (§3.3 — the query patient's own Stage-1 validation rows are excluded from the gate's
  memory before that patient's Phase-B prediction), **C3 is the first fusion condition in this lineage
  with no aggregate-MCCV leakage in its fusion layer.**
- **This is a real asymmetry, not an incidental detail:** if C3 outperforms C1/C2 in §8, part of the gap
  could be attributable to C3's stricter, more honest protocol rather than to the gating mechanism
  itself. The reverse is not a concern (a stricter protocol working *against* C3 and it still winning
  would be a stronger result). Reported as a limitation regardless of outcome, not adjusted for.

### 2.5 Limitation: the pooled evidence table over-represents patients unevenly

C3's evidence table (§3.3) pools each patient's validation-fold occurrence across the ~100 MCCV splits
into one table of rows (patient, split) → evidence vector. A patient who happens to land in the
validation set of more splits contributes proportionally more rows to the gate's memory. `iter_mccv_
splits` / the exp_4 MCCV design is not audited here for per-patient validation-membership balance — if
it is unbalanced, the gate's reliability estimates could be skewed toward over-represented patients. Not
corrected in this experiment (would require per-row reweighting); flagged as a limitation.

## 3. Conditions

All three conditions consume the **same** per-modality frozen experts (§2.1) at `label_smoothing=0.05`
(§2.2), hard arm only, and combine three probabilities into one. Only the combination rule and weight
source differ — this is the controlled variable.

| # | Condition | Weight source | Pooling rule | Purpose |
|---|-----------|---------------|--------------|---------|
| C1 | `fusion_arithmetic_smoothed` | Global simplex-grid search (exp_29 S1 construction, reused) | Arithmetic (`soft_vote`) | Matched-input **reference** for C2/C3; also a sanity check against exp_29's stored `0.7068` |
| C2 | `fusion_loglinear` | Global simplex-grid search (same protocol as C1, different scoring objective) | Log-linear / geometric (`logit_pool`, §3.2) | **H1**: does the Bayesian-evidence-combination reframing (product-of-experts) beat arithmetic pooling, holding weight-source and experts fixed? |
| C3 | `fusion_kdm_gate` | Non-parametric, per-patient (`BrentMemKDM`-based reliability gate, §3.3) | Arithmetic (`soft_vote`, same rule as C1) | **H2**: does an instance-adaptive, non-parametric weight beat a single global weight, holding pooling rule fixed? |

C2 isolates *pooling geometry* (vs. C1). C3 isolates *weight source* (vs. C1). Neither compounds both
changes in one condition — if both help, a natural follow-up (out of scope here, §9) combines them.

### 3.1 Per-modality experts (shared by all three conditions)

```python
def fit_predict_unimodal_smoothed(name, train_idx, val_idx):
    best = STAGE1_BEST_HARD[name]                      # loaded from exp_29's stage1_best_hparams.json
    X_tr, X_va = build_modality(name, best["rep"], train_idx, val_idx)
    m = BrentMemKDM(knn_k=best["knn_k"], label_smoothing=0.05)
    m.sigmas_ = {name: best["sigma"]}
    m.modality_order = [name]
    m.fit({name: X_tr}, Targets(y_binary=y_bin[train_idx], y_soft=y_bin[train_idx].astype(np.float32)))
    return m.predict_proba({name: X_va})[0, 1]
```

Run once via `run_loocv` per modality → `UNIMODAL_SMOOTHED_OOF["tab"|"mri"|"txt"]`, and once per MCCV
split (Stage-1 validation probabilities, for the weight/gate searches) → `WINNER_VAL_PROBS_SMOOTHED`,
mirroring exp_29's `winner_val_probs` exactly except for the added `label_smoothing=0.05`.

### 3.2 C2 — log-linear pooling

```python
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def logit_pool(probs: dict, weights: dict) -> np.ndarray:
    """Weighted geometric mean in probability space. probs[m]: (n, 2)."""
    names = list(probs.keys())
    fused_logit = sum(weights[m] * logit(probs[m][:, 1]) for m in names)
    p1 = 1.0 / (1.0 + np.exp(-fused_logit))
    return np.stack([1 - p1, p1], axis=1)
```

Weight search: identical structure to exp_29 S1's simplex search (`simplex_grid(3, 0.05)`, scored by
mean Macro-F1 over the 100 MCCV splits' `WINNER_VAL_PROBS_SMOOTHED`), with `soft_vote` replaced by
`logit_pool` in the scoring loop. Frozen best weights → `fusion_weights_loglinear.json`, applied to
`UNIMODAL_SMOOTHED_OOF` via `logit_pool` for the Phase-B LOOCV number.

### 3.3 C3 — non-parametric KDM reliability gate

**Evidence table** (built once, from `WINNER_VAL_PROBS_SMOOTHED`, §3.1): one row per (MCCV split, patient
in that split's validation set) — 1800 rows over 88 distinct patients (~20 rows/patient) in a design-time
check against the live cohort — columns `patient_idx`, `e_tab, e_mri, e_txt` (each modality's `P(yes)` on
that row), `y_true`, and `correct_tab, correct_mri, correct_txt` (`1{(e_m >= 0.5) == (y_true == 1)}` —
`>= 0.5`, matching `binary_metrics`'s own threshold convention; **not** `round(e_m)`, which is
banker's-rounding and maps `e_m == 0.5` to `0`, silently disagreeing with that convention exactly at the
label-smoothed midpoint `knn_k=3`/hard-targets can produce).

**Gate bandwidth — deduplicated by patient, not the raw evidence table.** A design-time diagnostic (a
throwaway pilot script, not `scripts/train.py`) built this exact table against the live cohort and found
`_sigma_from_knn` on the full 1800-row table gives `sigma=0.0241`, and after excluding a query patient's
own ~20 rows, the effective sample size (`(Σw)² / Σw²`) of the remaining kernel weights was **1.1–12**
across 8 sampled patients — the ~20 near-duplicate rows contributed by *other* patients that happen to be
close in evidence-space dominate, and the bandwidth ends up so narrow the "non-parametric average" is
functionally a 1–2-neighbor lookup. Computing `_sigma_from_knn` on a **deduplicated table (one row per
patient, mean evidence vector across that patient's split-appearances — 88 points)** instead gives
`sigma=0.170`, with ESS **76–357** on the same 8 patients — a stable, genuinely-non-parametric
neighborhood, confirmed empirically before write-up, not merely asserted.

**Gate estimator**, one independent instance per modality `m`:

```python
def modality_reliability(e_query, evidence_table, exclude_patient, modality, sigma):
    """Nadaraya-Watson estimate of P(modality is individually correct | evidence pattern),
    memory = evidence_table rows whose patient != exclude_patient (leak-free at prediction time)."""
    rows = evidence_table[evidence_table.patient_idx != exclude_patient]
    E = rows[["e_tab", "e_mri", "e_txt"]].values          # (n_rows, 3)
    d2 = ((E - e_query) ** 2).sum(axis=1)
    w = np.exp(-d2 / (2 * sigma ** 2))
    return float(np.average(rows[f"correct_{modality}"].values, weights=w))
```

`sigma` = `kdm.init._sigma_from_knn(E_dedup, sigma_mult=1.0)`, `E_dedup` = the 88-row per-patient-mean
table above, computed once (§2.4 — not Brent-searched). The memory `modality_reliability` averages over
is still the full 1800-row table (excluding the query patient) — only the *bandwidth* comes from the
deduplicated table; the estimator itself needs every split-appearance as a distinct observation. For
each LOOCV query patient `p`, compute `e(x_p)` from `UNIMODAL_SMOOTHED_OOF`, then `r_m(x_p) =
modality_reliability(e(x_p), evidence_table, exclude_patient=p, modality=m, sigma)` for each `m`, then
normalize to a simplex:

```python
r = np.array([r_tab, r_mri, r_txt])
w = r / r.sum() if r.sum() > 1e-8 else np.array([1/3, 1/3, 1/3])   # uniform fallback, documented not expected to trigger
```

`p_fused(x_p) = soft_vote({m: to_2col(UNIMODAL_SMOOTHED_OOF[m][p:p+1])}, dict(zip(["tab","mri","txt"], w)))`
— same arithmetic pooling rule as C1, only the weight source differs (§3, table). Per-patient `w(x_p)`
is saved (`gate_weights_per_patient.csv`) and summarized (`gate_weights_summary.json`: mean/std of
`w_tab, w_mri, w_txt` over the 88 LOOCV patients) — this mean/std is the non-parametric, per-patient
*reliability weight* (§1's precision note: not a Bayesian evidence-combination weight in the log-linear
sense — that's C2), reported alongside exp_29's fixed `(0.4, 0.1, 0.5)` for comparison (§8, context only
— a global summary of an instance-varying quantity is not the same claim as a single fixed triple, and
the comparison is descriptive, not a hypothesis test).

## 4. Reference / context values (not re-derived, loaded or cited from exp_29's own results)

- `exp_29 fusion_optimal_leakfree__hard = 0.7067941474721136` (`experiments/fagonzalezo/exp_29/results/
  loocv_metrics.json`) — historical context only (§2.2: not numerically comparable due to `label_
  smoothing`).
- `exp_29 fusion_weights_leakfree.json`: `tab=0.4, mri=0.1, txt=0.5` — compared descriptively against
  C3's gate weight summary (§3.3, §8).
- `exp_8 = 0.7171` — cited only as the lineage's long-standing unbeaten target; **not** this experiment's
  success bar (§1).

## 5. Step 0 — reproduction gates

- **G0** — reload `experiments/fagonzalezo/exp_29/results/stage1_best_hparams.json["hard"]`, refit each
  modality's frozen `(rep, knn_k, sigma)` at `label_smoothing=0` (not 0.05 — this gate checks the reload
  is faithful, independent of §2.2's new choice) through `run_loocv`, and assert Macro-F1 matches
  exp_29's own stored hard-arm unimodal values (§2.1 table) to `abs(diff) < 1e-9`.
- **G1** — at `label_smoothing=0` (not 0.05), C1's simplex-grid search machinery (`simplex_grid`,
  `soft_vote`, scored over `WINNER_VAL_PROBS_SMOOTHED` built with `label_smoothing=0`) must reproduce
  exp_29's own `fusion_weights_leakfree.json` (`tab=0.4, mri=0.1, txt=0.5`) and
  `fusion_optimal_leakfree__hard` (`0.7067941474721136`) bit-for-bit — this is a structural
  self-consistency check that C1's code is exp_29 S1's construction verbatim, with `label_smoothing` as
  the only new parameter (set to 0 here to isolate that claim).
- `train.py` asserts both gates with `abs(got - expected) < 1e-9`, aborting on failure, and writes
  `results/reproduction_gates.json`.

## 6. File Layout

```
experiments/exp_31/
├── DESIGN.md
├── IMPLEMENTATION.md          ← added after this file is approved
├── scripts/train.py            ← self-contained; --smoke flag
├── results/
│   ├── reproduction_gates.json       ← G0/G1 pass/fail + values
│   ├── fusion_weights_arithmetic.json    (C1)
│   ├── fusion_weights_loglinear.json     (C2)
│   ├── gate_sigma.json                   (C3 — the fixed data-driven bandwidth used)
│   ├── gate_weights_per_patient.csv      (C3 — w_tab/w_mri/w_txt per LOOCV patient)
│   ├── gate_weights_summary.json         (C3 — mean/std, vs. exp_29's fixed triple)
│   ├── loocv_metrics.json            ← C1/C2/C3 + unimodal_smoothed arms
│   ├── loocv_predictions.csv
│   ├── mcnemar.json                  ← H1 (C2 vs C1), H2 (C3 vs C1), context (C2 vs C3)
│   └── git_commit.txt
└── reports/
    ├── figures/   ← per-patient gate-weight distribution, confusion matrix, ROC overlay
    └── summary.md
```

## 7. Evaluation Protocol & Decision Rules

Same two-phase MCCV→LOOCV harness (`CLAUDE.md`; `src/evaluation/protocol.py`): Phase A = 100 MCCV
splits (weight/gate construction, §3); Phase B = 88-fold LOOCV, frozen fusion mechanism, per-modality
experts refit per fold (never re-fit hyperparameters). Metric: `binary_metrics` (Macro-F1 primary) +
`mcnemar_exact`, paired on the 88 LOOCV folds.

- **H1 (primary):** `fusion_loglinear__hard` (C2) exceeds `fusion_arithmetic_smoothed__hard` (C1) in
  LOOCV Macro-F1, with `mcnemar_exact` significance (p < 0.05) between their paired LOOCV predictions.
- **H2 (primary):** `fusion_kdm_gate__hard` (C3) exceeds `fusion_arithmetic_smoothed__hard` (C1) in
  LOOCV Macro-F1, with `mcnemar_exact` significance (p < 0.05) between their paired LOOCV predictions.
  Reported alongside the §2.4 leakage-asymmetry caveat — a significant win is evidence the mechanism
  helps, but not fully separable from C3's stricter protocol.
- **Context (not scored against H1/H2):** C2 vs. C3 pairwise McNemar (does pooling-rule change or
  weight-source change matter more); C1 vs. exp_29's `0.7068` (sanity check that `label_smoothing=0.05`
  does not materially change the arithmetic-pool number); C3's `gate_weights_summary.json` vs. exp_29's
  `(0.4, 0.1, 0.5)`.
- Given `std_macro_f1 ≈ 0.11` across MCCV splits in this cohort (exp_28/29's own estimate, unchanged
  cohort), any Phase-B improvement without significant McNemar is reported as **not established**, not a
  win — same standard as every prior experiment in this lineage.
- Per §1: **exceeding exp_8's 0.7171 is not a decision criterion here.** If some condition happens to
  reach it, report it as a bonus, not the headline.

## 8. Scope

**In scope:** C1/C2/C3 as specified (hard arm only); reloading exp_29's frozen per-modality winners
verbatim (no re-search); `label_smoothing=0.05` applied uniformly to all three modalities' experts;
G0/G1 reproduction gates; `mcnemar_exact` for H1/H2 and the C2-vs-C3 context comparison; the gate weight
summary (§3.3) as a descriptive, non-hypothesis-tested comparison to exp_29's fixed triple.

**Out of scope:** editing `src/methods/*` or `src/evaluation/*` (all new logic is local to
`experiments/exp_31/scripts/train.py`, matching exp_28/29's own scope discipline); re-searching any
per-modality `(rep, knn_k, sigma)` — those are exp_29's, taken as given; the soft (confidence-informed)
arm; Brent-searching or grid-searching the gate's evidence-space bandwidth (§2.4 — fixed data-driven
default only); searching `label_smoothing` itself (§2.2 — fixed at 0.05); combining C2's log-linear
pooling with C3's non-parametric weights in one condition (a natural follow-up if either H1 or H2 holds,
not attempted here to keep each condition's novelty isolated); auditing `iter_mccv_splits`/exp_4's
per-patient validation-membership balance (§2.5, flagged as a limitation, not corrected).

## 9. Next Steps

1. Review and accept this design.
2. `IMPLEMENTATION.md` (concrete build plan, exact execution command) for approval.
3. Implement `scripts/train.py`, run `--smoke`, then the full run.
