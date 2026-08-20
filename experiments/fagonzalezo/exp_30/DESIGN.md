# Experiment Design: Uncertainty Prediction Ability of BrentMemKDM (exp_30)
**Experiment**: experiments/exp_30/ · **Project**: pathology-reasoning · **Date**: 2026-08-19 · **Status**: Complete

---

## 1. Motivation — the KDM lineage has never honestly beaten classical ICI on confidence

Five consecutive experiments (exp_23, exp_25–29) evaluated the KDM/MemKDM family on the **binary
biopsy-decision** task; all five failed to beat exp_8's 0.7171 LOOCV Macro-F1 with significance.
exp_29 (the `knn_k` truncation sweep) closed that thread. Meanwhile `BrentMemKDM` — the
zero-gradient, globally-fit, continuous-bandwidth model introduced in exp_28 and extended with
per-query k-NN truncation in exp_29 — has **never been evaluated on CHIMERA Task 1's other
objective**, the 3-class diagnostic confidence label (`uncertain`/`borderline`/`clear`). exp_28 and
exp_29 ran no confidence task at all.

A full audit of the confidence sub-lineage (exp_9–12, exp_17, exp_19, exp_23–27) turned up the fact
that motivates this experiment specifically:

> **Every row of exp_25, exp_26, and exp_27 (all 366 rows, across all three results files) is
> `target_informed: True`.** Their headline numbers — 0.4547, 0.5287, and exp_27's 0.5630 (the
> project's best confidence result to date) — are all obtained from models whose soft targets
> encode the confidence label via `CONFIDENCE_CERTAINTY_MAP` (`build_targets(cohort.y_binary,
> cohort.confidence, certainty_map=CONFIDENCE_CERTAINTY_MAP)`, `exp_27/scripts/train.py:112`).
> The best **non-target-informed** confidence result anywhere in this project is still **exp_24's
> 0.4368** (`a_hard__multivariate_7signal`), which remains short of exp_17's classical Composite
> Fuzzy ICI baseline of **0.4470**.

So no KDM-family model has ever beaten classical ICI on an honest, non-target-informed footing.
`BrentMemKDM`'s hard arm is structurally clean for this purpose: with `build_targets(y_binary)`
(no certainty map), `x_train = y_train = w_train = False`, and σ selected in Phase A against
**binary** macro-F1 on `y_binary` alone, nothing in the model ever touches the `confidence` column.
exp_29 already ran exactly this arm for the binary task and recorded `target_informed: false` for
all six hard conditions — this experiment reuses that same non-leakage property for a different
target.

**exp_30 is therefore the first clean shot at exp_17's 0.4470 since exp_24** — and it brings a
mechanism exp_24 could not have had: exp_29's `knn_k` makes the retrieved neighborhood *literal*
(a real, indexable set of `k` memory points), so neighborhood-level uncertainty signals — label
disagreement among the retrieved neighbors, weight concentration over them — become definable for
the first time in this lineage.

**H1 (primary, non-target-informed).** At least one `BrentMemKDM` **hard-arm** confidence
predictor exceeds **both** exp_24's 0.4368 (best clean KDM result) **and** exp_17's 0.4470
(classical ICI baseline) on 3-class LOOCV Macro-F1 — i.e. the first KDM-family model to beat
classical ICI without consuming the confidence label anywhere in its own construction.

**H2 (secondary, mechanism).** Family-C neighborhood signals (§4), definable only under `knn_k`
truncation, add information beyond the whole-memory particle signals: a head trained on
families A–C at a finite `knn_k` beats the same head restricted to families A+B at `knn_k=None`.

**H3 (secondary, protocol).** Report the first significance test ever run on this task in this
project. No confidence-task experiment to date (exp_9–12, 17, 19, 23–27) has run one; every ranking
in the lineage, including the reference numbers this experiment targets, is an unguarded point
estimate.

## 2. Reference numbers (verified against results files, not `INDEX.md` summaries)

| Reference | Macro-F1 | Non-target-informed? | Source |
|---|---|---|---|
| All-`clear` constant predictor (collapse floor) | **0.2593** | — | class balance (56/18/14 of 88) |
| exp_17 Composite Fuzzy ICI | **0.4469706011059394** | ✅ yes (head not held out — see §5) | `exp_17/results/loocv_confidence_metrics.json` |
| exp_24 `a_hard` 7-signal held-out trees | **0.4368** | ✅ yes — best clean KDM number | `exp_24/results/confidence_metrics.json` |
| exp_23 `log_marginal_hard` | 0.3340457088136029 | ✅ yes | `exp_23/results/confidence_metrics.json` |
| exp_25 `joint_tab_mri`/`1d`/`w_max` | 0.4547 | ❌ no | `exp_25/results/...` |
| exp_26 late-fusion 23-signal held-out trees | 0.5287 | ❌ no | `exp_26/results/...` |
| exp_27 `unimodal_tab`/`{h_total,log_marginal}` | 0.5630 | ❌ no | `exp_27/results/...` |

Class balance, N=88 labeled cohort: clear 56 (63.6%), borderline 18 (20.5%), uncertain 14 (15.9%),
identical cohort to the binary task (`src/evaluation/data.py`'s `CONFIDENCE_CODE_MAP`).

exp_25/26/27's `target_informed=True` is set unconditionally the moment `build_targets(...,
certainty_map=CONFIDENCE_CERTAINTY_MAP)` is used to construct the model at all (`mem_kdm.py:373`
propagates `Targets.soft_from_confidence` into `MemKDM.target_informed`) — every downstream signal
in those experiments inherits confidence information both directly (through `c_y`) and indirectly
(through the σ fitted against that soft loss). This experiment reports provenance **per signal**,
not as one blanket flag, because some soft-arm signals (family C, §4) remain non-target-informed
even when the model itself is fit on soft targets.

## 3. Background — what's reused

Reuses `src/evaluation/{data,protocol,metrics,reporting}.py` and `src/methods/{base,mem_kdm,
brent_mem_kdm}.py` exactly as exp_29 left them, same `Data/preprocessed_old/task1/` schema, same
N=88 complete-case cohort. `BrentMemKDM.uncertainty_signals()`, `.fit_confidence()`, and
`.predict_confidence()` are already implemented (`brent_mem_kdm.py:800-900`) and already dispatch
to `extract_particle_signals` (whole-memory) or `_knn_signals` (truncated) — no new entry points
are needed for families A/B/D. Confidence heads reuse `src/methods/base.py`'s
`fit_meta_thresholds_safe`/`apply_meta_thresholds` (1-D) and `fit_predict_heldout_trees`
(multivariate), the exact two heads exp_24–27 used.

### 3.1 What's new: Family C requires an additive code change

`_knn_signals` (`brent_mem_kdm.py:825-847`) currently retrieves the k nearest memory points via
`_knn_submodel`, which returns the neighbor indices, then **discards them**
(`model, _nbr = _knn_submodel(...)`). Family C (§4) needs those indices kept and turned into three
new signals. This is additive — new dict keys on `uncertainty_signals`' return value — so every
exp_28/29 call site (which never reads these keys) stays bit-exact. Implemented and verified in the
implementation phase, not here.

### 3.2 `utils/embedding-kit/` is still empty — same caveat as exp_28/29, unchanged

Not relevant to this experiment's MRI arm choice (§4 restricts to `{raw_l2, pca90_l2}` as exp_28/29
did) but stated for consistency with prior DESIGN.md files.

## 4. Conditions — narrow, pre-registered

Narrowed deliberately (not exp_29's full 7-point `knn_k` grid): the confidence task adds a large
new selection surface (signal family × head) on top of `knn_k`, and exp_29 §2.2 already
demonstrated that a wide grid on this N=88 cohort risks fitting Phase-A noise. Pre-registering a
small grid keeps H1/H2 falsifiable rather than a search for the best-looking cell.

**Modalities (4):** `tab`, `mri` (`{raw_l2, pca90_l2}`, exp_28/29's restriction), `txt`
(`{500, 2000, None}` max_features), plus the trimodal late-fusion composite (needed only for
family D's `composite_reliability_index`/`inter_modality_variance`).

**`knn_k` (3 candidates, fixed, not searched):** `{5, 20, None}`. `None` is the exp_24–27 control
(whole-memory, family C undefined). 5 and 20 bracket exp_29's winning `knn_k` range for these
modalities without landing on the degenerate low end (§6.1). σ itself is still Brent-searched per
`(modality, knn_k)` as in exp_28/29 — only `knn_k` is a fixed grid, matching
`run_brent_search`'s own contract that `knn_k` is swept in an outer loop, never Brent-searched.

**Arms (2):** hard (`build_targets(y_binary)`, non-target-informed, carries H1) and soft
(`build_targets(y_binary, confidence, CONFIDENCE_CERTAINTY_MAP)`, target-informed, reported only
for lineage comparability with exp_25–27, never used for H1).

⚠️ **`knn_k=1` is excluded from the condition grid and used only as a degeneracy gate (G3, §7)** —
see §6.1.

## 5. Signal families

| Family | Signals | Status | Provenance |
|---|---|---|---|
| A — collapsed | margin `\|p−0.5\|`, `h_total` | exists | non-target-informed in hard arm |
| B — particle | `h_aleatoric`, `h_epistemic`, `h_weights`, `log_ess`, `w_max`, `log_marginal` | exists (`mem_kdm.py:484`) | non-target-informed in hard arm; `h_aleatoric`/`h_epistemic` degenerate to 0 under hard targets with `label_smoothing=0` (§6.4) |
| C — neighborhood (**new**) | label disagreement (entropy/variance of `y_binary`) over the k retrieved neighbors; weight concentration (Gini or normalized entropy) over the k; distance to the k-th neighbor | **to add**, §3.1 | non-target-informed in **both** arms — a pure input + binary-train-label statistic (§6.5) |
| D — multimodal | `composite_reliability_index` (ICI, `mem_kdm.py:538`), `inter_modality_variance` (`mem_kdm.py:550`) | exists | computed on the trimodal fusion of families A/B, same provenance as its inputs |

Family C staying non-target-informed even in the soft arm is deliberate and important: it means H2
can be tested on fully honest footing, independent of the arm/leakage question H1 is about.

## 6. Risks — stated explicitly, not deferred to the report

### 6.1 `knn_k` degenerates signals — the sharpest hazard in this design

At `k=1`, the single retrieved neighbor's weight normalizes to 1 for every query, so
`h_weights ≡ 0`, `log_ess ≡ 0`, `w_max ≡ 1` — three signals collapse to constants regardless of
modality or query. Under hard targets, additionally `h_total = h_aleatoric = h_epistemic ≡ 0`
(§6.4). **Only `log_marginal` varies at `k=1`.** At `k=3`, `h_weights, log_ess ∈ [0, ln 3]` — still
coarse. This is why the condition grid (§4) starts at `k=5`: exp_29's soft-arm `mri` winner was
`knn_k=1` (`stage1_best_hparams.json`), so naively inheriting exp_29's frozen configs would hand
this experiment a near-constant signal set on that arm/modality without anyone noticing.

### 6.2 `log_marginal` is not comparable across `k`

Under truncation, `log_marginal` is a sum over `k_eff` components, not `n_train`
(`brent_mem_kdm.py:829-835`) — its scale shifts with `k`. Thresholds/trees are refit per `(knn_k,
arm)` cell as part of Phase A, which is correct; the risk is only in any post-hoc comparison of raw
`log_marginal` values across `k` cells, which this experiment must not do.

### 6.3 The soft arm is doubly target-informed, and one signal is a near-direct label read-out

Under `CONFIDENCE_CERTAINTY_MAP` (`clear=1.00, borderline=0.50, uncertain=0.25`),
`h_aleatoric = Σ_j w_j · H_b(y_soft_j)`, and `H_b(y_soft_j)` takes exactly three values keyed to
each neighbor's own confidence class: clear → 0, borderline → 0.5623, uncertain → 0.6616 nats,
strictly decreasing in confidence. Soft-arm `h_aleatoric` (and `h_epistemic`, which subtracts it
from `h_total`) is therefore a kernel-weighted read-out of neighbors' confidence labels, not a
derived uncertainty quantity. Every soft-arm row involving these two signals must say so
explicitly, not just carry a `target_informed=True` flag.

### 6.4 `h_total` in this frozen regime is a monotone function of the prediction margin

With `x_train=y_train=w_train=False` and identity encoders, the Nadaraya-Watson reduction gives
`p_mean = p₁` exactly (`brent_mem_kdm.py:17-54`), so `h_total = H_b(p₁)`, strictly decreasing in
`|p₁ − 0.5|`. Consequently `{h_total, log_marginal}` — exp_27's own best ablation, target-informed
there — is structurally the pair (prediction margin, input kernel density), the same family as
exp_17's ICI `= 2·margin · (1 − 2·std)` with inter-modality agreement swapped for input density.
Report this plainly rather than presenting the signals as unrelated; a comparison to exp_17 is in
some sense a comparison between two members of the same family.

### 6.5 Family C's non-target-informed status is a real property, not a labeling convenience

Family C signals are computed from the k retrieved neighbors' **binary** labels (`y_binary`, never
`confidence`) and from input distances alone — they consume nothing that touches the confidence
column even when the underlying `BrentMemKDM` was fit on soft targets, because the neighbor
*retrieval* (§3.1's `_topk_neighbors`, kernel value in X-space) doesn't depend on `c_y` at all.
State this explicitly with a worked example in the report, not just asserted.

### 6.6 Selection surface at N=88

Even narrowed: 4 modalities × 3 `knn_k` × 2 arms × 2 heads × (up to 4 signal-family combinations
per head) is still a non-trivial Phase-A surface on a cohort with `std_macro_f1 ≈ 0.11` on the
binary task (exp_28/29's own estimate; the confidence task's equivalent noise floor is unmeasured
and should be estimated in Phase A rather than assumed). exp_29 §2.2's rule carries over verbatim:
**a Phase-B improvement without a significant test (H3) is reported as not established, not as a
win.**

### 6.7 Frozen-config leakage, accepted and stated

Unchanged from exp_17/24/27/28/29: Phase-A σ selection spans all 100 MCCV splits, which
collectively cover the whole cohort, so the frozen `(knn_k, σ)` used in every LOOCV fold of Phase B
was informed, in aggregate, by data including that fold's own patient. Not claimed leak-free.

### 6.8 Compute

In knn mode, every `uncertainty_signals`/`predict_proba` call fits a **fresh `MemKDM` per query
row** (`_knn_submodel`, `brent_mem_kdm.py:319`). Confidence-signal extraction over 88 LOOCV folds ×
4 modalities × 2 finite `knn_k` values × 2 arms is a materially larger compute budget than exp_29's
binary sweep, which only needed a scalar probability per fold. A `--smoke` timing check (small
fold/split subset) is required before launching the full run, and its wall-clock extrapolation
must be reported before the full run is approved to execute.

## 7. Reproduction gates (Step 0)

- **G0** — `python scripts/verify_brent_mem_kdm.py` (full run): all checks pass (56/56 as of
  exp_29; re-verify the count against whatever commit `results/git_commit.txt` names).
- **G1** — recompute exp_17's Composite Fuzzy ICI from `exp_16/results/oof_predictions.csv`
  (columns `prob_{tabular,mri,text}_fuzzy`) using `composite_reliability_index` + the
  MCCV-mean-threshold head, and reproduce **0.4469706011059394** exactly.
- **G2** — reproduce exp_24's `a_hard__multivariate_7signal` = **0.4368** from stored/recomputed
  signals, or if not exactly reproducible (e.g. due to library version drift in the trainable KDM
  path), document the discrepancy explicitly rather than silently adopting a different number as
  the target.
- **G3 (new, degeneracy self-check)** — at `knn_k=1` (used only as a gate input, never a condition,
  §4), assert the exact degeneracies predicted in §6.1: `h_weights ≡ 0`, `log_ess ≡ 0`,
  `w_max ≡ 1` (any modality, any σ in bounds), and under hard targets additionally `h_total ≡ 0`.
  This validates both the existing particle-signal code and the new family-C code against the same
  truncation semantics exp_29's own G3 verified for predictions.
- **G4 (new)** — family-C signals are *absent from the returned dict* at `knn_k=None` (no
  neighbor set exists to define them), not silently computed over an arbitrary top-k substitute.
- `train.py` asserts G1/G2 with `abs(got - expected) < 1e-9` (looser than exp_29's `1e-12` given
  the extra recomputation step through a differently-shaped pipeline) and G3/G4 as exact equality /
  key-absence checks, aborting on any failure.

## 8. File Layout

```
experiments/exp_30/
├── DESIGN.md
├── IMPLEMENTATION.md          ← added after this file is approved
├── scripts/train.py            ← self-contained; --smoke flag (reuses exp_28/29's script structure)
├── results/
│   ├── reproduction_gates.json       ← G0-G4 pass/fail + values
│   ├── phasea_sigma_grid_{tab,mri,txt}.csv   ← per-(modality,knn_k) Brent search, reused from exp_29
│   ├── stage1_best_hparams.json      ← per-modality winning (rep, knn_k, sigma*), binary-selected
│   ├── loocv_signals.csv             ← per-patient OOF values for every signal, every (modality,knn_k,arm)
│   ├── confidence_metrics.json       ← Macro-F1/accuracy/rho per (modality,knn_k,arm,head,signal-set)
│   ├── confidence_predictions.csv    ← per-patient predicted confidence class, for H3
│   ├── significance.json             ← H3: mcnemar_exact + permutation test vs. exp_17/exp_24
│   └── git_commit.txt
└── reports/
    ├── figures/   ← signal distributions by confidence class, knn_k degeneracy check, confusion matrix
    └── summary.md
```

## 9. Evaluation Protocol & Decision Rules

Two-phase MCCV→LOOCV harness (`CLAUDE.md`'s protocol; `src/evaluation/protocol.py`), extended with
a signal-extraction step between Phase A and the confidence head:

1. **Phase A₁ (σ, binary objective):** per `(modality, knn_k)`, `run_brent_search` selecting on
   **binary** macro-F1 (`y_binary`) over 100 MCCV splits — identical mechanics to exp_28/29. σ
   selection never sees `confidence`, which is what keeps the hard arm non-target-informed end to
   end.
2. **Phase B (signals):** 88-fold LOOCV, frozen `(knn_k, σ)` per `(modality, arm)`, model refit per
   fold, reading `uncertainty_signals(X_va)` on the single held-out row → genuine leave-one-out
   signal vectors for every family.
3. **Phase A₂ (confidence head):** fit `fit_meta_thresholds_safe`/`apply_meta_thresholds` (1-D) and
   `fit_predict_heldout_trees` (multivariate, families A+B vs. A+B+C) on the OOF signals from step 2,
   across all 100 MCCV splits — never reduced, or the heldout-tree "every patient gets ≥1 held-out
   vote" invariant breaks.
4. **Metrics:** `confidence_metrics` (3-class Macro-F1 primary, accuracy, Spearman ρ), wrapped in
   the `safe_confidence_metrics` NaN guard exp_25–27 used for degenerate/single-class predictions.

- **H1 (primary):** at least one hard-arm confidence predictor (any modality, `knn_k ∈ {5,20,None}`,
  either head) exceeds both 0.4368 and 0.4470, **with** the H3 significance test (below) confirming
  the margin over at least the *harder* of the two comparators (exp_24, since it shares the
  non-target-informed property this experiment is being held to).
- **H2 (secondary):** for at least one modality/knn_k/arm cell, a held-out-tree head over families
  A+B+C beats the same head restricted to A+B at the same `knn_k` (finite) — compared against the
  `knn_k=None` A+B-only control, with `mcnemar_exact` on per-patient correctness.
- **H3 (secondary, protocol):** report `mcnemar_exact` (adapted to 3-class via per-patient
  correctness, exp_25–27's convention for the binary task carried over) plus a permutation test on
  the Macro-F1 delta, against exp_17's recomputed predictions (G1) and exp_24's recomputed
  predictions (G2). This is the first time this comparison has ever been guarded by a test in this
  project.
- Given the binary task's `std_macro_f1 ≈ 0.11` at N=88 (exp_28/29) and no prior measurement of the
  confidence task's own noise floor, **a Phase-B improvement without significance is reported as
  not established, not as a win** — same standard as exp_29 held itself to (§6.6).
- Soft-arm rows are reported in full, for lineage comparability with exp_25–27, but are explicitly
  excluded from deciding H1 (§1).

## 10. Scope

**In scope:** hard and soft arms across the 4-modality × 3-`knn_k` grid (§4); signal families A–D
(§5), with family C newly implemented as an additive `_knn_signals` change; both confidence heads;
reproduction gates G0–G4; the H1/H2/H3 comparisons and their significance tests as defined in §9;
a `--smoke` compute-budget check before the full run (§6.8).

**Out of scope:** editing `src/evaluation/*` or `src/methods/{base,mem_kdm}.py` beyond what family
C requires in `brent_mem_kdm.py` (additive only); the new-schema data
(`Data/preprocessed/task1/`); calibration/selective-prediction metrics on the binary task (a
distinct question, deferred to a future experiment if wanted); widening the `knn_k` grid beyond
`{5, 20, None}` (a follow-up if H2 shows a real, significant effect worth resolving more finely);
re-litigating the binary-task H1/H2 verdicts from exp_23–29.

## 11. Next Steps

1. Review and accept this design.
2. `IMPLEMENTATION.md` (concrete build plan: the family-C addition to `brent_mem_kdm.py`, new
   `scripts/verify_brent_mem_kdm.py` checks for G3/G4, and `experiments/exp_30/scripts/train.py`) —
   in Claude Code plan mode, not this skill.
3. Implement, run `--smoke`, confirm the compute-budget extrapolation from §6.8, then the full run.
