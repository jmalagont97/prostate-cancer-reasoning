# Experiment Design: Task3 Embedding Fusion (T3.2/T3.3)
**Experiment**: experiments/exp_5/
**Project**: challenge_chimera_v3
**Date**: 2026-08-25
**Status**: Design (not yet run)

---

## 1. Context and hypothesis

`exp_3`/`exp_4` built a tabular+text baseline (age/psa/cleaned-dre + 8 surgical-pathology fields,
Cox penalizer=0.3) scoring LOOCV `ranking_score` (C-index) = **0.7655** — this experiment's target
to beat. MRI, biopsy-slide, and prostatectomy-slide embeddings (frozen, provided by the challenge)
sit completely unused. The plan (`docs/model_training_plan.md` T3.2/T3.3) names embedding fusion as
the single biggest untested lever left for task3.

**Hypothesis, stated skeptically rather than optimistically** (worth being explicit about, given
this project's repeated finding that naive feature expansion hurts at this sample size — task1's
`exp_1`/`exp_2` refuted feature expansion outright, and only a carefully-scoped feature set ever
won): embeddings may add **little marginal value** on top of the current baseline, because
`exp_3`'s pathology-text extraction (ISUP grade, stage, margins, EPE, SVI, LN status) already
decodes much of what a prostatectomy-slide embedding would encode about the same tissue — grade and
morphology are largely the same signal, one machine-read, one human-reported. **MRI is the one
source with a real prior for adding independent, non-redundant signal**: it's a different modality
(imaging, not tissue), a different time point (pre-operative, not post-surgical), and task3's
structured-prompt.json carries no imaging-derived field at all (unlike task1's `pirads`). Biopsy-
slide embeddings sit in between — pre-surgical tissue, but the same modality (histopathology) as
what the post-surgical text already captures, and 24/75 cases (32%) have *no* valid biopsy-slide
embedding at all (real missingness, not an artifact — see §3).

Refuted (per-source) if that source's fused LOOCV C-index does not exceed 0.7655. Promoted (per-
source or combined) only if it does, following this project's standing promotion-gated discipline —
no partial credit for "plausible" a priori reasoning above, which is why it's tested rather than
assumed.

---

## 2. Data confirmed (not assumed) this session

| Source | Shape | Coverage |
|---|---|---|
| `embeddings/mri.npz` | `(75, 1024)`, one vector per case | 75/75 (always present) |
| `embeddings/biopsy_slides.npz` | `(75, 3, 960)`, up to 3 slide vectors/case | 51/75 have ≥1 valid slide (24 have **zero**: 0→24, 1→7, 2→13, 3→31 cases) |
| `embeddings/prostatectomy_slides.npz` | `(75, 3, 960)`, up to 3 slide vectors/case | 75/75 have ≥1 valid slide (1→31, 2→29, 3→15 cases) |

## 3. Methodology

### 3.1 Pooling (fixed, not a design choice — there's only one sensible option without a
learned aggregator, which is out of scope at n=75)
- MRI: already a single vector per case, no pooling needed.
- Biopsy/prostatectomy slides: **masked mean-pool** over the valid slide vectors per case (per the
  plan's own T3.2/T3.3 rationale). For the 24 biopsy-slide cases with zero valid slides: pooled
  vector imputed as the **training fold's mean pooled vector** (not zero — a zero vector is an
  arbitrary point in embedding space, not a neutral one) plus an explicit `has_biopsy_slides` binary
  flag column, so the model can learn to discount the imputed placeholder. Mirrors this project's
  consistent convention for structural missingness elsewhere (`dre_not_done`, `pirads_na`,
  `ln_positive`'s NaN-when-unassessed) — flag it, don't silently guess.

### 3.2 Dimensionality reduction (the real design decision)
960–1024 raw dims against **n=74 training cases per LOOCV fold** is exactly the curse-of-
dimensionality trap task1's `exp_1`/`exp_2` already burned real effort confirming is real on this
kind of small-n clinical data. PCA is required, and **must be fit inside each LOOCV training fold
only** (never on all 75 cases up front — that's the exact leakage class `exp_1`'s
`clinician_shortlist` condition was caught doing, collapsing AUROC 0.724→0.363 once fixed). Swept
component counts: **k ∈ {2, 5, 10}** — deliberately small, single-digit-to-low-double-digit range,
not a wide sweep, given the established prior that more dimensions hurts more often than it helps
here.

### 3.3 Conditions (staged, not a single giant cross-product)

**Stage A** — fixed model (`cox_p03`, the current promoted default) + fixed k=5, one embedding
source at a time plus the current baseline feature set:
1. `baseline` (no embeddings — reproduces 0.7655, sanity check)
2. `baseline + mri_pca5`
3. `baseline + biopsy_pca5` (+ `has_biopsy_slides` flag)
4. `baseline + prostatectomy_pca5`
5. `baseline + all_three_pca5`

**Stage B** — for whichever source(s) beat baseline in Stage A only: sweep k ∈ {2, 10} around 5 to
check the component count isn't the reason it won or lost (a source that only wins at one arbitrary
k value is a weaker finding than one that wins across the swept range).

**Stage C** — only if Stage A/B finds a real winner: re-sweep the Cox penalizer (same 7-point grid
`exp_4` used) on the new best feature set, since the optimal regularization strength may shift once
feature count changes — don't assume `cox_p03`'s penalizer=0.3 still holds.

If nothing in Stage A beats 0.7655, stop there and report the refutation — no reason to run
Stage B/C on a source that already lost.

### 3.4 Protocol
LOOCV only (primary and only protocol here — this project stopped reporting 5-fold as anything but
a continuity check once `exp_1` showed it misleading at this n, and there's no prior published
5-fold number for an embedding-fused task3 model to stay comparable to anyway).

### 3.5 Metric
`evaluate.py`'s own `concordance_index`, exactly as `exp_4` used it (literal predicted months, not
partial hazard) — same real-contract scoring, not a proxy.

## 4. Promotion rule

Same as `exp_3`/`exp_4`: only replace `task3_config.py`'s feature set / `DEFAULT_MODEL_KEY` if the
new condition's LOOCV C-index measurably beats 0.7655. Given `exp_4`'s own gain (+0.017) turned out
not to be significant under bootstrap, any embedding win here should get the same bootstrap check
before being described as decisive, not just a point-estimate comparison.

## 5. What this experiment deliberately does NOT attempt

- No learned/attention-based pooling (needs way more than 75 cases to train safely).
- No fine-tuning of the embeddings themselves (frozen, as provided).
- No fusion of embeddings with each other beyond simple concatenation post-PCA (e.g. no
  cross-modal attention) — out of scope at this n.
