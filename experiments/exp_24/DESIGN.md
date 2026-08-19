# Experiment Design: Particle-Set Uncertainty Decomposition for the Tabular KDM
**Experiment**: experiments/exp_24/ · **Project**: pathology-reasoning · **Date**: 2026-08-18 · **Status**: Complete

---

## 1. Hypothesis

A `KDMLayer`'s output `rho_y` is not a probability vector — it is a packed density matrix of shape
`(bs, n_comp, dim_y+1)`, i.e. a weighted **particle set** `{(w_j(x), p_j)}` with `n_comp = n_train`
particles. Particle `j` carries its own predictive distribution `p_j = normalize(c_y_j)²` (Born rule
on the learned/frozen prototype label vector `c_y_j`) and a per-input mixture weight `w_j(x)` from the
RBF kernel. `dm2discrete` (`own_libs/kdm/kdm/utils.py:58-71`, confirmed against the installed package)
computes `probs = Σ_j w_j p_j` — the mixture **mean** — and discards everything about how the
particles disagree.

exp_23 built its uncertainty story entirely on that collapsed mean: `entropy = H(probs)` and
`log_marginal = log P(x)`. Both underperformed the `exp_17` Composite-ICI baseline (Macro-F1 0.4470)
on the secondary confidence-prediction objective — best non-target-informed exp_23 signal was
`log_marginal_hard` at 0.3340 (`experiments/exp_23/results/confidence_metrics.json`). The unexploited
information is entirely in the particle set's internal structure.

**H1 (primary).** A signal derived from the particle set's internal structure — rather than only its
collapsed mixture mean — predicts the clinician's 3-class `confidence` annotation with higher Macro-F1
than both exp_17 (0.4470) and exp_23's best non-target-informed signal (0.3340).

**H2 (secondary).** A genuinely learned aleatoric/epistemic decomposition of predictive entropy
(§2.2) carries information beyond the collapsed `h_total` alone, evidenced by (a) significant Spearman
correlation of `h_aleatoric` against `confidence`, and (b) the full multivariate confidence head
outperforming an ablation restricted to exp_23's original two signals.

**H3 (structural, must be confirmed before H1/H2 are interpretable).** exp_23's hard arm (one-hot
`c_y` init, any `x_train`/`y_train` setting) admits *no* learnable aleatoric/epistemic split, because
one-hot amplitude vectors are an exact fixed point of the Born-rule Jacobian. See §2.1.

## 2. Background — why exp_23's hard arm cannot answer H1/H2 as-is

### 2.1 The one-hot fixed point

For a real vector `v`, let `u = v / ‖v‖` and `p = u²` (elementwise). The Jacobian is

```
∂p_i/∂v_k = 2 u_i (δ_ik − u_i u_k) / ‖v‖
```

At a one-hot `u = e_1` this is identically zero for every `(i, k)`: `c_y` sits at an exact gradient
fixed point of the amplitude→probability map, regardless of `x_train`/`y_train`/`w_train` or the loss
used downstream. Verified three independent ways during design review:

1. **Analytic** — the Jacobian above vanishes identically at any one-hot `u`.
2. **exp_23's own grid** — `experiments/exp_23/results/grid_search_results.csv`: every one of the 16
   hard-arm `(sigma_mult, x_train, encoder)` combinations is **bit-identical** between `y_train=True`
   and `y_train=False` (e.g. `sigma_mult=2.0, x_train=False, encoder=linear` → `0.596540 / 0.121817`
   both ways). All 16 soft-arm pairs differ.
3. **Direct gradient probe** against the installed `own_libs/kdm` package: one-hot init →
   `max|grad c_y| = 0.000e+00` after a forward+backward NLL pass; ε-smoothed-amplitude init on the
   same data → `3.838e-03`.

Consequence for the textbook decomposition `H(Σ_j w_j p_j) = Σ_j w_j H(p_j) + I` (Jensen's
inequality on entropy's concavity — the standard aleatoric/epistemic split for MC ensembles, e.g.
Depeweg et al. 2018 "BALD"): on exp_23's hard arm, `p_j` is exactly one-hot for every `j`, so
`H(p_j) = 0` for all `j`, giving `h_aleatoric ≡ 0` and `h_epistemic ≡ h_total` — a relabelling of
exp_23's existing `entropy_hard`, not new information. On the soft arm the split is non-degenerate,
but `p_j` is initialised from the `confidence` column itself (exp_23 §2.2), making `h_aleatoric`
target-informed by construction, not evidence for H1/H2.

### 2.2 Resolution — a third, ε-smoothed arm

Add **Arm C**: identical structural role to exp_23's hard arm (`x_train=False`, hard binary-outcome
loss, no `confidence` annotation anywhere in training), but with `c_y` initialised at

```
p_j = [ (1 − y_j)(1 − ε) + ε/2,  y_j(1 − ε) + ε/2 ],   c_y_j = √p_j
```

instead of a raw one-hot. This starts `c_y` a small, controlled distance off the fixed point (at
`ε = 0` it collapses to exp_23's hard arm exactly — used as an in-script identity check, not a grid
point). With `y_train=True`, minimising `F.nll_loss` on the true binary outcome moves each particle
toward the responsibility-weighted average outcome of the training points it is queried on
(`probs_i = Σ_j w_ij p_j`), so `h_aleatoric` on Arm C measures genuine **learned, label-free local
outcome heterogeneity** — the KDM equivalent of "how much do this patient's nearest analogues in
label-space actually disagree about biopsy outcome," entirely independent of the `confidence`
annotation being predicted.

## 3. Experimental Setup

### 3.1 Dataset — identical to exp_23, no changes
`Data/preprocessed_old/task1/` (with the same `resolve_data_dir()` fallback to
`data/chimera26/preprocessed/task1/`), N=88 labeled complete-case cohort, 54 yes / 34 no, same 12-dim
feature space (7 numeric + one-hot `dre`), same `experiments/exp_4/results/mccv_design.csv` MCCV
harness, same `confidence` 3-class target (clear 56 · borderline 18 · uncertain 14 on this cohort).

### 3.2 Three arms

| Arm | `c_y` init | `x_train` | `y_train` | Loss | Config source |
|---|---|---|---|---|---|
| **A — hard** | one-hot `[1−y, y]` | False | False | `F.nll_loss` | frozen, copied from `experiments/exp_23/results/best_hparams.json` (`sigma_mult=2.0, encoder=linear`) |
| **B — soft** | `[√(1−ỹ), √ỹ]` | False | False | soft CE | frozen, copied from `experiments/exp_23/results/best_hparams.json` (`sigma_mult=2.0, encoder=identity`) |
| **C — smoothed-hard** | ε-smoothed amplitude (§2.2) | False | **True** | `F.nll_loss` | **new — selected by Phase A, §3.3** |

Arms A and B need no Phase A of their own: their configurations already went through exp_23's 100-split
MCCV grid search and are re-read from its `results/best_hparams.json`, not re-swept. `x_train=False` is
fixed for all three arms — both exp_23 arms selected it, and pinning prototype positions to the actual
training points is what makes "one particle = one training patient" a meaningful reading for the
particle-level signals below.

### 3.3 Phase A — Arm C only (100 MCCV splits)

| Hyperparameter | Values |
|---|---|
| `sigma_mult` | 0.5, 1.0, 2.0 |
| `ε` (label-smoothing) | 0.05, 0.10, 0.20 |
| encoder | `nn.Identity()`, `nn.Linear(12, 8)` |

**Total: 3 × 3 × 2 = 18 configurations.** `sigma_mult=0.25` is dropped from exp_23's original grid: it
scored 0.487–0.518 mean Macro-F1 for both exp_23 arms, clearly dominated by `sigma_mult ∈ {1.0, 2.0}`
(≥0.58 for both arms) — not worth the extra 6 configs × 100 splits. Fixed: `w_train=True`,
`sigma_trainable=True`, Adam `lr=1e-3`, 300 full-batch epochs — identical training regime to exp_23.
Select argmax mean validation Macro-F1 across the 18 configs; freeze.

### 3.4 Phase B — LOOCV (88 folds), R=10 seeds, all three arms

Identical harness to exp_23 §5: `LeaveOneOut()` over the 88-row cohort, frozen config per arm, no
re-fitting. R=10 seeds for any arm whose selected encoder is `linear` (stochastic `nn.Linear` init);
R=1 if `identity` is selected (deterministic full-batch Adam, no random parameter anywhere in the
model — exp_23 DESIGN.md §3's determinism argument applies unchanged). Per fold/seed, in addition to
the class probabilities already computed by `dm2discrete`, extract the full particle set before
collapse:

```python
rho_y = model.kdm(pure2dm(model.encoder(x)))      # (bs, n_comp, dim_y+1)
w, v  = dm2comp(rho_y); w = w / w.sum(-1, keepdim=True)
p     = F.normalize(v, p=2, dim=-1, eps=1e-12) ** 2
```

giving seven per-patient signals (averaged over seeds exactly as exp_23 averages `entropy_mean` /
`log_marginal_mean`):

| # | Signal | Definition | Status on Arm A |
|---|---|---|---|
| 1 | `h_total` | `H(Σ_j w_j p_j)` | reproduces exp_23 `entropy_hard` |
| 2 | `h_aleatoric` | `Σ_j w_j H(p_j)` | **identically 0** (§2.1) — informative on Arm C only |
| 3 | `h_epistemic` | `h_total − h_aleatoric` | **≡ `h_total`** on Arm A |
| 4 | `h_weights` | `−Σ_j w_j log w_j` (entropy of the mixture weights) | non-degenerate |
| 5 | `log_ess` | `−log Σ_j w_j²` (log effective-#-particles) | non-degenerate |
| 6 | `w_max` | `max_j w_j` | non-degenerate |
| 7 | `log_marginal` | existing `model.kdm.log_marginal(...)`, unchanged from exp_23 | non-degenerate |

Signals 4–6 depend only on the weight profile `w(x)` — the kernel's local support geometry — not on
`c_y`, so they are unaffected by the §2.1 degeneracy on any arm and are new relative to exp_23
regardless of which arm they're read from. They are conceptually distinct from `log_marginal`: weight
*shape* (how concentrated the local support is) vs. total *mass* (how close `x` is overall).

## 4. Secondary Objective — Diagnostic Confidence from Particle Signals

Two confidence heads, both frozen at Phase A / applied without refitting at Phase B (`CLAUDE.md`
two-phase protocol):

**4.1 — 1D per-signal meta-thresholds.** Verbatim reuse of exp_23's `fit_1d_confidence_signal` /
`apply_1d_confidence_signal` / `score_confidence` (`experiments/exp_23/scripts/train.py:506-587`,
itself a documented, faithful superset of `exp_17`'s Phase A/B — see exp_23 §5). One row per
`(arm, signal)`. Entropy-family signals are sign-flipped exactly as exp_23 does (`-entropy_mean`) so
the `h_total` row on Arm A is byte-comparable to exp_23's published `entropy_hard` row.

**4.2 — Multivariate frozen tree ensemble (new).** Per MCCV split `i`, fit
`DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)` on the full 7-signal
vector restricted to `split_i == 0` (training rows of that split); freeze all 100 trees; do not refit
in Phase B.

⚠️ **Held-out voting, not exp_17's whole-cohort application.** exp_17 (and exp_23's 1D head, §4.1)
apply the mean of the 100 frozen thresholds to *all* 88 patients — each patient's own `confidence`
label influenced ~80 of the 100 fits that produced the average, but that influence is diluted into two
scalars and is treated as benign by the existing protocol. A 100-tree depth-3 ensemble has far more
capacity to fit individual patients, so applying the same whole-cohort rule here would convert a
tolerated asymmetry into real leakage. Fix: **patient `i`'s Phase-B prediction is the majority vote
only over the trees whose split had `split_i == 1`** (i.e., trees that never saw patient `i` in
training) — roughly 20 of 100 trees per patient, mirroring the out-of-fold discipline used everywhere
else in this codebase. This asymmetry with exp_17/§4.1 is stated explicitly in `reports/summary.md`.

Also fit the identical held-out-voted ensemble restricted to `{h_total, log_marginal}` — exp_23's
original two signals — as an ablation, so any Macro-F1 gain from the full 7-signal ensemble is
attributable to the new particle-structure signals specifically, not to ensembling itself.

Both heads report: 3-class Macro-F1, accuracy, Spearman ρ vs. `confidence` (`uncertain=0,
borderline=1, clear=2`), against the `exp_17` baseline (0.4470/57.95%/0.2790) and exp_23's best
non-target-informed signal (`log_marginal_hard`, 0.3340).

## 5. File Layout for This Experiment

```
experiments/exp_24/
├── DESIGN.md                  ← this file
├── IMPLEMENTATION.md          ← build plan (added after this file is approved)
├── scripts/
│   └── train.py                ← Phase A (Arm C only) + Phase B (all 3 arms) + confidence heads
├── results/
│   ├── best_hparams.json             ← Arm C selection (sigma_mult, epsilon, encoder) + Arm A/B provenance
│   ├── grid_search_results.csv       ← Arm C, 18 configs × 100 splits
│   ├── loocv_metrics.json            ← binary LOOCV metrics, 3 arms, reproduction deltas vs exp_23, McNemar
│   ├── oof_particle_signals.csv      ← 88 rows × (arm × 7 signals) + confidence_annotation
│   ├── confidence_metrics.json       ← §4.1 (1D) and §4.2 (multivariate + ablation) results
│   ├── degeneracy_check.json         ← H3 evidence (§2.1 reproduced in-script)
│   └── git_commit.txt
└── reports/
    ├── figures/
    │   ├── arm_c_grid_search_curves.png     ← Macro-F1 vs sigma_mult, faceted by epsilon/encoder
    │   ├── cy_drift.png                      ← ‖c_y_after − c_y_init‖ per particle, Arms A vs C
    │   ├── particle_signal_scatter.png       ← h_aleatoric vs h_epistemic, colored by confidence
    │   ├── signal_correlation_heatmap.png    ← 7-signal correlation matrix
    │   └── confidence_confusion_matrix.png   ← 3×3 CM, best confidence head
    └── summary.md
```

## 6. Evaluation Protocol & Decision Rules

Two-phase harness, identical structure to exp_23/`CLAUDE.md` §"Two-phase leak-free evaluation
protocol": scaler/encoder refit inside every split/fold; Phase B never re-fits hyperparameters or
thresholds. Arms A/B skip Phase A (configs already frozen by exp_23); Arm C runs its own Phase A per
§3.3.

- **Decision rule (H1, primary):** best non-target-informed row across §4.1 and §4.2 vs. `exp_17`
  (0.4470) and exp_23's best non-target-informed row (0.3340).
- **Decision rule (H2):** (a) Spearman ρ of Arm C's `h_aleatoric` vs. `confidence`, significance at
  p<0.05; (b) full 7-signal multivariate head (§4.2) vs. the `{h_total, log_marginal}` ablation head,
  same protocol, same trees-per-patient rule.
- **Decision rule (H3, structural):** in-script assertions, §7 items 1–3, must pass before any
  H1/H2 result is reported as informative — a failed assertion here invalidates the aleatoric/epistemic
  columns for that arm, not the whole experiment.
- **Secondary metrics (binary LOOCV, all 3 arms):** Macro-F1, Accuracy, Sensitivity, Specificity,
  AUROC, Brier, vs. the recomputed Fuzzy KNN reference (0.6364, read from
  `experiments/exp_23/results/oof_predictions.csv` rather than re-run) and exp_23's arms (hard 0.5636,
  soft 0.6694), with McNemar's exact test for Arm C vs. the KNN reference.

## 7. Known Pitfalls to Avoid (identified during design, not to be repeated)

- **Do not report `h_aleatoric`/`h_epistemic` on Arm A as informative** — they are algebraically
  identical to `0` and `h_total` respectively (§2.1). They are computed and asserted-near-zero as the
  H3 structural check, not treated as a result.
- **Arm B's particle signals remain target-informed**, exactly as exp_23 labels `entropy_soft` /
  `log_marginal_soft` — carried through unchanged in every results table with that flag.
- **Do not apply exp_17's whole-cohort threshold-application rule to the multivariate head** (§4)
  — capacity mismatch makes it leakage there even though it's tolerated for a 2-scalar 1D threshold.
- **Do not re-run the Fuzzy KNN reference or exp_23's Arm A/B Phase B** — read `reference_knn_p`,
  `reference_knn_pred`, and `best_hparams.json` from `experiments/exp_23/results/`. Re-deriving numbers
  that are already frozen and committed wastes the ~1.5–2h exp_23 already spent on them and risks a
  spurious mismatch from environment drift.
- **`c_y` in `own_libs/kdm/kdm/layers/kdm_layer.py`'s `KDMLayer.__init__`** is initialised as
  `torch.full((n_comp, dim_y), sqrt(1/dim_y))` before `init_kdm_layer` overwrites it — irrelevant here
  since `init_kdm_layer` always runs before the first optimizer step (exp_23 pitfall carried forward),
  but worth remembering if Arm C's `check_roundtrip` probe is reused from exp_23's `init_and_check`.

## 8. Scope

**In scope:** old schema (`Data/preprocessed_old/task1/`) only, exp_23-comparable. Binary biopsy
decision (secondary metric only, not the primary objective of this experiment) + 3-class diagnostic
confidence via particle-set uncertainty (primary objective, H1/H2).

**Explicitly out of scope:** new schema (`Data/preprocessed/task1/`); MRI/text modalities and late
fusion; `KDMRegressModel`; feature-attribution (exp_20–22 territory); re-running exp_13's Fuzzy KNN or
exp_23's Arm A/B Phase B (read from their committed results instead, §7); any change to Arms A/B's
frozen hyperparameters (only Arm C is swept).

## 9. Reproducibility Checklist

- [ ] Random seeds: Phase A single-seed per split (deterministic given split assignment), matching
      exp_23; Phase B explicit `seed ∈ {0..9}` per fold for stochastic (linear-encoder) arms, recorded
      per-seed in `loocv_metrics.json`.
- [ ] H3 structural assertions pass (§2.1/§6) before H1/H2 rows are written.
- [ ] Config and scripts saved in `scripts/`.
- [ ] Grid search results logged to `results/grid_search_results.csv` (Arm C only).
- [ ] **Git commit hash recorded** — `git log -1 --format="%H %s" > results/git_commit.txt` before
      execution.

## 10. Next Steps

1. Review and accept this experiment plan.
2. Once accepted, produce `IMPLEMENTATION.md` (concrete build plan + exact execution command) for
   approval.
3. Once accepted, implement `scripts/train.py`, run the 5-split Phase-A pilot to validate runtime
   before committing to the full 100-split run (recorded in `IMPLEMENTATION.md`, not a design change),
   then execute `exp_24`.
