# Experiment Design: MemKDM with training restricted to sigma-only, frozen unimodally (exp_27)
**Experiment**: experiments/exp_27/ · **Project**: pathology-reasoning · **Date**: 2026-08-19 · **Status**: Complete

---

## 1. Motivation — Phase B currently trains, not just evaluates

Every `MemKDM` in `exp_23`–`exp_26` fits `x` (prototype positions), `y` (prototype label content), `w`
(prototype weights), and — whenever `kernel_trainable=True` (the default for `KernelSpec`, and hardcoded
`True` for every unimodal Phase B condition in `exp_25`/`exp_26`) — `sigma` (kernel bandwidth), via 300
epochs of Adam on that fold's own training labels. This happens identically in Phase A (MCCV hyperparameter
search) **and** Phase B (LOOCV final evaluation): every one of the 88 LOOCV folds in `exp_25`/`exp_26`
re-optimizes some subset of `{x, y, w, sigma}` against that fold's 87 training labels before predicting the
held-out patient.

This conflates two things `CLAUDE.md`'s protocol keeps separate for every other method in this project
("Phase B... [never] re-fit hyperparameters or thresholds — that would leak"): Phase A is supposed to be
where fitting/selection happens, Phase B is supposed to be where the *frozen* result of that fitting is
evaluated. For a memory-based model, "fitting" and "having the data as prototypes" are inseparable — a
LOOCV fold's `x`/`y` necessarily differ from every other fold's, because the training set differs — but
`w` and `sigma` are not tied to which patients are in the fold the same way, and letting them additionally
be *gradient-optimized per fold using that fold's own labels* means Phase B is not a pure evaluation of a
frozen model, it is 88 independent small fits, each seeing its own held-out patient's 87 neighbors' labels
during optimization. This is not leakage of the held-out patient's own label (LOOCV correctly excludes that
patient throughout), but it does mean Phase B's numbers reflect 88 different label-supervised optimization
runs rather than one frozen configuration applied 88 times — not comparable to `exp_13`–`17`'s Fuzzy-KNN
lineage, where Phase B applies a single frozen bandwidth/threshold with no per-fold optimization at all.

**H (single, methodological).** Restricting training to the one place it is well-defined for a memory-based
model — `sigma` (a scalar per modality, not tied to which specific patients are in memory), fit once
unimodally during Stage 1's Phase A and frozen thereafter — and never training `x`, `y`, or `w` anywhere
(Phase A or Phase B), makes Phase B a genuine frozen-parameter evaluation, at the cost of an accepted bias
(§2.3): the frozen `sigma` is an average of values each fit on MCCV training folds that overlap the LOOCV
cohort, the same bias `exp_17`'s frozen meta-thresholds carry. Under this protocol, does `exp_25`/`exp_26`'s
H1 verdict (`joint_trimodal` does not beat the best unimodal model, leak-free late fusion, or `exp_23`'s
0.6694) still hold?

## 2. Background — what changes vs. exp_26, and why

Built from `exp_26/scripts/train.py`; same cohort, feature builders, MCCV/LOOCV harness
(`src/evaluation/*`), `MemKDM`/`LateFusionMemKDM` (`src/methods/mem_kdm.py`), one small `src/` fix (§2.4).
Differs in four ways plus that fix, all traceable to "only sigma is ever trained, and only once,
unimodally":

### 2.1 `w_train=False` everywhere (was hardcoded `True`, never searched)

`kdm.init.init_kdm_layer` already sets `layer.c_w.fill_(1.0 / layer.n_comp)` at construction — with
`w_train=False` this is not a new behavior to implement, it is simply never overwritten by gradient descent.
Every prototype keeps uniform weight `1/n` for the rest of that fit's life. This removes the one component
that had **no** grid-search justification in `exp_23`–`26` (it was fixed `True` by default, not evaluated
as a hyperparameter) and, per the user direction motivating this experiment, cannot be meaningfully
"transferred" from a Phase-A fit into Phase B's differently-sized folds the way a scalar bandwidth can.

### 2.2 `x_train=False, y_train=False` everywhere (previously Stage-1/Stage-2 grid dimensions)

`exp_23`'s own Stage-1/Stage-2 grids already searched `y_train ∈ {False, True}` (and `x_train` in Stage 2)
and consistently selected `False` for the winning configs across `exp_23`, `exp_25`, and `exp_26` — training
`x`/`y` never won. Per user direction, this experiment does not re-litigate that with a fresh grid search;
it fixes both `False` and removes them as grid dimensions entirely. `c_x`/`c_y` come directly from each
fold's own encoded training features / (label-smoothed, for `confidence_arm`) soft-target amplitudes —
data-driven init only, exactly `kdm.init.init_kdm_layer`'s un-overridden output.

### 2.3 Sigma is trained exactly once per modality — unimodally, in Stage 1 — then frozen everywhere else

This is the crux, and the one place this experiment accepts a documented bias:

- **Stage 1** (unimodal, 100-split MCCV): grid over `sigma_mult` (and `rep` for `mri`/`txt`) only. Every fit
  has `kernel_trainable=True` — `sigma` starts at `sigma_mult × _sigma_from_knn(fold's own ~70-patient
  training block)` and is refined by the same 300-epoch Adam loop as before (the *only* parameter with a
  nonzero gradient now, since `x_train=y_train=w_train=False`). `evaluate_fn` records both `macro_f1` (for
  selection, unchanged rule) and the fold's post-training fitted `sigma` (`MemKDM.kernel_params()`) —
  `protocol.run_mccv_grid` already aggregates every key `evaluate_fn` returns into `mean_<key>`/`std_<key>`
  columns (`src/evaluation/protocol.py:46-49`), so this needs no `src/` change, just one extra field in the
  returned dict. For the winning `sigma_mult` config, `mean_fitted_sigma` (already a column in
  `stage1_grid_search.csv`) **is** "the sigma learned in Phase A" for that modality — call it
  `STAGE1_MEAN_SIGMA[modality]`.
- **Stage 2** (joint conditions): no grid search — `sigma_scale` (§2.5) is gone and nothing else remains to
  search. Each joint condition's product kernel is built directly from `STAGE1_MEAN_SIGMA` per modality
  (`KernelSpec(sigma=STAGE1_MEAN_SIGMA[m], trainable=False)`), `rep` per modality from Stage 1's winners,
  `x_train=y_train=w_train=False`. Stage 2's 100-split MCCV pass still runs, purely to report each
  condition's mean/std Macro-F1 (informational — there is nothing to select among).
- **Phase B** (LOOCV, 88 folds, all 8 conditions): identical construction to Stage 2's evaluation model —
  `KernelSpec(sigma=STAGE1_MEAN_SIGMA[m], trainable=False)`, `x_train=y_train=w_train=False`. The *same*
  frozen `STAGE1_MEAN_SIGMA[m]` number is reused for all 88 folds (not recomputed per fold via
  `_sigma_from_knn` the way `exp_26` did) — this is the accepted bias: `STAGE1_MEAN_SIGMA` was averaged over
  100 MCCV training folds that collectively cover the full cohort, so every LOOCV fold's frozen bandwidth
  was informed, in aggregate, by data that includes that fold's own held-out patient. Explicitly not
  claimed leak-free; explicitly the same shape of bias `exp_17`'s frozen meta-thresholds carry, which is
  the basis for comparability invoked in §1.
- **What was explicitly ruled out**: fitting one model on the full 88-patient cohort to extract a single
  "frozen" sigma. For a memory-based model this doesn't reduce bias relative to the MCCV-average approach —
  it's worse, since the model's memory *would literally contain* the very patient later queried in every
  Phase-B fold (the degenerate case `_check_amplitude_roundtrip` exists to catch), not just an indirect
  training-fold overlap.

### 2.4 Required `src/methods/mem_kdm.py` fix — `MemKDM.fit()` on zero trainable parameters

With sigma frozen (`trainable=False`) and `x_train=y_train=w_train=False`, Stage 2 and Phase B models have
nothing left to optimize. `kdm.layers.kdm_layer.KDMLayer.__init__` and
`kdm.layers.rbf_kernel_layer.RBFKernelLayer.__init__` always construct `c_x`/`c_y`/`c_w`/`raw_sigma` as
`nn.Parameter` — only `requires_grad` varies with `x_train`/`y_train`/`w_train`/`trainable` — so
`model.parameters()` is never actually empty; what's empty is the *requires_grad=True* subset. Verified
that in this all-`False` case, `loss.backward()` raises `RuntimeError: element 0 of tensors does not
require grad and does not have a grad_fn` (the whole forward graph has no gradient-requiring leaf), not an
`Adam([])` construction error as initially assumed. No existing experiment (`exp_23`–`26`) ever exercised
this path — something was always trainable. Fix: guard the optimizer construction and training loop on
`any(p.requires_grad for p in model.parameters())`; skip both when nothing needs a gradient (the model is
then fully determined by its data-driven init, which is exactly Phase B's intent here). ~4 lines, no
behavior change for any existing config. This is the only `src/` change in this experiment.

### 2.5 `sigma_scale` (exp_25/26's Stage-2 grid dimension) removed entirely

`sigma_scale ∈ {0.5, 1.0, 2.0}` is the exact same set `MRI_GRID`/`TXT_GRID` already grid-search directly as
`sigma_mult` in Stage 1 — it re-explores the same relative range as a second multiplier stacked on Stage
1's per-modality winner, only for `mri`/`txt`. With `x_train`/`y_train`/`kernel_trainable` no longer Stage-2
grid dimensions either (§2.2, §2.3), `sigma_scale` would be the *only* thing left to search in Stage 2 —
searching a redundant re-scaling of an already-searched value. Dropped; §2.3 covers what replaces it.

## 3. Grids (down from exp_26's 8/12/18/24 — TAB/MRI/TXT/STAGE2)

| Modality | Grid | # configs |
|---|---|---|
| `tab` | `sigma_mult ∈ {0.25, 0.5, 1.0, 2.0}` | 4 |
| `mri` | `rep ∈ {raw_l2, pca90_l2}` × `sigma_mult ∈ {0.5, 1.0, 2.0}` | 6 |
| `txt` | `rep ∈ {(500,.9), (2000,.9), (None,.9)}` × `sigma_mult ∈ {0.5, 1.0, 2.0}` | 9 |
| Stage 2 (any joint condition) | none — deterministic given Stage 1 winners | 1 |

Fixed everywhere: `encoder="identity"` (unchanged from `exp_26`), `x_train=False`, `y_train=False`,
`w_train=False`, `label_smoothing=0.0` except `confidence_arm` (`0.10`, unchanged from `exp_26`),
`lr=1e-3`, `epochs=300`.

## 4. Step 0 — reproduction gate (unchanged mechanism, now a divergent-by-design reference)

Kept exactly as in `exp_25`/`exp_26`: a standalone fit with `KernelSpec(sigma_mult=2.0)` (defaults:
`trainable=True`, `w_train=True`) through the harness, asserting LOOCV Macro-F1 == `0.6694214876033058`.
This is a code-path sanity check (the harness/data loading still reproduces `exp_23`'s historical number
under `exp_23`'s original model), **not** a claim that `exp_27`'s own `unimodal_tab` condition will match
it — under §2.1–2.3's fixed flags, `unimodal_tab` is a materially different model (no `w` training, no
per-fold sigma refinement) and is expected to diverge from 0.6694. If it happens to match, that is a
finding (disabling `w`/per-fold-sigma-training was inert for `tab`), not a failure of this gate.

## 5. File Layout

```
experiments/exp_27/
├── DESIGN.md
├── IMPLEMENTATION.md          ← added after this file is approved
├── scripts/train.py            ← built from exp_26/scripts/train.py; --smoke flag
├── results/
│   ├── reproduction_check.json
│   ├── stage1_grid_search.csv        ← 19 configs × 100 splits, incl. mean/std fitted_sigma
│   ├── stage1_best_hparams.json      ← per-modality winner incl. STAGE1_MEAN_SIGMA
│   ├── stage2_grid_search.csv        ← 5 conditions × 1 config × 100 splits (informational)
│   ├── best_hparams.json
│   ├── fusion_weights.json
│   ├── loocv_metrics.json
│   ├── confidence_metrics.json
│   ├── oof_predictions.csv
│   ├── oof_particle_signals.csv
│   └── git_commit.txt
└── reports/
    ├── figures/   ← stage1 grid curves (Macro-F1 and fitted_sigma vs sigma_mult), confusion matrix,
    │                 ROC overlay, particle signal scatter
    └── summary.md
```

## 6. Evaluation Protocol & Decision Rules

Same H1-style three-way comparison as `exp_25`/`exp_26` (§1): `joint_trimodal` LOOCV Macro-F1 vs. best
unimodal, `late_fusion_optimal`, and `exp_23`'s 0.6694, with `mcnemar_exact` (`joint_trimodal` vs.
`late_fusion_optimal`) as the significance criterion. Confidence task (secondary, H2-style) reported as
context against the same comparators used in `exp_25`/`exp_26`, not re-litigated as a fresh hypothesis.

## 7. Scope

**In scope:** re-running Stage 1 (smaller grid, `w_train=False` fixed) and Stage 2 (no grid, sigma
transferred from Stage 1) under the protocol in §2–3; Phase B LOOCV with zero gradient training of any
kind; re-evaluating H1 under this protocol.

**Out of scope:** any new hypothesis beyond re-testing H1 under the corrected protocol; re-opening
`x_train`/`y_train`/`kernel_trainable` as grid dimensions (§2.2, §2.3 — fixed by user direction, not
re-derived here); the new-schema data (`Data/preprocessed/task1/`); `embedkit_*` representations
(unavailable in this checkout, same caveat as `exp_25`/`exp_26`).

## 8. Next Steps

1. Review and accept this design.
2. `IMPLEMENTATION.md` (concrete build plan, exact execution command) for approval.
3. Implement `scripts/train.py`, run `--smoke`, then the full run.
