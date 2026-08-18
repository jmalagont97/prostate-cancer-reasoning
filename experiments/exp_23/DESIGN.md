# Experiment Design: Tabular KDM (Kernel Density Matrix) Biopsy Decision Prediction
**Experiment**: experiments/exp_23/ · **Project**: pathology-reasoning · **Date**: 2026-08-18 · **Status**: Draft

---

## 1. Hypothesis

The canonical tabular model, **exp_13 Tabular Fuzzy KNN**, is a memory-based model with a *frozen* memory: the
training points are the prototypes, the neighborhood is a hard top-k cut, and neither the metric nor the
prototype content is learned. It reached LOOCV Macro-F1 **0.6364** (k=1, uniform, euclidean), beating the
hard-KNN baseline (exp_5, 0.6333) by +0.0031 — one flipped case.

A **Kernel Density Matrix (KDM)** is the natural generalization of that exact model class: it keeps a memory of
support points (`c_x`), but makes the prototypes, their targets (`c_y`), their weights (`c_w`), and the RBF
kernel bandwidth (`σ`) all differentiable and trained end-to-end by gradient descent, replacing the hard
k-neighborhood with a soft, learned-bandwidth kernel over the whole memory. It also emits predictive entropy and
`log_marginal` (input log-likelihood log P(x)) as a byproduct — a learned, principled alternative to the
project's hand-crafted Composite Reliability Index (ICI, exp_9–exp_19).

**H1 (primary).** Replacing exp_13's frozen KNN memory with a KDM whose prototypes, weights, and bandwidth are
learned end-to-end — on the identical 88-patient cohort, identical 12-dim feature space, identical MCCV/LOOCV
splits — will produce LOOCV binary predictions that differ from Fuzzy KNN's by more than chance (McNemar test),
with the better KDM arm's mean LOOCV Macro-F1 exceeding the *recomputed* Fuzzy KNN reference.

**H2 (secondary).** KDM's native uncertainty signals (predictive entropy, `log_marginal`), routed through the
same 1D/2D balanced-decision-tree meta-threshold machinery as exp_17, will predict the urologist's 3-class
diagnostic confidence annotation with higher Macro-F1 and stronger Spearman rank correlation than exp_17's
Composite Fuzzy ICI (Macro-F1 0.4470, accuracy 57.95%, ρ=0.2790, p=0.0085).

## 2. Experimental Setup

### 2.1 Dataset (old schema, exp_13-comparable — see §7 for why not the new schema)
- Tabular data: `Data/preprocessed_old/task1/clinical_data_tabular.csv` (script resolves this path with a
  fallback to the legacy `data/chimera26/preprocessed/task1/`, which does not exist in this checkout — see §6).
- Biopsy decision target: `Data/preprocessed_old/task1/biopsy_decision.csv` (`biopsy_decision`, N=195).
- Clinical reasoning annotations: `Data/preprocessed_old/task1/clinical_reasoning.csv` (`confidence`).
- MCCV split design: `experiments/exp_4/results/mccv_design.csv` (`split_0..split_99`, 0=train/1=val/absent=excluded).
- Cohort: **N=88 labeled complete-case cohort**, **54 `yes` / 34 `no`**, matching exp_5/exp_13. (exp_13's own
  `reports/summary.md` states 56/32 — that is a bug in exp_13, confirmed against the raw data; exp_23 will not
  reproduce it.) `confidence` on this cohort: clear 56 · borderline 18 · uncertain 14.
- Features: 7 numeric (`age, psa, vol, pirads, psad, psav, psap`) + 1 categorical (`dre`, 5 levels observed in
  cohort: Normal, Nodus, Abnormal, Not done, Suspicious) → **12 dims** after one-hot encoding.
- `'NONE'` is not in pandas' default `na_values`; the loader uses `dtype=str, keep_default_na=False` wherever
  missingness matters, matching what the exploration for this experiment verified directly against the CSVs.

### 2.2 Soft-target formulation (Arm B only, identical to exp_13)
- Certainty weights: `clear → c=1.00`, `borderline → c=0.50`, `uncertain → c=0.25`, unannotated → `c=1.00`.
- `ỹ = 0.50 + 0.50·c` if `y=1`, `ỹ = 0.50 − 0.50·c` if `y=0` → six distinct values
  `{0, 0.25, 0.375, 0.625, 0.75, 1.0}`.
- **Ceiling on interpretability:** with `clear`=56/88, most `ỹ` are exactly 0 or 1. Arm A (hard) and Arm B (soft)
  can only differ on the 32 non-`clear` patients — bounding how large any soft-vs-hard gap can legitimately be,
  the same "one flipped case" ceiling that makes exp_13-vs-exp_5 hard to over-interpret.

### 2.3 Preprocessing
- Numeric features: `MinMaxScaler`, fit on train only, inside every split/fold.
- Categorical (`dre`): `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, fit on train only.
- `np.hstack` → 12-dim input to the KDM encoder.

### 2.4 Model — `KDMClassModel` (not `MemKDMClassModelWrapper`)
```python
KDMClassModel(encoded_size, dim_y=2, encoder, n_comp, sigma, sigma_trainable=True,
               min_sigma=1e-3, x_train=..., y_train=..., w_train=True)
```
- **`dim_y=2`**: probabilities → Macro-F1/AUROC/Brier exactly as exp_13; entropy; `log_marginal`. No
  `KDMRegressModel` arm.
- **`n_comp = n_train`** (structural rule, not swept): the entire training set is the prototype memory — 70 in
  Phase A, 87 in Phase B. This is leak-free (depends on no validation outcome) and makes KDM the direct
  structural analogue of KNN: same memory, learned prototypes/weights/bandwidth instead of frozen ones.
- **Init**: `init_kdm_layer(model.kdm, enc_train, c_y_train, init_sigma=True, sigma_mult=...)` before the first
  optimizer step, always. `c_y` uses **amplitude encoding** — `dm2discrete` L2-normalizes then squares each
  component vector, so soft targets must be stored as `[√(1−ỹ), √ỹ]`, not `[1−ỹ, ỹ]`. Verified empirically:
  round-trip error at narrow σ is 0.00000 for the `√` encoding vs. 0.15012 for the raw-probability encoding.
- **Why not `MemKDMClassModelWrapper`**: (1) faiss hard-crashes alongside torch in this environment
  (`OMP: Error #15`, exit 139 — all four standard workarounds fail); (2) at N=88 approximate NN buys nothing
  over exact; (3) `MemKDMClassModel.forward` hardcodes `F.one_hot(y_neigh.long(), ...)`, which **silently**
  truncates any soft label <1 to class 0 — no error raised. `KDMClassModel` with `n_comp=n_train` already is the
  memory-based model in the sense that matters here.

### 2.5 Two target arms
| Arm | `c_y` init | Loss |
|---|---|---|
| **A — hard** | one-hot `[1−y, y]` | `F.nll_loss(log(probs.clamp_min(1e-7)), y)` |
| **B — soft** | `[√(1−ỹ), √ỹ]` | hand-written soft CE: `-(t·log(probs.clamp_min(1e-7))).sum(-1).mean()`, `t=[1−ỹ,ỹ]` |

`F.nll_loss` does not accept soft targets (`RuntimeError: 0D or 1D target tensor expected`) — Arm B's loss is
written explicitly rather than reusing the library's classification snippet verbatim. Two arms isolate whether
any gain over Fuzzy KNN comes from KDM's learned memory itself or from the soft-target formulation.

## 3. Hyperparameter Sweep Grid (Phase A, 100 MCCV splits)

| Hyperparameter | Values |
|---|---|
| `sigma_mult` | 0.25, 0.5, 1.0, 2.0 |
| `x_train` (prototype positions trainable) | True, False |
| `y_train` (label content trainable) | True, False |
| encoder | `nn.Identity()` (12→12), `nn.Linear(12, 8)` |

Fixed: `w_train=True`, `sigma_trainable=True`, Adam `lr=1e-3`, 300 full-batch epochs (n≈70, no `DataLoader`).
**Total: 4 × 2 × 2 × 2 = 32 configurations per arm, 64 total.**

`x_train`/`y_train` are swept as independent axes rather than one bundled "capacity" knob so that any result is
attributable to prototype-pinning vs. label-pinning specifically — the latter is the operative "learning under
label uncertainty" question for Arm B. The constructor's `sigma` argument is always overwritten by
`init_kdm_layer(..., init_sigma=True)`, so it is not swept; `sigma_mult` is the meaningful bandwidth axis (an
uncalibrated init produced a bandwidth broad enough to collapse every prediction to ≈0.5 in a pilot check).

⚠️ **Correction during implementation review:** determinism is per-*encoder*, not confined to one corner. With
`encoder="identity"`, `c_x`, `c_y`, `c_w`, and σ are all set from data by `init_kdm_layer`, and training is
deterministic full-batch Adam — there is no randomly initialized parameter anywhere in the model regardless of
`x_train`/`y_train`. Only `encoder="linear"` introduces genuine randomness, via `nn.Linear`'s weight init. So
**every `identity`-encoder config (16 of the 32 per arm — half the grid: 4 `sigma_mult` × 2 `x_train` × 2
`y_train`) is deterministic**, not just `x_train=False, y_train=False`. If Phase A selects an `identity` config,
Phase B's seed-stability report states that explicitly
rather than reporting a spurious std (see §5) — and McNemar's test, already the primary decision rule for
exactly this reason, becomes the only inferential tool available in that case.

Runtime: 100 splits × 32 configs × 2 arms on a ≤87×87 kernel. Plan is to pilot on 5 splits and extrapolate before
committing to the full 100-split run (recorded in IMPLEMENTATION.md, not a design change).

## 4. File Layout for This Experiment
```
experiments/exp_23/
├── DESIGN.md                  ← this file
├── IMPLEMENTATION.md          ← build plan (added after this file is approved)
├── scripts/
│   └── train.py                ← MCCV grid search (2 arms) & LOOCV evaluation, incl. recomputed Fuzzy KNN reference
├── results/
│   ├── best_hparams.json             ← optimal KDM config per arm (sigma_mult, x_train, y_train, encoder)
│   ├── grid_search_results.csv       ← mean Macro-F1 across 32 configs × 2 arms
│   ├── loocv_metrics.json            ← LOOCV metrics per arm, per seed + mean/std; recomputed Fuzzy KNN reference
│   ├── oof_predictions.csv           ← out-of-fold probabilities & predictions (both arms + reference)
│   ├── confidence_metrics.json       ← secondary objective: 3-class confidence prediction from entropy/log_marginal
│   └── git_commit.txt                ← `git log -1 --format="%H %s"`
└── reports/
    ├── figures/
    │   ├── grid_search_curves.png       ← Macro-F1 vs sigma_mult, faceted by x_train/y_train/encoder
    │   ├── confusion_matrix.png         ← LOOCV 2x2, best KDM arm
    │   ├── roc_curve.png                ← KDM vs. recomputed Fuzzy KNN overlay
    │   ├── confidence_confusion_matrix.png  ← 3x3 confidence prediction
    │   └── uncertainty_scatter.png      ← entropy vs. log_marginal, colored by confidence annotation
    └── summary.md                    ← final report contrasting exp_23 vs. exp_13/exp_17
```

## 5. Evaluation Protocol & Decision Rules

Two-phase harness, identical structure to exp_13 (`CLAUDE.md` §"Two-phase leak-free evaluation protocol"):
scaler/encoder refit **inside** every split/fold; Phase B never re-fits hyperparameters.

- **Phase A — MCCV (100 splits, `exp_4/results/mccv_design.csv`)**: per split, per arm, per config — train,
  threshold `p(yes) ≥ 0.50`, score Macro-F1. Select argmax mean Macro-F1 per arm → freeze.
- **Phase B — LOOCV (88 folds), R=10 seeds**: `LeaveOneOut()` over the 88-row cohort, frozen config per arm, no
  re-fitting. KDM is stochastic (Adam, `Linear`-encoder init); at N=88 a single-seed Macro-F1 can swing several
  points, so Phase B runs **R=10 seeds** and reports mean ± std (unless the selected config is the deterministic
  corner above, in which case std is reported as exactly 0/not applicable).
- **Recomputed Fuzzy KNN reference, inside this script.** exp_13's published LOOCV Macro-F1 (0.6364) was produced
  against `data/chimera26/preprocessed/task1/`, a path absent from this checkout — comparing it to a model fed
  from `Data/preprocessed_old/task1/` would rest on an unverified assumption of identical inputs. `train.py`
  therefore re-runs exp_13's exact pipeline (`KNeighborsRegressor`, 66 configs × 100 splits) on the same loaded
  frames, so KDM and the reference see byte-identical rows, splits, and folds. Both the recomputed number and
  exp_13's published 0.6364 are reported side by side.

**Decision rule (primary):** McNemar's test on the 88 paired LOOCV binary predictions, best KDM arm (mode vote
across R=10 seeds) vs. recomputed Fuzzy KNN — the correct test for paired binary decisions at this N, and it
remains informative even in the deterministic-config corner where a seed-std comparison would degenerate to a
bare inequality.

**Decision rule (secondary):** mean LOOCV Macro-F1 of the better KDM arm vs. the recomputed Fuzzy KNN reference,
with AUROC and Brier reported alongside (exp_13 Brier = 0.2908; a learned-bandwidth kernel is expected to
calibrate better even where Macro-F1 moves little).

**Baseline to beat:** `exp_13` Tabular Fuzzy KNN — MCCV mean Macro-F1 0.6117, LOOCV Macro-F1 **0.6364**, Accuracy
65.91%, Sensitivity 0.7407, Specificity 0.5294, AUROC 0.6304, Brier 0.2908 (recomputed value used for the actual
test; published value reported alongside).

**Secondary metrics:** Accuracy, Sensitivity, Specificity, AUROC, Brier Score — identical schema to
`exp_13/results/loocv_metrics.json`.

### 5.1 Secondary objective — diagnostic confidence from native uncertainty
From the **Arm A (hard-label)** OOF model only: `entropy = -(p·log p).sum(-1)` and
`log_p_x = model.kdm.log_marginal(pure2dm(model.encoder(x)))`. This objective is carried on Arm A, not Arm B,
because Arm B's soft targets are *derived from* the `confidence` column — the very thing being predicted; using
Arm B's uncertainty here would not support a claim that KDM "recovers" clinician confidence, even though it is
not per-sample leakage (held-out confidence is never used in Phase B). Arm B's version is still reported,
explicitly labeled **target-informed**.

Both signals (plus a joint 2D `[entropy, log_p_x]` variant) are routed through exp_17's existing machinery: Phase
A fits `DecisionTreeClassifier(class_weight='balanced')` (1D per signal, 2D joint) over the 100 MCCV splits to
learn mean meta-thresholds; Phase B applies the **frozen** thresholds to LOOCV OOF signals for 3-class
prediction. Metrics: 3-class Macro-F1, accuracy, Spearman ρ vs. `confidence` coded
`uncertain=0, borderline=1, clear=2`. **Baseline to beat:** exp_17 — Macro-F1 0.4470, accuracy 57.95%,
ρ=0.2790 (p=0.0085).

## 6. Known Pitfalls to Avoid (identified during design, not to be repeated)
- exp_13 never wrote `results/git_commit.txt` despite listing it in its own checklist — exp_23 will.
- exp_13's `reports/summary.md` reports the class totals as 56/32; the correct totals are 54/34.
- exp_13's summary-writer uses an f-string `\text{{std}}` without a raw prefix, emitting a literal tab character
  into the markdown — exp_23 uses a raw string.
- `experiments/exp_20/scripts/train.py`'s custom `DistanceWeightedFuzzyKNN` uses `1/d²` weighting (not sklearn's
  `1/d`), a reduced 5-feature set, and median imputation — it is not used here; the recomputed reference uses
  exp_13's own `sklearn.neighbors.KNeighborsRegressor` pipeline verbatim.
- The recomputed Fuzzy KNN's *selected config* is not gated on exactly matching exp_13's `k=1, uniform, euclidean`
  — `k=1 uniform` and `k=1 distance` are byte-identical in exp_13's own `grid_search_results.csv`, and `k=3
  uniform` is within 0.0006 mean Macro-F1, so argmax among near-ties is not a stable identity check across
  pandas versions. The real check is that the recomputed LOOCV Macro-F1 equals 0.6364 with confusion matrix
  `tp=40, tn=18, fp=16, fn=14`.

## 7. Scope

**In scope:** old schema (`Data/preprocessed_old/task1/`) only, for exp_13-comparability. Binary biopsy decision
(primary) + 3-class diagnostic confidence via native uncertainty (secondary).

**Explicitly out of scope:** new schema (`Data/preprocessed/task1/`) features; MRI/text modalities and late
fusion; `KDMRegressModel`; feature-attribution (exp_20–22 territory); re-running exp_13 itself (only its pipeline
is recomputed inline for a same-input reference).

## 8. Reproducibility Checklist
- [ ] Random seeds: Phase A single-seed per split (deterministic given split assignment); Phase B explicit
      `seed ∈ {0..9}` per fold, recorded per-seed in `loocv_metrics.json`.
- [ ] Config and scripts saved in `scripts/`.
- [ ] Grid search results logged to `results/grid_search_results.csv` (both arms).
- [ ] **Git commit hash recorded** — `git log -1 --format="%H %s" > results/git_commit.txt` before execution.

## 9. Next Steps
1. Review and accept this experiment plan.
2. Once accepted, produce `IMPLEMENTATION.md` (concrete build plan + exact execution command) for approval.
3. Once accepted, implement `scripts/train.py` and execute `exp_23`.
