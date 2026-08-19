# Experiment Design: Multimodal Memory-Based KDM (`MemKDM`)
**Experiment**: experiments/exp_25/ · **Project**: pathology-reasoning · **Date**: 2026-08-18 · **Status**: Complete

---

## 1. Hypothesis

The `src/` method/harness layer (uncommitted; extracted after exp_23/exp_24 shared ~476 identical lines)
generalizes exp_23/exp_24's tabular-only `KDMClassModel` into `MemKDM`: one class covering 1..N
modalities, each carrying its own RBF-kernel bandwidth, combined as a product kernel
`k(x, c) = Π_m k_m(x_m, c_m)`. A unimodal `MemKDM` is the one-entry case of the same code path, so it
reproduces exp_23's soft-arm number bit-for-bit — the reproduction gate in §3.3 Step 0 checks this
before any multimodal claim is made.

**H1 (primary, decision).** A joint product-kernel `MemKDM` over {tabular, MRI, text} — tuned in two
stages (§3): unimodal bandwidths/encoders first (Stage 1), then joint memory/bandwidth-ratio hyperparameters
second (Stage 2) — beats (a) the best unimodal `MemKDM` from Stage 1, (b) `LateFusionMemKDM` with
fusion weights selected leak-free, and (c) exp_23 Arm B (LOOCV Macro-F1 **0.6694**), on LOOCV Macro-F1.
*Supported* = beats all three **and** McNemar vs. `LateFusionMemKDM` p<0.05. *Partial* = beats on point
estimate without significance. *Refuted* = does not beat (a) or (b).

**H2 (secondary, confidence).** Particle-set signals (`mem_kdm.extract_particle_signals`) from a
*shared* multimodal particle set carry more diagnostic-confidence information than exp_24's
tabular-only particle set. Because supervision is soft everywhere in this experiment (§2.4), every
`MemKDM` here has `target_informed=True` (`Targets.soft_from_confidence=True`,
`src/methods/base.py:51-55`) — so the only apples-to-apples comparator is another target-informed
number: exp_23's `entropy_soft` = **0.4164**. exp_24's best non-target-informed head (0.4368) and
exp_17's Composite ICI (0.4470) are reported alongside as context, explicitly marked *not*
like-for-like — beating them is not evidence for H2 on its own.

**H3 (structural).** A shared particle set (`MemKDM`, one product kernel, one `n_comp=n_train` memory)
is not equivalent to per-modality particle sets combined post hoc (`LateFusionMemKDM`, three
independent memories): the joint model's `h_epistemic`/`log_ess`/`w_max` are computed over one weight
profile that already reflects cross-modality agreement, whereas `LateFusionMemKDM.composite_ici` is an
inter-modality spread statistic computed *after* each modality already collapsed its own particle set.
Reported as a qualitative contrast (do the two uncertainty stories agree on which patients are
hard?), not a significance test — there is no shared null hypothesis to test between two different
uncertainty constructions.

## 2. Background

### 2.1 Why this is wiring, not new model code

Everything H1–H3 need already exists in the uncommitted `src/` package:

- `src/methods/mem_kdm.py` — `MemKDM` (product-kernel joint model), `LateFusionMemKDM` (per-modality
  `MemKDM` + weighted soft voting, the exp_17/exp_16-comparable baseline), `extract_particle_signals`
  (generalizes exp_24's function over a modality dict), `composite_reliability_index`,
  `simplex_grid`/`soft_vote`/`search_fusion_weights` (fusion utilities).
- `src/methods/base.py` — the `Method` protocol, `Targets`, and the meta-threshold/held-out-tree
  confidence-head plumbing shared with exp_23/exp_24.
- `src/evaluation/{data,protocol,metrics,reporting}.py` — cohort loading for the **old** schema
  (`Data/preprocessed_old/task1/`), the MCCV/LOOCV harness with the explicit tie-break exp_13–24 never
  had, scoring, and artifact writers.

exp_25 is therefore `DESIGN.md` → approval → `IMPLEMENTATION.md` → approval → one `scripts/train.py`
that calls these APIs, per `CLAUDE.md`'s required workflow. `nn.Identity()` and cosine-via-L2-norm
carry over from `src/`'s existing conventions unchanged; nothing here proposes a new kernel, encoder,
or loss.

### 2.2 Two axes named "phase" — kept distinct throughout

| | Meaning | Machinery |
|---|---|---|
| **Stage 1** | tune each modality independently | Phase A only (100-split MCCV), per modality |
| **Stage 2** | tune the joint model, then evaluate it once | Phase A over the reduced joint space, then Phase B (LOOCV, 88 folds) |
| **Phase A / Phase B** | this repo's MCCV-search / LOOCV-final-eval split (`CLAUDE.md`) | orthogonal to Stage 1/2 — both stages contain their own Phase A; only Stage 2 reaches Phase B |

### 2.3 The sigma-transfer decision — the leak-free crux

`MemKDM.kernel_params()` (`mem_kdm.py:426-434`) returns a **fitted** `sigma` value, and
`MemKDM.from_unimodal()` (`:436-452`) propagates it directly; at `:339-342` an explicit `sigma` on a
`KernelSpec` bypasses that fold's own `_sigma_from_knn` computation. If Stage 1 fit on the full cohort
and Stage 2 froze those fitted numbers via `from_unimodal`, every Phase-B LOOCV fold would carry a
bandwidth that had already seen its own held-out patient — the same failure mode
`search_fusion_weights`' docstring names as exp_16's biggest defect, just one level down in the model.

**exp_25 transfers `sigma_mult` (the hyperparameter), not `sigma` (the fitted number).** Stage 2 builds
its `kernels` dict directly rather than through `from_unimodal`:

```python
kernels = {m: KernelSpec(sigma_mult=stage1_best[m]["sigma_mult"] * sigma_scale.get(m, 1.0),
                          trainable=cfg["kernel_trainable"], sigma=None)   # sigma=None is the point
           for m in subset}
encoders = {m: EncoderSpec(**stage1_best[m]["encoder"]) for m in subset}
model = MemKDM(kernels=kernels, encoders=encoders, x_train=cfg["x_train"], y_train=cfg["y_train"], seed=seed)
```

so `sigma` is recomputed from each fold's own training data via `_sigma_from_knn` every time — §3.3's
leak-free property holds regardless of `kernel_trainable`. `MemKDM.from_unimodal` and
`mem_kdm.search_fusion_weights` are consequently **not called** by exp_25 (see §7 for why the latter
doesn't fit this design's per-split stash either); both remain as-is in `src/` for future use.

### 2.4 Soft supervision, and its consequence for the confidence task

Per user direction: soft supervision (`data.build_targets(..., certainty_map=CONFIDENCE_CERTAINTY_MAP)`)
is the default for every arm in this experiment, decision task and confidence task alike. This sets
`Targets.soft_from_confidence=True` and therefore `MemKDM.target_informed=True` on every model fit here
— `base.py`'s `Method.target_informed` docstring states a model fit on confidence-derived targets
"cannot validly be used to predict confidence" in the *non-target-informed* sense exp_17/exp_23/exp_24
use as their headline confidence number. exp_25 does not carry a separate hard-supervised arm to avoid
this; instead every confidence result is written with an explicit `"target_informed": true` field and
compared only against other target-informed numbers as the primary claim (§1 H2), with
non-target-informed historical numbers shown as context only.

### 2.5 The `h_aleatoric` degeneracy — inherited from exp_24, not assumed away

`extract_particle_signals`' docstring: `h_aleatoric` is identically 0 whenever every particle's own
`p_j` is one-hot, i.e. whenever `y_soft` is binary and `label_smoothing == 0` — exp_24's central finding
(H3 there). With soft targets, 56/88 patients are `clear` → `y_soft ∈ {0,1}` exactly, so only the 32
non-`clear` patients' prototypes sit off the one-hot fixed point. This is a **per-prototype** property,
not a per-cohort one: a query patient whose nearest neighbors in the joint kernel are dominated by
`clear` prototypes will show `h_aleatoric ≈ 0` regardless of `label_smoothing`, exactly as on exp_24's
Arm A. §3.3 Step 0 measures how much this actually bites (fraction of patients with non-trivial
`h_aleatoric`, correlation with soft-prototype weight mass) *before* deciding whether Stage 2 needs a
label-smoothed confidence arm (§3.5).

## 3. Experimental Setup

### 3.1 Dataset — identical cohort to exp_23/24, extended to three modalities

`Data/preprocessed_old/task1/` via `data.resolve_data_dir()`; `data.load_cohort(..., load_mri=True,
load_text=True)`. N=88 labeled complete-case cohort, 54 `yes` / 34 `no`, same
`experiments/exp_4/results/mccv_design.csv` MCCV harness (100 `split_*` columns), same `confidence`
3-class target (clear 56 · borderline 18 · uncertain 14).

| Modality | Builder | Dim / representations swept |
|---|---|---|
| `tab` | `data.build_tabular_features(..., dre_categories=cohort.dre_categories)` | fixed, 12-D (7 numeric MinMax + one-hot `dre`, 5 levels) |
| `mri` | `data.build_mri_features(..., pca_variance=...)` | 1024-D raw or PCA-90%, both L2-normalized |
| `txt` | `data.clean_texts_spacy` once at cohort level, then `data.build_text_features(...)` | TF-IDF `max_features ∈ {500, 2000, None}`, PCA-90%, L2-normalized |

**Hard constraint — `dre_categories` must be passed explicitly**, not inferred per split.
`src/evaluation/data.py:190-196`: 3 of `dre`'s 5 levels are singletons in the 88-patient cohort, so
inferring categories from the training subset yields <5 one-hot columns on 49/100 MCCV splits and 3/88
LOOCV folds, crashing `init_kdm_layer`'s shape-checked `copy_`. Every `MemKDM` in this experiment that
includes `tab` uses the full known category set.

Cosine-similarity behaviour for MRI/text comes from L2-normalizing those blocks in preprocessing (as
`src/` already does), not from a cosine kernel — `KDMLayer` squares kernel values, so a parameterless
cosine kernel ties `cos=-1` and `cos=+1` at weight 1.0 either way (`mem_kdm.py` module docstring,
verified directly in the prior `src/` extraction pass: `CosineKernelLayer([1,0],[-1,0]) = -1.0`, squared
`= 1.0`).

### 3.2 Scope note — `embedkit` unavailable in this checkout

`utils/embedding-kit/` is empty, so exp_14's `embedkit_unsup` MRI representation and exp_15's
`embedkit_*` text representations cannot be used here. exp_25's unimodal MRI/text numbers are therefore
**not directly comparable** to exp_14/exp_15's published values (which used those representations), and
any downstream comparison to exp_16/exp_17 (which reused exp_14/exp_15's frozen configs) inherits the
same caveat. This experiment's MRI/text representations are limited to `{raw, PCA-90%}` × L2-norm.

### 3.3 Step 0 — reproduction gate and degeneracy probe (before Stage 1)

Fit a tabular-only `MemKDM(kernels={"tab": KernelSpec(sigma_mult=2.0)}, x_train=False, y_train=False,
encoders={"tab": EncoderSpec("identity")})` on soft targets through the exp_25 harness over the 88
LOOCV folds and **assert** Macro-F1 == `0.6694214876033058` within 1e-6 — exp_23's soft-arm winner,
mirroring exp_24's own in-script reproduction asserts (`exp_24/scripts/train.py:772-779`). Run one fit
per modality subset with `check_roundtrip=True` (default `False` in `MemKDM.__init__`). If the
reproduction assert fails, stop — nothing downstream is interpretable, since it would mean the `src/`
extraction or the exp_25 wiring diverges from exp_23's committed numbers.

On the same tabular reproduction fit, additionally record: `float((h_aleatoric > 1e-6).mean())`, median
`h_aleatoric`, and its Spearman correlation with each patient's total prototype weight mass on the 32
non-`clear` prototypes (§2.5). Write everything to `results/reproduction_check.json`.

### 3.4 Stage 1 — unimodal Phase A (100 MCCV splits, per modality)

Common fixed: `lr=1e-3`, `epochs=300`, `w_train=True`, `label_smoothing=0.0`, `x_train=False`, soft
targets. `x_train=False` mirrors exp_23/24's every selected config.

| Modality | Grid | # configs |
|---|---|---|
| `tab` | `sigma_mult ∈ {0.25,0.5,1.0,2.0}` × `encoder ∈ {identity, linear(12→8)}` × `y_train ∈ {False,True}` | 16 |
| `mri` | `rep ∈ {raw_l2, pca90_l2}` × `sigma_mult ∈ {0.5,1.0,2.0}` × `encoder ∈ {identity, linear(→32)}` × `y_train ∈ {False,True}` | 24 |
| `txt` | `max_features ∈ {500,2000,None}` × `sigma_mult ∈ {0.5,1.0,2.0}` × `encoder ∈ {identity, linear(→32)}` × `y_train ∈ {False,True}` | 36 |

Selected via `protocol.run_mccv_grid` + `protocol.select_best` (argmax mean validation Macro-F1, ties
broken by lowest std then lowest `cfg_id` — the explicit rule exp_13–24 never had). Features are cached
per `(split, modality, representation)` and reused across configs sharing that representation.

**Side effect kept for Stage 2's fusion-weight search (§3.5):** while running each modality's Phase A,
stash that modality's per-split validation-set probabilities at its own winning config — a
`(100, n_val, 2)`-shaped structure, not a global `(n, 2)` array (a training row appears in ~20 different
splits' validation sets with ~20 different probability values).

Artifacts: `results/stage1_grid_search.csv`, `results/stage1_best_hparams.json`.

### 3.5 Stage 2 — joint Phase A, then Phase B

Four multimodal **conditions**, each with its own Phase A over a reduced joint grid:
`{tab,mri}`, `{tab,txt}`, `{mri,txt}`, `{tab,mri,txt}`.

Per-condition Phase-A grid (per-modality encoder/representation frozen from Stage 1):

| Hyperparameter | Values |
|---|---|
| `sigma_scale` | 0.5, 1.0, 2.0 |
| `x_train` | False, True |
| `y_train` | False, True |
| `kernel_trainable` | False, True |

**Total: 3 × 2 × 2 × 2 = 24 configurations per condition.**

`sigma_scale` multiplies every **non-tabular** block's transferred `sigma_mult` (tabular fixed at scale
1.0 as the reference), and is the piece Stage 1 structurally cannot provide: in a product kernel
`Π_m RBF_m`, the *ratio* of per-modality bandwidths sets each modality's effective influence on the
combined weight profile, and independently-tuned per-modality `sigma_mult` values (chosen against
different-dimensional, differently-scaled feature blocks) have no reason to already balance —
`sigma_scale` is the multimodal analogue of exp_16's fusion-weight search, one level inside the kernel
rather than after prediction. `kernel_trainable=True` covers overlapping ground by letting σ move during
joint training; sweeping both separates "the init ratio was wrong" from "training could not fix it".
Per §2.3, `sigma=None` throughout regardless of these settings, so the per-fold init is always
recomputed from that fold's own training data.

Selected via `protocol.run_mccv_grid` + `protocol.select_best`, same rule as Stage 1.

**Confidence arm (exp_24 Arm-C analogue) — conditional on §3.3's degeneracy probe.** If
`(h_aleatoric > 1e-6).mean()` comes back low on the reproduction fit, add one *additional* trimodal
condition at `label_smoothing=0.10` (per `mem_kdm.smooth`, which perturbs only the memory-prototype init
target, never the training loss target), with its own 24-config Phase A, evaluated **only on the
confidence task** — kept out of the decision-task selection exactly as exp_24 kept its Arm C separate
from Arms A/B, so the primary Macro-F1 selection metric is unaffected.

**Leak-free fusion-weight search (`late_fusion_optimal` condition).** `mem_kdm.search_fusion_weights`
is not used: it indexes `probs[m][val_idx]` against global `(n,2)` arrays, which does not fit the
Stage-1 stash's per-split `(100, n_val, 2)` shape (§3.4). Instead, a local helper in `train.py` iterates
`mem_kdm.simplex_grid(3, 0.05)` (231 weight vectors on the 3-modality simplex), `soft_vote`s the stashed
per-split validation probabilities at each vector, scores mean MCCV Macro-F1 across the 100 splits, and
freezes the argmax — zero extra model fits, and leak-free by construction since it only ever touches
Stage-1 validation folds. This directly fixes exp_16's defect: its fusion weights were selected by
`sort_values("macro_f1")` over the same 88 LOOCV out-of-fold predictions it then reported.

**Phase B (LOOCV, 88 folds, all configs frozen — refit per fold, never refit hyperparameters):**

| Condition | Model |
|---|---|
| `unimodal_tab` / `unimodal_mri` / `unimodal_txt` | Stage-1 winner, single-modality `MemKDM` |
| `joint_tab_mri` / `joint_tab_txt` / `joint_mri_txt` / `joint_trimodal` | Stage-2 winner, product-kernel `MemKDM` |
| `late_fusion_equal` | `LateFusionMemKDM`, weights = 1/3 each, members = Stage-1 winners |
| `late_fusion_optimal` | `LateFusionMemKDM`, weights frozen from the search above |
| *(context row, not evaluated)* | exp_16's published Fuzzy-KNN late fusion, 0.6813 — quoted with its selection-on-OOF leak flagged (`search_fusion_weights` docstring), excluded from H1's three criteria |

**Stochasticity.** `linear` encoders consume RNG via `nn.Linear`'s init. Any condition whose selected
encoder(s) include `linear` runs Phase B with `n_seeds=10`, reporting exp_24's `loocv_metrics.json`
shape (`deterministic`, `n_seeds`, `mode_vote_agreement`, `per_seed_macro_f1`,
`macro_f1_std_across_seeds`); headline prediction is the mean soft probability across seeds, thresholded
at 0.50. Conditions with only `identity` encoders run `n_seeds=1` (deterministic full-batch Adam, no
random parameter in the model — exp_23's determinism argument, unchanged).

### 3.6 Confidence task (Stage 2, secondary — all results explicitly target-informed, §2.4)

Per condition: collect LOOCV-OOF particle signals (`MemKDM.uncertainty_signals`, 7 signals; or
`LateFusionMemKDM.uncertainty_signals`, `composite_ici` + prefixed per-member signals), then:

- **1D per-signal meta-thresholds** — `base.fit_meta_thresholds_safe` / `apply_meta_thresholds` per
  signal over the 100 MCCV splits, best chosen by Phase-A Macro-F1 (mirrors `mem_kdm._best_1d_key`'s
  approach, reimplemented locally in `train.py` rather than imported — that helper does not filter on
  `target_informed`, which is moot here since every model is target-informed already, §2.4).
- **Multivariate held-out tree ensemble** — `base.fit_predict_heldout_trees` on the full 7-signal
  vector, plus the `{h_total, log_marginal}` ablation (exp_24's ablation, isolating whether any gain
  is attributable to the new weight-geometry signals specifically).

Scored with `metrics.confidence_metrics` (3-class Macro-F1, accuracy, Spearman ρ vs. `confidence`).
Every row in `results/confidence_metrics.json` carries `"target_informed": true`.

## 4. File Layout for This Experiment

```
experiments/exp_25/
├── DESIGN.md                  ← this file
├── IMPLEMENTATION.md          ← build plan (added after this file is approved)
├── scripts/
│   └── train.py                ← Stage 1 (3 unimodal Phase A's) + Stage 2 (4 joint Phase A's,
│                                   fusion-weight search, Phase B) + confidence heads; --smoke flag
├── results/
│   ├── reproduction_check.json       ← exp_23 Arm-B reproduction assert + h_aleatoric degeneracy probe
│   ├── stage1_grid_search.csv        ← 76 configs × 100 splits, 3 modalities
│   ├── stage1_best_hparams.json      ← per-modality winner (sigma_mult, encoder, y_train)
│   ├── stage2_grid_search.csv        ← up to 4-5 conditions × 24 configs × 100 splits
│   ├── best_hparams.json             ← per-condition Stage-2 winner + Stage-1 provenance
│   ├── fusion_weights.json           ← simplex-grid search result for late_fusion_optimal
│   ├── loocv_metrics.json            ← binary LOOCV metrics, all conditions, seed info, McNemar
│   ├── confidence_metrics.json       ← 1D + multivariate heads, all conditions, target_informed=true
│   ├── oof_predictions.csv           ← per-patient probabilities, all conditions
│   ├── oof_particle_signals.csv      ← per-patient 7-signal particle set, all MemKDM conditions
│   └── git_commit.txt
└── reports/
    ├── figures/
    │   ├── stage1_grid_search_curves.png   ← Macro-F1 vs sigma_mult, per modality
    │   ├── stage2_grid_search_curves.png   ← Macro-F1 vs sigma_scale, per condition
    │   ├── confusion_matrix.png            ← LOOCV 2x2, best condition
    │   ├── roc_curve.png                   ← joint_trimodal vs late_fusion_optimal vs unimodal_tab overlay
    │   ├── particle_signal_scatter.png     ← h_aleatoric vs h_epistemic, colored by confidence
    │   └── confidence_confusion_matrix.png ← 3x3 CM, best confidence head
    └── summary.md
```

## 5. Evaluation Protocol & Decision Rules

Two-phase harness (`CLAUDE.md` "Two-phase leak-free evaluation protocol"), instantiated via
`src/evaluation/protocol.py`: scaler/encoder refit inside every split/fold; Phase B never re-fits
hyperparameters, thresholds, or fusion weights.

- **Decision rule (H1, primary):** `joint_trimodal`'s LOOCV Macro-F1 vs. the best of
  {`unimodal_tab/mri/txt`}, `late_fusion_optimal`, and exp_23 Arm B (0.6694). `mcnemar_exact`
  (`joint_trimodal` vs. `late_fusion_optimal`) for the significance criterion.
- **Decision rule (H2):** best target-informed row in `confidence_metrics.json` vs. exp_23's
  `entropy_soft` (0.4164); exp_24 (0.4368) and exp_17 (0.4470) reported as non-like-for-like context.
- **Decision rule (H3):** qualitative — per-patient agreement/disagreement between `joint_trimodal`'s
  `h_epistemic` ranking and `late_fusion_optimal`'s `composite_ici` ranking (Spearman ρ between the two
  signals, plus a scatter, `particle_signal_scatter.png`); no significance threshold, since there is no
  shared null between two different uncertainty constructions.
- **Reproduction gate (blocking, §3.3):** Step 0's assert must pass before any other result is reported.
- **Secondary metrics (binary LOOCV, all conditions):** Macro-F1, Accuracy, Sensitivity, Specificity,
  AUROC, Brier — same schema as `exp_23/24/results/loocv_metrics.json`.

## 6. Known Pitfalls to Avoid (identified during design, not to be repeated)

- **Split-tuple arity.** `protocol.iter_mccv_splits` yields 3-tuples `(split_idx, train_idx, val_idx)`.
  `run_mccv_grid` tolerates that, but `base.fit_meta_thresholds`, `base.fit_predict_heldout_trees`, and
  `mem_kdm.search_fusion_weights` all unpack 2-tuples and raise on a 3-tuple. `train.py` materializes
  `SPLITS = [(tr, va) for _, tr, va in iter_mccv_splits(df_design, 100)]` once and passes `SPLITS`
  everywhere those functions are used.
- **`mem_kdm.simplex_grid` hard-fails on `n_modalities != 3`** — fine for the trimodal
  `late_fusion_optimal` condition; there is deliberately no bimodal late-fusion condition (the bimodal
  *joint* `MemKDM` conditions already cover that comparison, and a 2-weight sweep is trivial and adds
  nothing to H1).
- **`from_unimodal` and `search_fusion_weights` go unused, deliberately** (§2.3, §3.5) — not oversights.
- **`src/` is not on `sys.path`**, and `src/evaluation/data.py:167` does
  `from src.methods.base import Targets` — `train.py` inserts the project root at the head of
  `sys.path` before any `src.*` import.
- **`check_roundtrip` defaults to `False`** in `MemKDM.__init__` — turned on deliberately in §3.3, not
  relied on implicitly elsewhere (it is per-fit overhead, not needed on every one of the ~17,000+ fits).
- **`dre_categories=None` is not a safe default** for any condition including `tab` (§3.1) — always pass
  `cohort.dre_categories` explicitly.
- Two pre-existing `src/` issues noted during exploration, **out of scope to fix in exp_25**:
  `base.py:71`'s docstring cites a non-existent `mem_kdm.best_signal` (the real function is the private
  `_best_1d_key`, which does not filter on `target_informed` — moot here since every exp_25 model is
  target-informed, §2.4); `data.build_targets` never populates `Targets.y_conf` (harmless here, since
  `MemKDM.fit_confidence` takes `y_conf` as a separate argument).
- `src/` is currently **uncommitted** (`git status`: `?? src/`). It should be committed — with its
  `.logbook.md`/`.discussion.md` entries — before or alongside exp_25's implementation, so
  `results/git_commit.txt` points at a tree that actually contains the method layer being evaluated.

## 7. Scope

**In scope:** old schema (`Data/preprocessed_old/task1/`) only, exp_23/24-comparable. Binary biopsy
decision (primary, H1) + 3-class diagnostic confidence via multimodal particle signals (secondary, H2),
both under soft supervision throughout (§2.4) + a structural contrast between shared and per-modality
particle sets (H3).

**Explicitly out of scope:** new schema (`Data/preprocessed/task1/`); `embedkit_*` MRI/text
representations (§3.2, unavailable in this checkout); `KDMRegressModel`; a separate hard-supervised arm
for the confidence task (§2.4 — the user's explicit direction is soft supervision everywhere, with the
target-informed caveat carried instead); feature-attribution (exp_20–22 territory); re-running
exp_13/16/17's Fuzzy-KNN pipelines (their published numbers are quoted, not recomputed, except for the
exp_23 reproduction check in §3.3 which exists specifically to validate the new code path).

## 8. Reproducibility Checklist

- [ ] Random seeds: Stage 1 & 2 Phase A single-seed per split (deterministic given split assignment);
      Phase B explicit `seed ∈ {0..9}` per fold for any condition with a `linear` encoder, recorded
      per-seed in `loocv_metrics.json`.
- [ ] Reproduction gate (§3.3) passes before any other result is written or reported.
- [ ] Config and scripts saved in `scripts/`.
- [ ] Grid search results logged to `results/stage1_grid_search.csv` and `results/stage2_grid_search.csv`.
- [ ] Fusion weights logged to `results/fusion_weights.json` with the MCCV split scores that produced them.
- [ ] **Git commit hash recorded** — `git log -1 --format="%H %s" > results/git_commit.txt` before
      execution, from a commit that includes `src/`.

## 9. Next Steps

1. Review and accept this experiment plan.
2. Once accepted, produce `IMPLEMENTATION.md` (concrete build plan, measured pilot-timing gate per §7 of
   the approved planning discussion, and exact execution command) for approval.
3. Once accepted, implement `scripts/train.py`, run the reproduction gate and a `--smoke` pilot to
   validate runtime and correctness before committing to the full run, then execute `exp_25`.
