# Experiment Report: Multimodal Memory-Based KDM (`MemKDM`)
**Experiment**: experiments/exp_25/ · **Project**: pathology-reasoning · **Report date**: 2026-08-18 ·
**Plan date**: 2026-08-18 · **Status**: Complete

---

## 1. Summary

exp_25 evaluated whether a shared-memory, product-kernel `MemKDM` over {tabular, MRI, text} — tuned in two
stages, unimodal bandwidths first (Stage 1) then joint memory/bandwidth-ratio hyperparameters second
(Stage 2) — beats the best unimodal model, a leak-free late-fusion baseline, and exp_23's tabular-only
KDM (LOOCV Macro-F1 0.6694) on the biopsy-decision task (H1). It does not: the trimodal joint model
(LOOCV Macro-F1 0.5860) underperforms both the best unimodal model (text, 0.6081) and leak-free late
fusion (0.6071), and falls well short of exp_23 (0.6694) — **H1 is refuted**, cleanly and without a close
call. The secondary confidence-prediction objective (H2) shows a modest, non-significance-tested gain
over the correct like-for-like baseline (0.4547 vs. exp_23's target-informed 0.4164), driven by a
`joint_tab_mri` bimodal condition, not the trimodal model. The structural contrast (H3) between the
joint model's own uncertainty and late fusion's composite ICI shows the two are weakly and
non-significantly related (Spearman ρ=−0.117, p=0.276), consistent with them capturing different things,
though the correlation is too weak to support a stronger claim than "not obviously the same signal."

---

## 2. Hypotheses & Verdicts

**H1 (primary, decision).** *"A joint product-kernel `MemKDM` over {tabular, MRI, text} ... beats (a) the
best unimodal `MemKDM` from Stage 1, (b) `LateFusionMemKDM` with fusion weights selected leak-free, and
(c) exp_23 Arm B (0.6694), on LOOCV Macro-F1."*

**Verdict: ❌ Refuted.** `joint_trimodal` LOOCV Macro-F1 = **0.5860**, below (a) unimodal_txt (0.6081),
(b) late_fusion_optimal (0.6071), and (c) exp_23 Arm B (0.6694). It does not even beat the bimodal
`joint_tab_mri` (0.6048). McNemar `joint_trimodal` vs. `late_fusion_optimal`: b=7, c=10, p=0.629 — not
significant, but the direction (c>b) is consistent with the point estimate: late fusion wins more
discordant pairs. Combining three modalities into one shared kernel did not help, and specifically hurt
relative to the strongest single modality (text) on this 88-patient cohort.

**H2 (secondary, confidence).** *"Particle-set signals from a shared multimodal particle set carry more
diagnostic-confidence information than exp_24's tabular-only particle set,"* measured against the correct
target-informed comparator (exp_23's `entropy_soft`, 0.4164) since every exp_25 model is target-informed
under soft supervision (§2.4 in DESIGN.md).

**Verdict: ⚠️ Partially supported, weak margin.** Best target-informed row overall: `joint_tab_mri` /
1D-signal `w_max` = **0.4547** (accuracy 57.95%, ρ=0.202) — beats the like-for-like baseline (0.4164) by
+0.038, and, as context only (not the test), also nominally exceeds exp_24's non-target-informed 0.4368
and exp_17's 0.4470. No significance test was specified for H2 in DESIGN.md and none is reported here —
this is a single point comparison at N=88 with std_macro_f1 ~0.10-0.12 typical for this cohort's MCCV
grids (`select_best`'s own justification for its tie-break rule), so the margin should not be
over-read. Notably the win comes from a **bimodal** condition (`joint_tab_mri`), not the trimodal model
H1 was about — `joint_trimodal`'s own best confidence row does not appear in the top 10.

**H3 (structural).** *"A shared particle set is not equivalent to per-modality particle sets combined post
hoc,"* assessed as a qualitative Spearman contrast between `joint_trimodal`'s `h_epistemic` and
`late_fusion_equal`'s `composite_ici`, per patient.

**Verdict: ⚠️ Weakly consistent with the claim, not strongly evidenced.** Spearman ρ = −0.117 (p=0.276,
`h_epistemic` vs. `composite_ici`); ρ = −0.245 (p=0.021, `h_total` vs. `composite_ici` — nominally
significant, negative direction, i.e. higher joint-model total entropy loosely tracks *lower* late-fusion
composite reliability). Neither correlation is strong. This is consistent with "the two constructions
capture different things" but is equally consistent with "the two constructions are just noisy" — DESIGN.md
pre-registered this as a qualitative contrast with no significance threshold, so no stronger claim is made.

---

## 3. Experimental Setup (as run)

Built exactly as `IMPLEMENTATION.md` specifies, with two internal simplifications the script's own
docstring documents (neither changes DESIGN.md's protocol, hyperparameters, or leak-free guarantees):
feature rebuilding is not cached per (split, modality, representation) — matches the established repo
convention of exp_13–24, which never cache either; and the fusion-weight search uses a small dedicated
post-hoc re-fit of the three Stage-1 winners across the 100 MCCV splits, rather than the stash-based
design IMPLEMENTATION.md §1.4/§1.5 sketched (the stash approach required reconstructing a config key from
a `select_best` row's coerced pandas dtypes, which was fragile; the post-hoc pass is simpler, still
strictly leak-free, and costs ~300 extra fits).

Two further deviations from `IMPLEMENTATION.md`, both flagged inline in `train.py` where they occur and
none affecting the pre-registered hypotheses:
- Confidence-head MCCV splits (`fit_meta_thresholds_safe`, `fit_predict_heldout_trees`) always use the
  full 100 MCCV splits, even conceptually — this was forced during `--smoke` testing (a 5-split reduction
  breaks `fit_predict_heldout_trees`'s "every patient gets ≥1 held-out vote" invariant at N=88) and
  applies unconditionally in the real run too, matching exp_24's own documented precedent for the
  identical reason.
- `confidence_metrics`' `spearman_rho`/`spearman_pvalue` are coerced from NaN to `null` when a head's
  predictions collapse to one class, rather than letting `write_json`'s strict encoder reject the file —
  this is the exact bug `reporting._StrictEncoder` was built to catch (it exists because
  `exp_24/results/confidence_metrics.json` shipped with bare NaN); the fix belongs at the metric source,
  not as an `allow_nan=True` bypass.

- **Dataset**: `Data/preprocessed_old/task1/`, N=88 (54 yes / 34 no), tabular (12-D) + MRI (1024-D raw or
  ~11-D PCA-90%) + text (TF-IDF, spaCy-lemmatized, PCA-90%, ~54–57-D). Soft targets throughout
  (`certainty_map={"clear":1.0,"borderline":0.5,"uncertain":0.25}`).
- **Model**: `MemKDM`, `n_comp = n_train` (70 in Phase A, 87 in Phase B), product RBF kernel.
- **Stage 1** (unimodal Phase A, 100-split MCCV): 16/24/36 configs for tab/mri/txt.
- **Stage 2** (joint Phase A, then Phase B): 24 configs × 5 conditions (4 decision-task + 1
  `label_smoothing=0.10` confidence-only arm, run unconditionally per the implementation-review deviation
  recorded in `IMPLEMENTATION.md` §1.5).
- **Training**: Adam, lr=1e-3, 300 full-batch epochs, `sigma=None` throughout (recomputed from each
  fold's own training data — the leak-free sigma-transfer design from `DESIGN.md` §2.3).
- **Hardware**: local `pytorch` conda env, single-threaded CPU. Actual runtime: **101.3 min** (estimate
  was 1.5–2.5h — landed at the low end).
- **Deviations from plan**: the two internal simplifications above; none change what was measured.

---

## 4. Code Version

| Item | Value |
|---|---|
| `src/` commit | `b601b2a671ed42b27cf10f0c0cddccc6409c8216` — "Extract src/methods and src/evaluation: memory-based multimodal KDM (MemKDM)" |
| `results/git_commit.txt` | same hash, recorded at end of run |

`src/` was uncommitted when the run was designed; committed (by explicit user confirmation) before the
full run's `record_git_commit` call, so the recorded hash correctly reflects the evaluated code.

---

## 5. Results

### 5.1 Reproduction gate (Step 0 — blocking; passed before any other result was computed)

| Check | Result |
|---|---|
| exp_23 Arm B reproduction (tab-only, `sigma_mult=2.0, identity`) | **0.6694214876033058** — exact match to exp_23's published value (diff 0.0) |
| `check_roundtrip` (tab / mri-pca90 / txt) | all passed |
| `frac_nonzero_h_aleatoric` | **0.977** — the a-priori concern that most `h_aleatoric` would be degenerate (56/88 patients are `clear`, sitting at the one-hot fixed point per exp_24's H3) did **not** materialize; only 2/88 patients show near-zero `h_aleatoric`, so the confidence-arm's label-smoothing condition was well-motivated to run |

### 5.2 Stage 1 — unimodal winners (MCCV Phase A, mean over 100 splits)

| Modality | Config | mean_macro_f1 | std_macro_f1 |
|---|---|---|---|
| `tab` | `sigma_mult=2.0, encoder=linear, y_train=False` | 0.6044 | 0.1082 |
| `mri` | `sigma_mult=1.0, encoder=identity, y_train=False, rep=pca90_l2` | 0.5012 | 0.1121 |
| `txt` | `sigma_mult=0.5, encoder=identity, y_train=False, rep=(max_features=2000, pca=0.90)` | 0.6065 | 0.1009 |

Leak-free fusion weights (post-hoc search over the Stage-1 winners' stashed validation probabilities,
231-point simplex grid): **tab=0.50, mri=0.25, txt=0.25**, mean MCCV Macro-F1 = 0.6232.

### 5.3 Stage 2 — joint winners (MCCV Phase A, mean over 100 splits)

| Condition | `sigma_scale` | `x_train` | `y_train` | `kernel_trainable` | mean_macro_f1 |
|---|---|---|---|---|---|
| `tab_mri` | 2.0 | False | **True** | False | 0.6283 |
| `tab_txt` | 2.0 | **True** | False | False | 0.6132 |
| `mri_txt` | 2.0 | **True** | False | **True** | 0.5334 |
| `tab_mri_txt` | 2.0 | **True** | False | False | 0.6095 |
| `confidence_arm` (`label_smoothing=0.10`) | 2.0 | **True** | False | False | 0.6093 |

All five winners selected `sigma_scale=2.0` — the widest bandwidth in the grid — suggesting the search may
be bandwidth-starved at the top of its range; not re-swept here (out of scope for this run) but worth
noting for `DESIGN.md`'s "Next Steps."

### 5.4 Phase B — LOOCV, binary decision task (primary metric: Macro-F1)

| Condition | Macro-F1 | AUROC | Brier | Deterministic | n_seeds |
|---|---|---|---|---|---|
| `unimodal_tab` | 0.5978 | 0.6699 | 0.2273 | No | 10 |
| `unimodal_mri` | 0.5152 | 0.4439 | 0.2932 | Yes | 1 |
| `unimodal_txt` | **0.6081** | 0.6759 | 0.2155 | Yes | 1 |
| `joint_tab_mri` | 0.6048 | 0.6868 | 0.2171 | No | 10 |
| `joint_tab_txt` | 0.5978 | 0.6618 | 0.2443 | No | 10 |
| `joint_mri_txt` | 0.5200 | 0.5300 | 0.2501 | Yes | 1 |
| `joint_trimodal` | 0.5860 | 0.6514 | 0.2497 | No | 10 |
| `late_fusion_equal` | 0.5997 | 0.6596 | 0.2214 | No | mixed (tab:10, mri:1, txt:1) |
| `late_fusion_optimal` | **0.6071** | 0.6661 | 0.2169 | No | mixed (tab:10, mri:1, txt:1) |
| `confidence_arm` (context only, excluded from H1) | 0.5860 | 0.6514 | 0.2417 | No | 10 |
| exp_23 Arm B (context, not re-run) | **0.6694** | 0.6498 | 0.2263 | Yes | — |

`joint_trimodal` and `confidence_arm` show identical Macro-F1 (0.5860) and AUROC (0.6514) despite
different `label_smoothing` (0.0 vs. 0.10) and genuinely different per-patient probabilities (max
|Δprob| = 0.080, no row identical) — verified not a bug: the two conditions' predictions happen to share
the same rank order (→ identical AUROC) and the same 0.50-threshold crossings (→ identical Macro-F1),
plausible given both share the same winning Stage-2 hyperparameters and `x_train=True` (only `c_y`'s
fixed init differs).

**McNemar's exact test:**

| Comparison | b | c | p-value |
|---|---|---|---|
| `joint_trimodal` vs. `late_fusion_optimal` (H1's significance criterion) | 7 | 10 | 0.629 |
| `joint_trimodal` vs. exp_23 Arm B (context) | 6 | 14 | 0.115 |

Neither reaches significance, but both point in the same direction as the Macro-F1 comparison — the
joint trimodal model is not distinguishable from chance-level disagreement with either baseline, and
where it does disagree, it more often loses.

### 5.5 Confidence task (secondary, all rows `target_informed: true`)

Top rows by 3-class Macro-F1 (full table: `results/confidence_metrics.json`, 10 conditions × 3 heads):

| Condition | Head | Signal(s) | Macro-F1 | Accuracy | Spearman ρ |
|---|---|---|---|---|---|
| `joint_tab_mri` | 1D | `w_max` | **0.4547** | 57.95% | 0.202 |
| `joint_tab_mri` | multivariate (7-signal) | all | 0.4486 | 50.00% | 0.051 |
| `unimodal_txt` / `late_fusion_*` | 1D | `log_ess` (txt) | 0.4418 | 45.45% | 0.175 |
| `unimodal_mri` / `late_fusion_*` | 1D | `log_ess` (mri) | 0.4404 | 59.09% | 0.086 |
| `unimodal_txt` | multivariate ablation | `{h_total, log_marginal}` | 0.4264 | 47.73% | 0.185 |

Context baselines (not like-for-like — see H2 verdict): exp_23 `entropy_soft` (target-informed) = 0.4164;
exp_24 best non-target-informed = 0.4368; exp_17 Composite ICI (non-target-informed) = 0.4470.

### 5.6 Figures

`reports/figures/`: `stage1_grid_search_curves_{tab,mri,txt}.png`, `stage2_grid_search_curves_{tab_mri,
tab_txt,mri_txt,tab_mri_txt,confidence_arm}.png`, `confusion_matrix.png` (`joint_trimodal`, LOOCV),
`roc_curve.png` (`joint_trimodal` vs. `late_fusion_optimal` vs. `unimodal_tab` overlay),
`particle_signal_scatter.png` (`joint_trimodal` `h_aleatoric` vs. `h_epistemic`, colored by confidence).

---

## 6. Statistical Analysis

- **H1's test**: McNemar's exact test (`scipy.stats.binomtest`) on paired LOOCV binary predictions,
  `joint_trimodal` vs. `late_fusion_optimal` — the pre-registered significance criterion. p=0.629, not
  significant. N=88 discordant-pair count (b+c=17) is small, as expected at this cohort size.
- **H2, H3**: no significance test was pre-registered in `DESIGN.md` beyond the Spearman correlations
  already reported in §2/§5.5 (p=0.276 for `h_epistemic`/`composite_ici`, p=0.021 for
  `h_total`/`composite_ici`).
- **Per-seed values**: available for every stochastic (`linear`-encoder) condition
  (`per_seed_macro_f1`, 10 entries each, in `loocv_metrics.json`) but no additional significance test
  across conditions was pre-registered on these, so none is computed here beyond the reported
  `macro_f1_std_across_seeds`.

---

## 7. Comparison to Expected Results

| Expected (DESIGN.md §1) | Observed | Match? |
|---|---|---|
| H1: `joint_trimodal` beats best unimodal, late fusion, and exp_23 (0.6694), McNemar p<0.05 vs. late fusion | 0.5860 < 0.6081 (unimodal) < 0.6071 (late fusion) < 0.6694 (exp_23); McNemar p=0.629 | ❌ Refuted |
| H2: best target-informed confidence row beats exp_23 `entropy_soft` (0.4164) | 0.4547 > 0.4164 (+0.038) | ⚠️ Nominally met, weak margin, no significance test |
| H3: joint vs. late-fusion uncertainty are qualitatively distinct (no threshold pre-registered) | ρ=−0.117 (p=0.276), weak and non-significant | ⚠️ Consistent with, not strong evidence for |

---

## 8. Missing Data & Caveats

All planned Stage 1/Stage 2 conditions, the reproduction gate, the degeneracy probe, the fusion-weight
search, Phase B for every condition, and both confidence heads (1D + multivariate, plus ablation)
completed. **All planned runs completed** — nothing in `DESIGN.md`'s scope was skipped.

Caveats carried from `DESIGN.md` and confirmed still applicable:
- `embedkit_*` MRI/text representations were unavailable (`utils/embedding-kit/` empty in this checkout);
  exp_25's unimodal MRI/text numbers are not directly comparable to exp_14/exp_15's published values.
- Every exp_25 model is target-informed (soft supervision throughout, per user direction) — H2's numbers
  are not comparable to exp_17/exp_24's headline non-target-informed confidence numbers as a fair test,
  only as context, per `DESIGN.md` §2.4.
- MRI's Stage-1 winner (mean_macro_f1 0.5012) is barely above chance-level for a 2-class task — MRI is
  the weakest modality throughout every table in this report (`unimodal_mri` 0.5152, `joint_mri_txt`
  0.5200), consistent with exp_14's own finding that MRI-alone was the weakest exp_13–17 modality.

---

## 9. Conclusions & Next Steps

**What this experiment established:**
- The `src/` extraction (`MemKDM`, the two-phase MCCV/LOOCV harness) is faithful: the reproduction gate
  matched exp_23 to 16 significant digits before any multimodal claim was made.
- Naively combining three modalities into one shared product-kernel memory **does not help** on this
  88-patient cohort, and specifically underperforms the strongest single modality (text) and a
  leak-free-weighted late-fusion baseline. The weight-transfer/bandwidth-ratio machinery
  (`sigma_scale`) built for this experiment consistently selected the widest bandwidth in its grid across
  all five Stage-2 conditions — a pattern, not five independent coincidences.
- A joint bimodal (tab+MRI) particle set gives a modest confidence-prediction edge over the correct
  target-informed baseline, but the effect is not large and was not the condition H1 was built around.

**What remains uncertain:**
- Whether `MemKDM`'s multimodal underperformance is a genuine ceiling for this cohort size (N=88, 3
  modalities, `n_comp` shared across a widening feature space) or an artifact of the `sigma_scale` grid
  not reaching wide enough — every winner sat at the grid's edge (§5.3).
- Whether the text modality's strength (best unimodal, 0.6081) reflects real signal or the smaller,
  cleaner feature space (~54-57D vs. MRI's ~11-1024D) simply suiting a memory-based kernel better at
  N≈70-87 training prototypes.

**Recommended follow-up:**
- A narrow re-sweep of `sigma_scale` extending past 2.0 (e.g. {2.0, 4.0, 8.0}) on `joint_trimodal` and
  `joint_tab_mri` specifically, to check whether the edge-of-grid pattern in §5.3 is a real optimum or a
  truncated search.
- If the goal remains a multimodal decision model, `joint_tab_mri` (the strongest joint condition, and
  the source of H2's confidence gain) is a more promising target than the trimodal model for a follow-up
  `exp_26`.
- To set up either, use the `ml-experiment-planner` skill.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Seeds logged | ✅ (`per_seed_macro_f1`, 10 seeds per stochastic condition, `loocv_metrics.json`) |
| Configs versioned | ✅ (`stage1_best_hparams.json`, `best_hparams.json`) |
| Git commit recorded | ✅ (`b601b2a...`, §4) |
| Checkpoints saved | ❌ (not designed to — matches exp_13–24 convention; models are refit per-fold, not persisted) |
| Environment frozen | ⚠️ Partial — `conda run -n pytorch`, no `requirements.txt`/lockfile committed (repo-wide convention, see `CLAUDE.md`) |
| Experiment tracker linked | N/A — this repo's tracker is `experiments/INDEX.md` |
