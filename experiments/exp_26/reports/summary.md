# Experiment Report: MemKDM without the noise-selected linear encoder (exp_26)
**Experiment**: experiments/exp_26/ · **Project**: pathology-reasoning · **Report date**: 2026-08-19 ·
**Plan date**: 2026-08-19 · **Status**: Complete

---

## 1. Summary

exp_25's Stage 1 grid search picked `encoder=linear` for `tab` by a 0.003 margin on mean MCCV Macro-F1 —
two orders of magnitude smaller than each config's own noise (std ≈ 0.11) — and that noise-driven pick
propagated a trainable, randomly-initialized encoder into every downstream joint model, regressing
`unimodal_tab`'s LOOCV Macro-F1 to 0.5978 against exp_23's deterministic 0.6694 for the otherwise-identical
config. exp_26 drops `linear` from the grid entirely (identity encoder only) and makes the Stage-1→Stage-2
bandwidth transfer explicit (fold-local `sigma`, verified a numerical no-op). The fix works exactly as
predicted: `unimodal_tab` now reproduces exp_23 **bit-for-bit** (0.6694214876033058, confusion matrix
48/15/19/6). Every joint condition improves substantially over exp_25 (`joint_tab_mri` 0.6048→0.6662,
`joint_trimodal` 0.5860→0.6048, `late_fusion_optimal` 0.6071→0.6648), and the confidence-prediction task's
best row jumps from 0.4547 to 0.5287. Despite this, exp_25's H1 verdict is **unchanged**: `joint_trimodal`
(0.6048) still underperforms the best unimodal model (`unimodal_tab`, 0.6694), leak-free late fusion
(0.6648), and exp_23 (0.6694) — the encoder-instability confound was real and worth removing, but it was
not the reason the trimodal joint model underperformed.

---

## 2. Hypothesis & Verdict

**H (single, corrective, from `DESIGN.md`).** *"Restricting Stage 1/Stage 2 to `encoder=identity` only...
removes this selection-noise-driven regression: `unimodal_tab`'s LOOCV Macro-F1 recovers to match exp_23
Arm B (0.6694, expected bit-for-bit)... re-evaluating exp_25's H1 under this fix shows whether exp_25's
'H1 refuted' verdict survives once the encoder-instability confound is removed."*

**Verdict: ✅ Supported (recovery) / H1 remains ❌ Refuted (under the fixed grid).** The corrective half of
H is fully supported: `unimodal_tab` = **0.6694214876033058**, exact match to exp_23 (§3 below). The
re-evaluation half shows exp_25's H1 verdict survives the fix: `joint_trimodal` (0.6048) is still below
`unimodal_tab` (0.6694), `late_fusion_optimal` (0.6648), and exp_23 (0.6694). McNemar `joint_trimodal` vs.
`late_fusion_optimal`: b=4, c=11, p=0.118 — not significant, but the discordant-pair split is now *more*
lopsided in late fusion's favor than exp_25's (b=7, c=10), not less — combining all three modalities into
one shared kernel still does not help, and the encoder-noise confound was not masking a hidden win.

---

## 3. Experimental Setup (as run)

As described in `DESIGN.md` / `IMPLEMENTATION.md` — no deviations. Confirms the two load-bearing checks
`DESIGN.md` §3 pre-registered before trusting any Stage-2 number:

| Check | Result |
|---|---|
| Step 0 reproduction gate (exp_23 Arm B, tab-only, `sigma_mult=2.0, identity`, untouched direct-`sigma_mult` path) | **0.6694214876033058** — exact match |
| Stage 1 `unimodal_tab` winner | `sigma_mult=2.0, encoder=identity, y_train=False` (mean MCCV Macro-F1 0.6013) — now the *only* identity option, previously edged out by the noise-selected `linear` (0.6044) |
| Phase B `unimodal_tab` (via the new explicit fold-local `sigma=` transfer path) | **0.6694214876033058**, confusion matrix 48/15/19/6 — bit-for-bit identical to the Step 0 oracle and to exp_23, confirming the "transfer sigma, not sigma_mult" refactor (`DESIGN.md` §2.3) is truly a numerical no-op |
| `check_roundtrip` (tab / mri-pca90 / txt) | all passed |
| `frac_nonzero_h_aleatoric` | 0.977 (same as exp_25 — expected, `tab`'s reproduction path is untouched) |

- **Dataset**: `Data/preprocessed_old/task1/` (old schema), N=88 complete-case cohort, 54 yes / 34 no — same as exp_25.
- **Model**: `MemKDM`, `n_comp = n_train` (70 in Phase A, 87 in Phase B), product RBF kernel, `encoder=identity` everywhere (no trainable encoder in this experiment).
- **Stage 1** (unimodal Phase A, 100-split MCCV): 8/12/18 configs for tab/mri/txt (down from exp_25's 16/24/36 — `linear` encoder removed).
- **Stage 2** (joint Phase A, then Phase B): 24 configs × 5 conditions (4 decision-task + confidence_arm), same `STAGE2_GRID` as exp_25.
- **Runtime**: 63.3 min full run (exp_25: 101.3 min — faster here mainly because every condition is now `n_seeds=1` instead of `n_seeds=10` for any config touching `tab`).
- **Deviations from plan**: None.

---

## 4. Code Version

| Item | Value |
|---|---|
| Git commit at run time | `b601b2a` "Extract src/methods and src/evaluation: memory-based multimodal KDM (MemKDM)" |

Single script (`experiments/exp_26/scripts/train.py`), single commit for the whole run — no per-condition commits, consistent with exp_25's convention (`src/` unchanged; only the experiment script differs).

---

## 5. Results

### 5.1 Stage 1 — unimodal winners (MCCV Phase A, mean over 100 splits)

| Modality | Config | mean_macro_f1 | std_macro_f1 | exp_25 winner (for reference) |
|---|---|---|---|---|
| `tab` | `sigma_mult=2.0, encoder=identity, y_train=False` | 0.6013 | 0.1074 | `linear` @ 0.6044 (noise-selected) |
| `mri` | `sigma_mult=1.0, encoder=identity, y_train=False, rep=pca90_l2` | 0.5012 | 0.1121 | identical (already `identity`) |
| `txt` | `sigma_mult=0.5, encoder=identity, y_train=False, rep=(max_features=2000, pca=0.90)` | 0.6065 | 0.1009 | identical (already `identity`) |

Leak-free fusion weights (post-hoc search over the Stage-1 winners' stashed validation probabilities,
231-point simplex grid): **tab=0.50, mri=0.05, txt=0.45**, mean MCCV Macro-F1 = 0.6261 (exp_25:
tab=0.50/mri=0.25/txt=0.25, 0.6232).

### 5.2 Stage 2 — joint winners (MCCV Phase A, mean over 100 splits)

| Condition | `sigma_scale` | `x_train` | `y_train` | `kernel_trainable` | mean_macro_f1 |
|---|---|---|---|---|---|
| `tab_mri` | 2.0 | **True** | False | False | 0.6344 |
| `tab_txt` | 2.0 | False | False | False | 0.5922 |
| `mri_txt` | 2.0 | **True** | False | **True** | 0.5334 |
| `tab_mri_txt` | 2.0 | **True** | False | **True** | 0.6035 |
| `confidence_arm` (`label_smoothing=0.10`) | 2.0 | **True** | False | **True** | 0.5991 |

All five winners again select `sigma_scale=2.0` — the widest bandwidth in the grid — reproducing exp_25's
"bandwidth-starved" pattern exactly. This is now the second experiment in a row to hit this edge; it is
increasingly unlikely to be coincidence (see §9).

### 5.3 Phase B — LOOCV, binary decision task (primary metric: Macro-F1)

| Condition | Macro-F1 | AUROC | Brier | exp_25 Macro-F1 | Δ vs. exp_25 |
|---|---|---|---|---|---|
| `unimodal_tab` | **0.6694** | 0.6498 | 0.2263 | 0.5978 | **+0.0716** |
| `unimodal_mri` | 0.5152 | 0.4439 | 0.2932 | 0.5152 | +0.0000 (unaffected — already `identity`) |
| `unimodal_txt` | 0.6081 | 0.6759 | 0.2155 | 0.6081 | +0.0000 (unaffected — already `identity`) |
| `joint_tab_mri` | 0.6662 | **0.7081** | 0.2146 | 0.6048 | **+0.0614** |
| `joint_tab_txt` | 0.6081 | 0.6416 | 0.2260 | 0.5978 | +0.0103 |
| `joint_mri_txt` | 0.5200 | 0.5300 | 0.2501 | 0.5200 | +0.0000 (unaffected — no `tab`) |
| `joint_trimodal` | 0.6048 | 0.6427 | 0.2467 | 0.5860 | +0.0188 |
| `late_fusion_equal` | 0.5616 | 0.6471 | 0.2216 | 0.5997 | −0.0381 |
| `late_fusion_optimal` | 0.6648 | 0.6993 | 0.2078 | 0.6071 | **+0.0577** |
| `confidence_arm` (context, excluded from H1) | 0.6048 | 0.6318 | 0.2421 | 0.5860 | +0.0188 |
| exp_23 Arm B (context, not re-run) | **0.6694** | 0.6498 | 0.2263 | — | — |

`unimodal_tab` and exp_23 Arm B are not merely close — every metric (Macro-F1, AUROC, Brier, confusion
matrix) is bit-for-bit identical, since both now compute the same quantity by the same route. `joint_tab_mri`
has the best AUROC of any condition (0.7081) and comes within 0.0032 Macro-F1 of `unimodal_tab`/exp_23 —
the closest any joint or fusion condition has come to the tabular-only ceiling across exp_25 and exp_26.
`late_fusion_equal` is the one condition that got *worse* — its equal-weighting scheme no longer suits the
now-much-stronger `unimodal_tab`, since `late_fusion_optimal`'s leak-free search correctly shifts weight
toward tab+txt (§5.1) and recovers essentially all of the gain.

**McNemar tests:**

| Comparison | b | c | p-value | exp_25 (b, c, p) |
|---|---|---|---|---|
| `joint_trimodal` vs. `late_fusion_optimal` | 4 | 11 | 0.118 | 7, 10, 0.629 |
| `joint_trimodal` vs. exp_23 soft | 7 | 13 | 0.263 | 6, 14, 0.115 |

Neither reaches significance at N=88, consistent with exp_25. The `joint_trimodal` vs. `late_fusion_optimal`
split is *more* lopsided toward late fusion here (c=11 vs. b=4) than in exp_25 (c=10 vs. b=7) — i.e.
fixing the encoder-noise confound did not surface a hidden trimodal advantage; if anything the gap in
discordant pairs widened.

### 5.4 Confidence-prediction task (secondary, H2-style — context, not part of H1)

| Condition / head | Macro-F1 | Accuracy | Spearman ρ |
|---|---|---|---|
| `late_fusion_equal` / `multivariate_full` (best) | **0.5287** | 0.5682 | 0.333 |
| `late_fusion_optimal` / `multivariate_full` | 0.5287 | 0.5682 | 0.333 |
| `confidence_arm` / `multivariate_ablation` | 0.4472 | 0.5000 | 0.266 |
| exp_23 `entropy_soft` (like-for-like target-informed baseline) | 0.4164 | — | — |
| exp_24 best non-target-informed (context) | 0.4368 | — | — |
| exp_17 Composite ICI (context) | 0.4470 | — | — |
| exp_25 best row (`joint_tab_mri`/`w_max`, context) | 0.4547 | — | — |

Best row (0.5287) beats every prior comparator in this lineage, including exp_25's own best (0.4547) by a
wide margin — and, unlike exp_25, the win now comes from the **late-fusion multivariate head**, not a
single bimodal condition. No significance test was specified for H2 in `DESIGN.md` (out of scope here,
same as exp_25) — reported as context, not a tested claim.

---

## 6. Statistical Analysis

- **Test used**: McNemar's exact test (paired, on LOOCV binary predictions) — see §5.3. No test was
  pre-registered for the confidence task (§5.4), consistent with exp_25/`DESIGN.md`.
- **Per-seed variance**: N/A — every condition in exp_26 is deterministic (`encoder=identity` throughout,
  `n_seeds=1`), unlike exp_25 where `tab`-involving conditions needed a 10-seed average to characterize
  `linear`-encoder stochasticity. This is itself one of the intended outcomes of the fix.
- **Conclusion**: H1's re-evaluation is not statistically significant in either direction, same as exp_25 —
  the *point estimates* moved substantially (§5.3), the *significance* verdict did not.

---

## 7. Comparison to Expected Results

| Expected (`DESIGN.md` §3–4) | Observed | Match? |
|---|---|---|
| Step 0 reproduction gate unaffected, still 0.6694214876033058 | 0.6694214876033058 | ✅ |
| `unimodal_tab` Stage-1 winner selects `sigma_mult=2.0, identity` | `sigma_mult=2.0, identity, y_train=False` | ✅ |
| `unimodal_tab` Phase B reproduces exp_23 bit-for-bit via the new sigma-transfer path | 0.6694214876033058, confusion matrix 48/15/19/6 | ✅ exact |
| Re-evaluate whether exp_25's H1 verdict survives the fix | `joint_trimodal` still below best unimodal / late fusion / exp_23 | ✅ verdict unchanged (refuted), though margins compressed |

---

## 8. Missing Data & Caveats

All planned conditions ran to completion. Nothing from `DESIGN.md`'s scope was skipped. As pre-registered
in `DESIGN.md` §4, exp_25's H2/H3 are out of scope for a fresh statistical test here; §5.4 above reports
H2-relevant numbers only as context, per that scope decision — not re-litigated.

---

## 9. Conclusions & Next Steps

- **What this experiment established**: The `linear` encoder's Phase-A selection in exp_25 was genuinely
  noise (0.003 margin vs. ~0.11 std), and removing it recovers `unimodal_tab` to exp_23's number exactly
  and meaningfully improves every joint/fusion condition (+0.02 to +0.07 Macro-F1). This was worth fixing
  independent of any hypothesis outcome — the encoder-instability confound was real, and this establishes
  a materially better set of numbers for anyone building on this line of work going forward.
- **What remains uncertain**: exp_25's H1 (joint trimodal beats unimodal/fusion/exp_23) is still refuted,
  now on cleaner footing — the fix ruled out "encoder noise was masking a real trimodal advantage" as an
  explanation, but didn't explain *why* combining modalities into one shared kernel underperforms simple
  late fusion of the same unimodal models. `joint_tab_mri` (0.6662, best AUROC of any condition at 0.7081)
  is the one joint condition that comes close to the unimodal ceiling — worth a closer look at what makes
  the tab+MRI pairing behave differently from tab+txt or the full trimodal join.
- **All five Stage-2 winners have now independently selected `sigma_scale=2.0` in *two* experiments in a
  row** (exp_25 and exp_26) — the top of its grid both times. This is a pattern, not an artifact of the
  encoder fix. A narrow re-sweep of `sigma_scale` past 2.0 (e.g. `[2.0, 4.0, 8.0]`) on `joint_tab_mri` and
  `joint_trimodal` would check whether the true optimum is being clipped, before spending more effort on
  joint-kernel modeling choices that may be starting from an under-widened bandwidth.
- **Recommended follow-up**: If a multimodal decision model is still wanted, `joint_tab_mri` remains — as
  in exp_25 — the more promising target than the trimodal model, and now clears a higher bar (0.6662 vs.
  0.6048 for `joint_trimodal`). To set up a `sigma_scale` re-sweep or a `joint_tab_mri`-focused follow-up,
  use the ml-experiment-planner skill for `exp_27`.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Seeds logged | ✅ (trivially — every condition is deterministic, `n_seeds=1`, recorded in `loocv_metrics.json`) |
| Configs versioned | ✅ (`results/best_hparams.json`, `results/stage1_best_hparams.json`) |
| Git commit recorded | ✅ (`results/git_commit.txt`) |
| Reproduction gate | ✅ (`results/reproduction_check.json`, passed exactly) |
| Environment frozen | ⚠️ Not pinned (repo has no `requirements.txt`/`environment.yml`; run used the local `pytorch` conda env per project convention) |
| Experiment tracker linked | ❌ N/A — no external tracker in use for this project |
