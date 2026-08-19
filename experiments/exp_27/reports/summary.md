# Experiment Report: MemKDM with training restricted to sigma-only, frozen unimodally (exp_27)
**Experiment**: experiments/exp_27/ · **Project**: pathology-reasoning · **Report date**: 2026-08-19 ·
**Plan date**: 2026-08-19 · **Status**: Complete

---

## 1. Summary

`exp_23`–`26`'s `MemKDM` trained some subset of `{x, y, w, sigma}` via gradient descent in **both** Phase A
(MCCV hyperparameter search) and Phase B (LOOCV final evaluation) — so Phase B was never a pure evaluation
of a frozen model, it was 88 independent label-supervised optimization runs. `exp_27` restricts training to
the one place it's well-defined for a memory-based model: `sigma`, fit once per modality, unimodally, in
Stage 1, and frozen everywhere else (`x_train=y_train=w_train=False` throughout; `sigma_scale` dropped as
redundant with Stage 1's own `sigma_mult` grid). Phase B then does zero gradient training of any kind. This
required one small `src/methods/mem_kdm.py` fix (§4) to handle the now-legitimate case of a model with no
gradient-requiring parameters. The fix worked as designed — Phase B is now a genuine frozen-parameter
evaluation — but it cost real decision-task performance: `unimodal_tab` dropped from `exp_26`'s 0.6694 to
**0.5993**, and every joint condition dropped further still. `joint_trimodal` (0.4943) now loses to
`late_fusion_optimal` (0.6174) with statistical significance for the first time in this lineage (McNemar
p=0.034, vs. `exp_25`/`26`'s p=0.629/0.118) — H1 is refuted more decisively than before, but on a
substantially weaker set of numbers overall. The confidence-prediction task moved the other way: its best
row improved to 0.5630, beating every prior comparator including `exp_26`'s 0.5287.

---

## 2. Hypothesis & Verdict

**H (single, methodological, from `DESIGN.md`).** *"Restricting training to sigma alone (fit once
unimodally in Stage 1, frozen everywhere else) makes Phase B a genuine frozen-parameter evaluation... Under
this protocol, does exp_25/26's H1 verdict (joint_trimodal does not beat the best unimodal model, leak-free
late fusion, or exp_23's 0.6694) still hold?"*

**Verdict: ✅ Protocol change confirmed working / H1 remains ❌ Refuted, now with statistical significance.**
The mechanical part of H is confirmed: Phase B runs with zero gradient training (verified via the
`src/methods/mem_kdm.py` fix, §4, and by `mean_sigma` in `stage1_best_hparams.json` being the single value
reused across all 88 LOOCV folds — no `_sigma_from_knn` recomputation, no `kernel_trainable` branch, per
`DESIGN.md` §2.3). The re-evaluation half shows H1 still refuted: `joint_trimodal` (0.4943) is below
`unimodal_txt` (0.6174), `late_fusion_optimal` (0.6174, which collapsed to pure-text weighting — §5.1), and
`exp_23` (0.6694). Unlike `exp_25`/`exp_26`, both McNemar comparisons now reach significance: `joint_trimodal`
vs. `late_fusion_optimal` p=0.034 (b=12, c=26) and vs. `exp_23` p=0.004 (b=5, c=20) — a cleaner negative
result, but arrived at via models that are themselves substantially weaker than `exp_26`'s (§5.3).

---

## 3. Experimental Setup (as run)

As described in `DESIGN.md`/`IMPLEMENTATION.md` — no deviations from the approved design. One required
`src/` change beyond the design's original scope-check (§4).

- **Dataset**: `Data/preprocessed_old/task1/` (old schema), N=88 complete-case cohort, 54 yes / 34 no — same
  as `exp_23`–`26`.
- **Model**: `MemKDM`, `encoder="identity"` everywhere, `x_train=y_train=w_train=False` throughout (Phase A
  and Phase B alike). Sigma is the only ever-trained parameter, and only in Stage 1 (unimodal, `epochs=300`,
  `lr=1e-3`, Adam) — Stage 2 and Phase B use `KernelSpec(sigma=<Stage 1's mean fitted sigma>, trainable=False)`,
  zero gradient steps.
- **Stage 1** (unimodal Phase A, 100-split MCCV): 4/6/9 configs for tab/mri/txt (down from `exp_26`'s
  8/12/18 — `y_train` dropped as a grid dimension). Each fit additionally records its post-training fitted
  `sigma` (`MemKDM.kernel_params()`); the winning config's `mean_fitted_sigma` across the 100 splits is
  "the sigma learned in Phase A," frozen for Stage 2/Phase B.
- **Stage 2** (joint conditions): no grid search (`sigma_scale`, `x_train`, `y_train`, `kernel_trainable`
  all removed — nothing left to search once sigma is transferred from Stage 1 unmodified). Each of the 5
  conditions (4 decision-task + `confidence_arm`) runs a single deterministic MCCV pass across the 100
  splits purely to report mean/std Macro-F1 — down from `exp_26`'s 24-config grid.
- **Runtime**: 8.7 min full run (`exp_26`: 63.3 min) — smaller grids (19 Stage-1 configs total vs. 38, no
  Stage-2 grid at all) and zero-epoch Phase B fits both contribute.
- **Deviations from plan**: None from the approved `DESIGN.md`/`IMPLEMENTATION.md`.

---

## 4. Code Version

| Item | Value |
|---|---|
| `results/git_commit.txt` (recorded by the run) | `b601b2a` "Extract src/methods and src/evaluation: memory-based multimodal KDM (MemKDM)" |
| **Actual code state at run time** | ⚠️ **Not the same tree.** `src/methods/mem_kdm.py` carries an uncommitted fix (`git status`: `M src/methods/mem_kdm.py`) required for this experiment (§3, `DESIGN.md` §2.4): `MemKDM.fit()` now guards optimizer construction/the training loop on `any(p.requires_grad for p in model.parameters())`, since `c_x`/`c_y`/`c_w`/`raw_sigma` are always `nn.Parameter` regardless of the `*_train`/`trainable` flags (only `requires_grad` varies) — with everything fixed `False`/non-trainable, the old unconditional code raised `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` on `loss.backward()`. `record_git_commit` records the tree's last **commit**, not its working-tree diff, so this run's actual code is `b601b2a` **plus** that uncommitted fix. `experiments/exp_27/` itself is also untracked (`?? experiments/exp_27/`). Recommend committing `src/methods/mem_kdm.py` and `experiments/exp_27/` together before treating this run as archival — see §8. |

---

## 5. Results

### 5.1 Stage 1 — unimodal winners (MCCV Phase A, mean over 100 splits)

| Modality | Config | mean_macro_f1 | std_macro_f1 | mean_sigma (frozen for Phase B) |
|---|---|---|---|---|
| `tab` | `sigma_mult=0.5` | 0.5671 | 0.1103 | 0.10978 |
| `mri` | `sigma_mult=1.0, rep=pca90_l2` | 0.4769 | 0.1008 | 0.41835 |
| `txt` | `sigma_mult=0.5, rep=(2000,0.9)` | 0.5995 | 0.1067 | 0.54623 |

Leak-free fusion-weight search (231-point simplex grid over the Stage-1 winners' stashed validation
probabilities) collapsed to **tab=0, mri=0, txt=1.0** (`exp_26`: tab=0.50/mri=0.05/txt=0.45) — under this
protocol, blending never beat pure text on the Stage-1 validation folds, so `late_fusion_optimal` is
identical to `unimodal_txt` in every downstream number.

### 5.2 Stage 2 — joint conditions (MCCV Phase A, mean over 100 splits, no selection — single config)

| Condition | mean_macro_f1 | std_macro_f1 |
|---|---|---|
| `tab_mri` | 0.5071 | 0.1146 |
| `tab_txt` | 0.5385 | 0.1116 |
| `mri_txt` | 0.5120 | 0.1129 |
| `tab_mri_txt` | 0.4692 | 0.0967 |
| `confidence_arm` (`label_smoothing=0.10`) | 0.4692 | 0.0967 |

No grid to report a "winner" from — these are the only configuration each condition ever runs, per
`DESIGN.md` §2.5.

### 5.3 Phase B — LOOCV, binary decision task (primary metric: Macro-F1)

| Condition | Macro-F1 | AUROC | Brier | exp_26 Macro-F1 | Δ vs. exp_26 |
|---|---|---|---|---|---|
| `unimodal_tab` | 0.5993 | 0.6683 | 0.2448 | 0.6694 | **−0.0701** |
| `unimodal_mri` | 0.4657 | 0.4172 | 0.2820 | 0.5152 | −0.0495 |
| `unimodal_txt` | 0.6174 | 0.6683 | 0.2337 | 0.6081 | +0.0093 |
| `joint_tab_mri` | 0.5394 | 0.6138 | 0.2401 | 0.6662 | **−0.1268** |
| `joint_tab_txt` | 0.5726 | 0.6612 | 0.2350 | 0.6081 | −0.0355 |
| `joint_mri_txt` | 0.4924 | 0.4880 | 0.2688 | 0.5200 | −0.0276 |
| `joint_trimodal` | 0.4943 | 0.6133 | 0.2531 | 0.6048 | **−0.1105** |
| `late_fusion_equal` | 0.5210 | 0.6754 | 0.2245 | 0.5616 | −0.0406 |
| `late_fusion_optimal` | 0.6174 | 0.6683 | 0.2337 | 0.6648 | −0.0474 |
| `confidence_arm` (context, excluded from H1) | 0.4943 | 0.6133 | 0.2531 | 0.6048 | −0.1105 |
| exp_23 Arm B (context, not re-run) | 0.6694 | 0.6498 | 0.2263 | — | — |

Every condition dropped relative to `exp_26` except `unimodal_txt` (+0.009, within noise). `unimodal_tab`'s
−0.070 drop is the load-bearing signal: this is the same feature builder, the same encoder, the same
`sigma_mult` grid values as `exp_26`, differing only in `w_train` (now always `False`) and in `sigma` never
being gradient-refined per LOOCV fold — so the components this experiment removed from training were doing
real work for `tab` specifically, not fitting noise. `joint_tab_mri` fell hardest (−0.127), losing the best
position it held in `exp_26` (0.6662, closest to the unimodal ceiling there).

**McNemar tests:**

| Comparison | b | c | p-value | exp_26 (b, c, p) |
|---|---|---|---|---|
| `joint_trimodal` vs. `late_fusion_optimal` | 12 | 26 | **0.0336** | 4, 11, 0.118 |
| `joint_trimodal` vs. exp_23 soft | 5 | 20 | **0.0041** | 7, 13, 0.263 |

Both comparisons reach significance here for the first time in this lineage (`exp_25`: p=0.629/0.115;
`exp_26`: p=0.118/0.263) — `late_fusion_optimal` and `exp_23` beat `joint_trimodal` by more than chance
under this protocol. This is a stronger, cleaner refutation of H1 than either prior experiment produced, but
it comes from a `joint_trimodal` that is itself weaker (0.4943 vs. `exp_26`'s 0.6048) — the significance
reflects a wider gap on a lower baseline, not `joint_trimodal` improving in absolute terms.

### 5.4 Confidence-prediction task (secondary, H2-style — context, not part of H1)

| Condition / head | Macro-F1 | Accuracy |
|---|---|---|
| `unimodal_tab` / `multivariate_ablation` (`{h_total, log_marginal}`) (best) | **0.5630** | 0.6591 |
| `joint_tab_txt` / `multivariate_ablation` | 0.5116 | 0.6250 |
| `unimodal_tab` / `1d` (`h_total`) | 0.4974 | 0.5568 |
| `late_fusion_equal` and `late_fusion_optimal` / `1d` (`tab__h_total`) | 0.4974 | 0.5568 |
| `unimodal_tab` / `multivariate_full` (all 7 particle signals) | 0.4603 | 0.5909 |
| exp_26 best row (`late_fusion_equal`/`multivariate_full`, context) | 0.5287 | — |
| exp_25 best row (`joint_tab_mri`/`w_max`, context) | 0.4547 | — |
| exp_23 `entropy_soft` (like-for-like target-informed baseline) | 0.4164 | — |
| exp_24 best non-target-informed (context) | 0.4368 | — |
| exp_17 Composite ICI (context) | 0.4470 | — |

Best row (0.5630) beats every prior comparator in this lineage, including `exp_26`'s own best (0.5287) —
and, unlike `exp_25`/`26`, the win now comes from `unimodal_tab`'s own particle signals (the ablation head,
not the full 7-signal set), not a joint or late-fusion condition. No significance test was specified for H2
in `DESIGN.md` (out of scope here, same as `exp_25`/`26`) — reported as context, not a tested claim.

---

## 6. Statistical Analysis

- **Test used**: McNemar's exact test (paired, on LOOCV binary predictions) — see §5.3. No test was
  pre-registered for the confidence task (§5.4), consistent with `exp_25`/`26`/`DESIGN.md`.
- **Per-seed variance**: N/A — every condition is deterministic (`n_seeds=1`, no `linear` encoder, and now
  additionally zero gradient training in Phase B — there is no stochastic element left to vary across
  seeds).
- **Conclusion**: unlike `exp_25`/`exp_26`, H1's re-evaluation **is** statistically significant here — both
  McNemar comparisons reject the null at p<0.05 (§5.3).

---

## 7. Comparison to Expected Results

| Expected (`DESIGN.md` §1, §4) | Observed | Match? |
|---|---|---|
| Step 0 reproduction gate unaffected, still 0.6694214876033058 | 0.6694214876033058 | ✅ |
| Phase B is a genuine frozen-parameter evaluation (zero gradient training) | Confirmed via `src/` fix (§4) and single frozen `mean_sigma` reused across all 88 folds | ✅ |
| `unimodal_tab` expected to diverge from 0.6694 under the new protocol, not required to match | 0.5993 (diverges, as flagged in `DESIGN.md` §4 as an acceptable outcome either way) | ✅ (divergence, not a failure) |
| Re-evaluate whether exp_25/26's H1 verdict survives | `joint_trimodal` still below best unimodal / late fusion / exp_23, now with McNemar significance | ✅ verdict unchanged (refuted), evidence strengthened |

---

## 8. Missing Data & Caveats

All planned Stage 1/Stage 2/Phase B/confidence-task conditions ran to completion — nothing from
`DESIGN.md`'s scope was skipped.

- ⚠️ **Uncommitted code at run time** (§4): `src/methods/mem_kdm.py`'s `MemKDM.fit()` fix and all of
  `experiments/exp_27/` are uncommitted. `results/git_commit.txt` therefore does not fully describe the code
  that produced these results. Recommend committing both together before relying on this run as an
  archival reference point.
- **Weaker absolute numbers, by design and by consequence.** `DESIGN.md` §1 explicitly accepted a bias
  (the frozen `mean_sigma` is averaged over MCCV training folds that collectively cover the full cohort) in
  exchange for a genuine frozen-parameter Phase B. What was *not* pre-registered as an expected consequence
  is how much decision-task performance would drop from disabling `w`/per-fold-sigma training (§5.3) — this
  is a genuine finding, not a caveat about missing data, but worth flagging as a limit on how far this
  report's numbers should be read as "the ceiling for a leaner MemKDM," rather than "evidence that `w`
  training was pure noise" (§9).

---

## 9. Conclusions & Next Steps

- **What this experiment established**: the protocol fix works exactly as designed — Phase B is now a
  genuine evaluation of parameters frozen at Phase A, with zero gradient training anywhere in Phase B
  (verified structurally, not just by the numbers). Re-evaluating H1 under this protocol gives a *cleaner*
  negative result than `exp_25`/`26` — both McNemar tests now reach significance — so the corrected protocol
  doesn't rescue the trimodal joint model; if anything it makes the case against it stronger.
- **What remains uncertain**: whether the performance lost relative to `exp_26` (`unimodal_tab` −0.070,
  `joint_tab_mri` −0.127) reflects `w`/per-fold-sigma training doing genuine, generalizable work, or
  reflects the specific bias introduced by freezing `sigma` from an MCCV-training-fold average rather than
  something closer to per-fold data. This experiment cannot distinguish those two explanations on its own —
  it only shows the combined effect.
- **The fusion-weight collapse to pure text (§5.1)** is itself notable: under a protocol where nothing but
  sigma is ever tuned, the leak-free search found no combination of modalities that beat text alone on the
  Stage-1 validation folds — a stronger version of `exp_25`/`26`'s "combining modalities doesn't help"
  finding, now visible even before Stage 2 or Phase B.
- **Recommended follow-up**: if the goal is disentangling "how much did disabling `w` training cost" from
  "how much did disabling per-fold sigma refinement cost," a follow-up experiment could freeze only one of
  the two (e.g. keep `w_train=True` in Phase A/Stage 1 but still transfer a single frozen sigma to Phase B)
  to isolate the effect. Otherwise, if a multimodal decision model is still wanted, this run gives no
  encouragement for `joint_tab_mri`/`joint_trimodal` over unimodal text or the collapsed late-fusion
  baseline — a materially different picture than `exp_26`'s recommendation to look at `joint_tab_mri`. To
  set up either follow-up, use the ml-experiment-planner skill for `exp_28`.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Seeds logged | ✅ (trivially — every condition is deterministic, `n_seeds=1`, no stochastic element remains) |
| Configs versioned | ✅ (`results/best_hparams.json`, `results/stage1_best_hparams.json`) |
| Git commit recorded | ⚠️ Recorded (`results/git_commit.txt`) but incomplete — see §4/§8, uncommitted `src/` fix not reflected |
| Reproduction gate | ✅ (`results/reproduction_check.json`, passed exactly, unaffected by this experiment's own model) |
| Environment frozen | ⚠️ Not pinned (repo has no `requirements.txt`/`environment.yml`; run used the local `pytorch` conda env per project convention) |
| Experiment tracker linked | ❌ N/A — no external tracker in use for this project |
