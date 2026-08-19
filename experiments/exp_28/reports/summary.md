# Experiment Report: BrentMemKDM re-evaluation of the exp_5–exp_8 hard-KNN generation (exp_28)
**Experiment**: experiments/exp_28/ · **Project**: pathology-reasoning · **Report date**: 2026-08-19 ·
**Plan date**: 2026-08-19 · **Status**: Complete

---

## 1. Summary

`exp_5`–`exp_8` (unimodal KNN + late-fusion soft voting) still hold the best published LOOCV Macro-F1 in
the repo (`exp_8`'s equal-weight trimodal fusion, 0.7171). This experiment replaced each modality's KNN
classifier with `BrentMemKDM` — a `MemKDM` whose only fitted quantity is a globally, once-fit RBF bandwidth
per modality, chosen by a continuous Brent search rather than a discrete grid — under the same MCCV→LOOCV
protocol. **H1 is refuted**: neither BrentMemKDM fusion arm (hard 0.6085, soft 0.6452) exceeds exp_8's
0.7171, and neither beats even this experiment's own recomputed, leak-free KNN fusion reference
(0.6613) with statistical significance (McNemar p=0.80 hard, p=1.0 soft). Per-modality, the tabular and
text arms underperform their exact published exp_5/exp_7 counterparts (tab 0.5810 vs. 0.6333; text 0.6951
vs. 0.6988); the MRI arm modestly beats its own recomputed reference (0.5458 vs. 0.5299) but not
significantly (p=0.307). The one bright spot — `fusion_optimal_leakfree` at 0.7068, using MCCV-selected
weights {tab:0.4, mri:0.1, txt:0.5} — comes close to exp_8's 0.7171 without reaching it, and is a secondary
condition, not the primary H1 comparison. `utils/embedding-kit/` is empty in this checkout, so exp_6's and
exp_8's published numbers are not reproducible targets here (§3/§8) — reported alongside recomputed honest
references throughout.

---

## 2. Hypothesis & Verdict

**H1 (primary, from `DESIGN.md` §1).** *"Replacing exp_5–8's per-modality KNN classifiers with
`BrentMemKDM`... will produce a fusion LOOCV Macro-F1 exceeding exp_8's 0.7171, with McNemar significance
against a recomputed leak-free KNN fusion reference."*

**Verdict: ❌ Refuted.** `fusion_equal__hard` = 0.6085 and `fusion_equal__soft` = 0.6452, both below
0.7171 by a wide margin (−0.109 and −0.072). Neither beats `knn_fusion_equal` (0.6613, this run's own
honest recomputed reference) with significance: McNemar `fusion_equal__hard` vs. `knn_fusion_equal`
b=7, c=9, p=0.804; `fusion_equal__soft` vs. `knn_fusion_equal` b=9, c=8, p=1.0.

**H2 (per-modality, secondary, from `DESIGN.md` §1/§8).** tab vs. 0.6333 and txt vs. 0.6988 are the
like-for-like comparisons; mri is judged against `knn_mri_pca` only.

**Verdict: ❌ Not supported for tab/txt, ⚠️ inconclusive-but-modestly-positive for mri.**
`tab__hard` 0.5810 (−0.0523 vs. exp_5), `txt__hard` 0.6951 (−0.0037 vs. exp_7, within noise).
`mri__hard` 0.5458 vs. `knn_mri_pca` 0.5299 (+0.0159), McNemar p=0.307 — not significant.

---

## 3. Experimental Setup (as run)

As described in `DESIGN.md`/`IMPLEMENTATION.md` — no deviations from the approved design.

- **Dataset**: `Data/preprocessed_old/task1/` (old schema), N=88 complete-case cohort, 54 yes / 34 no —
  same cohort as `exp_5`–`27`.
- **Model**: `BrentMemKDM` — `x_train=y_train=w_train=False` always (zero gradient training anywhere);
  the only fitted quantity is `sigma` per modality, chosen once by a global Brent search of mean MCCV
  Macro-F1 over 100 splits (`strategy="nested"` for unimodal, `n_prescan=15, maxiter=20`;
  `strategy="coordinate"` for the S2 joint condition, `n_prescan=7, maxiter=10, max_rounds=5`, 5 MCCV
  splits only per `DESIGN.md` §4). `BrentMemKDM.fit()` builds a zero-trainable-parameter model, so Phase B
  is deterministic — no seed averaging anywhere.
- **Representations**: `tab` — `build_tabular_features(dre_categories=cohort.dre_categories)` (mandatory
  fixed-width one-hot, `DESIGN.md` §4). `mri` — `pca_variance ∈ {None, 0.90}`; winner `raw_l2` (rep=`None`)
  both arms. `txt` — `max_features ∈ {500, 2000, None}`, `pca_variance=0.90`; winner
  `(500, 0.9)` both arms, matching exp_7's own winning `max_features`.
- **Supervision arms**: `hard` (`y ∈ {0,1}`) and `soft` (exp_13-style, clear=1.0/borderline=0.5/uncertain=0.25,
  `target_informed=True` — consumes the confidence label, labeled as such in every soft-arm results row).
- **Reference arms (recomputed, not published)**: `knn_mri_pca` (exp_6's own `pca` pipeline,
  `k=1, uniform, euclidean`) and `knn_fusion_equal` (equal-weight soft vote of exp_5's/exp_7's exact
  winning KNN + `knn_mri_pca`) — built because `utils/embedding-kit/` is empty, so exp_6's
  `embedkit_sup`-based 0.5335 and exp_8's fusion built on it (0.7171) are not reproducible in this
  checkout (`DESIGN.md` §2.1).
- **Step 0 gates**: `scripts/verify_brent_mem_kdm.py` full run — 32/32 checks passed (30 core + check 3
  unimodal-tab-vs-grid + check 5 determinism, both skipped under `--quick` but run here). G1 (exp_5 KNN)
  reproduced `0.6333333333333333` exactly; G2 (exp_7 KNN) reproduced `0.6987539367383268` exactly.
- **Runtime**: 1.9 min full run (100 MCCV splits × 2 supervision arms × up to 3 representations per
  modality for Phase A, 88-fold LOOCV × 6 unimodal/fusion conditions for Phase B) — the fast
  Nadaraya-Watson search path made the sigma search itself negligible relative to feature building.
- **Deviations from plan**: None from the approved `DESIGN.md`/`IMPLEMENTATION.md`.

---

## 4. Code Version

| Item | Value |
|---|---|
| `results/git_commit.txt` (recorded by the run) | `b601b2a` "Extract src/methods and src/evaluation: memory-based multimodal KDM (MemKDM)" |
| **Actual code state at run time** | ⚠️ **Not the same tree.** `src/methods/mem_kdm.py` carries the uncommitted `has_trainable` guard fix (`git status`: `M src/methods/mem_kdm.py`) that `BrentMemKDM.fit()` requires — without it, `loss.backward()` raises on the zero-`requires_grad` models this method always builds (`DESIGN.md` §2.4). `src/methods/brent_mem_kdm.py`, `scripts/verify_brent_mem_kdm.py`, `experiments/exp_25`–`27`, and `experiments/exp_28` itself are all untracked. `record_git_commit` records the tree's last **commit**, not its working-tree diff — this run's actual code is `b601b2a` **plus** the uncommitted `mem_kdm.py` fix plus all the untracked files above. Recommend committing all of it together (`DESIGN.md` §10) before treating this run as archival. |

---

## 5. Results

### 5.1 Phase A — per-modality Brent sigma search (mean over 100 MCCV splits)

| Arm | Modality | Winning rep | sigma* | sigma_mult | mean_macro_f1 | std_macro_f1 |
|---|---|---|---|---|---|---|
| hard | tab | — | 0.18401 | 0.6388 | 0.5748 | 0.1101 |
| hard | mri | raw_l2 | 0.06135 | 0.2264 | 0.5455 | 0.1010 |
| hard | txt | (500, 0.9) | 0.29316 | 0.2280 | 0.6403 | 0.1119 |
| soft | tab | — | 0.12483 | 0.4333 | 0.5766 | 0.1134 |
| soft | mri | raw_l2 | 0.06135 | 0.2264 | 0.5104 | 0.1024 |
| soft | txt | (500, 0.9) | 0.26097 | 0.2030 | 0.6419 | 0.1067 |

`std_macro_f1 ≈ 0.10–0.11` throughout — consistent with `exp_13`/`23`'s prior observation that MCCV
split-to-split variance dwarfs typical inter-config gaps, and the basis for `DESIGN.md` §8's rule that a
Phase-B improvement without McNemar significance is *not established*.

**`sigma_mult` vs. the discrete grid this replaces.** `exp_27`'s frozen grid winners were `tab=0.5,
mri=1.0, txt=0.5` (searched over `{0.25, 0.5, 1.0, 2.0}` for tab, `{0.5, 1.0, 2.0}` for mri/txt).
BrentMemKDM's continuous search landed `tab` in the same range it would have (0.43–0.64), but found
`mri` (0.226) and `txt` (0.20–0.23) **below the smallest grid point either prior experiment ever tried**
(0.5) — the opposite direction from `exp_25`'s finding that all five Stage-2 conditions pinned the grid's
*top* edge (`sigma_scale=2.0`). Taken together, both findings say the same thing: a `{0.25, 0.5, 1.0, 2.0}`-
style discrete grid was too coarse and too narrowly centered for this problem, in both directions,
depending on modality — which is exactly the motivation `brent_mem_kdm.py`'s docstring gives for
building a continuous search in the first place.

### 5.2 S1 — leak-free fusion-weight search (MCCV validation probs only, hard arm)

Simplex weights selected by scoring the 100 Stage-1-winner MCCV validation probabilities (never LOOCV
output): **{tab: 0.40, mri: 0.10, txt: 0.50}**, mean MCCV Macro-F1 0.6610. Down-weights MRI, matches the
general pattern in `exp_5`–`8`/`23`–`27` that MRI is the weakest modality in this cohort, but — unlike
`exp_27`'s fusion search, which collapsed to pure text (weight 1.0) — still assigns meaningful weight to
both tab and txt.

### 5.3 S2 — joint_trimodal (product-kernel, coordinate search, hard arm)

Frozen sigmas: `tab=1.699 (mult 5.84), mri=0.0629 (mult 0.231), txt=4.081 (mult 3.17)`, MCCV search
score 0.5588 (153 evaluations, 5 splits per `DESIGN.md` §4's fixed cheap budget). LOOCV Macro-F1
**0.5372** — the weakest condition in the entire experiment, continuing the `exp_25`/`26`/`27` pattern
that a single shared product-kernel across all three modalities underperforms both unimodal and
late-fusion alternatives. Not scored against H1 (context only, `DESIGN.md` §8).

### 5.4 Phase B — LOOCV, binary decision task (primary metric: Macro-F1)

| Condition | Macro-F1 | Accuracy | AUROC | tp/tn/fp/fn | Reference | Reference Macro-F1 | Δ |
|---|---|---|---|---|---|---|---|
| `tab__hard` | 0.5810 | 0.6364 | 0.6471 | 44/12/22/10 | exp_5 (published) | 0.6333 | **−0.0523** |
| `mri__hard` | 0.5458 | 0.6136 | 0.5180 | 44/10/24/10 | `knn_mri_pca` (recomputed) | 0.5299 | +0.0159 |
| `txt__hard` | 0.6951 | 0.7159 | 0.6906 | 43/20/14/11 | exp_7 (published) | 0.6988 | −0.0037 |
| `fusion_equal__hard` | 0.6085 | 0.6705 | 0.7168 | 47/12/22/7 | `knn_fusion_equal` (recomputed) | 0.6613 | −0.0528 |
| `fusion_optimal_leakfree__hard` | 0.7068 | 0.7386 | 0.7467 | 47/18/16/7 | `knn_fusion_equal` (recomputed) | 0.6613 | +0.0455 |
| `joint_trimodal__hard` | 0.5372 | 0.6023 | 0.5158 | 43/10/24/11 | — | — | — |
| `tab__soft` ⚠️ target-informed | 0.6085 | 0.6705 | 0.6656 | 47/12/22/7 | exp_5 (published, hard) | 0.6333 | −0.0248 |
| `mri__soft` ⚠️ target-informed | 0.5191 | 0.5909 | 0.4777 | 43/9/25/11 | `knn_mri_pca` (recomputed) | 0.5299 | −0.0108 |
| `txt__soft` ⚠️ target-informed | 0.7068 | 0.7386 | 0.7380 | 47/18/16/7 | exp_7 (published, hard) | 0.6988 | +0.0080 |
| `fusion_equal__soft` ⚠️ target-informed | 0.6452 | 0.7045 | 0.7309 | 49/13/21/5 | `knn_fusion_equal` (recomputed) | 0.6613 | −0.0161 |
| exp_8 `Equal-Trimodal-Fusion` (published, not reproducible — §8) | 0.7171 | 0.7500 | 0.7821 | 48/18/16/6 | — | — | — |

**McNemar tests** (BrentMemKDM condition vs. its reference, exact binomial on paired LOOCV predictions):

| Comparison | b | c | p-value |
|---|---|---|---|
| `tab__hard` vs. `knn_tab` (exp_5) | 4 | 8 | 0.388 |
| `mri__hard` vs. `knn_mri_pca` | 15 | 9 | 0.307 |
| `txt__hard` vs. `knn_txt` (exp_7) | 8 | 8 | 1.000 |
| `fusion_equal__hard` vs. `knn_fusion_equal` | 7 | 9 | 0.804 |
| `fusion_optimal_leakfree__hard` vs. `knn_fusion_equal` | 10 | 6 | 0.454 |
| `tab__soft` vs. `knn_tab` | 5 | 6 | 1.000 |
| `mri__soft` vs. `knn_mri_pca` | 15 | 11 | 0.557 |
| `txt__soft` vs. `knn_txt` | 10 | 8 | 0.815 |
| `fusion_equal__soft` vs. `knn_fusion_equal` | 9 | 8 | 1.000 |

**No comparison in the entire experiment reaches significance** (all p > 0.3, most p > 0.5) — every
BrentMemKDM condition's difference from its KNN reference is consistent with chance under this cohort's
LOOCV sample size, in both directions.

---

## 6. Statistical Analysis

- **Test used**: McNemar's exact test (paired, on LOOCV binary predictions), per condition vs. its
  reference — see §5.4.
- **Per-seed variance**: N/A — every `BrentMemKDM` condition is deterministic (zero trainable parameters
  by construction; `DESIGN.md` §1 and verified in `scripts/verify_brent_mem_kdm.py` check 5).
- **Conclusion**: H1 is refuted both on the point estimate (0.6085/0.6452 vs. target 0.7171) and has no
  significant McNemar result to fall back on even against the honest recomputed reference (p=0.804/1.0).
  H2 is not supported for tab/txt (both below their published targets, and neither McNemar test reaches
  significance); mri shows a small positive point estimate that is not statistically distinguishable from
  its reference (p=0.307).

---

## 7. Comparison to Expected Results

| Expected (`DESIGN.md` §8) | Observed | Match? |
|---|---|---|
| At least one C4 arm exceeds exp_8's 0.7171 with McNemar significance vs. `knn_fusion_equal` | Neither arm exceeds 0.7171 (0.6085/0.6452); neither McNemar test is significant | ❌ |
| tab vs. 0.6333, txt vs. 0.6988 like-for-like | tab −0.0523, txt −0.0037 (both below) | ❌ |
| mri judged against `knn_mri_pca` only | +0.0159, not significant (p=0.307) | ⚠️ inconclusive |
| Phase-B improvement without significant McNemar reported as *not established*, not a win | Applied throughout §5.4/§6 — no win claimed anywhere despite `fusion_optimal_leakfree`'s positive point estimate | ✅ |
| Step 0 gates (G0/G1/G2) pass before trusting any BrentMemKDM number | All three passed exactly (`results/reproduction_gates.json`) | ✅ |
| `sigma_mult` reported against exp_27/exp_25 grid patterns as secondary readout | Done, §5.1 — found the opposite edge-of-grid pattern from exp_25 | ✅ |

---

## 8. Missing Data & Caveats

All planned Step-0 gates, Phase A searches, Phase B conditions (C1–C4 × hard/soft, S1, S2), reference
arms, and McNemar comparisons ran to completion — nothing from `DESIGN.md`'s scope was skipped.

- ⚠️ **Uncommitted code at run time** (§4): `src/methods/mem_kdm.py`'s fix and all of `src/methods/
  brent_mem_kdm.py`, `scripts/verify_brent_mem_kdm.py`, `experiments/exp_25`–`27`, `experiments/exp_28`
  are uncommitted/untracked. `results/git_commit.txt` does not fully describe the code that produced
  these results. Recommend committing all of it together (`DESIGN.md` §10) before relying on this run as
  an archival reference point.
- **exp_6/exp_8's published numbers are not reproducible in this checkout** (`utils/embedding-kit/` is
  empty) — every C2/C4 comparison in §5.4 uses this run's own recomputed `knn_mri_pca`/`knn_fusion_equal`
  as the honest reference; exp_6's 0.5335 and exp_8's 0.7171 are reported as literal published constants
  for context, not as reproduced numbers.
- **Soft-arm rows consume the confidence label** (`target_informed=True`) — flagged inline in §5.4 with
  ⚠️. `txt__soft`'s +0.0080 over exp_7's published 0.6988 should not be read as a pure method gain; it may
  partly reflect the additional confidence-label information.

---

## 9. Conclusions & Next Steps

- **What this experiment established**: replacing exp_5–8's KNN classifiers with a globally-fit,
  continuous-bandwidth `BrentMemKDM` does not recover or exceed the hard-KNN generation's numbers in this
  cohort. The primary fusion hypothesis (H1) is refuted on both the point estimate and every available
  significance test. This extends the `exp_25`/`26`/`27` finding — joint and fused KDM-family models
  underperforming simpler baselines — to a fourth, methodologically distinct model (continuous Brent
  sigma vs. gradient-trained sigma) and to a *different* baseline family (hard KNN, not Fuzzy KNN).
- **What this experiment established, structurally**: the continuous search does find sigmas the
  discrete `{0.25, 0.5, 1.0, 2.0}`-style grids in `exp_23`–`27` could not reach — here at the *low* edge
  for mri/txt (§5.1), complementing `exp_25`'s high-edge (`sigma_scale=2.0`) finding. The search
  mechanism works as designed (Step 0's 32/32 checks, exact G1/G2 reproduction); the negative H1 result is
  not an artifact of the search failing to explore properly.
- **What remains uncertain**: whether BrentMemKDM's weakness relative to KNN reflects something structural
  to memory-based single-bandwidth models on this cohort (echoing `exp_23`'s own "Partial" verdict against
  Fuzzy KNN), or whether a richer per-modality parameterization (e.g. `mem_kdm.py`'s trainable `x`/`y`/`w`,
  deliberately excluded here per `DESIGN.md` §9) would close the gap — this experiment cannot distinguish
  those explanations, since only `sigma` was ever free to vary.
- **`fusion_optimal_leakfree`'s 0.7068** (§5.2/§5.4) is the closest any condition here came to exp_8's
  0.7171, and unlike `exp_27`'s fusion search it did not collapse to a single modality — a mild positive
  signal for leak-free-weighted fusion specifically, worth a dedicated look if this line of work continues,
  but not sufficient on its own to revisit H1's verdict (it is a secondary condition, and still short of
  the target).
- **Recommended follow-up**: if BrentMemKDM is worth pursuing further against the hard-KNN generation, the
  natural next step is isolating whether `mri`'s below-grid sigma and `fusion_optimal_leakfree`'s
  near-target score generalize with a wider `bounds_mult` search or a richer representation grid (this run
  used only 2–3 representations per modality, `DESIGN.md` §4) — not a re-run of this exact design. To set
  up that follow-up, use the ml-experiment-planner skill for `exp_29`.

---

## 10. Reproducibility Record

| Item | Status |
|---|---|
| Seeds logged | ✅ (trivially — every `BrentMemKDM` condition is deterministic, zero trainable parameters) |
| Configs versioned | ✅ (`results/stage1_best_hparams.json`, `results/fusion_weights_leakfree.json`, `results/joint_trimodal_search.json`) |
| Git commit recorded | ⚠️ Recorded (`results/git_commit.txt`) but incomplete — see §4/§8, uncommitted code not reflected |
| Reproduction gates | ✅ (`results/reproduction_gates.json` — G0/G1/G2 all passed exactly) |
| Environment frozen | ⚠️ Not pinned (repo has no `requirements.txt`/`environment.yml`; run used the local `pytorch` conda env per project convention) |
| Experiment tracker linked | ❌ N/A — no external tracker in use for this project |
