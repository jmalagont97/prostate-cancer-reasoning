# Experiment Report: BrentMemKDM k-NN Truncation Sweep over the exp_28 Hard-KNN Generation

**Experiment**: experiments/exp_29/
**Project**: pathology-reasoning (CHIMERA 2026, Task 1)
**Report date**: 2026-08-19
**Plan date**: 2026-08-19
**Author**: TBD
**Status**: Complete

---

## 1. Summary

exp_29 added a discrete `knn_k` truncation dimension (`{1, 3, 5, 10, 20, 40, None}`) to `BrentMemKDM`, sweeping it alongside representation and sigma in Phase A, to test whether an intermediate neighborhood size between exp_28's 1-NN endpoint and its whole-memory endpoint could beat exp_8's late-fusion Macro-F1 of 0.7171. It doesn't: the best fusion arm (`fusion_equal__hard`, 0.6165) falls well short of 0.7171, and none of the four primary fusion/reference comparisons reach McNemar significance (H1 refuted). The secondary mechanism check (H2) is directionally mixed but statistically silent — 6 of 8 condition/arm pairs beat exp_28's own `knn_k=None` number, none by a significant margin (all McNemar p ≥ 0.39), so per the plan's own decision rule the truncation mechanism is **not established** to help, independent of H1.

---

## 2. Hypothesis & Verdict

**H1 (primary, from DESIGN.md §1):** "At least one `BrentMemKDM(knn_k=k)` fusion arm (hard or soft) will produce a LOOCV Macro-F1 exceeding exp_8's 0.7171, with `mcnemar_exact` significance against a recomputed leak-free KNN fusion reference."

**Verdict:** ❌ Refuted.

**Evidence:** `fusion_equal__hard` = 0.6165 (knn_k: tab=3, mri=40, txt=3) and `fusion_equal__soft` = 0.6400 (knn_k: tab=3, mri=1, txt=None) — both below the 0.7171 target, and neither beats `knn_fusion_equal` (0.6613, the recomputed leak-free KNN reference) with significance (McNemar p=0.804 hard, p=1.0 soft). The two secondary conditions come closer but are explicitly out of scope for H1 per DESIGN.md §8: `fusion_optimal_leakfree__hard` = 0.7068 (0.0103 short of target — identical to exp_28's own value) and `joint_trimodal__hard` = 0.5372.

**H2 (secondary, from DESIGN.md §1):** "For at least one modality, the best `knn_k` found in Phase A improves mean MCCV Macro-F1 over exp_28's own `knn_k=None` result for that modality/arm."

**Verdict:** ⚠️ Inconclusive / not established. Point estimates improve for 6 of 8 condition/arm pairs (see §5.3), but every one of those improvements is well inside McNemar noise (all p ≥ 0.39, most p > 0.6), and two pairs (`mri__hard`, `fusion_equal__soft`) move in the *wrong* direction. Per DESIGN.md §8's own standard — "a Phase-B improvement without a significant McNemar result is reported as not established, not as a win" — H2 is not established.

---

## 3. Experimental Setup (as run)

As described in DESIGN.md. `IMPLEMENTATION.md` contains no deviation notes, and the run log matches the plan's Step-0 → Phase A → Phase B → S1/S2 → metrics/McNemar → figures sequence exactly.

- **Dataset**: `Data/preprocessed_old/task1/`, N=88 complete-case cohort (yes=54, no=34), same cohort as exp_5–exp_28.
- **Model**: `BrentMemKDM` (Nadaraya-Watson RBF memory model, continuous Brent-searched bandwidth) with the new `knn_k` per-query k-NN truncation option (`k_eff = min(k, n_train)`), swept over `{1, 3, 5, 10, 20, 40, None}`.
- **Search**: Phase A = 100 MCCV splits, grid over `(representation, knn_k)` × 1-D Brent sigma search per grid point; selection by mean Macro-F1 (descending), std tie-break (ascending). Phase B = 88-fold LOOCV with the frozen winning `(representation, knn_k, sigma)` triple, refit per fold.
- **Hardware / runtime**: local `pytorch` conda env; full run completed in 2.1 minutes.
- **Deviations from plan**: None. One implementation bug was found and fixed mid-session (see §8) but did not require a design or build change — only a one-line fix to shared evaluation code.

---

## 4. Code Version

| Component | Git commit | Commit message |
|-----------|-----------|-----------------|
| `results/git_commit.txt` (recorded by the run) | `189926fc920b6ff86f48c3636b672c054c466f17` | Add knn_k option to BrentMemKDM: per-query k-NN truncation of the memory |
| Repo HEAD at time of this report | `189926fc920b6ff86f48c3636b672c054c466f17` | (same — matches) |

⚠️ Note: the `binary_metrics` fix described in §8 was applied to `src/evaluation/metrics.py` as an uncommitted working-tree edit for this run (`np.clip` before `brier_score_loss`). It is not yet committed — `results/git_commit.txt` names `189926f`, which predates that fix. The fix must be committed before this git-commit record can be trusted as a complete description of the code that produced these results.

---

## 5. Results

### 5.1 Primary Metric — LOOCV Macro-F1 (Phase B, N=88 all rows)

| Condition | Arm | `knn_k` (winning) | Macro-F1 | vs. target 0.7171 |
|---|---|---|---|---|
| `fusion_equal` | hard | tab=3, mri=40, txt=3 | 0.6165 | −0.1006 |
| `fusion_equal` | soft | tab=3, mri=1, txt=None | 0.6400 | −0.0771 |
| `fusion_optimal_leakfree` (S1, context) | hard | (per-modality winners above) | 0.7068 | −0.0103 |
| `joint_trimodal` (S2, context) | hard | n/a (`knn_k=None`, joint kernel) | 0.5372 | −0.1799 |
| `unimodal_tab` | hard | 3 | 0.6333 | — |
| `unimodal_tab` | soft | 3 | 0.6848 | — |
| `unimodal_mri` | hard | 40 | 0.5286 | — |
| `unimodal_mri` | soft | 1 | 0.5269 | — |
| `unimodal_txt` | hard | 3 | **0.7091** | — |
| `unimodal_txt` | soft | None | 0.7068 | — |
| `knn_tab` (reference) | — | n/a | 0.6333 | — |
| `knn_mri_pca` (reference) | — | n/a | 0.5299 | — |
| `knn_txt` (reference) | — | n/a | 0.6988 | — |
| `knn_fusion_equal` (reference) | — | n/a | 0.6613 | — |

> Success threshold from plan (H1): a `fusion_equal` arm ≥ 0.7171 with McNemar significance. **Not met by either arm.** The single best Phase-B result across the whole experiment is `unimodal_txt__hard` (0.7091), a unimodal condition — not the fusion arm H1 requires.

### 5.2 McNemar — H1 (vs. recomputed KNN references)

| Comparison | b | c | p-value |
|---|---|---|---|
| `fusion_equal__hard` vs `knn_fusion_equal` | 7 | 9 | 0.804 |
| `fusion_equal__soft` vs `knn_fusion_equal` | 11 | 12 | 1.0 |
| `fusion_optimal_leakfree__hard` vs `knn_fusion_equal` (context) | 12 | 8 | 0.503 |
| `tab__hard` vs `knn_tab` | 0 | 0 | 1.0 |
| `tab__soft` vs `knn_tab` | 10 | 8 | 0.815 |
| `mri__hard` vs `knn_mri_pca` | 13 | 9 | 0.523 |
| `mri__soft` vs `knn_mri_pca` | 10 | 8 | 0.815 |
| `txt__hard` vs `knn_txt` | 8 | 7 | 1.0 |
| `txt__soft` vs `knn_txt` | 10 | 8 | 0.815 |

No comparison reaches p < 0.05.

### 5.3 McNemar — H2 (vs. exp_28's own `knn_k=None` predictions)

| Condition/arm | exp_28 `knn_k=None` | exp_29 best `knn_k` | Δ | Direction | McNemar b/c | p-value |
|---|---|---|---|---|---|---|
| `tab__hard` | 0.5810 | 0.6333 (k=3) | +0.0523 | ↑ | 8/4 | 0.388 |
| `tab__soft` | 0.6085 | 0.6848 (k=3) | +0.0763 | ↑ | 9/6 | 0.607 |
| `mri__hard` | 0.5458 | 0.5286 (k=40) | −0.0172 | ↓ | 0/2 | 0.5 |
| `mri__soft` | 0.5191 | 0.5269 (k=1) | +0.0078 | ↑ | 4/6 | 0.754 |
| `txt__hard` | 0.6951 | 0.7091 (k=3) | +0.0140 | ↑ | 1/0 | 1.0 |
| `txt__soft` | 0.7068 | 0.7068 (k=None) | 0.0000 | = | 0/0 | 1.0 |
| `fusion_equal__hard` | 0.6085 | 0.6165 | +0.0080 | ↑ | 3/3 | 1.0 |
| `fusion_equal__soft` | 0.6452 | 0.6400 | −0.0052 | ↓ | 6/8 | 0.791 |

Directionally positive in 5 of 8 pairs (2 flat/negative, 1 unchanged), but every comparison is far from significance — consistent with the plan's own noise estimate (`std_macro_f1 ≈ 0.11` across MCCV splits).

### 5.4 Phase-A `knn_k` curves (figures available)

- `reports/figures/phasea_knn_curve_tab.png`, `_mri.png`, `_txt.png` — mean MCCV Macro-F1 vs. `knn_k` per modality/arm.
- `reports/figures/phasea_tab.png`, `_mri.png`, `_txt.png` — Phase-A sigma-search curves (carried over from exp_28's figure set).
- `reports/figures/confusion_matrix.png`, `roc_curve.png` — Phase-B diagnostics for the primary fusion condition.

Per Phase-A grid data (`results/phasea_grid_{tab,mri,txt}.csv`): for `tab`, mean MCCV Macro-F1 peaks at `knn_k=3` (hard: 0.6231, soft: 0.6346) and falls monotonically toward `knn_k=None` (hard: 0.5748, soft: 0.5766) — the clearest truncation effect in the sweep, matching the motivating single-modality pilot in DESIGN.md §4. For `mri`, the curve is comparatively flat and noisy (hard best at `knn_k=40`, 0.5455, barely above `knn_k=None`'s 0.5455; soft best at `knn_k=1`, 0.5303). For `txt`, `knn_k=3` edges out `knn_k=None` by only 0.0040–0.0212 depending on arm — a small, likely-noise-level margin.

---

## 6. Statistical Analysis

- **Test used**: `mcnemar_exact` (exact binomial McNemar test, `scipy.stats.binomtest`), paired on the 88 LOOCV folds — the test specified in DESIGN.md §8, appropriate for paired binary-classifier comparisons on a shared cohort.
- **Significance threshold**: p < 0.05, per DESIGN.md §8.
- **Result**: 0 of 13 McNemar comparisons (9 in §5.2, 8 in §5.3, with overlap) reach p < 0.05. The lowest p-value in the entire experiment is 0.388 (`tab__hard` vs `exp28_knn_k_none`).
- **Conclusion**: no result in this experiment is statistically distinguishable from its comparator. This is a single-run LOOCV design (deterministic per fold, no seed variation), so "N seeds" does not apply here; the McNemar test is the experiment's only significance mechanism, per its own protocol, and it is uniformly negative.

---

## 7. Comparison to Expected Results

| Expected (DESIGN.md) | Observed | Match? |
|---|---|---|
| §1: some `fusion_equal` `knn_k` arm exceeds 0.7171 with significance (H1) | Best fusion arm 0.6400 (soft); no significance anywhere | ❌ |
| §1: `knn_k` truncation improves ≥1 modality's Phase-A mean MCCV F1 over exp_28 (H2, point-estimate only) | tab clearly improves in Phase A (0.5748→0.6346 soft); mri/txt marginal | ✅ (point estimate only, not McNemar-confirmed) |
| §4: tab single-modality pilot motivates `knn_k=3` as the peak (0.6346 at k=3 vs 0.5766 at `knn_k=None`) | Full grid reproduces this exactly: `tab` `soft` `knn_k=3`=0.6346, `knn_k=None`=0.5766 (bit-for-bit match to the pilot cited in DESIGN.md §4) | ✅ |
| §6 G0–G3 reproduction gates | All 4 pass (56/56 verify checks; G1=0.633333…; G2=0.698754…; G3=0 mismatches) | ✅ |
| §2.2 risk: wider Phase-A grid (up to 21 points for `txt`) increases noise-fitting risk | Realized: `txt`'s `knn_k` margins over `knn_k=None` are small (0.004–0.021) and not McNemar-significant vs. exp_28 — consistent with the flagged risk, not a genuine signal | ⚠️ (risk materialized, as anticipated) |

---

## 8. Missing Data & Caveats

All planned conditions completed. Two notable operational events during execution, not scope changes:

1. **A run crashed mid-execution on the first attempt, before this run.** `experiments/exp_29/scripts/train.py`'s Phase-B metrics/McNemar step calls `binary_metrics` (`src/evaluation/metrics.py`), which passes soft-voted probabilities straight to `sklearn.metrics.brier_score_loss`. The new `knn_k`-truncated `BrentMemKDM` path produced a float32 rounding overshoot (`1.0000001192092896` instead of `1.0`) in at least one fusion arm's output, which sklearn's strict `[0,1]` validation rejected, raising an unhandled `ValueError` after Phase A and unimodal Phase B had already completed successfully. The `results/` files present at that point were stale leftovers from an earlier `--smoke` run (giveaway: `total_cases: 6` for every condition, matching `train.py`'s `--smoke` fold count) — not partial output of the crashed run, since the crash occurred before any of the final `results/` writes. **Fix applied**: `src/evaluation/metrics.py:28` now clips probabilities to `[0,1]` before the Brier-score call (`np.clip(p_soft, 0.0, 1.0)`). A `--smoke` run confirmed the fix end-to-end before the full run (documented in this report's §4) was launched; all results in this report are from that full run, which completed without error. This is a change to shared evaluation code, used by exp_23–exp_28 as well — it is currently only a working-tree edit, not committed (§4).
2. **`joint_trimodal_search.json`'s `score` (0.5588)** reflects the S2 product-kernel search objective and is not directly comparable to the binary-metric Macro-F1 columns above; it is reported here only as raw provenance, not interpreted further, matching DESIGN.md §3's framing of S2 as context.

Everything DESIGN.md §7/§9 specified as in-scope was produced: `reproduction_gates.json` (G0–G3, all passed), `phasea_grid_{tab,mri,txt}.csv`, `stage1_best_hparams.json`, `loocv_metrics.json`, `loocv_predictions.csv` (88 rows), `mcnemar.json` (H1 and H2 comparisons), `fusion_weights_leakfree.json`, `joint_trimodal_search.json`, and all figures listed in §5.4.

---

## 9. Conclusions & Next Steps

**What this experiment established:**
- `knn_k` truncation, as implemented, does not close the gap to exp_8's 0.7171 fusion target — the best fusion-arm result (0.6400) remains well below it, and no comparison in the experiment reaches statistical significance.
- The `knn_k=3` peak seen in the single-modality tab pilot (DESIGN.md §4) reproduces cleanly across the full grid for `tab`, but the same pattern is much weaker for `mri` and `txt`, and even `tab`'s clear Phase-A signal does not translate into a McNemar-significant Phase-B gain over exp_28's `knn_k=None`.

**What remains uncertain:**
- Whether the small, consistently-positive-but-non-significant point estimates across 5 of 8 H2 comparisons reflect a real (if weak) truncation effect that this N=88 cohort is underpowered to confirm, or pure Phase-A selection noise given the widened `(representation, knn_k)` grid (DESIGN.md §2.2's own stated risk).
- Whether a joint (product-kernel) `knn_k` — explicitly out of scope here (DESIGN.md §9) — would behave differently than the independent per-modality truncation tested; `joint_trimodal` performed worst of all conditions (0.5372) but used `knn_k=None` throughout, so this experiment says nothing about a truncated joint kernel.

**Recommended follow-up:**
- Commit the `binary_metrics` clipping fix (`src/evaluation/metrics.py`) and `src/methods/brent_mem_kdm.py`'s `knn_k` addition together, so future experiments in this lineage build on a clean, reproducible base (§4's outstanding caveat).
- Given four consecutive experiments in the KDM/MemKDM lineage (exp_23, exp_25–29) have now failed to beat exp_8's 0.7171 with significance, treat 0.7171 as a strong, well-defended baseline rather than continuing to search the same family of memory-kernel variants. A follow-up worth planning: audit *why* exp_8's original late-fusion KNN reaches 0.7171 while every faithfully-reproduced or generalized KNN/KDM variant in this checkout tops out near 0.66–0.71 — e.g., whether exp_8's number depended on the unreproducible `embedkit_sup` MRI representation (§2.1 caveat, carried since exp_28) in a way that inflates it relative to what's achievable with `{raw_l2, pca90_l2}` alone.
- To set up that audit or any new variant, use the `ml-experiment-planner` skill.

---

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Deterministic run (no seeds needed — `BrentMemKDM.fit()` is deterministic) | ✅ |
| Configs versioned (self-contained `train.py`, no external config files) | ✅ |
| Git commit recorded | ⚠️ Recorded (`189926f`), but the Brier-clip fix in `src/evaluation/metrics.py` is not yet committed — see §4/§8 |
| Checkpoints saved | N/A (memory-based model, no checkpoint artifact beyond the frozen `(rep, knn_k, sigma)` triple in `stage1_best_hparams.json`) |
| Environment frozen | ⚠️ No `requirements.txt`/`environment.yml` in repo (per `CLAUDE.md`); run used the local `pytorch` conda env |
| Experiment tracker linked | N/A (no external tracker used in this project) |
