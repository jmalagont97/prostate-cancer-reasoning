# Experiment Report: Uncertainty Prediction Ability of BrentMemKDM

**Experiment**: experiments/exp_30/
**Project**: pathology-reasoning (CHIMERA 2026, Task 1)
**Report date**: 2026-08-19
**Plan date**: 2026-08-19
**Author**: TBD
**Status**: Complete

---

## 1. Summary

exp_30 asked whether `BrentMemKDM`'s internal uncertainty signals — extended with a new `knn_k`-only "neighborhood" signal family (C) — could predict the urologist's 3-class diagnostic confidence better than the two honest standing references in this project: exp_17's classical Composite Fuzzy ICI (0.4470) and exp_24's best non-target-informed KDM result (0.4368). The best non-target-informed row (`tab`/`knn_k=20`/multivariate signals A+B+C) reached **0.4747**, beating both references on point estimate — but neither comparison reaches statistical significance (McNemar p=1.0 vs. exp_17, p=0.110 vs. exp_24), so per this project's own standard, **H1 is not established**. The more robust finding is **H2**: adding family-C neighborhood signals to a finite-`knn_k` model produces a **significant** improvement over the same head restricted to whole-memory signals, in **all three modalities** (tab p=0.024, mri p=0.043, txt p=0.0015) — the first statistically significant result ever obtained on this project's confidence-prediction task, in either direction.

---

## 2. Hypothesis & Verdict

**H1 (primary, from DESIGN.md §1):** "At least one `BrentMemKDM` hard-arm confidence predictor exceeds both exp_24's 0.4368 and exp_17's 0.4470 on 3-class LOOCV Macro-F1 — the first KDM-family model to beat classical ICI without consuming the confidence label."

**Verdict:** ⚠️ Not established. `tab`/`knn_k=20`/multivariate_ABC reaches 0.4747 — a genuine point-estimate improvement over both 0.4470 (+0.0278) and 0.4368 (+0.0379) — but McNemar exact tests against both recomputed reference predictions are non-significant (vs. exp_17: b=20, c=21, p=1.0; vs. exp_24: b=21, c=11, p=0.110), and a paired permutation test on the Macro-F1 delta agrees (p=0.725 and p=0.570 respectively). DESIGN.md §9's decision rule is explicit that an improvement without significance is reported as not established, not a win.

**H2 (secondary, mechanism, from DESIGN.md §1):** "Family-C neighborhood signals... add information beyond the whole-memory particle signals: a head including them beats the same head restricted to families A/B at `knn_k=None`."

**Verdict:** ✅ Supported, with significance, in all three modalities. Comparing each modality's best finite-`knn_k` multivariate_ABC row against its own multivariate_AB row at `knn_k=None`:

| Modality | ABC (best finite k) | AB (`knn_k=None`) | Δ | McNemar p |
|---|---|---|---|---|
| `tab` | 0.4747 (k=20) | 0.3928 | +0.0819 | **0.0237** |
| `mri` | 0.3322 (k=20) | 0.3166 | +0.0157 | **0.0428** |
| `txt` | 0.4031 (k=5) | 0.2525 | +0.1506 | **0.0015** |

This is the primary positive result of the experiment.

**H3 (secondary, protocol, from DESIGN.md §1):** "Report the first significance test ever run on this task in this project."

**Verdict:** ✅ Done. Every H1/H2 comparison above is backed by both `mcnemar_exact` and a 1000-permutation test on the Macro-F1 delta (`results/significance.json`). No prior confidence-task experiment (exp_9-12, 17, 19, 23-27) ran either.

---

## 3. Experimental Setup (as run)

As described in DESIGN.md, with the implementation decisions recorded in IMPLEMENTATION.md (representation fixed per modality by the same rule as exp_28/29's Phase A selection, then the pre-registered `knn_k ∈ {5, 20, None}` grid searched independently at that fixed representation, rather than selected to a single winner).

- **Dataset**: `Data/preprocessed_old/task1/`, N=88 complete-case cohort (same as exp_5-29). Confidence class balance: uncertain 14 (15.9%), borderline 18 (20.5%), clear 56 (63.6%).
- **Model**: `BrentMemKDM` with the new `_neighborhood_signals` family-C addition (`src/methods/brent_mem_kdm.py`), evaluated at `knn_k ∈ {5, 20, None}` per modality (`tab`, `mri`, `txt`), plus a family-D composite (`composite_reliability_index`, `inter_modality_variance`) derived from the three modalities' own predictions at each `knn_k`.
- **Search**: Phase A1 = 100 MCCV splits, Brent sigma search per `(modality, representation, knn_k)`, scored on **binary** macro-F1 only (never the confidence label) — this is what keeps the hard arm non-target-informed end to end. Phase B = 88-fold LOOCV, frozen `(representation, knn_k, sigma)`, extracting the full uncertainty-signal dict per fold. Phase A2 = confidence heads (1-D meta-threshold and multivariate held-out-tree, both from `src/methods/base.py`) fit on the Phase-B OOF signals across the full 100 MCCV splits.
- **Hardware / runtime**: local `pytorch` conda env; full run completed in 2.1 minutes (in line with the `--smoke` run's ~1-minute Phase-B extrapolation).
- **Deviations from plan**: One implementation gap found and fixed during the full run (not the smoke run) — see §8.

---

## 4. Code Version

| Component | Git commit | Commit message |
|-----------|-----------|-----------------|
| `results/git_commit.txt` (recorded by the run) | `c80b81b8e0e04dd2dff94ddf2e072a0ef5f7b2c0` | Add exp_29: BrentMemKDM knn_k truncation sweep (H1 refuted, H2 not established) |

⚠️ This commit **predates** the code this experiment actually ran on. The family-C addition to `src/methods/brent_mem_kdm.py`, the four new checks in `scripts/verify_brent_mem_kdm.py`, and `experiments/exp_30/scripts/train.py` itself were all uncommitted working-tree changes at run time — `record_git_commit` only records HEAD, with no mechanism to flag a dirty tree. **These changes must be committed** before this git-commit record can be trusted as a complete description of the code that produced these results (same caveat class as exp_29 §2.4/§4, not yet resolved by tooling in this repo).

---

## 5. Results

### 5.1 Primary Metric — Confidence Task LOOCV Macro-F1 (hard arm, non-target-informed)

Top 8 of 108 hard-arm rows (`results/confidence_metrics.json`):

| Condition | `knn_k` | Head | Signal set | Macro-F1 |
|---|---|---|---|---|
| `tab` | 20 | multivariate_ABC | margin+7 particle+2 family-C | **0.4747** |
| `mri` | 20 | 1d | `log_ess` | 0.4479 |
| `mri` | 20 | 1d | `h_weights` | 0.4456 |
| `txt` | None | 1d | `log_marginal` | 0.4086 |
| `txt` | 20 | 1d | `log_marginal` | 0.4080 |
| `txt` | 5 | multivariate_ABC | margin+7 particle+2 family-C | 0.4031 |
| `tab` | None | multivariate_AB | margin+7 particle | 0.3928 |
| `txt` | 20 | 1d | `w_max` | 0.3920 |

> Reference targets: exp_17 Composite Fuzzy ICI = 0.4470; exp_24 best non-target-informed = 0.4368. Best row (0.4747) exceeds both on point estimate; significance below (§5.2).

### 5.2 Significance (H1, H3) — `results/significance.json`

| Comparison | b | c | McNemar p | Permutation Δ | Permutation p |
|---|---|---|---|---|---|
| best hard row vs. exp_17 (G1) | 20 | 21 | 1.0 | +0.0278 | 0.725 |
| best hard row vs. exp_24 (G2) | 21 | 11 | 0.110 | +0.0379 | 0.570 |

Neither reaches p < 0.05. `exp_24` is numerically closer (both the McNemar and permutation p-values are smaller), consistent with `exp_24` and this experiment's `tab` condition sharing more structural similarity (both are tabular-modality, particle-signal-based) than either does with `exp_17`'s three-modality classical ICI.

### 5.3 H2 — Family-C Contribution (the established result)

See table in §2. All three modalities show a significant McNemar result for finite-`knn_k` (family A+B+C) vs. whole-memory (family A+B only, `knn_k=None`), with `txt` showing both the largest effect size (+0.1506 Macro-F1) and the strongest significance (p=0.0015).

### 5.4 Secondary — Soft Arm (target-informed, lineage context only, not scored against H1)

Top row: `tab`/`knn_k=5`/1d/`h_weights` = **0.4928**. For context against the target-informed lineage (all soft-arm/target-informed numbers, not a fair comparison to this experiment's own hard-arm H1 result): exp_25 = 0.4547, exp_26 = 0.5287, exp_27 = 0.5630. exp_30's soft-arm best sits between exp_25 and exp_26 — a reasonable result, but not a new project-wide best, and (per DESIGN.md §6.3) soft-arm `h_weights`/`h_aleatoric`-family signals are doubly target-informed in this task (the underlying model consumes the confidence label AND the label is being predicted), so this number is reported only for lineage continuity, not as a claim.

### 5.5 Figures

- `reports/figures/phasea_{tab,mri,txt}.png` — Phase A1 sigma-search curves per modality, grouped by arm and `knn_k`.
- `reports/figures/confusion_matrix.png` — 3-class confusion matrix for the best hard-arm row (`tab`/`knn_k=20`/multivariate_ABC).

---

## 6. Statistical Analysis

- **Tests used**: `mcnemar_exact` (paired, exact binomial, on per-patient correctness — generalizes cleanly from the binary task to 3-class since it only compares `pred == y_true`) and a 1000-iteration paired permutation test on the Macro-F1 delta, both defined in `experiments/exp_30/scripts/train.py` (the permutation test is new to this experiment; `mcnemar_exact` is reused unmodified from `src/evaluation/metrics.py`).
- **Significance threshold**: p < 0.05, matching every prior experiment in this lineage.
- **Result**: 2 of 2 H1 comparisons non-significant; 3 of 3 H2 comparisons significant. This is the first time any confidence-task comparison in this project has reached significance in either direction.
- This is a single deterministic LOOCV run (no seed variation — `BrentMemKDM.fit()` is deterministic), so per-seed variance doesn't apply; McNemar and the permutation test are the only significance mechanisms, exactly as DESIGN.md specified.

---

## 7. Comparison to Expected Results

| Expected (DESIGN.md) | Observed | Match? |
|---|---|---|
| §1 H1: a non-target-informed hard-arm row beats both 0.4368 and 0.4470, with significance | Point estimate beats both (0.4747); neither comparison significant | ⚠️ Partial (point estimate only) |
| §1 H2: family-C signals improve over A+B-only at `knn_k=None`, at least one modality | Significant in **all three** modalities, not just one | ✅ Exceeded |
| §1 H3: first significance test on this task | McNemar + permutation test run and reported for every H1/H2 comparison | ✅ |
| §6.1 risk: `knn_k=1` degenerates several signals to constants | Excluded from the condition grid as designed; used only as gate G3's input | ✅ (risk avoided by design) |
| §6.3 risk: soft-arm `h_aleatoric` is a near-direct confidence-label read-out | Confirmed structurally true (not separately re-derived here; carried from exp_24/DESIGN.md's analysis) — soft-arm results reported as context only, never used for H1 | ✅ (risk respected) |
| §6.8: compute budget requires a `--smoke` check before the full run | `--smoke` extrapolated ~1.0 min for full Phase B; full run (including Phase A1/A2, not smoke-tested) completed in 2.1 min — well within budget | ✅ |

---

## 8. Missing Data & Caveats

All planned conditions completed after one mid-implementation fix, documented here for transparency:

1. **First full-run attempt crashed** in Phase A2's metrics writer: `confidence_metrics` (`src/evaluation/metrics.py:34`) computes `spearmanr(y_conf, pred)`, which returns `NaN` for a degenerate/constant prediction (a real occurrence across 216 rows, given several 1-D signal/`knn_k` combinations produce near-constant thresholded output). The strict JSON writer (`reporting._check_finite`) correctly rejected the resulting `NaN`. **Fix applied**: added `safe_confidence_metrics` (a local wrapper in `experiments/exp_30/scripts/train.py`, mapping non-finite `spearman_rho`/`spearman_pvalue` to `None`) — the same convention exp_25-27 already used for exactly this situation, which the initial implementation had omitted. The `--smoke` run never exercised this path (Phase A2 is skipped under `--smoke` by design, §3.6), so the gap wasn't caught until the full run. All results in this report are from the corrected full run, which completed without error.
2. **The recorded `git_commit.txt` predates the code actually run** — see §4. Must be resolved (commit the working-tree changes) before this record is trusted for future reproduction.
3. `results/loocv_signals.csv` (88 rows, ~174 signal columns) and `results/confidence_predictions.csv` are both written and match the expected row/condition counts exactly (216 = 108 hard + 108 soft rows in `confidence_metrics.json`, cross-checked against the 3-modality × 3-`knn_k` × signal-count arithmetic in IMPLEMENTATION.md §3.3).

Everything DESIGN.md §7/§10 specified as in-scope was produced: `reproduction_gates.json` (G0-G4, all passed), `stage1_best_hparams.json`, `phasea_sigma_grid_{tab,mri,txt}.csv`, `loocv_signals.csv`, `confidence_metrics.json`, `confidence_predictions.csv`, `significance.json`, and the figures listed in §5.5.

---

## 9. Conclusions & Next Steps

**What this experiment established:**
- The `knn_k`-only neighborhood signal family (label disagreement among retrieved neighbors, k-th-neighbor distance) is not redundant with existing particle-set signals — it contributes real, statistically significant information to confidence prediction in all three modalities tested. This is the first significant result of any kind on this project's confidence-prediction task.
- `BrentMemKDM`'s best non-target-informed confidence result (0.4747) is now the project's best non-target-informed number, ahead of exp_24's 0.4368 and exp_17's 0.4470 — but not with statistical confidence, so it should be treated as a promising point estimate, not a settled improvement.

**What remains uncertain:**
- Whether H1's point-estimate lead over exp_17/exp_24 would become significant with more statistical power (the McNemar tests here have relatively small `b+c` — 41 and 32 discordant pairs out of 88 — typical for this cohort size) or whether it is itself noise, given exp_29 §2.2's standing observation that `std_macro_f1 ≈ 0.11` on the binary task at this N.
- Whether family C's significant contribution (H2) is specific to `tab`'s particular representation and the `{5, 20}` grid points tested, or would generalize to a wider `knn_k` sweep — this experiment deliberately used a narrow, pre-registered grid (DESIGN.md §4) and did not search it.
- Whether the soft-arm's best number (0.4928) reflects genuine signal beyond exp_25's comparable value, given both are target-informed and not directly comparable to this experiment's own H1 test.

**Recommended follow-up:**
1. Commit the outstanding working-tree changes (`src/methods/brent_mem_kdm.py`'s family-C addition, `scripts/verify_brent_mem_kdm.py`'s new checks, and `experiments/exp_30/`) so `git_commit.txt` correctly describes the code that produced these results (§4/§8's outstanding item).
2. A natural follow-up experiment: widen the `knn_k` grid around the `tab`/`k=20` and `txt`/`k=5` winners (e.g. `{3, 10, 15, 20, 30}`) to test whether H1's point-estimate lead firms up into significance with a finer search — but pre-register the grid narrowly again, given DESIGN.md §6.6's standing caution about this cohort's noise floor.
3. To set up that follow-up, use the `ml-experiment-planner` skill.

---

## 10. Reproducibility Record

| Item | Status |
|------|--------|
| Deterministic run (no seeds needed — `BrentMemKDM.fit()` is deterministic) | ✅ |
| Configs versioned (self-contained `train.py`) | ✅ |
| Git commit recorded | ⚠️ Recorded (`c80b81b`), but predates the actual code run — see §4/§8 |
| Checkpoints saved | N/A (memory-based model; the frozen `(representation, knn_k, sigma)` triples are in `stage1_best_hparams.json`) |
| Environment frozen | ⚠️ No `requirements.txt`/`environment.yml` in repo (per `CLAUDE.md`); run used the local `pytorch` conda env |
| Experiment tracker linked | N/A (no external tracker used in this project) |
