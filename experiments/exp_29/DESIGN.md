# Experiment Design: BrentMemKDM k-NN truncation sweep over the exp_28 hard-KNN generation (exp_29)
**Experiment**: experiments/exp_29/ · **Project**: pathology-reasoning · **Date**: 2026-08-19 · **Status**: Complete

---

## 1. Motivation — exp_28 refuted H1, but landed on the mechanism this experiment tests

`exp_28` replaced exp_5–8's per-modality KNN classifiers with `BrentMemKDM` (whole-memory,
`n_comp = n_train`) and asked whether a continuous, globally-fit-once RBF bandwidth alone would
recover or exceed exp_8's 0.7171 LOOCV Macro-F1. It didn't: `fusion_equal` reached 0.6085 (hard) /
0.6452 (soft), and no condition beat its own recomputed KNN reference with McNemar significance
(all p > 0.3). Two things from that result motivate this experiment specifically, not a repeat of
exp_28:

- exp_28's own secondary condition `fusion_optimal_leakfree` (leak-free MCCV-selected fusion
  weights) reached **0.7068** — 0.0103 short of the 0.7171 target, the closest any condition in
  the KDM/MemKDM lineage (exp_23–28) has come.
- exp_28's continuous sigma search found `mri`/`txt` bandwidths (`sigma_mult` 0.20–0.23) **below**
  every prior discrete grid point — i.e. the whole-memory Nadaraya-Watson average was already
  leaning hard on a small effective neighborhood, without a mechanism to make that neighborhood
  literal (BrentMemKDM's kernel still touches every one of the ~70-87 training points, just at
  very unequal weight).

This session added `knn_k` to `BrentMemKDM` (`src/methods/brent_mem_kdm.py`): retrieve each
query's `k` nearest memory points by kernel value and apply the exact same computation to only
those `k`, with `k_eff = min(k, n_train)` replacing `n_train` in the normalizing divisor (module
docstring's k-NN section). Two boundary facts make this the natural next test, not a new method:

- `knn_k=1` (with `label_smoothing=0`, hard targets) is **exactly** a 1-NN classifier — verified
  bit-identical to `sklearn.neighbors.KNeighborsClassifier(n_neighbors=1, weights="uniform",
  metric="euclidean")` per-fold over 15 MCCV splits (0 prediction mismatches) — i.e. the same
  computation exp_28's own `knn_tab`/`knn_txt`/`knn_mri_pca` reference arms compute.
- `knn_k=None` (or `knn_k >= n_train`) is exp_28's whole-memory model, unchanged (verified
  bit-exact; `scripts/verify_brent_mem_kdm.py` is 56/56 passing, including the 24 new knn-mode
  checks).

`knn_k` therefore makes `BrentMemKDM` a **continuous family with exp_28's KNN references at one
end and exp_28's own BrentMemKDM at the other**. exp_28 tested both ends independently and both
underperformed 0.7171; this experiment tests whether an intermediate `k` — sharper than
whole-memory averaging, softer than a single nearest neighbor — does better than either end,
sweeping `k` as an additional Phase-A dimension inside the *same* MCCV→LOOCV harness.

**H1 (single, primary — identical target to exp_28's H1, now with `knn_k` as a Phase-A search
dimension).** At least one `BrentMemKDM(knn_k=k)` fusion arm (hard or soft) will produce a LOOCV
Macro-F1 exceeding exp_8's **0.7171**, with `mcnemar_exact` significance against a recomputed
leak-free KNN fusion reference.

**H2 (secondary, mechanism check).** For at least one modality, the best `knn_k` found in Phase A
improves mean MCCV Macro-F1 over exp_28's own `knn_k=None` result for that modality/arm — i.e. the
truncation mechanism itself helps, independent of whether H1 is met. This is the more locally
falsifiable claim: exp_28 already measured the `knn_k=None` endpoint precisely (§ below), so this
experiment's own Phase-A curve is compared directly against those numbers, not re-estimated.

## 2. Background — what's reused, what's new, what's still not reproducible

Reuses `src/evaluation/{data,protocol,metrics,reporting}.py` and
`src/methods/{base,mem_kdm,brent_mem_kdm}.py` exactly as exp_28 left them, reading
`Data/preprocessed_old/task1/` (same schema, same `N=88` complete-case cohort, same
`resolve_data_dir`/`load_cohort` path handling). The only code delta since exp_28 is additive:
`BrentMemKDM`/`run_brent_search`/`_FoldCache` gained the optional `knn_k` parameter (default
`None`, so every exp_28 call site is unaffected) — no existing method-module code was rewritten.

### 2.1 `utils/embedding-kit/` is still empty — same caveat as exp_28 §2.1, unchanged

`exp_6`'s winning MRI representation (`embedkit_sup`) and exp_8's fusion built on it remain
unreproducible in this checkout. This experiment reuses exp_28's resolution verbatim: the MRI
representation search is restricted to `{raw_l2, pca90_l2}` (`build_mri_features`'s
`pca_variance ∈ {None, 0.90}`), and exp_6/exp_8's published numbers are reported as
published-but-not-reproducible-here, alongside recomputed KNN reference arms (§5, reused from
exp_28's script).

### 2.2 The `knn_k` grid adds a second discrete Phase-A selection axis, on top of representation

exp_28 selected each modality's representation by mean MCCV Macro-F1 (best of `{raw_l2, pca90_l2}`
for `mri`, `{500, 2000, None}` for `txt`, one fixed representation for `tab`), then ran a
continuous Brent sigma search inside the winning representation. This experiment adds `knn_k` as a
second axis **inside the same selection**: for each `(representation, knn_k)` pair, run
`run_brent_search(..., knn_k=knn_k)` (unchanged Brent search, just scored under truncation) and
select the best `(representation, knn_k, sigma*)` triple by the same rule exp_28 used — mean
Macro-F1 descending, std ascending tie-break. `knn_k` is never Brent-searched itself (it's
discrete); this is a grid sweep wrapping the existing continuous search, exactly as `knn_k`'s own
module docstring specifies ("a caller sweeps `k` in an outer loop, running one Brent sigma-search
per `k`").

**Risk, stated plainly:** widening Phase A's selection grid (representation × `knn_k`, up to
3 × 7 = 21 points for `txt`) increases the chance of selecting a configuration that fits Phase-A
noise rather than a genuine effect, on top of the frozen-sigma leakage already accepted in exp_28
§2.3 (below). §8's decision rule treats this explicitly: a Phase-B improvement without McNemar
significance is **not established**, not a win — the same standard exp_28 held itself to, now more
important given the larger grid.

### 2.3 The frozen (representation, knn_k, sigma) triple is not leak-free — same accepted bias as exp_28 §2.3

Unchanged from exp_28: Phase-A selection uses all 100 MCCV splits, which collectively cover the
whole labeled cohort, so the frozen configuration used in every LOOCV fold of Phase B was
informed, in aggregate, by data including that fold's own held-out patient. Not claimed leak-free;
stated here and in the summary, same as exp_17/exp_27/exp_28.

### 2.4 `knn_k` is implemented and verified, but not yet committed

`src/methods/brent_mem_kdm.py` (the `knn_k` option) and `scripts/verify_brent_mem_kdm.py` (the new
knn-mode checks) are both currently uncommitted working-tree changes (`git status`). Both must be
committed before this experiment's final run, so `results/git_commit.txt` names the code actually
evaluated — the same requirement exp_28 §2.4 stated for its own `has_trainable` fix.

### 2.5 `scripts/verify_brent_mem_kdm.py` has already been run in full, unlike exp_28 §2.6

Unlike exp_28 (whose G0 gate was the script's first-ever run), the full suite already ran this
session: **56/56 checks pass**, including 24 new knn-mode checks (k=1 closed-form identity across
sigma regimes, `knn_k=None` bit-exactness, `k >= n_train` equivalence, fast-vs-exact-per-query
agreement at the lower/center/upper sigma bounds, determinism). Running it again as Step 0 (§6)
before the final run remains required — it re-validates against whatever commit `results/
git_commit.txt` will name (§2.4), not the working-tree state checked during development.

### 2.6 Soft targets consume the confidence label — same caveat as exp_28 §2.5, unchanged

Both supervision arms (`hard`, exp_5–8's original; `soft`, exp_13-style
clear=1.0/borderline=0.5/uncertain=0.25) are run for every condition, and every soft-arm row is
labeled `target_informed=True`, exactly as exp_28 did.

## 3. Conditions

Same four primary conditions as exp_28 §3, each now searched over the `(representation, knn_k)`
grid in Phase A instead of representation alone; same two supervision arms.

| # | Condition | Modalities | Phase-A search | Replaces | Target |
|---|-----------|-----------|-----------------|----------|--------|
| C1 | `tab` | tabular | `knn_k` grid × 1-D Brent | exp_5 / exp_28's `tab` | exp_28's `tab__hard` 0.5810 / `tab__soft` 0.6085 (H2); 0.6333 published (context) |
| C2 | `mri` | MRI (`{raw_l2, pca90_l2}`) | rep × `knn_k` grid × 1-D Brent | exp_6 / exp_28's `mri` | exp_28's `mri__hard` 0.5458 / `mri__soft` 0.5191 (H2); recomputed `knn_mri_pca` 0.5299 |
| C3 | `txt` | text (`{500, 2000, None}` max_features) | rep × `knn_k` grid × 1-D Brent | exp_7 / exp_28's `txt` | exp_28's `txt__hard` 0.6951 / `txt__soft` 0.7068 (H2); 0.6988 published (context) |
| C4 | `fusion_equal` | tab+mri+txt, soft-voted 1/3 each, each arm's own winning `(rep, knn_k, sigma*)` | independent per-modality searches above | exp_8 / exp_28's `fusion_equal` | **0.7171** (H1, primary); exp_28's `fusion_equal__hard` 0.6085 / `__soft` 0.6452 (H2); recomputed `knn_fusion_equal` 0.6613 |

Secondary (hard arm only, reused structure from exp_28 §3, `knn_k=None` — not part of this
experiment's added dimension, kept as exp_28 ran them):

| # | Condition | Purpose |
|---|-----------|---------|
| S1 | `fusion_optimal_leakfree` | Same construction as exp_28 (simplex fusion weights chosen on Stage-1 MCCV validation probabilities only), rebuilt on this experiment's `knn_k`-winning per-modality arms instead of exp_28's whole-memory ones. exp_28's own value (0.7068) is the direct comparison. |
| S2 | `joint_trimodal` | Single product-kernel `BrentMemKDM(strategy="coordinate")` over all three modalities, **`knn_k=None`** (whole-memory) — unchanged from exp_28. A joint per-query neighbor retrieval across a product kernel is a materially different design question (which modality's distance dominates neighbor selection) and is explicitly out of scope here (§9); S2 exists only as exp_28's own context carried forward, not as a `knn_k` condition. |

## 4. Representation grid, `knn_k` grid, and search budget

Representation grid unchanged from exp_28 §4:

| Modality | Representation grid | Source |
|---|---|---|
| `tab` | one representation: `build_tabular_features(..., dre_categories=cohort.dre_categories)` | mandatory, not optional (exp_28 §4's `init_kdm_layer` shape-crash rationale, unchanged) |
| `mri` | `pca_variance ∈ {None, 0.90}` (`raw_l2`, `pca90_l2`) | `build_mri_features` |
| `txt` | `max_features ∈ {500, 2000, None}`, `pca_variance=0.90` | `build_text_features` |

**`knn_k` grid (new dimension, this experiment):** `{1, 3, 5, 10, 20, 40, None}` per modality —
7 points from 1-NN to a value comfortably above the ~70-87-row training folds in every MCCV split
(so the grid's own top end double-checks the `knn_k=None` endpoint rather than only trusting the
`k_eff = min(k, n_train)` equivalence proven in `scripts/verify_brent_mem_kdm.py`). Matches the
range already exercised in this session's own 100-MCCV-split Phase-A-style sweep on `tab`, soft
arm only (`build_targets(cohort.y_binary, cohort.confidence, CONFIDENCE_CERTAINTY_MAP)`, the same
target exp_28's soft arm used): `knn_k=None` reproduced exp_28's own
`results/stage1_best_hparams.json` `tab`/`soft` entry bit-for-bit (`sigma_mult=0.43333`,
`mean_macro_f1=0.57660`), confirming the sweep script matches exp_28's own numbers at the shared
endpoint, and the sweep found mean MCCV Macro-F1 peaking at `knn_k=3` (0.6346), above both
`knn_k=1` (0.6117) and `knn_k=None` (0.5766). This is one modality, one arm, one representation,
Phase A only — not a Phase-B LOOCV result and not evidence for H1 or H2 on its own — but it is a
positive, exact-endpoint-anchored signal that motivates running the full grid rather than treating
the choice of `knn_k` range as speculative.

**Search budget:** for `tab`, 7 `knn_k` points × 1 representation × 1-D Brent search
(`n_prescan=15, maxiter=20`, ~35 evaluations each, unchanged per-search cost from exp_28 since
`knn_k` truncation adds only an `argpartition` over an already-computed distance matrix — see
`brent_mem_kdm.py`'s `_FoldCache.probs`). For `mri`, 7 × 2 = 14 searches; for `txt`, 7 × 3 = 21
searches — a 7× multiple of exp_28's Phase-A cost per modality, still on the fast Nadaraya-Watson
path (the same path this session's `scripts/verify_brent_mem_kdm.py --quick` ran, 54 checks
including several full MemKDM torch fits, in ~7 seconds total). `--smoke` (§6) confirms this
estimate empirically before the full 100-split run. S2 (`joint_trimodal`) keeps exp_28's own
`strategy="coordinate"`, reduced 5-split budget, unchanged.

## 5. Reference arms (recomputed, reused from exp_28's own script)

Same construction as exp_28 §5, reused verbatim (KNN reference arms don't have a `knn_k` — they
already **are** the `k=1`/`k=3` endpoints this experiment interpolates around):

- `knn_mri_pca` — `KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean")` on
  exp_6's `pca` representation. Honest exp_6 reference for C2.
- `knn_fusion_equal` — equal-weight soft vote of exp_5's tabular KNN + `knn_mri_pca` + exp_7's text
  KNN (0.6613 in exp_28). Honest exp_8 reference for C4, and the direct McNemar comparator for H1.

## 6. Step 0 — reproduction gates

- **G0** — `python scripts/verify_brent_mem_kdm.py` (full run, not `--quick`) against the commit
  named in `results/git_commit.txt` (§2.4): all checks pass (56/56 as last run this session,
  including the 24 knn-mode checks — re-verify the count against the committed code, don't assume
  it stays 56).
- **G1** — exp_5's KNN (`k=3, uniform, euclidean`) through `run_loocv` reproduces Macro-F1
  `0.6333333333333333`, confusion 46/14/20/8 — identical gate to exp_28 §6.
- **G2** — exp_7's KNN (`max_features=500`, TF-IDF → PCA@0.90, `k=1, uniform, cosine`) reproduces
  `0.6987539367383268`, confusion 42/21/13/12 — identical gate to exp_28 §6.
- **G3 (new, self-consistency rather than a fixed-number gate)** — `BrentMemKDM(knn_k=1)` on
  `tab`, hard arm, any sigma in the search bounds, produces **identical LOOCV predictions** to
  `KNeighborsClassifier(n_neighbors=1, weights="uniform", metric="euclidean")` fit on the exact
  same `dre_categories`-fixed representation this experiment's C1 uses — the `knn_k=1` ≡ 1-NN
  identity (§1), re-verified on the full 88-fold LOOCV this experiment actually runs, rather than
  the 15-MCCV-split ad hoc check done this session. This is deliberately **not** compared against
  G1's number: G1 reproduces exp_5's *original* pipeline (`dre_categories=None`, inferred per
  split), while C1 uses the `dre_categories=cohort.dre_categories`-fixed pipeline exp_28 §4 made
  mandatory — the two encodings can legitimately break nearest-neighbor ties differently, so G3
  checks `BrentMemKDM(knn_k=1)` against a plain 1-NN fit on *its own* representation, not against
  G1's differently-encoded number.
- exp_6/exp_8 have no gate (§2.1, unchanged from exp_28). `train.py` asserts G1/G2 with
  `abs(got - expected) < 1e-12` and G3 with zero prediction mismatches, aborting on any failure.

## 7. File Layout

```
experiments/exp_29/
├── DESIGN.md
├── IMPLEMENTATION.md          ← added after this file is approved
├── scripts/train.py            ← self-contained; --smoke flag (reuses exp_28's script structure)
├── results/
│   ├── reproduction_gates.json       ← G0/G1/G2/G3 pass/fail + values
│   ├── phasea_grid_tab.csv, _mri.csv, _txt.csv   ← now includes a knn_k column
│   ├── stage1_best_hparams.json      ← per-modality winning (rep, knn_k, sigma*)
│   ├── loocv_metrics.json            ← all 4 primary conditions × 2 arms + S1/S2 + reference arms
│   ├── loocv_predictions.csv
│   ├── mcnemar.json                  ← vs. KNN references AND vs. exp_28's own knn_k=None rows (H2)
│   └── git_commit.txt
└── reports/
    ├── figures/   ← per-modality Macro-F1-vs-knn_k curve (new), sigma curves, confusion matrix, ROC
    └── summary.md
```

## 8. Evaluation Protocol & Decision Rules

Same two-phase MCCV→LOOCV harness as exp_28 (`CLAUDE.md`'s protocol; `src/evaluation/protocol.py`):
Phase A = 100 MCCV splits, select on mean Macro-F1 (now over the `(representation, knn_k)` grid);
Phase B = 88-fold LOOCV, frozen `(representation, knn_k, sigma)`, model re-fit per fold,
hyperparameters never re-fit. Metrics: `binary_metrics` (Macro-F1 primary) + `mcnemar_exact`
against each condition's KNN counterpart.

- **H1 (primary):** at least one `BrentMemKDM(knn_k=k)` arm of C4 (`fusion_equal`, hard or soft,
  any `k` in the grid) exceeds exp_8's 0.7171 LOOCV Macro-F1, with `mcnemar_exact` significance
  vs. `knn_fusion_equal` (0.6613).
- **H2 (secondary, per condition):** compare each condition's best-`knn_k` Phase-B Macro-F1
  against exp_28's own `knn_k=None` number for that exact condition/arm (§3 table) — reported with
  `mcnemar_exact` between this experiment's predictions and exp_28's stored
  `results/loocv_predictions.csv` where fold alignment permits (same LOOCV protocol, same cohort
  order), not just a point-estimate comparison.
- Given `std_macro_f1 ≈ 0.11` across MCCV splits in this cohort (exp_28 §8, unchanged cohort), a
  Phase-B improvement without a significant McNemar result is reported as **not established**, not
  as a win — same standard exp_28 held itself to, doubly important given §2.2's wider grid.
- S1/S2 are reported as context, not scored against H1/H2 (same as exp_28).

## 9. Scope

**In scope:** re-running C1–C4 (both supervision arms) with `knn_k` swept per §4, and S1/S2 (hard
only, `knn_k=None`) exactly as exp_28 constructed them but rebuilt on this experiment's winning
per-modality arms; recomputed MRI/fusion reference arms (unchanged from exp_28); reproduction gates
G0–G3; `mcnemar_exact` against each condition's KNN counterpart (H1) and against exp_28's own
`knn_k=None` predictions where alignable (H2); reporting the winning `knn_k` per condition/arm
alongside `sigma_mult` (as exp_28 did for sigma alone).

**Out of scope:** editing `src/evaluation/*` or `src/methods/{base,mem_kdm}.py` (only
`brent_mem_kdm.py`'s already-implemented, already-committed-pending `knn_k` addition is used);
the new-schema data (`Data/preprocessed/task1/`); reproducing exp_6/exp_8's embedkit-based
numbers or exp_8's eval-set-leaky weight sweep (both unchanged from exp_28); a joint-kernel
`knn_k` for S2 (§3); Brent-searching `knn_k` itself (it is swept as a discrete grid, per
`brent_mem_kdm.py`'s own module docstring); seed averaging (`BrentMemKDM.fit()` remains
deterministic in both whole-memory and knn modes).

## 10. Next Steps

1. Review and accept this design.
2. `IMPLEMENTATION.md` (concrete build plan, exact execution command, likely a close adaptation of
   exp_28's `scripts/train.py` with the `(representation, knn_k)` grid substituted for exp_28's
   representation-only Phase-A grid) for approval — in Claude Code plan mode, not this skill.
3. Commit `src/methods/brent_mem_kdm.py`'s `knn_k` addition and
   `scripts/verify_brent_mem_kdm.py`'s new knn-mode checks (§2.4, currently uncommitted).
4. Implement `scripts/train.py`, run `--smoke`, then the full run.
