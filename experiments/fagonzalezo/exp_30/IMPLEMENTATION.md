# Implementation Plan: exp_30 — Uncertainty Prediction Ability of BrentMemKDM

## Context

`experiments/exp_30/DESIGN.md` is approved. It tests whether `BrentMemKDM`-derived signals predict
the 3-class diagnostic confidence label better than the honest standing best (exp_24's
non-target-informed 0.4368, and exp_17's classical ICI 0.4470), using a new `knn_k`-only
neighborhood signal family (C) as the mechanism, over a narrow pre-registered grid
(4 modalities × `knn_k ∈ {5,20,None}` × 2 arms).

This plan covers the three deferred pieces from DESIGN.md §11.2: the additive family-C code in
`src/methods/brent_mem_kdm.py`, new checks in `scripts/verify_brent_mem_kdm.py`, and
`experiments/exp_30/scripts/train.py`.

## Part 1 — Family-C signals in `src/methods/brent_mem_kdm.py`

Key facts driving this design:

- `_knn_signals` (`:825-847`) already loops per query row, calls `_knn_submodel(...)` to get a
  fitted `k_eff`-memory `MemKDM` and discards its neighbor indices (`model, _nbr = ...`), then reads
  `model.uncertainty_signals(x_row)` — which **already returns `h_weights`, `log_ess`, `w_max`
  computed over the truncated k-particle mixture**. So DESIGN.md §5's "weight concentration" bullet
  for family C is **already covered by existing family B under `knn_k` truncation** — no new code
  needed for it. The genuinely new, non-redundant family-C signals are: (1) neighbor **label**
  disagreement (`y_binary`, which family B never touches) and (2) k-th-neighbor distance (a raw
  geometric quantity, distinct from `log_marginal`'s log-sum-exp aggregate).
- `BrentMemKDM.fit()`'s knn branch (`:805-812`) stores `self._train_y_soft` but never
  `targets.y_binary`. In the **soft** arm, `y_soft_train ≠ y_binary_train` (it encodes confidence
  via `CONFIDENCE_CERTAINTY_MAP`), so family C must read `y_binary_train` specifically to stay
  non-target-informed in both arms (DESIGN.md §6.5) — this requires storing it.
- `_knn_submodel` (`:319-347`) computes `expo` (the per-neighbor kernel exponent) internally via
  `_topk_neighbors`, then discards it, returning only `(model, nbr)`. Its only other call site is
  `_TorchScorer.score` (`:478`, unpacked as `sub_model, _nbr = ...`).

**Change 1 — `_knn_submodel`, return `expo` at the selected neighbors (ascending order, from
`_topk_neighbors`'s own sort):**
```python
def _knn_submodel(..., x_row, label_smoothing=0.0, seed=0) -> tuple:
    ...
    nbr = _topk_neighbors(expo, k_eff)[0]
    expo_nbr = expo[0, nbr]          # NEW
    ...
    return model, nbr, expo_nbr      # CHANGED: 3-tuple
```
Update the one other caller, `_TorchScorer.score` (`:478`):
```python
sub_model, _nbr, _expo_nbr = _knn_submodel(...)
```

**Change 2 — new module-level helper**, placed near `_topk_neighbors`:
```python
def _h_b(p: float) -> float:
    """Binary entropy in nats, exact zero at p=0/1 (no log(0) via clipping —
    matters for G3's exact-degeneracy check at knn_k=1)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def _neighborhood_signals(y_binary_train: np.ndarray, nbr: np.ndarray, expo_nbr: np.ndarray) -> dict:
    """Family-C (exp_30): signals definable only because `knn_k` retrieves a
    literal, finite neighbor set. Both inputs are non-target-informed even
    when the underlying model was fit on confidence-derived soft targets:
    `y_binary_train` is the raw biopsy label (never `confidence`), and
    `expo_nbr` is a pure input-space kernel exponent, independent of
    `y_soft`/`c_y`. `expo_nbr` is ascending (nearest first, from
    `_topk_neighbors`), so its last entry is the farthest retrieved
    neighbor — the k-th-neighbor distance in kernel-exponent units."""
    p = float(y_binary_train[nbr].mean())
    return {
        "nbr_label_entropy": _h_b(p),
        "nbr_kth_expo": float(expo_nbr[-1]),
    }
```

**Change 3 — `BrentMemKDM.fit()` knn branch (`:805-812`)**, store `y_binary`:
```python
self._train_y_binary = np.asarray(targets.y_binary, dtype=np.int64)   # NEW line
self._train_y_soft = np.clip(np.asarray(targets.y_soft, dtype=np.float32), 0.0, 1.0)
```

**Change 4 — `_knn_signals` (`:825-847`)**, collect the two new keys:
```python
def _knn_signals(self, X: Modalities) -> dict:
    X_train, y_soft_train = self._require_knn_fit()
    y_binary_train = self._train_y_binary
    n_val = len(np.asarray(X[self.modality_order[0]]))
    fam_c_keys = ("nbr_label_entropy", "nbr_kth_expo")
    collected = {key: [] for key in ("probs",) + PARTICLE_SIGNAL_NAMES + fam_c_keys}
    for i in range(n_val):
        x_row = {m: np.asarray(X[m])[i:i + 1] for m in self.modality_order}
        model, nbr, expo_nbr = _knn_submodel(X_train, y_soft_train, self.modality_order, self.sigmas_,
                                              self.knn_k, x_row, label_smoothing=self.label_smoothing, seed=self.seed)
        sig = model.uncertainty_signals(x_row)
        for key in ("probs",) + PARTICLE_SIGNAL_NAMES:
            collected[key].append(sig[key][0])
        fam_c = _neighborhood_signals(y_binary_train, nbr, expo_nbr)
        for key in fam_c_keys:
            collected[key].append(fam_c[key])
    return {key: (np.stack(vals, axis=0) if key == "probs" else np.array(vals))
            for key, vals in collected.items()}
```

**Blast radius check:** `_knn_submodel`'s new 3rd return value only affects its two callers, both
updated in this change. `_knn_signals`' two new dict keys are additive — every exp_28/29 caller
only reads `signals["probs"]` or specific `PARTICLE_SIGNAL_NAMES` entries, never iterates the dict
keys, so nothing existing breaks. The whole-memory path (`knn_k=None`, `_require_fit().
uncertainty_signals(X)` → `MemKDM.uncertainty_signals` → `extract_particle_signals`) is untouched —
family C is structurally absent there, which is exactly what DESIGN.md's G4 checks.

Import `math` is already present at module top (`brent_mem_kdm.py:97`).

## Part 2 — `scripts/verify_brent_mem_kdm.py`: new Check 7

Modeled directly on the existing `check_knn_truncation` (`:149-212`, Check 6). Add after it:

```python
# ---------------------------------------------------------------------------
# Check 7 — family-C neighborhood signals (exp_30)
# ---------------------------------------------------------------------------
def check_neighborhood_signals(folds: list, mods: list, rng: np.random.Generator) -> None:
    f0 = folds[0]
    sigmas = _sigma_ref_per_modality(folds, mods)
    y_binary_train = (f0.y_soft_train >= 0.5).astype(int)  # hard-arm folds only; this check is arm-agnostic by construction

    # (a) k=1: nbr_label_entropy is EXACTLY 0 (single neighbor, no disagreement possible).
    x_row = {m: np.asarray(f0.X_val[m])[0:1] for m in mods}
    _model, nbr, expo_nbr = _knn_submodel(f0.X_train, f0.y_soft_train, mods, sigmas, 1, x_row)
    sig = _neighborhood_signals(y_binary_train, nbr, expo_nbr)
    check("family-C k=1 nbr_label_entropy == 0", sig["nbr_label_entropy"] == 0.0)

    # (b) synthetic exact case: construct a 3-point neighbor set with 2 label-1, 1 label-0 and check
    # nbr_label_entropy == H_b(2/3) exactly (bypasses retrieval, tests the formula directly).
    y_synth = np.array([1, 1, 0, 0, 0])
    nbr_synth = np.array([0, 1, 2])
    expo_synth = np.array([0.1, 0.2, 0.3])
    sig_synth = _neighborhood_signals(y_synth, nbr_synth, expo_synth)
    expected = _h_b(2.0 / 3.0)
    check("family-C synthetic H_b(2/3) exact", abs(sig_synth["nbr_label_entropy"] - expected) < 1e-12)
    check("family-C nbr_kth_expo picks last (farthest) entry", sig_synth["nbr_kth_expo"] == 0.3)

    # (c) k_eff = min(k, n_train) still governs family C when k > n_train (reuses existing truncation
    # semantics; must not crash or silently use a wrong neighbor count).
    n_train = len(f0.y_soft_train)
    _model2, nbr2, expo_nbr2 = _knn_submodel(f0.X_train, f0.y_soft_train, mods, sigmas, n_train * 10, x_row)
    check("family-C k>=n_train uses full memory", len(nbr2) == n_train)
```

Import `_neighborhood_signals`/`_h_b` alongside the existing `brent_mem_kdm` imports at module top
(`:52-55`). Wire into `main()`'s check sequence (after the existing `print("\n=== Check 6:...")`
block, `:340-342`):
```python
print("\n=== Check 7: family-C neighborhood signals (exp_30) ===")
check_neighborhood_signals(build_folds(cohort, cleaned_texts, targets.y_soft, small_splits, ["tab"], {}), ["tab"], rng)
```

Total check count moves from 56 to 60 (56 + 4 new). `exp_30/scripts/train.py`'s G0 gate must not
hardcode "56" — read the pass/fail state from the subprocess return code and "checks passed" string
match, exactly as `exp_29/scripts/train.py`'s G0 already does (it never hardcodes a count either).

## Part 3 — `experiments/exp_30/scripts/train.py`

Structure closely follows `experiments/exp_29/scripts/train.py` (511 lines) — same imports, cohort
loading, `build_modality` dispatcher, `--smoke` flag pattern, `record_git_commit` — with these
exp_30-specific sections replacing exp_29's binary-fusion-focused ones:

### 3.1 Reused verbatim from exp_29
- Cohort/spaCy/splits setup (`:108-125`)
- `build_modality(name, rep, train_idx, val_idx)` dispatcher (`:127-135`)
- G0 (verify script subprocess) and its pass/fail parsing
- Phase A₁ σ search structure (`:220-264`) — but scored on **binary** macro-F1 only (`y_binary`),
  restricted to the pre-registered grid: `knn_k ∈ {5, 20, None}` (not exp_29's 7-point sweep), each
  modality's own representation grid unchanged from exp_28/29 (`tab`: fixed; `mri`:
  `{raw_l2, pca90_l2}`; `txt`: `{500,2000,None} × 0.90`)

### 3.2 New: Phase B signal extraction
For each `(modality, knn_k, arm)`, 88-fold LOOCV (`run_loocv_folds`, exp_29's local smoke-aware
variant of `protocol.run_loocv`), refit `BrentMemKDM(knn_k=k)` per fold with the frozen `(rep,
sigma)` from Phase A₁, and on each held-out row call `.uncertainty_signals(X_va)` (not just
`predict_proba`) — collect the full per-signal OOF vector (families A/B/D always; family C only
when `knn_k` is finite). Write `results/loocv_signals.csv`: one row per patient, one column per
`(modality, knn_k, arm, signal)`.

Family D (`composite_reliability_index`, `inter_modality_variance` from `mem_kdm.py:538,550`) is
computed once per `(knn_k, arm)` from the three modalities' own `p_mean`/margin at that same
`knn_k` — reuse the functions directly, no new code.

### 3.3 New: Phase A₂ confidence heads
For every `(modality, knn_k, arm)` cell and signal-family combination (A+B always; A+B+C only where
family C exists, i.e. finite `knn_k`):
- 1-D head: `fit_meta_thresholds_safe(signal, y_conf, SPLITS_2T)` → `apply_meta_thresholds(...)`,
  one row per individual signal (`src/methods/base.py:180,200`).
- Multivariate head: `fit_predict_heldout_trees(signal_matrix, y_conf, SPLITS_2T)`
  (`base.py:209`), one row for the A+B-only matrix and one for the A+B+C matrix at each finite
  `knn_k` — this A+B vs A+B+C pair, both under LOOCV signals from the *same* `knn_k`, is exactly
  H2's comparison.

`SPLITS_2T` = `[(train_idx, val_idx) for _, train_idx, val_idx in iter_mccv_splits(df_design, 100)]`
— stripped to 2-tuples, matching `base.py`'s documented `splits` contract. Never reduced under
`--smoke` (mirrors exp_27's own invariant, DESIGN.md §9).

### 3.4 Gates G1/G2/G3/G4

**G1 — exp_17 Composite Fuzzy ICI, target `0.4469706011059394`.** Verified end-to-end (no KDM
retrain, no torch import needed — pure numpy/pandas/sklearn). Two independent routes, run both and
assert they agree:

Route A — score the stored per-patient predictions directly:
```python
oof_conf = pd.read_csv(PROJECT_ROOT / "experiments/exp_17/results/oof_confidence_predictions.csv")
m = {"uncertain": 0, "borderline": 1, "clear": 2}
g1_route_a = f1_score(oof_conf.ground_truth_confidence.map(m), oof_conf.predicted_confidence.map(m),
                       average="macro", zero_division=0)
```

Route B — full recompute from exp_16's OOF probabilities (exercises the ICI formula and threshold
loop, not just a cached scalar):
```python
oof = pd.read_csv(PROJECT_ROOT / "experiments/exp_16/results/oof_predictions.csv")
rea = pd.read_csv(DATA_DIR / "clinical_reasoning.csv")
oofL = oof[oof.patient_id.isin(cohort.pids)].sort_values("patient_id").reset_index(drop=True)
reaL = rea[rea.patient_id.isin(cohort.pids)].sort_values("patient_id").reset_index(drop=True)
assert (oofL.patient_id.values == cohort.pids).all() and len(oofL) == 88   # exp_17 never asserted this

p1, p2, p3 = (oofL[c].values for c in ["prob_tabular_fuzzy", "prob_mri_fuzzy", "prob_text_fuzzy"])
pm = (p1 + p2 + p3) / 3.0
std = np.sqrt(((p1 - pm) ** 2 + (p2 - pm) ** 2 + (p3 - pm) ** 2) / 3.0)
ici = np.clip((2.0 * np.abs(pm - 0.50)) * (1.0 - 2.0 * std), 0.0, 1.0)
y = reaL.confidence.map(m).values

t1s, t2s = [], []
for i in range(100):
    tr = df_design_labeled[f"split_{i}"].values == 0
    dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)
    dt.fit(ici[tr].reshape(-1, 1), y[tr])
    th = np.sort(dt.tree_.threshold[dt.tree_.threshold != -2])
    if len(th) >= 2: a, b = th[0], th[1]
    elif len(th) == 1: a, b = th[0], th[0] + 0.1
    else: a, b = 0.10, 0.30
    t1s.append(a); t2s.append(b)
t1, t2 = float(np.mean(t1s)), float(np.mean(t2s))
pred = np.where(ici < t1, 0, np.where(ici < t2, 1, 2))
g1_route_b = f1_score(y, pred, average="macro", zero_division=0)

assert g1_route_a == g1_route_b == 0.4469706011059394
```
(`composite_reliability_index` from `mem_kdm.py` is deliberately **not** imported here — that module
imports `torch` at module level, which is unavailable outside the `pytorch` conda env; the inline
6-line formula above was verified numerically identical to the shared helper.)

**G2 — exp_24 `a_hard__multivariate_7signal`, target `0.4368347338935575`.** One route, verified,
no torch/KDM needed — recompute the held-out-tree ensemble from exp_24's stored input signals:
```python
S = ["h_total", "h_aleatoric", "h_epistemic", "h_weights", "log_ess", "w_max", "log_marginal"]
sig = pd.read_csv(PROJECT_ROOT / "experiments/exp_24/results/oof_particle_signals.csv")
sig = sig[sig.patient_id.isin(cohort.pids)].sort_values("patient_id").reset_index(drop=True)
assert (sig.patient_id.values == cohort.pids).all() and len(sig) == 88

mat = np.stack([sig[f"a_hard_{s}"].values for s in S], axis=1)
y_g2 = sig.y_conf.values.astype(int)
pred_g2, votes_g2 = fit_predict_heldout_trees(mat, y_g2, SPLITS_2T)   # src/methods/base.py, reused directly
g2_score = f1_score(y_g2, pred_g2, average="macro", zero_division=0)
assert g2_score == 0.4368347338935575 and votes_g2.min() == 11
```

**G3** (train.py-level, mirrors exp_29's own G3 pattern of verifying on the actual run rather than
an ad hoc check): after Phase B signal extraction, assert `loocv_signals[knn_k=5 or 20]["h_weights"]`
is NOT constant (sanity that truncation isn't accidentally degenerate at the chosen grid points),
and separately, using a **local, isolated** `knn_k=1` fit (not part of the condition grid, run only
for this gate) assert the exact degeneracies: `h_weights≡0`, `log_ess≡0`, `w_max≡1`, hard-arm
`h_total≡0`, and `nbr_label_entropy` reproduces the single-neighbor entropy exactly (`== 0.0`).

**G4**: assert `"nbr_label_entropy" not in <knn_k=None signal dict>.keys()` — trivially true by
construction (Part 1's blast-radius note) but asserted explicitly per DESIGN.md §7.

All four gates use exact equality (`==`), not a tolerance — G1/G2 held exactly under this
environment's library versions (sklearn 1.7.2, scipy 1.15.3, numpy 2.2.6, pandas 2.3.3) per the
verification that produced this plan; a tolerance would only mask drift, per the verification
agent's explicit recommendation. `train.py` asserts G0-G4, aborting on any failure, exactly as
exp_28/29 do.

⚠️ **Environment note confirmed during verification:** `torch` is not installed in the default
interpreter; only the `pytorch` conda env has it (per project memory). `mem_kdm.py` imports torch at
module level, so any code path importing it (e.g. `composite_reliability_index`,
`inter_modality_variance` for family D) requires running under `conda activate pytorch` — same as
every other script in this project. G1/G2 specifically avoid this dependency by using the inline
formula / `base.py`'s torch-free `fit_predict_heldout_trees`, so they would also run standalone if
ever needed outside that env, but `exp_30/scripts/train.py` as a whole still requires it (same as
exp_28/29) since Phase A/B fit real `BrentMemKDM`/`MemKDM` instances.

⚠️ **Path case-sensitivity note confirmed during verification:** `resolve_data_dir`'s fallback to
`Data/preprocessed_old/task1` only resolves on this machine because APFS is case-insensitive
(on-disk the directory is `data/preprocessed_old/task1`). Not an exp_30-specific issue — inherited
from every prior experiment — but worth knowing if this is ever run on a case-sensitive filesystem.

### 3.5 New: Significance (H3)
`mcnemar_exact` (`src/evaluation/metrics.py:47`) on per-patient correctness (`pred == y_conf`,
correctness as a binary outcome per patient) between this experiment's best row and (a) the G1
predictions and (b) the G2 predictions (`pred_g2` from above — already computed by the gate, reused
directly rather than recomputed). Plus a permutation test on the Macro-F1 delta (1000 label
permutations) — new, small, self-contained function in `train.py`, not `src/evaluation/metrics.py`
(avoids growing shared code for an exp_30-specific comparison).

### 3.6 `--smoke` and compute-budget check (DESIGN.md §6.8)
`--smoke`: 5 MCCV splits, 6 LOOCV folds (exp_29's pattern), restricted to one modality (`tab`) and
`knn_k=5` only. Must print elapsed wall-clock and extrapolate to the full grid (4 modalities × 3
`knn_k` × 2 arms × 88 folds) before the full run is launched — since knn-mode `uncertainty_signals`
fits a fresh `MemKDM` per query row (`_knn_submodel`), this is a materially larger budget than
exp_29's scalar-probability sweep.

## File Layout

```
experiments/exp_30/
├── DESIGN.md
├── IMPLEMENTATION.md          ← this file
├── scripts/train.py
├── results/
│   ├── reproduction_gates.json       ← G0-G4 pass/fail + values
│   ├── phasea_sigma_grid_{tab,mri,txt}.csv
│   ├── stage1_best_hparams.json
│   ├── loocv_signals.csv
│   ├── confidence_metrics.json
│   ├── confidence_predictions.csv
│   ├── significance.json
│   └── git_commit.txt
└── reports/
    ├── figures/
    └── summary.md
```

## Verification (end-to-end)

1. `python scripts/verify_brent_mem_kdm.py` (full run, `pytorch` conda env) — expect 60/60 (56
   existing + 4 new family-C checks), including the new Check 7.
2. `python experiments/exp_30/scripts/train.py --smoke` — completes without error, prints the
   compute-budget extrapolation, all four gates (G0-G4) pass on the smoke subset.
3. Review the smoke extrapolation before launching the full run (DESIGN.md §6.8) — if it implies an
   excessive wall-clock, narrow the grid further (e.g. drop one modality or one `knn_k` point)
   rather than running an unreviewed multi-hour job.
4. `python experiments/exp_30/scripts/train.py` (full run) — G0-G4 pass, `results/loocv_signals.csv`
   has 88 rows, `results/confidence_metrics.json` has one row per `(modality, knn_k, arm, head,
   signal-set)` cell, `results/significance.json` has the H3 comparisons against G1/G2.
5. Confirm G1 reproduces `0.4469706011059394` and G2 reproduces `0.4368347338935575` exactly (`==`,
   not a tolerance).
