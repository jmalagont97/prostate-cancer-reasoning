# Implementation Plan: Particle-Set Uncertainty Decomposition for the Tabular KDM
**Experiment**: experiments/exp_24/ · **Project**: pathology-reasoning · **Date**: 2026-08-18 · **Status**: Draft

---

## 0. Estimated runtime (derived from exp_23's measured per-model timing, same machine)

exp_23's `IMPLEMENTATION.md` §0 measured, on this machine (`/Users/fgonza/miniforge3/envs/pytorch/bin/python`,
`torch 2.10.0`), `n_train≈70-88, dim=12, n_comp≈70-88`, 300 full-batch epochs: **Identity encoder 0.66 s/model,
Linear(12,8) encoder 0.14 s/model**. exp_24's grid is much smaller than exp_23's (no re-sweep for Arms A/B):

| Stage | Models | Encoder mix | Est. time |
|---|---|---|---|
| Phase A (Arm C only): 18 configs × 100 splits | 1,800 | 900 identity + 900 linear | ≈ 0.66×900 + 0.14×900 ≈ **12 min** |
| Phase B, Arm A (frozen, `encoder=linear`, R=10 seeds): 88×10 | 880 | linear | ≈ **2 min** |
| Phase B, Arm B (frozen, `encoder=identity`, R=1 seed): 88×1 | 88 | identity | ≈ **1 min** |
| Phase B, Arm C (selected config, worst case linear, R=10): 88×10 | 880 | linear or identity | ≈ **2–6 min** |
| Degeneracy check (§1.5) + confidence heads (sklearn trees, sub-second each) | — | — | ≈ **2 min** |

**Total ≈ 20–25 min single-threaded.** No pilot-then-reduce step or multiprocessing needed at this scale;
still run a 5-split Phase-A smoke test first (§2) purely to catch implementation bugs before the full run, per
`DESIGN.md` §10 — not for timing.

---

## 1. Script: `experiments/exp_24/scripts/train.py`

Single self-contained script (repo convention — zero cross-experiment imports; verified by grep across
`experiments/*/scripts/*.py`), monolithic `main()`, mirroring exp_23's structure.

### 1.1 Setup, paths, reused constants
Copy verbatim from `experiments/exp_23/scripts/train.py`:
- `resolve_data_dir()` (lines 47–55) — same two-candidate fallback (`data/chimera26/...` then
  `Data/preprocessed_old/task1/`).
- `NUM_COLS`, `CAT_COLS` (lines 58–59).
- `load_cohort(data_dir)` (lines 79–126) — identical cohort assembly, identical `N=88, 54/34` asserts, identical
  `y_soft`/`c_weights`/`y_conf` construction (needed for Arm B and the confidence heads).

New constants:
```python
SIGMA_MULTS_C = [0.5, 1.0, 2.0]
EPS_LIST_C = [0.05, 0.10, 0.20]
GRID_C = [
    {"sigma_mult": sm, "eps": eps, "encoder": enc, "x_train": False, "y_train": True}
    for sm in SIGMA_MULTS_C for eps in EPS_LIST_C for enc in ["identity", "linear"]
]
assert len(GRID_C) == 18
```
`sigma_mult=0.25` dropped from exp_23's original range (DESIGN.md §3.3): dominated in both exp_23 arms
(0.487–0.518 vs ≥0.58 for `sigma_mult∈{1.0,2.0}`).

### 1.2 Feature construction (copy verbatim)
`build_features_fixed_categories(df, train_idx, val_idx, dre_categories)` (`exp_23/scripts/train.py:148–173`) —
the fixed-category-width `OneHotEncoder` variant KDM requires. `dre_categories = sorted(df_tab_labeled["dre"].unique())`,
asserted `len == 5`, exactly as exp_23.

### 1.3 Target/amplitude encodings
Copy verbatim: `to_amplitude_hard(y_binary_subset)` (176–178), `to_amplitude_soft(y_soft_subset)` (181–183).

New:
```python
def to_amplitude_hard_smoothed(y_binary_subset, eps):
    """eps=0.0 reduces exactly to to_amplitude_hard — used as the H3 structural identity."""
    y = y_binary_subset.astype(np.float32)
    p1 = y * (1 - eps) + eps / 2
    p0 = 1 - p1
    return np.stack([np.sqrt(p0), np.sqrt(p1)], axis=1).astype(np.float32)
```

### 1.4 KDM model builder and training step (copy verbatim)
`build_kdm(cfg, n_comp)` (262–271) — reused unmodified for all three arms; `cfg` only needs
`encoder`/`x_train`/`y_train` keys, present in both the Arm A/B JSON-loaded configs and `GRID_C` entries.
`init_and_check(model, X_tr, c_y_tr, sigma_mult, check_roundtrip=False)` (274–304) and
`train_kdm(model, X_tr, arm, y_binary_tr=None, y_soft_tr=None, lr=1e-3, epochs=300)` (307–321) — reused
unmodified. `arm` here means *loss selection* (`"hard"` → `F.nll_loss`, `"soft"` → soft CE); Arm C uses
`arm="hard"` despite its smoothed init, per `DESIGN.md` §2.2.

### 1.5 Particle-set signal extraction (new)
```python
from kdm.utils import pure2dm, dm2comp

def extract_particle_signals(model, X, eps=1e-7):
    """Per-sample signals from the KDM's output density matrix, before dm2discrete collapses it.
    Reproduces dm2discrete's own probs exactly (own_libs/kdm/kdm/utils.py:58-71: normalized weights,
    L2-normalized+squared vectors), so h_total is bit-identical to H(model(x))."""
    model.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(X, dtype=torch.float32)
        enc = model.encoder(Xt)
        rho_x = pure2dm(enc)
        rho_y = model.kdm(rho_x)                                  # (n, n_comp, dim_y+1)
        w, v = dm2comp(rho_y)
        w = w / w.sum(-1, keepdim=True)
        p = F.normalize(v, p=2, dim=-1, eps=1e-12) ** 2            # (n, n_comp, dim_y)

        p_mean = (w.unsqueeze(-1) * p).sum(dim=1)                  # == dm2discrete(rho_y)
        h_total = -(p_mean * torch.log(p_mean.clamp_min(eps))).sum(-1)

        h_particles = -(p * torch.log(p.clamp_min(eps))).sum(-1)   # (n, n_comp)
        h_aleatoric = (w * h_particles).sum(dim=1)
        h_epistemic = h_total - h_aleatoric

        h_weights = -(w * torch.log(w.clamp_min(eps))).sum(-1)
        log_ess = -torch.log((w ** 2).sum(-1).clamp_min(eps))
        w_max = w.max(dim=-1).values

        log_marginal = model.kdm.log_marginal(rho_x)
    return {
        "probs": p_mean.numpy(), "h_total": h_total.numpy(), "h_aleatoric": h_aleatoric.numpy(),
        "h_epistemic": h_epistemic.numpy(), "h_weights": h_weights.numpy(), "log_ess": log_ess.numpy(),
        "w_max": w_max.numpy(), "log_marginal": log_marginal.numpy(),
    }
```
`h_total`/`h_epistemic` are asserted, not just reported, to equal `H(model(x))` on a spot sample in `main()`
(catches a sign/axis bug in `dm2comp`/`F.normalize` immediately rather than downstream in a metric table).

### 1.6 Phase A — Arm C grid search (100 MCCV splits)
`kdm_phase_a_arm_c(df_tab_labeled, df_design_labeled, y_binary, dre_categories, n_splits=100)` — same
structure as `exp_23/scripts/train.py:327-389`'s `kdm_phase_a`, restricted to `GRID_C` (18 configs) and
`c_y_tr = to_amplitude_hard_smoothed(y_binary_tr, cfg["eps"])`, loss always `"hard"`. `torch.manual_seed(42)`
once per split (not per config) — paired comparison across configs within a split, exp_23's convention.
Select argmax mean validation Macro-F1 → freeze `best_cfg_c`.

### 1.7 Phase B — LOOCV (88 folds), all three arms, with particle signals
```python
def kdm_phase_b_particles(df_tab_labeled, y_binary, y_soft, loss_arm, c_y_builder, cfg, dre_categories,
                           n_seeds=10, track_drift=False):
    """Generalizes exp_23's kdm_phase_b (train.py:395-484): same LOOCV/seed/mode-vote/metrics structure,
    but (a) c_y init is supplied by c_y_builder(y_binary_tr, y_soft_tr) instead of being hardcoded per
    arm, letting Arms A/B/C share one loop, and (b) extract_particle_signals replaces the
    entropy/log_marginal-only extraction, adding the 5 new signals."""
```
Per fold/seed: `torch.manual_seed(seed)` immediately before `build_kdm` (exp_23's determinism-critical
ordering, §1.7 note there — the only randomness anywhere in the pipeline is `nn.Linear`'s init, which must
be the first random draw consumed after the seed is set). If `track_drift`, record
`(c_y_after - c_y_init).abs().max()` per fold — feeds `cy_drift.png` and the pitfall check in `DESIGN.md` §7.
Returns `p_mean`, `y_pred`, binary metrics (identical schema to exp_23's), and `signals_mean` (7 arrays,
seed-averaged, mirroring how exp_23 seed-averages `entropy_mean`/`log_marginal_mean`).

Per-arm calls in `main()`:
```python
cfg_a = {**best_hparams_exp23["kdm"]["hard"], }             # sigma_mult, x_train=False, y_train=False, encoder
res_a = kdm_phase_b_particles(df_tab_labeled, y_binary, y_soft, "hard",
                               lambda yb, ys: to_amplitude_hard(yb), cfg_a, dre_categories, n_seeds=10)

cfg_b = {**best_hparams_exp23["kdm"]["soft"]}
res_b = kdm_phase_b_particles(df_tab_labeled, y_binary, y_soft, "soft",
                               lambda yb, ys: to_amplitude_soft(ys), cfg_b, dre_categories, n_seeds=10)

res_c = kdm_phase_b_particles(df_tab_labeled, y_binary, y_soft, "hard",
                               lambda yb, ys: to_amplitude_hard_smoothed(yb, best_cfg_c["eps"]),
                               best_cfg_c, dre_categories, n_seeds=10, track_drift=True)
```
`best_hparams_exp23` is loaded from `experiments/exp_23/results/best_hparams.json` — **not re-swept**
(`DESIGN.md` §3.2/§7). `n_seeds=10` requested uniformly; `kdm_phase_b_particles` internally collapses to a
single deterministic seed when `cfg["encoder"] == "identity"`, exactly as exp_23.

**Reproduction checks** (validate the copied pipeline before trusting new numbers):
```python
assert abs(res_b["metrics"]["macro_f1"] - 0.6694214876033058) < 1e-6   # Arm B: fully deterministic
assert abs(res_a["metrics"]["macro_f1"] - 0.5636363636363637) < 1e-4   # Arm A: same seeds/order ⇒ expect exact match; loose tolerance only for library/version drift
```
Both arms are fully deterministic pipelines given identical seeds and call order (`torch.manual_seed(seed)`
resets global RNG state independent of history — verified during design review), so a mismatch beyond the
tolerance means the copy diverged from exp_23, not natural stochasticity, and must be debugged before
proceeding.

### 1.8 Structural check (H3) — `degeneracy_check` (new)
```python
def degeneracy_check(df_tab_labeled, y_binary, dre_categories, encoder="linear", sigma_mult=2.0, epochs=300):
    """Confirms DESIGN.md §2.1: a one-hot c_y is a gradient fixed point even with y_train=True and
    gradients flowing. Trains one full-cohort model (n_comp=88, structural check only — never scored,
    no leakage concern) at eps=0.0 (== to_amplitude_hard exactly) with y_train=True and asserts c_y does
    not move."""
    idx = np.arange(len(y_binary))
    X_tr, _ = build_features_fixed_categories(df_tab_labeled, idx, idx[:1], dre_categories)
    cfg = {"encoder": encoder, "x_train": False, "y_train": True}
    c_y_tr = to_amplitude_hard_smoothed(y_binary, eps=0.0)
    torch.manual_seed(0)
    model = build_kdm(cfg, n_comp=len(idx))
    init_and_check(model, X_tr, c_y_tr, sigma_mult, check_roundtrip=False)
    c_y_before = model.kdm.c_y.detach().clone()
    train_kdm(model, X_tr, "hard", y_binary_tr=y_binary, epochs=epochs)
    max_drift = (model.kdm.c_y.detach() - c_y_before).abs().max().item()
    return {"max_cy_drift_eps0": max_drift, "passed": bool(max_drift < 1e-9)}
```
Written to `results/degeneracy_check.json`; `main()` asserts `passed`. Also asserts
`max(res_a["signals_mean"]["h_aleatoric"]) < 1e-6` and `np.allclose(res_a["signals_mean"]["h_epistemic"], res_a["signals_mean"]["h_total"])` directly on the Phase B Arm A output — the same claim, checked twice (a
clean isolated construction and the actual LOOCV output), since §7 forbids reporting those two Arm-A columns
as informative and this is the enforcement point.

### 1.9 Confidence heads
**1D per-signal (copy verbatim):** `fit_1d_confidence_signal` (506–567), `apply_1d_confidence_signal`
(570–576), `score_confidence` (579–587) from `exp_23/scripts/train.py`, applied to each of the 7 signals ×
3 arms = 21 rows. Entropy-family signals (`h_total`, `h_epistemic`) sign-flipped (`-signal`) before fitting,
matching exp_23's `-entropy_mean` convention, so the Arm-A `h_total` row is byte-comparable to exp_23's
published `entropy_hard` row.

⚠️ **Discovered during the smoke test, not anticipated in `DESIGN.md`:** `fit_1d_confidence_signal` can
raise on two inputs it doesn't gracefully resolve — every split's depth-2 tree collapsing to a constant
prediction across the full sweep (`RuntimeError`), or an exact tie in the per-split direction vote
(`AssertionError` on `np.sign(0)`). exp_23's own four signals never hit either case, but a smoke run with
a reduced seed count did, on both failure modes in succession, confirming they're reachable in practice, not
just in principle. `fit_1d_confidence_signal_safe` wraps the verbatim function (which is left unmodified)
and falls back to 33rd/67th-percentile thresholds with `direction=+1` on either exception, flagging
`"degenerate_fallback": true` in that row's output so it's visible in `confidence_metrics.json` and callable
out in `summary.md` rather than silently accepted.

**Multivariate frozen tree ensemble (new):**
```python
def fit_multivariate_confidence_head(signal_matrix, y_conf, df_design_labeled, n_splits=100,
                                      max_depth=3, random_state=42):
    """One DecisionTreeClassifier per MCCV split, trained on that split's train rows only. Patient i's
    Phase-B prediction is the majority vote over ONLY the trees whose split had i in validation (split_i
    == 1) -- i.e. trees that never saw patient i during fitting. Required because a depth-3 tree over 7
    signals has far more memorization capacity than exp_17's 2-scalar 1D threshold, so exp_17's
    whole-cohort application (tolerated there because dilution over 100 splits makes any one patient's
    influence negligible) would leak here. Split design confirmed fixed 70 train / 18 val per split, every
    patient a validation member 11-31 times (mean 20.5) across the 100 splits -- always >=1 held-out vote."""
    n = signal_matrix.shape[0]
    votes = [[] for _ in range(n)]
    for split_idx in range(n_splits):
        split_vals = df_design_labeled[f"split_{split_idx}"].values
        train_mask, val_idx = split_vals == 0, np.where(split_vals == 1)[0]
        dt = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=random_state)
        dt.fit(signal_matrix[train_mask], y_conf[train_mask])
        for i, p in zip(val_idx, dt.predict(signal_matrix[val_idx])):
            votes[i].append(int(p))
    assert all(len(v) > 0 for v in votes), "a patient received zero held-out votes"
    final_pred = np.array([int(np.bincount(v, minlength=3).argmax()) for v in votes])
    votes_per_patient = np.array([len(v) for v in votes])
    return final_pred, votes_per_patient
```
Called twice per arm: once on the full 7-signal matrix (headline result), once on the
`[h_total, log_marginal]` 2-column ablation (exp_23's original signal set) under the identical held-out-vote
protocol — isolates whether any gain is from the new signals or from ensembling itself. Scored with
`score_confidence`. Both variants + the 1D rows write into `results/confidence_metrics.json`; Arm B's rows
are flagged `"target_informed": true` throughout, matching exp_23's convention for `entropy_soft`/`log_marginal_soft`.

### 1.10 Artifacts
`results/`: `best_hparams.json` (Arm C selection + copied Arm A/B provenance), `grid_search_results.csv`
(Arm C, 18×100), `loocv_metrics.json` (3-arm binary metrics + `fuzzy_knn_reference` copied verbatim from
`exp_23/results/loocv_metrics.json` + McNemar Arm C vs. reference), `oof_particle_signals.csv` (88 rows ×
3 arms × 7 signals + `confidence_annotation`), `confidence_metrics.json`, `degeneracy_check.json`,
`git_commit.txt`. `reports/figures/`: `arm_c_grid_search_curves.png`, `cy_drift.png` (bar of per-particle
`|c_y_after − c_y_before|`, Arm-A-style eps=0 model vs. Arm C's actual selected-config model, both from
full-cohort fits), `particle_signal_scatter.png` (Arm C `h_aleatoric` vs. `h_epistemic`, colored by
`confidence`), `signal_correlation_heatmap.png` (Arm C's 7×7 signal correlation), `confidence_confusion_matrix.png`
(best-scoring head). `reports/summary.md` via the `ml-experiment-reporter` skill once results exist.

---

## 2. Command Lines

### Environment check (run first)
```bash
/Users/fgonza/miniforge3/envs/pytorch/bin/python -c "import kdm, torch, sklearn, scipy, pandas, numpy; print('ok')"
```

### Smoke test (bug-catching, not timing — §0 already grounds timing)
Run with `n_splits=5` for Phase A and a 6-fold subset of Phase B (env var or CLI flag `--smoke`), confirm no
exceptions, confirm `degeneracy_check` passes, confirm Arm B's tiny-N reproduction is in the right ballpark,
before committing to the full run.

### Execution
```bash
git log -1 --format="%H %s" > experiments/exp_24/results/git_commit.txt
/Users/fgonza/miniforge3/envs/pytorch/bin/python experiments/exp_24/scripts/train.py
```
Expected wall time: **~20–25 min** (§0). Single process; acceptable to run in the background.

---

## 3. Post-Execution (per repo workflow, after results exist and are reviewed)
1. `reports/summary.md` via `ml-experiment-reporter`.
2. Add **both** `exp_23` and `exp_24` rows to `experiments/INDEX.md` (exp_23 has committed results but was
   never indexed — its own `IMPLEMENTATION.md` §3 lists this as outstanding).
3. Append `.logbook.md` entries for both experiments (Spanish, `## [YYYY-MM-DD] - Title` /
   `Current Project State` / `Specific Advancements` / `Next Tasks`, matching the exp_22 entry's style).
4. Insert `.discussion.md` entries for both experiments **adjacent to the existing exp_22 entry (~line 118),
   not at end-of-file** — the file is not chronologically ordered at the tail (English,
   `Principal Investigator Proposal` / `Co-Investigator Analysis` / `Consensus`).
