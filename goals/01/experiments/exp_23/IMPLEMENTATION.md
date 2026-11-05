# Implementation Plan: Tabular KDM Biopsy Decision Prediction
**Experiment**: experiments/exp_23/ · **Project**: pathology-reasoning · **Date**: 2026-08-18 · **Status**: Draft

---

## 0. Measured runtime (grounds the plan below, not an estimate)

Timed directly on this machine (`/Users/fgonza/miniforge3/envs/pytorch/bin/python`, `torch 2.10.0`, 4 threads,
10 CPU cores) with `n_train=70`, `dim=12`, `n_comp=70`, 300 full-batch epochs:

| Encoder | Time / model (300 epochs) |
|---|---|
| `Identity` | 0.66 s |
| `Linear(12,8)` | 0.14 s |

Primary Phase A sweep: 100 splits × 32 configs × 2 arms = 6400 models ≈ **0.7–1.2 h single-threaded**. Phase B:
88 folds × 2 arms × 10 seeds = 1760 models (n_comp≈87, slightly larger) ≈ **20–30 min**. Both comfortably
tractable single-threaded in well under the design's contingency budget — **no pilot-then-reduce step, no
multiprocessing, and no early stopping are needed**; DESIGN.md's "pilot on 5 splits and extrapolate" caveat is
satisfied by this measurement and the grid runs as specified in DESIGN.md §3 without modification.

---

## 1. Script: `experiments/exp_23/scripts/train.py`

Single script, run once, mirroring exp_13's monolithic `main()` structure. Sections below are the order of
execution.

### 1.1 Setup & path resolution
```python
script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
for candidate in [project_root/"data"/"chimera26"/"preprocessed"/"task1",
                   project_root/"Data"/"preprocessed_old"/"task1"]:
    if candidate.exists():
        data_dir = candidate; break
else:
    raise FileNotFoundError(...)
```
`pytorch` env only (`import kdm` fails elsewhere). Imports include `copy` (for the deepcopy'd round-trip probe in
§1.5). `torch.manual_seed(seed)` is called once per `(arm, seed, fold)` iteration in Phase B, **before**
`build_kdm` — never globally, and never inside `train_kdm` — because the only source of randomness in the whole
pipeline is `nn.Linear`'s weight init, which happens during model construction, not during training; setting the
seed any later would leave it controlling nothing (see §1.7).

### 1.2 Load, align, and validate the cohort (copy `exp_13/scripts/train.py:33–55` verbatim)
Load `clinical_data_tabular.csv`, `biopsy_decision.csv`, `clinical_reasoning.csv` with
`dtype=str, keep_default_na=False` for the columns where `'NONE'` matters, then cast numerics; load
`experiments/exp_4/results/mccv_design.csv` (plain `read_csv`, no missingness concern — no `'NONE'` tokens).
Align on `patient_id`, filter to `biopsy_decision != "NONE"`.

```python
assert len(y_binary) == 88, f"expected N=88, got {len(y_binary)}"
assert int((y_binary==1).sum()) == 54 and int((y_binary==0).sum()) == 34, "class-count mismatch"
```
Abort loudly on failure — this is the identity check on the loaded data, not a warning.

### 1.3 Feature & target construction
- `num_cols = ["age","psa","vol","pirads","psad","psav","psap"]`, `cat_cols = ["dre"]`.
- `build_features(df, train_idx, val_idx) -> X_tr, X_va`: `MinMaxScaler` + `OneHotEncoder(handle_unknown="ignore",
  sparse_output=False)`, both fit on `train_idx` only, `np.hstack`. Returns 12-dim arrays. Used identically by the
  KDM loops and the reference KNN loop below. `assert np.isfinite(X_tr).all() and np.isfinite(X_va).all()` right
  after construction — a NaN here would otherwise propagate silently through the RBF kernel into NaN loss instead
  of raising.
- Soft target `y_soft` — exp_13's exact formula (`clear:1.00, borderline:0.50, uncertain:0.25`, `.fillna(1.00)`,
  `ỹ = 0.5 ± 0.5·c`).
- `to_amplitude(y_binary=None, y_soft=None) -> c_y (n,2) float32`: hard → `F.one_hot(y_binary,2).float()`
  (already unit-norm, no sqrt needed); soft → `np.stack([sqrt(1-y_soft), sqrt(y_soft)], axis=1)`.

### 1.4 Recomputed Fuzzy KNN reference (exp_13's exact pipeline, inline)
Copy `exp_13/scripts/train.py:76–246` verbatim, factoring the Phase B LOOCV loop into a small helper so it can be
re-invoked with a forced config (needed below):
```python
def knn_loocv_macrof1_and_cm(k, weights, metric):
    """Runs exp_13's Phase B LOOCV loop (lines 209-236) for one fixed (k, weights, metric)."""
    ...  # identical body to exp_13:209-236, parameterized instead of reading best_hparams
    tn, fp, fn, tp = confusion_matrix(oof_y_true, oof_y_pred, labels=[0, 1]).ravel()
    return f1_score(oof_y_true, oof_y_pred, average="macro"), (tp, tn, fp, fn), oof_p_soft, oof_y_pred
```
Grid: `k∈[1,3,5,7,9,11,13,15,17,21,25]×weights[uniform,distance]×metric[euclidean,manhattan,cosine]` = 66
configs; Phase A MCCV → `best_hparams_reference`; Phase B → `knn_loocv_macrof1_and_cm(**best_hparams_reference)`.
Runs in seconds (sklearn `KNeighborsRegressor`). Store under `results/loocv_metrics.json["fuzzy_knn_reference"]`.
```python
macro_f1_ref, cm_ref, oof_p_soft_reference, oof_y_pred_reference = knn_loocv_macrof1_and_cm(**best_hparams_reference)
assert abs(macro_f1_ref - 0.6364) < 0.01, \
    f"recomputed Fuzzy KNN reference Macro-F1 {macro_f1_ref:.4f} diverges from exp_13's published 0.6364 — data-path substitution not input-identical"

if cm_ref != (40, 18, 16, 14):   # (tp, tn, fp, fn)
    print(f"MISMATCH: selected config = {best_hparams_reference}, got (tp,tn,fp,fn)={cm_ref}")
    print(df_grid_reference.head(5).to_string())
    # Disambiguate tie-breaking (benign — k=1 uniform and k=1 distance are tied in exp_13's own
    # grid_search_results.csv) from an actual data difference (fatal): force exp_13's exact published
    # config and recheck before concluding the loaded inputs differ.
    _, cm_forced, _, _ = knn_loocv_macrof1_and_cm(k=1, weights="uniform", metric="euclidean")
    assert cm_forced == (40, 18, 16, 14), \
        f"reference confusion matrix mismatch even with exp_13's exact published config forced: {cm_forced} — data-path substitution not input-identical"
```
Per DESIGN.md §6, this checks the recomputed **Macro-F1 and confusion matrix**, not the selected `(k, weights,
metric)` — a different *selected* config alone is not fatal, since the primary sweep can legitimately land on a
statistical tie. The forced-config recheck above is what separates that benign case from the fatal one; only the
fatal one should stop the run.

### 1.5 KDM model builder and training step
```python
def build_kdm(cfg, n_comp):
    # encoded_size derives from cfg["encoder"] rather than being a separate argument — a caller passing
    # the wrong encoded_size for a given encoder would fail fast on a shape mismatch, but there's no reason
    # to allow that footgun at all when it's fully determined by cfg.
    if cfg["encoder"] == "identity":
        encoder, encoded_size = nn.Identity(), 12
    else:
        encoder, encoded_size = nn.Linear(12, 8), 8
    return KDMClassModel(encoded_size=encoded_size, dim_y=2, encoder=encoder, n_comp=n_comp,
                          sigma=0.5, sigma_trainable=True,
                          x_train=cfg["x_train"], y_train=cfg["y_train"], w_train=True)

def init_and_check(model, X_tr, c_y_tr, sigma_mult, check_roundtrip=False):
    with torch.no_grad():
        enc_sub = model.encoder(torch.as_tensor(X_tr, dtype=torch.float32))
    init_kdm_layer(model.kdm, enc_sub.detach(), torch.as_tensor(c_y_tr, dtype=torch.float32),
                    init_sigma=True, sigma_mult=sigma_mult)
    if check_roundtrip:
        probe = copy.deepcopy(model)   # never touch the live model's sigma
        probe.kernel.sigma = 0.05
        with torch.no_grad():
            probs = probe(torch.as_tensor(X_tr, dtype=torch.float32))
        target = c_y_tr[:, 1] ** 2   # ỹ recovered from stored amplitude
        assert np.abs(probs[:, 1].numpy() - target).max() < 1e-4, "amplitude round-trip failed"

def train_kdm(model, X_tr, arm, y_binary_tr=None, y_soft_tr=None, lr=1e-3, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X_tr, dtype=torch.float32)
    for _ in range(epochs):
        probs = model(Xt)
        if arm == "hard":
            loss = F.nll_loss(torch.log(probs.clamp_min(1e-7)), torch.as_tensor(y_binary_tr, dtype=torch.long))
        else:  # soft
            t = torch.as_tensor(np.stack([1 - y_soft_tr, y_soft_tr], axis=1), dtype=torch.float32)
            loss = -(t * torch.log(probs.clamp_min(1e-7))).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model
```
The amplitude round-trip assertion (`check_roundtrip=True`) runs **once**, on the first split × first config of
each arm, immediately post-init with σ temporarily narrowed — never on a trained model (per DESIGN.md §"known
pitfalls", the property does not hold once `x_train`/σ have moved).

### 1.6 Phase A — MCCV grid search (100 splits × 32 configs × 2 arms)
```python
grid = [{"sigma_mult": sm, "x_train": xt, "y_train": yt, "encoder": enc}
        for sm in [0.25, 0.5, 1.0, 2.0]
        for xt in [True, False]
        for yt in [True, False]
        for enc in ["identity", "linear"]]   # len(grid) == 32
```
For `arm in ["hard","soft"]`, for `split_idx in range(100)`: build `X_tr, X_va` via §1.3; for each of the 32
configs: `torch.manual_seed(42)` **immediately before** `build_kdm` — every config within a split then draws
`nn.Linear`'s init from the same fixed state, so the 32-way ranking is a paired comparison rather than confounded
by init luck across configs, and the sweep is reproducible → `build_kdm` → `init_and_check` (round-trip check
only at `split_idx==0 and cfg_idx==0`) → `train_kdm` (single seed — Phase A stays single-seed per split per
DESIGN.md §5, the 100-split average already controls variance) → predict `X_va` → threshold `p(yes)≥0.5` →
Macro-F1/Acc/Sens/Spec. Aggregate to
`grid_search_results.csv` (columns: `arm, cfg_id, sigma_mult, x_train, y_train, encoder, mean_macro_f1,
std_macro_f1, mean_acc, mean_sens, mean_spec`), select per-arm argmax → `best_hparams.json` (keyed by arm).

### 1.7 Phase B — LOOCV, R=10 seeds

⚠️ **Determinism is per-encoder, not per-corner.** With `encoder="identity"` there is *no* randomly initialized
parameter anywhere in the model — `c_x`, `c_y`, `c_w`, and σ are all set from data by `init_kdm_layer`, and
training is deterministic full-batch Adam. `x_train`/`y_train` only control whether those deterministic tensors
receive deterministic updates; they don't introduce randomness. **Every `encoder=="identity"` config is
deterministic**, not just `x_train=False, y_train=False`. Only `encoder=="linear"` configs have a genuine random
draw (`nn.Linear(12,8)`'s weight init) — and only if that draw actually varies across the 10 seeds, which
requires the seed to be set *before* it, not inside `train_kdm` (see below). This corrects DESIGN.md §3's
narrower note, which named only one deterministic corner; if `identity` wins Phase A, McNemar is the only
inferential tool available, which is exactly why it is the primary decision rule rather than a fallback.

For `arm in ["hard","soft"]`, `seed in range(10)`, `LeaveOneOut()` over the 88-row cohort:
```python
torch.manual_seed(seed)          # BEFORE build_kdm — the only randomness is nn.Linear's init inside it
model = build_kdm(best_cfg, ...)
init_and_check(model, X_tr, c_y_tr, best_cfg["sigma_mult"], check_roundtrip=False)
train_kdm(model, X_tr, arm, ...)   # no seed arg — setting it here would be too late, see above
```
Build `X_tr, X_va` via §1.3 with the frozen per-arm config from §1.6 (**no re-fitting of hyperparameters**),
`n_comp=87`. Record `probs_va`, `entropy_va = -(probs_va*log(probs_va.clamp_min(1e-7))).sum(-1)`, and
`log_p_x_va = model.kdm.log_marginal(pure2dm(model.encoder(X_va)))`.

**For Arm A only**, also evaluate the same model in-sample: `probs_tr = model(X_tr)`, `entropy_tr =
-(probs_tr*log(probs_tr.clamp_min(1e-7))).sum(-1)`, `log_p_x_tr = model.kdm.log_marginal(pure2dm(model.encoder(X_tr)))`.
This is what §1.8's 2D joint variant fits on — it needs the 87 training patients' in-sample signals from *this
same fold's* model, which is otherwise nowhere else computed; store `(entropy_tr, log_p_x_tr)` per
`(fold, seed)` rather than discarding them after the fold's held-out prediction is recorded.

If `best_cfg["encoder"] == "identity"`: run the fold **once** (seed is inert), report `"deterministic": true` and
omit a std rather than computing 0.0 from 10 identical runs. Otherwise run all 10 seeds.

Per arm: `oof_p_soft[seed, patient]`, `oof_entropy[seed, patient]`, `oof_log_marginal[seed, patient]`. Reduce
across seeds: `p_mean = oof_p_soft.mean(0)`, `entropy_mean`, `log_marginal_mean`. **Hard predictions and every
downstream metric use `y_pred = (p_mean >= 0.50)`** — matching exp_13's own threshold protocol and keeping
AUROC/Brier (computed from `p_mean`) and Macro-F1/accuracy/confusion-matrix (computed from `y_pred`) describing
the *same* classifier. Separately compute `mode_vote = mode of (oof_p_soft>=0.5) across seeds` and report
`agreement_with_mean_threshold = mean(y_pred == mode_vote)` as a stability diagnostic only — it does not feed any
headline metric.

Metrics computed on `p_mean`/`y_pred` against `oof_y_true`: `macro_f1, accuracy, sensitivity, specificity,
auroc, brier_score, tp, tn, fp, fn, total_cases` — same key set as `exp_13/results/loocv_metrics.json`, nested
per arm plus `"fuzzy_knn_reference"` (§1.4) in one `results/loocv_metrics.json`.

**McNemar's test** (primary decision rule). `statsmodels` is **not installed** in the `pytorch` env (checked
directly) — implemented manually rather than adding a new dependency: build the 2×2 contingency table of
`(kdm_correct, knn_correct)` booleans over the 88 paired predictions (`y_pred` from §1.7, not `mode_vote`) for
the better-of-two-arms KDM vs. the reference, take the two discordant-pair counts `b` (KDM right/KNN wrong) and
`c` (KDM wrong/KNN right), and

**Multiplicity note:** DESIGN.md's decision rule names "the better KDM arm," so both arms' McNemar tests are
computed and reported (not just the smaller p-value) — testing two arms against the same reference is two shots
at significance, and `summary.md` states that plainly rather than silently presenting only the significant one.

For the exact test itself,
report the exact two-sided binomial test p-value on `min(b,c) ~ Binomial(b+c, 0.5)` via
`scipy.stats.binomtest(min(b,c), b+c, 0.5)` — the standard exact form of McNemar's test for small discordant
counts (the χ² approximation is unreliable when `b+c` is small, which is plausible here at N=88). Store
`{"b":..., "c":..., "statistic": min(b,c), "pvalue":...}` in `loocv_metrics.json["mcnemar"]`.

### 1.8 Secondary objective — confidence prediction from native uncertainty

**Data source: the LOOCV OOF signal vectors from §1.7, not a new pass.** exp_17's own Phase A does not compute a
validation split at all for the confidence sub-task — it takes the single 88-length **LOOCV OOF** `ici_fuzzy`
vector (produced once, upstream, by exp_16) and for each of the 100 MCCV splits fits a tree on
`ici_fuzzy[train_mask]` only, never evaluating a held-out portion in Phase A (validation happens only once, in
Phase B, over the full 88). Reusing that pattern literally — rather than computing a separate per-split
validation signal — means exp_23's `entropy_oof`/`log_marginal_oof` (88-length, from §1.7, averaged across the
R=10 seeds) are subset by each split's `train_mask` exactly as `ici_fuzzy` is. This is both the structurally
faithful replication ("exp_17's exact machinery") and the cheaper option — no extra 100-split training pass.

Four 1D signals, each run through the pattern below independently:

| Signal | Definition |
|---|---|
| `entropy` (Arm A) | `-entropy_oof` from Arm A (negated so higher = more confident, matching the convention the threshold code assumes) |
| `log_marginal` (Arm A) | raw `log_marginal_oof` from Arm A |
| `entropy` (Arm B) | as above, from Arm B — reported **target-informed** |
| `log_marginal` (Arm B) | as above, from Arm B — reported **target-informed** |

**Phase A — per split** (`DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)` on
`signal_oof[train_mask].reshape(-1,1)` vs. `y_conf[train_mask]`, exactly exp_17's fit call):
- Extract sorted `tree_.threshold[tree_.threshold != -2]` → up to 2 cut points `(t1, t2)`.
- **Fallback, scale-relative instead of exp_17's magic ICI numbers, and monotone by construction.** exp_17's
  literal fallback (`0.10, 0.30`, and `t1+0.1`) is calibrated to ICI's `[0,1]` range; `entropy` lives in
  `[0, ln 2]` and `log_marginal` observed around `[-40, -5]` in the smoke test — applying `0.10/0.30` to
  `log_marginal` would classify every patient "clear" with no error raised. Instead, on `len(thresholds) < 2`:
  `t1, t2 = np.percentile(signal_oof[train_mask], [33, 67])`; if exactly 1 threshold is found, keep it as `t1`
  and set `t2 = max(np.percentile(signal_oof[train_mask], 67), t1 + (signal_oof[train_mask].max() - t1) / 2)` —
  guarantees `t2 > t1` by construction regardless of where `t1` falls in the training range, rather than a
  conditional retry that can still invert. **Count and record how often the fallback fires per signal** — a high
  count means the frozen thresholds are quantile-based rather than tree-derived, which changes how much weight
  the number deserves.
- **Direction, determined empirically per split via rank correlation and frozen by majority vote — not
  assumed.** Sweep `np.linspace(signal_oof.min(), signal_oof.max(), 50)`, reshape, `tree.predict`.
  **Degenerate-sweep guard first**: if `tree.predict(sweep)` has a single unique value (the same all-one-class
  case the `len(thresholds)==0` fallback exists for), skip this split's direction vote entirely (tally it as
  "degenerate" alongside the non-monotone count) — `spearmanr` on a constant array returns NaN, and
  `warnings.filterwarnings("ignore")` (inherited from exp_13/exp_17) would silently swallow scipy's
  `ConstantInputWarning`, so an unguarded NaN would propagate through `sign()` and flip the *else* branch on in
  Phase B, silently inverting every one of the 88 predictions with no visible error. Otherwise set
  `rho, _ = spearmanr(np.arange(50), tree.predict(sweep))` (tuple-unpacked, not `.statistic`, for scipy-version
  safety — matches exp_17's own call convention) and `direction_s = np.sign(rho)`; rank correlation degrades
  gracefully on a non-monotone region order (e.g. a depth-2 tree emitting class order `1, 0, 2` along the sweep,
  which a plain non-decreasing/non-increasing check has no branch for). Count and record non-monotone splits
  (`abs(rho) < 0.1`) per signal alongside `fallback_count_of_100` and the degenerate-sweep count. After 100
  splits, freeze `direction = np.sign(np.nansum(direction_s over surviving splits))`, and
  `assert direction in (1, -1)` (not `0` or `nan`) before Phase B uses it — a tie or an all-degenerate signal
  should raise, not silently pick a branch. `entropy` is already pre-oriented so `+1` is expected; for
  `log_marginal` this is a genuine empirical question, not an assumption, and the frozen sign is reported as a
  finding in `summary.md`.

Average `(t1, t2)` over the 100 splits → `(τ̄1, τ̄2)`, frozen alongside the frozen `direction`.

**Phase B** — apply directionally on the full 88-length OOF vector:
```python
if direction == +1:
    pred = np.where(val < t1_bar, 0, np.where(val < t2_bar, 1, 2))
else:
    pred = np.where(val < t1_bar, 2, np.where(val < t2_bar, 1, 0))
```

**Joint 2D variant** (`[entropy, log_marginal]`, Arm A only — omitted for Arm B since it adds no signal beyond
the 1D pair already marked target-informed): follows the exp_12/exp_22 *local-fit* pattern instead of the 1D
frozen-threshold pattern above — no MCCV meta-threshold search. The fit happens **inside §1.7's LOOCV loop
itself**, using the `(entropy_tr, log_p_x_tr)` in-sample pairs that loop now retains per `(fold, seed)`: for each
fold, a fresh `DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)` is fit on the 87
training patients' `[entropy_tr, log_p_x_tr]` (deliberately *in-sample*, not OOF, unlike the 1D signals above —
this mirrors exp_12/exp_22's own practice and is noted here explicitly so the difference reads as intentional,
not a mismatch with §1.8's 1D treatment) plus their known `confidence` labels, then applied to predict the
held-out patient's class. This is leak-free — the held-out patient's label is never used to fit — and mirrors
how the KDM model and scalers are themselves refit per fold. **Seed handling:** since a `linear`-encoder config
produces a different in-sample signal matrix per seed, the 2D tree is fit and applied once per `(fold, seed)`,
and the held-out patient's final predicted class is the majority vote across the R seeds (or the single
deterministic result if the frozen config is `identity`) — the same seed-reduction pattern as the primary
biopsy-decision prediction in §1.7.

All five signals' metrics (`macro_f1, accuracy, spearman_rho, spearman_pvalue, total_cases`, plus for the four
1D signals `meta_threshold_1/2, direction, fallback_count_of_100`) go into `results/confidence_metrics.json`,
each compared inline against the exp_17 baseline (Macro-F1 0.4470, accuracy 57.95%, ρ=0.2790, p=0.0085).

### 1.9 Artifacts
- `results/best_hparams.json` — `{"hard": {...}, "soft": {...}}`.
- `results/grid_search_results.csv` — 64 rows (32 configs × 2 arms).
- `results/loocv_metrics.json` — `{"kdm_hard": {...per-seed + mean/std...}, "kdm_soft": {...}, "fuzzy_knn_reference": {...}, "mcnemar": {...}}`.
- `results/oof_predictions.csv` — `patient_id, ground_truth_biopsy, confidence_annotation, certainty_weight,
  kdm_hard_p_mean, kdm_hard_pred, kdm_soft_p_mean, kdm_soft_pred, reference_knn_p, reference_knn_pred`.
- `results/confidence_metrics.json` — per §1.8.
- `results/git_commit.txt` — `git log -1 --format="%H %s"`, written **before** the run starts.
- `reports/figures/grid_search_curves.png` — Macro-F1 vs `sigma_mult`, one line per `(x_train,y_train,encoder)`
  combination, faceted by arm (2 subplots).
- `reports/figures/confusion_matrix.png` — best KDM arm, LOOCV.
- `reports/figures/roc_curve.png` — KDM (both arms) vs. recomputed Fuzzy KNN, overlaid.
- `reports/figures/confidence_confusion_matrix.png` — best confidence signal, 3×3.
- `reports/figures/uncertainty_scatter.png` — entropy (x) vs. log_marginal (y), Arm A, colored by
  `confidence_annotation`.
- `reports/summary.md` — contrasts exp_23 vs. exp_13 (primary) and vs. exp_17 (secondary), states the McNemar
  result plainly, and explicitly notes if the selected primary config was the deterministic corner.

Report-writing uses raw f-strings for any LaTeX (`rf"...$\text{{std}}$..."`) — exp_13's non-raw string emitted a
literal tab character; exp_23 does not repeat it. Class totals in prose are 54/34, never exp_13's incorrect
56/32.

---

## 2. Command Lines

### Environment check (run first)
```bash
/Users/fgonza/miniforge3/envs/pytorch/bin/python -c "import kdm, torch, sklearn, scipy, pandas, numpy; print('ok')"
```

### Execution
```bash
git log -1 --format="%H %s" > experiments/exp_23/results/git_commit.txt
/Users/fgonza/miniforge3/envs/pytorch/bin/python experiments/exp_23/scripts/train.py
```

Expected wall time: **~1.5–2 h total** (§0), single process, no background execution needed but acceptable to run
in the background given the length.

---

## 3. Post-Execution (per repo workflow, after results exist and are reviewed)
1. Add the `exp_23` row to `experiments/INDEX.md`.
2. Append a `.logbook.md` entry (Spanish, `## [YYYY-MM-DD] - Title` / Current Project State / Specific
   Advancements / Next Tasks format).
3. Append a `.discussion.md` entry (English, `### [YYYY-MM-DD] - Topic` / Principal Investigator Proposal /
   Co-Investigator Analysis / Consensus format) — insert adjacent to the existing exp_22 entry (2026-08-06), not
   at end-of-file, since `.discussion.md` is not chronologically ordered at the tail.
