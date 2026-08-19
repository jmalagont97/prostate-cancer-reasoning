# Implementation Plan: Multimodal Memory-Based KDM (`MemKDM`)
**Experiment**: experiments/exp_25/ · **Project**: pathology-reasoning · **Date**: 2026-08-18 · **Status**: Complete

---

## 0. Measured runtime (grounds the plan below, not an estimate)

Per `DESIGN.md` §9 / the approved planning discussion's "pilot timing — a hard gate", per-fit cost was
measured directly on this machine (`conda run -n pytorch`, `torch 2.10.0`) by calling `MemKDM.fit`
through the real `src/` APIs on the actual cohort, `n_train=70`, 300 full-batch epochs, before writing
this document — not extrapolated from exp_23's tabular-only numbers:

| Modality / representation | Encoder | s / model |
|---|---|---|
| `tab` (12-D) | identity | 0.61 |
| `tab` (12-D) | linear(12→8) | 0.13 |
| `mri` raw (1024-D) | identity | 0.22 |
| `mri` raw (1024-D) | linear(1024→32) | 0.19 |
| `mri` pca90 (≈11-D) | identity | 0.11 |
| `txt` pca90 (≈54–57-D, spaCy-lemmatized — measured after install, §2) | identity | 0.11 |
| joint trimodal, `mri`=pca90 | identity, all modalities | 0.15 |
| joint trimodal, `mri`=raw | identity, all modalities | 0.23 |

**Key finding: cost is driven by `n_comp` (≈70 in Phase A, ≈87 in Phase B), not by modality count or
input dimensionality** — a 1024-D raw-MRI fit (0.22s) is *cheaper* than the 12-D tabular fit (0.61s),
because `PCA-90%`/kernel construction cost is dwarfed by the `n_comp × n_comp` kernel evaluation inside
`KDMLayer`, and `tab`'s `identity` encoder happens to be the slowest single-modality case measured.
Budget below uses 0.61s as a conservative per-fit ceiling for any identity-encoder condition and scales
Phase B by `(87/70)² ≈ 1.55×` for the larger `n_comp`.

| Stage | Fits | Est. time |
|---|---|---|
| Step 0 (reproduction gate + degeneracy probe): 88×10 seeds, tab only | 880 | ≈ 2 min |
| Stage 1 Phase A: `tab` 16×100, `mri` 24×100, `txt` 36×100 = 7,600 | 7,600 | ≈ 20–25 min |
| Stage 2 Phase A: 5 conditions × 24 configs × 100 (4 decision-task + 1 confidence-arm, unconditional — §1.5 deviation) | 12,000 | ≈ 35–40 min |
| Phase B: 8 unique fit-slots (3 unimodal shared with late fusion + 4 joint + 1 confidence-arm, §1.6) × 88 folds × ≤10 seeds | ≤ 7,040 | ≈ 20–35 min (worst case: every winner is `linear`) |
| Confidence heads (sklearn trees, sub-second each) + fusion-weight search (in-memory recombination of stashed probs, no new fits) | — | ≈ 5 min |

**Total ≈ 1.5–2.5 h single-threaded** — comfortably under the approved plan's 4 h pruning gate, so
**none of the three grid cuts it specified are needed**; Stage 1/Stage 2 run exactly as `DESIGN.md` §3.4/§3.5
specify. Acceptable to run in the background given the length; no multiprocessing needed.

---

## 1. Script: `experiments/exp_25/scripts/train.py`

Single self-contained script (repo convention), monolithic `main()`. Unlike exp_23/24, this experiment
**imports the model/harness layer from `src/` rather than reimplementing it** — `DESIGN.md` §2.1 states
exp_25 is wiring, not new model code, and that is reflected directly in this section being much shorter
than exp_23/24's equivalent.

### 1.1 Setup, paths, imports

```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))          # src/evaluation/data.py:167 does `from src.methods.base import Targets`

from src.evaluation.data import (
    resolve_data_dir, load_cohort, build_targets, build_tabular_features,
    build_mri_features, clean_texts_spacy, build_text_features, OLD_SCHEMA,
    CONFIDENCE_CERTAINTY_MAP,
)
from src.evaluation.protocol import iter_mccv_splits, run_mccv_grid, select_best, run_loocv
from src.evaluation.metrics import binary_metrics, confidence_metrics, mcnemar_exact
from src.evaluation.reporting import write_json, record_git_commit, plot_confusion_matrix, plot_roc_curves, plot_grid_search_curves, plot_signal_scatter
from src.methods.base import Targets, fit_meta_thresholds_safe, apply_meta_thresholds, fit_predict_heldout_trees
from src.methods.mem_kdm import (
    MemKDM, LateFusionMemKDM, KernelSpec, EncoderSpec, extract_particle_signals,
    composite_reliability_index, simplex_grid, soft_vote, smooth, to_amplitude,
    PARTICLE_SIGNAL_NAMES,
)
```

`conda run -n pytorch python ...` only (`import kdm` fails elsewhere — verified: `kdm` resolves to
`~/Documents/research/code/own_libs/kdm/`, absent from every other local env).

**Text column, confirmed against the live data:** `cohort.df_text` has columns
`["patient_id", "clinical_prompt_text"]` — `clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)`.

**Split-tuple materialization (`DESIGN.md` §6, "split-tuple arity"):**
```python
SPLITS_3T = list(iter_mccv_splits(df_design_labeled, n_splits=100))          # for run_mccv_grid
SPLITS_2T = [(tr, va) for _, tr, va in SPLITS_3T]                            # for fit_meta_thresholds*, fit_predict_heldout_trees
```

### 1.2 Cohort load and per-representation feature builders

```python
data_dir = resolve_data_dir(PROJECT_ROOT)
cohort = load_cohort(data_dir, PROJECT_ROOT, OLD_SCHEMA, load_mri=True, load_text=True)
assert len(cohort.dre_categories) == 5
targets = build_targets(cohort.y_binary, cohort.confidence, certainty_map=CONFIDENCE_CERTAINTY_MAP)   # soft everywhere, DESIGN.md §2.4
cleaned_texts = clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)                        # once, cohort-level
```

Per-`(split, modality, representation)` feature cache, built once and reused across every config sharing
that representation (`DESIGN.md` §3.4):

```python
def build_modality(name, rep, train_idx, val_idx):
    if name == "tab":
        return build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
    if name == "mri":
        pca_variance = None if rep == "raw_l2" else 0.90
        return build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=pca_variance)
    if name == "txt":
        max_features, pca_variance = rep          # rep = (max_features, 0.90)
        return build_text_features(cleaned_texts, train_idx, val_idx, max_features=max_features, pca_variance=pca_variance)
```

`FEATURE_CACHE[(split_idx, name, rep)] = (X_tr, X_va)`, populated lazily on first use within Stage 1's
grid loop, keyed so Stage 2 can reuse the same entries for its frozen representation.

**Discovered during implementation review — encoder `out_dim` must be representation-specific, not a
single fixed 32.** `mri`'s `pca90_l2` representation reduces to ≈11-D on this cohort (measured, §0);
`EncoderSpec("linear", out_dim=32)` on an 11-D input is an *upsampling* projection, not the compression
`DESIGN.md` §3.4 intends. Concrete `out_dim` per modality/representation (mirrors `tab`'s 12→8 ratio,
does not change the grid's semantic axis count):

| Modality / rep | `linear` out_dim |
|---|---|
| `tab` | 8 (per `DESIGN.md`, unchanged) |
| `mri` `raw_l2` (1024-D) | 32 |
| `mri` `pca90_l2` (≈11-D) | 8 |
| `txt` any `(max_features, 0.90)` | 8 |

**`txt` dimensionality, confirmed after installing spaCy (§2):** `clean_texts_spacy` on the 88-document
cohort runs in 2.4s; post-PCA-90% dimensionality is 54-D (`max_features=500`), 57-D (`max_features=2000`
and `max_features=None` — TF-IDF vocabulary is small enough on this cohort that 2000 and unrestricted
agree). Nearly identical to the un-lemmatized pilot's 58-D estimate, confirming §0's "cost driven by
`n_comp`, not vocabulary size" finding — the runtime budget above is unaffected by lemmatization.

### 1.3 Step 0 — reproduction gate and `h_aleatoric` degeneracy probe

Per `DESIGN.md` §3.3, runs before Stage 1:

```python
def reproduction_gate():
    model = MemKDM(kernels={"tab": KernelSpec(sigma_mult=2.0)}, encoders={"tab": EncoderSpec("identity")},
                    x_train=False, y_train=False, epochs=300, lr=1e-3, seed=0)
    def evaluate_fn(train_idx, val_idx):
        X_tr, X_va = build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
        m = MemKDM(kernels={"tab": KernelSpec(sigma_mult=2.0)}, encoders={"tab": EncoderSpec("identity")},
                   x_train=False, y_train=False, epochs=300, lr=1e-3, seed=0)
        m.fit({"tab": X_tr}, Targets(y_binary=cohort.y_binary[train_idx], y_soft=targets.y_soft[train_idx]))
        p = m.predict_proba({"tab": X_va})[:, 1]
        sig = m.uncertainty_signals({"tab": X_va})
        return {"pred": p[0], "signals": {k: v[0] for k, v in sig.items() if k != "probs"}}
    oof_pred, oof_signals = run_loocv(evaluate_fn, n=len(cohort.y_binary))
    metrics = binary_metrics(cohort.y_binary, oof_pred)
    assert abs(metrics["macro_f1"] - 0.6694214876033058) < 1e-6, \
        f"exp_23 Arm B reproduction failed: got {metrics['macro_f1']}"

    h_al = oof_signals["h_aleatoric"]
    degeneracy = {
        "frac_nonzero_h_aleatoric": float((h_al > 1e-6).mean()),
        "median_h_aleatoric": float(np.median(h_al)),
    }
    write_json({"reproduction_macro_f1": metrics["macro_f1"], **degeneracy}, results_dir / "reproduction_check.json")
    return degeneracy
```

`run_loocv`'s `evaluate_fn` refits per fold (never reuses a whole-cohort model, per `protocol.py`'s
contract). One additional fit with `MemKDM(..., check_roundtrip=True)` is run separately (not inside the
LOOCV loop — it's a one-shot init-time assertion, not a per-fold metric) for `tab`, `mri` (`pca90_l2`),
and `txt` individually. **If the reproduction assert fails, `main()` raises and stops before Stage 1.**

**Verified standalone, exactly as written above, before this document was finalized** (not merely
assumed from the earlier `verify_slow.log` reproduction, which predates this experiment and did not go
through `src.evaluation.protocol.run_loocv` specifically): `run_loocv(evaluate_fn, n=88)` +
`binary_metrics` on `tab`-only `sigma_mult=2.0, encoder=identity` reproduces exp_23 Arm B to
`macro_f1 == 0.6694214876033058` exactly (`diff == 0.0`), confirming the `run_loocv` code path itself —
not just the underlying `MemKDM.fit` — is faithful.

`degeneracy["frac_nonzero_h_aleatoric"]` gates whether Stage 2's conditional `label_smoothing=0.10`
condition (§1.6) is added.

### 1.4 Stage 1 — unimodal Phase A

```python
TAB_GRID = [{"sigma_mult": sm, "encoder": enc, "y_train": yt}
            for sm in [0.25, 0.5, 1.0, 2.0] for enc in ["identity", "linear"] for yt in [False, True]]
MRI_GRID = [{"rep": rep, "sigma_mult": sm, "encoder": enc, "y_train": yt}
            for rep in ["raw_l2", "pca90_l2"] for sm in [0.5, 1.0, 2.0]
            for enc in ["identity", "linear"] for yt in [False, True]]
TXT_GRID = [{"rep": (mf, 0.90), "sigma_mult": sm, "encoder": enc, "y_train": yt}
            for mf in [500, 2000, None] for sm in [0.5, 1.0, 2.0]
            for enc in ["identity", "linear"] for yt in [False, True]]
assert len(TAB_GRID) == 16 and len(MRI_GRID) == 24 and len(TXT_GRID) == 36

def stage1_evaluate(name, cfg, split_idx, train_idx, val_idx, stash):
    rep = cfg.get("rep", None)
    X_tr, X_va = build_modality(name, rep, train_idx, val_idx)
    enc_spec = EncoderSpec("identity") if cfg["encoder"] == "identity" else EncoderSpec("linear", out_dim=OUT_DIM[name][rep])
    m = MemKDM(kernels={name: KernelSpec(sigma_mult=cfg["sigma_mult"])}, encoders={name: enc_spec},
               x_train=False, y_train=cfg["y_train"], epochs=300, lr=1e-3, seed=0)
    m.fit({name: X_tr}, Targets(y_binary=cohort.y_binary[train_idx], y_soft=targets.y_soft[train_idx]))
    p_va = m.predict_proba({name: X_va})
    stash[(name, tuple(sorted(cfg.items())), split_idx)] = (val_idx, p_va)     # for the fusion-weight search, §1.6
    y_pred = (p_va[:, 1] >= 0.50).astype(int)
    return {"macro_f1": f1_score(cohort.y_binary[val_idx], y_pred, average="macro", zero_division=0)}

for name, grid in [("tab", TAB_GRID), ("mri", MRI_GRID), ("txt", TXT_GRID)]:
    # Every stage1_evaluate call below passes seed=0, and MemKDM.fit calls torch.manual_seed(self.seed)
    # internally as its first line (mem_kdm.py:298) — so every config within a split draws nn.Linear's
    # init from the same RNG state, reproducing exp_23's paired-comparison guarantee (exp_23/
    # IMPLEMENTATION.md §1.6: "torch.manual_seed(42) immediately before build_kdm" for every config in a
    # split) via MemKDM's own seed handling rather than an explicit per-split reseed in train.py.
    df_grid = run_mccv_grid_with_stash(name, grid, SPLITS_3T, STAGE1_STASH)
    best_row = select_best(df_grid, primary_metric="macro_f1")
    STAGE1_BEST[name] = {
        **best_row.to_dict(),
        "sigma_mult": float(best_row["sigma_mult"]),
        "y_train": bool(best_row["y_train"]),
        "rep": tuple(best_row["rep"]) if name == "txt" else (None if pd.isna(best_row.get("rep")) else best_row.get("rep")),
    }
```

**Dtype coercion is required, not defensive.** `select_best` returns `df_grid.sort_values(...).iloc[0]`,
a `pandas.Series` — `sigma_mult` comes back as `np.float64`, `y_train` as `np.bool_` (or an
object-dtype column, depending on what forced upcasting in the DataFrame). `write_json`'s
`_StrictEncoder` (`reporting.py:15-28`) handles `np.floating`/`np.integer`/`np.ndarray` but **not**
`np.bool_` — writing `STAGE1_BEST[name]` into `stage1_best_hparams.json` uncoerced raises
`TypeError: Object of type bool_ is not JSON serializable` at artifact-write time, not at fit time, so
it would surface late in a multi-hour run. Coerce at extraction, once, as above. `rep` needs the same
care for a different reason: `TXT_GRID`'s `rep` field is a `(max_features, pca_variance)` tuple; verify
after `run_mccv_grid_with_stash` that it survives the DataFrame round-trip as a tuple (not stringified)
before `OUT_DIM[name][rep]` is used to key the encoder's `out_dim` in §1.5/§1.6.

**Stage 1 does *not* call `protocol.run_mccv_grid` directly.** `run_mccv_grid`'s
`evaluate_fn(cfg, train_idx, val_idx)` callback signature has no `split_idx` parameter, but the stash
needs the split index to key each `(name, cfg, split_idx)` entry. `run_mccv_grid_with_stash` is a ~15-line
local function in `train.py` with the *same* per-cfg-per-split aggregation and output shape as
`run_mccv_grid` (so `select_best` still applies unchanged), but iterating `enumerate(SPLITS_3T)` directly
instead of delegating to `protocol.run_mccv_grid` — a local reimplementation of that one loop, not a
change to `protocol.py`, which must stay method/experiment-agnostic and gets no stash concept added to it.

Artifacts: `results/stage1_grid_search.csv` (concatenated `df_grid` per modality, `modality` column
added), `results/stage1_best_hparams.json`.

### 1.5 Stage 2 — joint Phase A

```python
CONDITIONS = [["tab", "mri"], ["tab", "txt"], ["mri", "txt"], ["tab", "mri", "txt"]]
STAGE2_GRID = [{"sigma_scale": ss, "x_train": xt, "y_train": yt, "kernel_trainable": kt}
               for ss in [0.5, 1.0, 2.0] for xt in [False, True] for yt in [False, True] for kt in [False, True]]
assert len(STAGE2_GRID) == 24

def build_joint_kernels_encoders(subset, cfg):
    kernels, encoders = {}, {}
    for m in subset:
        base_mult = STAGE1_BEST[m]["sigma_mult"]
        scale = 1.0 if m == "tab" else cfg["sigma_scale"]          # DESIGN.md §3.5: tab is the reference
        kernels[m] = KernelSpec(sigma_mult=base_mult * scale, trainable=cfg["kernel_trainable"], sigma=None)
        rep = STAGE1_BEST[m].get("rep")
        enc = STAGE1_BEST[m]["encoder"]
        encoders[m] = EncoderSpec("identity") if enc == "identity" else EncoderSpec("linear", out_dim=OUT_DIM[m][rep])
    return kernels, encoders

def stage2_evaluate(subset, cfg, train_idx, val_idx, label_smoothing=0.0):
    X_tr, X_va = {}, {}
    for m in subset:
        rep = STAGE1_BEST[m].get("rep")
        X_tr[m], X_va[m] = build_modality(m, rep, train_idx, val_idx)
    kernels, encoders = build_joint_kernels_encoders(subset, cfg)
    m = MemKDM(kernels=kernels, encoders=encoders, x_train=cfg["x_train"], y_train=cfg["y_train"],
               label_smoothing=label_smoothing, epochs=300, lr=1e-3, seed=0)
    m.fit(X_tr, Targets(y_binary=cohort.y_binary[train_idx], y_soft=targets.y_soft[train_idx]))
    y_pred = (m.predict_proba(X_va)[:, 1] >= 0.50).astype(int)
    return {"macro_f1": f1_score(cohort.y_binary[val_idx], y_pred, average="macro", zero_division=0)}

for subset in CONDITIONS:
    df_grid = run_mccv_grid(STAGE2_GRID, lambda cfg, tr, va: stage2_evaluate(subset, cfg, tr, va), SPLITS_3T)
    STAGE2_BEST[tuple(subset)] = select_best(df_grid, "macro_f1").to_dict()

df_grid_c = run_mccv_grid(STAGE2_GRID, lambda cfg, tr, va: stage2_evaluate(["tab", "mri", "txt"], cfg, tr, va, label_smoothing=0.10), SPLITS_3T)
STAGE2_BEST["confidence_arm"] = select_best(df_grid_c, "macro_f1").to_dict()
```

**Deviation from `DESIGN.md` §3.5, discovered during implementation review:** the confidence-arm
condition is run **unconditionally**, not gated on `degeneracy["frac_nonzero_h_aleatoric"]`. `DESIGN.md`
left the gating threshold unspecified, and no principled cutoff (what fraction of non-trivial
`h_aleatoric` makes the split "worth it") was established during design — an invented number there would
make the experiment's shape depend on an unjustified constant. §0's measured runtime shows comfortable
headroom under the 4h budget gate for the extra 2,400 fits, so the condition always runs; Step 0's
degeneracy probe is still recorded in `reproduction_check.json` and used in `summary.md` to interpret
`h_aleatoric`'s results, just not to decide whether they're computed.

**`confidence_arm`'s Phase-A selection metric is decision-task Macro-F1** (same `stage2_evaluate` scoring
as every other Stage-2 condition), even though §1.6 excludes this condition's decision metrics from H1
and only its particle signals feed the confidence task. This selects the config that best predicts
biopsy decision, then reuses it for confidence prediction — not obviously the right target for a
condition whose only reported use is confidence. It is, however, **exactly exp_24's Arm C structure**
(Arm C's own Phase A selected on decision Macro-F1; its signals then fed the confidence heads,
`exp_24/DESIGN.md` §3.3/§4) — kept for consistency with that precedent rather than introduced by
oversight. Selecting on the confidence task directly would require nesting a confidence-head fit inside
Stage 2's Phase A loop, a materially bigger change not justified by this experiment's scope.

**Leak-free fusion-weight search** (`DESIGN.md` §3.5 — `search_fusion_weights` is not used, §2.3/§6):

```python
def search_fusion_weights_local(stash, best_cfg_key_fn):
    grid = simplex_grid(3, step=0.05)
    scores = np.zeros(len(grid))
    for split_idx in range(100):
        val_idx, probs = None, {}
        for name in ["tab", "mri", "txt"]:
            key = best_cfg_key_fn(name)
            v_idx, p_va = stash[(name, key, split_idx)]
            val_idx, probs[name] = v_idx, p_va
        for i, (wt, wm, wx) in enumerate(grid):
            p_fused = soft_vote(probs, dict(tab=wt, mri=wm, txt=wx))
            y_pred = (p_fused[:, 1] >= 0.50).astype(int)
            scores[i] += f1_score(cohort.y_binary[val_idx], y_pred, average="macro", zero_division=0)
    scores /= 100
    best_i = int(np.argmax(scores))
    return {"weights": dict(zip(["tab", "mri", "txt"], grid[best_i])), "mean_macro_f1": float(scores[best_i])}

fusion_weights = search_fusion_weights_local(STAGE1_STASH, lambda name: tuple(sorted(STAGE1_BEST[name].items())))
write_json(fusion_weights, results_dir / "fusion_weights.json")
```

Artifacts: `results/stage2_grid_search.csv`, `results/best_hparams.json` (Stage-1 + Stage-2 winners, one
JSON, keyed by condition name), `results/fusion_weights.json`.

### 1.6 Phase B — LOOCV, 88 folds

**Shared unimodal fits, computed once per fold and reused across four conditions** (`unimodal_tab/mri/txt`
and both `late_fusion_*` conditions), per `DESIGN.md`'s efficiency note in the runtime budget (§0):

`build_joint_kernels_encoders` (§1.5) is a **Stage-2 builder** — it applies `sigma_scale` and
`kernel_trainable` from a Stage-2 config, and `STAGE1_BEST[name]` is a `select_best` row (`.to_dict()`
on a `pandas.Series`) carrying grid-search *output* columns (`cfg_id`, `mean_macro_f1`, `std_macro_f1`,
...), not a config dict `build_joint_kernels_encoders` expects. Unimodal Phase B builds its
`KernelSpec`/`EncoderSpec` directly from the three fields it actually needs instead of routing through
that function:

```python
def build_unimodal_kernel_encoder(name):
    best = STAGE1_BEST[name]
    kernel = KernelSpec(sigma_mult=best["sigma_mult"], sigma=None)              # trainable=True, the dataclass default — matches Stage 1's fit
    rep = best.get("rep")                                                       # None for tab; a (max_features, 0.90) tuple for txt — verify it survives .to_dict() as a tuple, not a stringified object
    encoder = EncoderSpec("identity") if best["encoder"] == "identity" else EncoderSpec("linear", out_dim=OUT_DIM[name][rep])
    return kernel, encoder

def phase_b_unimodal(name, n_seeds):
    kernel, encoder = build_unimodal_kernel_encoder(name)
    def evaluate_fn(train_idx, val_idx):
        rep = STAGE1_BEST[name].get("rep")
        X_tr, X_va = build_modality(name, rep, train_idx, val_idx)
        p_seeds, sig_seeds = [], []
        for seed in range(n_seeds):
            m = MemKDM(kernels={name: kernel}, encoders={name: encoder}, x_train=False,
                       y_train=STAGE1_BEST[name]["y_train"], epochs=300, lr=1e-3, seed=seed)
            m.fit({name: X_tr}, Targets(y_binary=cohort.y_binary[train_idx], y_soft=targets.y_soft[train_idx]))
            p_seeds.append(m.predict_proba({name: X_va})[0])
            sig = m.uncertainty_signals({name: X_va})
            sig_seeds.append({k: v[0] for k, v in sig.items() if k != "probs"})
        p_mean = np.mean(p_seeds, axis=0)
        sig_mean = {k: float(np.mean([s[k] for s in sig_seeds])) for k in sig_seeds[0]}   # averaged across seeds, not last-seed
        return {"pred": p_mean[1], "signals": {**sig_mean, "p_mean": p_mean[1]}}
    return run_loocv(evaluate_fn, n=len(cohort.y_binary))

N_SEEDS = {m: (10 if STAGE1_BEST[m]["encoder"] == "linear" else 1) for m in ["tab", "mri", "txt"]}
unimodal_oof = {m: phase_b_unimodal(m, N_SEEDS[m]) for m in ["tab", "mri", "txt"]}
```

**Joint conditions**, one `MemKDM` per condition, follow the identical seed-accumulation pattern (`p_seeds`/
`sig_seeds` lists, meaned after the loop — never a bare `sig = m.uncertainty_signals(...)` left bound to
whichever seed's model the loop last constructed) with `build_joint_kernels_encoders(subset,
STAGE2_BEST[subset])` for the kernels/encoders, and `N_SEEDS[subset] = 10` if any member modality's
Stage-1-selected encoder is `linear`, else `1`.

**Late-fusion conditions** — reuse `unimodal_oof`'s per-fold, per-modality probabilities directly (no
new `MemKDM` fits): `late_fusion_equal` = `soft_vote(probs, {m: 1/3 for m in ...})`;
`late_fusion_optimal` = `soft_vote(probs, fusion_weights["weights"])`.

**`confidence_arm`** (§1.5's unconditional trimodal `label_smoothing=0.10` condition) also runs its own
Phase B — same loop shape as the joint conditions, `STAGE2_BEST["confidence_arm"]`,
`label_smoothing=0.10` passed through to `MemKDM`. Its `binary_metrics`/`mcnemar_exact` are computed and
recorded like every other condition (for completeness in `loocv_metrics.json`) but are **excluded from
H1's decision-rule comparisons** (`DESIGN.md` §3.5/§5) — only its particle signals feed §1.7's
confidence task.

Per condition, write the `deterministic`/`n_seeds`/`mode_vote_agreement`/`per_seed_macro_f1`/
`macro_f1_std_across_seeds` fields to `loocv_metrics.json`, matching exp_24's shape, computed only when
`n_seeds > 1` (else `"deterministic": true`, no std).

`mcnemar_exact` computed for `joint_trimodal` vs. `late_fusion_optimal` (H1's significance criterion,
`DESIGN.md` §5) and for every condition vs. exp_23 Arm B (context).

### 1.7 Confidence task

Per condition (all `MemKDM`/`LateFusionMemKDM` instances from §1.6, plus the conditional
`confidence_arm` from §1.5 if it ran): collect the Phase-B OOF `uncertainty_signals` already stashed in
`unimodal_oof[...]["signals"]` / the joint-condition equivalent, then:

```python
def confidence_1d(signals, y_conf):
    best_key, best = None, None
    for key in PARTICLE_SIGNAL_NAMES:                    # or composite_ici / member signals for LateFusionMemKDM
        thr = fit_meta_thresholds_safe(signals[key], y_conf, SPLITS_2T)
        pred = apply_meta_thresholds(signals[key], thr)
        m = confidence_metrics(y_conf, pred)
        if best is None or m["macro_f1"] > best["macro_f1"]:
            best_key, best = key, {**m, "signal": key, "target_informed": True}
    return best

def confidence_multivariate(signals, y_conf, keys):
    S = np.stack([signals[k] for k in keys], axis=1)
    pred, votes = fit_predict_heldout_trees(S, y_conf, SPLITS_2T)
    return {**confidence_metrics(y_conf, pred), "keys": keys, "min_votes": int(votes.min()), "target_informed": True}
```

Run `confidence_multivariate` twice per condition: full 7-signal (`MemKDM`) / composite set
(`LateFusionMemKDM`), and the `{h_total, log_marginal}` ablation (exp_24's ablation). Every row in
`results/confidence_metrics.json` carries `"target_informed": true` explicitly (`DESIGN.md` §2.4) — not
inferred by the reader from context.

### 1.8 Artifacts

`results/`: `reproduction_check.json`, `stage1_grid_search.csv`, `stage1_best_hparams.json`,
`stage2_grid_search.csv`, `best_hparams.json`, `fusion_weights.json`, `loocv_metrics.json`,
`confidence_metrics.json`, `oof_predictions.csv`, `oof_particle_signals.csv`, `git_commit.txt`.
`reports/figures/`: `stage1_grid_search_curves.png` (via `reporting.plot_grid_search_curves`, one call
per modality), `stage2_grid_search_curves.png` (one call per condition, faceted), `confusion_matrix.png`
(best condition, via `reporting.plot_confusion_matrix`), `roc_curve.png` (`joint_trimodal` vs.
`late_fusion_optimal` vs. `unimodal_tab`, via `reporting.plot_roc_curves`), `particle_signal_scatter.png`
(`h_aleatoric` vs. `h_epistemic`, colored by `confidence`, via `reporting.plot_signal_scatter`),
`confidence_confusion_matrix.png`. `reports/summary.md` via the `ml-experiment-reporter` skill once
results exist.

---

## 2. Command Lines

### Environment check (run first — this found a real gap)

```bash
conda run -n pytorch python -c "import kdm, torch, sklearn, scipy, pandas, numpy; print('ok')"
conda run -n pytorch python -c "import spacy; spacy.load('en_core_web_sm')"
```

**Resolved during planning, per explicit user direction.** `spacy` was not installed in the `pytorch`
conda env (nor in `base`, `jupyter-book`, `paper-audit`, `qgen`, or `tf2` on this machine) — exp_13–16,
which used spaCy, ran on a different (remote) machine; this was the first time the text pipeline had
been exercised in the local `pytorch` env. User chose to install it (over using un-lemmatized text or
dropping the text modality):

```bash
conda run -n pytorch pip install spacy                          # done — spacy 3.8.15
conda run -n pytorch python -m spacy download en_core_web_sm    # done — en_core_web_sm 3.8.0
```

Both checks above now pass. `clean_texts_spacy` on the 88-document cohort measured 2.4s (§0).

### Smoke test (bug-catching, not timing — §0 already grounds timing)

`--smoke`: `SPLITS_3T`/`SPLITS_2T` truncated to 5 splits, `N_SEEDS` capped at 3, Phase B run on a 6-fold
subset of the 88. Confirms no exceptions, confirms the reproduction gate still passes (§1.3 is **not**
skipped under `--smoke` — it is the one check that must hold even in a fast run), confirms every artifact
file is written, before committing to the full run.

```bash
conda run -n pytorch python experiments/exp_25/scripts/train.py --smoke
```

### Execution

```bash
git log -1 --format="%H %s" > experiments/exp_25/results/git_commit.txt
conda run -n pytorch python experiments/exp_25/scripts/train.py
```

Expected wall time: **~1.5–2.5 h** (§0). Single process; acceptable to run in the background.

`git_commit.txt` must point at a commit that includes `src/` (`DESIGN.md` §6 — `src/` is currently
uncommitted, `git status`: `?? src/`). **Commit `src/` (with its pending `.logbook.md`/`.discussion.md`
entries) before this run**, or the recorded hash will not actually contain the method layer being
evaluated.

---

## 3. Post-Execution (per repo workflow, after results exist and are reviewed)

1. `reports/summary.md` via `ml-experiment-reporter`, evaluated against `DESIGN.md` §1's pre-registered
   H1/H2/H3 decision rules.
2. Add the `exp_25` row to `experiments/INDEX.md`.
3. Append a `.logbook.md` entry (Spanish, `## [YYYY-MM-DD] - Title` / Current Project State / Specific
   Advancements / Next Tasks, matching the exp_23/24 entries' style).
4. Insert a `.discussion.md` entry (English, `### [YYYY-MM-DD] - Topic` / Principal Investigator Proposal
   / Co-Investigator Analysis / Consensus) **adjacent to the exp_24 entry (~line 146), not at
   end-of-file** — the file is not chronologically ordered at the tail (confirmed during exp_23/24
   planning).
