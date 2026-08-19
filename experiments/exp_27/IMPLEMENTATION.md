# Implementation Plan: MemKDM with training restricted to sigma-only, frozen unimodally (exp_27)
**Experiment**: experiments/exp_27/ · **Status**: Complete

See `DESIGN.md` for motivation and scope. This is a targeted diff against `exp_26/scripts/train.py` (cohort
loading, `iter_mccv_splits`, figure generation, `oof_particle_signals.csv`, confidence-task mechanics are
reused verbatim — not re-specified here) plus one `src/methods/mem_kdm.py` fix (§0).

## 0. `src/methods/mem_kdm.py` — `MemKDM.fit()` guard on zero trainable parameters

Per `DESIGN.md` §2.4. `c_x`/`c_y`/`c_w`/`raw_sigma` are always `nn.Parameter` regardless of
`x_train`/`y_train`/`w_train`/`trainable` (only `requires_grad` varies) — so `model.parameters()` is never
empty; `torch.optim.Adam(model.parameters(), ...)` construction never raises. The actual failure when
nothing is trainable: `loss.backward()` raises `RuntimeError: element 0 of tensors does not require grad
and does not have a grad_fn`, since no tensor in the forward graph requires grad. Current code
(`mem_kdm.py:350-362`):

```python
c_y_before = kdm_layer.c_y.detach().clone() if self.y_train else None
t = torch.as_tensor(np.stack([1 - y_soft, y_soft], axis=1), dtype=torch.float32)
opt = torch.optim.Adam(model.parameters(), lr=self.lr)
model.train()
for _ in range(self.epochs):
    probs = model(Xt)
    loss = -(t * torch.log(probs.clamp_min(1e-7))).sum(-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

if c_y_before is not None:
    self.cy_drift_max_ = float((kdm_layer.c_y.detach() - c_y_before).abs().max().item())
```

New:

```python
c_y_before = kdm_layer.c_y.detach().clone() if self.y_train else None
has_trainable = any(p.requires_grad for p in model.parameters())
if has_trainable:
    t = torch.as_tensor(np.stack([1 - y_soft, y_soft], axis=1), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=self.lr)
    model.train()
    for _ in range(self.epochs):
        probs = model(Xt)
        loss = -(t * torch.log(probs.clamp_min(1e-7))).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

if c_y_before is not None:
    self.cy_drift_max_ = float((kdm_layer.c_y.detach() - c_y_before).abs().max().item())
```

(`t` moves inside the `if` since it's only used there.) No behavior change when `has_trainable` is
`True` — same optimizer (still passed the full, unfiltered `model.parameters()`, matching the pre-existing
behavior of Adam silently skipping any individual `requires_grad=False` param), same loop, same loss.
`model.eval()`/`model.train()` mode doesn't affect forward-pass numerics here (no dropout/batchnorm in
`_MemKDMCore`), so leaving `model.train()` uncalled in the no-trainable-params branch is inert;
`predict_proba`/`uncertainty_signals` call the model directly without checking `.training`.

## 1. `experiments/exp_27/scripts/train.py` — diff against `exp_26/scripts/train.py`

### 1.1 Grids

```python
TAB_GRID = [{"sigma_mult": sm} for sm in [0.25, 0.5, 1.0, 2.0]]
MRI_GRID = [{"rep": rep, "sigma_mult": sm}
            for rep in ["raw_l2", "pca90_l2"] for sm in [0.5, 1.0, 2.0]]
TXT_GRID = [{"rep": (mf, 0.90), "sigma_mult": sm}
            for mf in [500, 2000, None] for sm in [0.5, 1.0, 2.0]]
assert len(TAB_GRID) == 4 and len(MRI_GRID) == 6 and len(TXT_GRID) == 9
```

`STAGE2_GRID` becomes a single trivial config so the existing `run_mccv_grid`/`select_best` plumbing (and
`stage2_grid_search.csv`'s schema) needs no special-casing:

```python
STAGE2_GRID = [{}]
```

`CONDITIONS`, `JOINT_KEYS`, `LATE_FUSION_NAMES` unchanged.

### 1.2 Stage 1 `evaluate_fn` — capture fitted sigma, drop `y_train`/`encoder` branching

```python
def stage1_evaluate_factory(name):
    def evaluate_fn(cfg, train_idx, val_idx):
        rep = cfg.get("rep")
        X_tr, X_va = bm(name, rep, train_idx, val_idx)
        m = MemKDM(kernels={name: KernelSpec(sigma_mult=cfg["sigma_mult"])}, encoders={name: EncoderSpec("identity")},
                   x_train=False, y_train=False, w_train=False, epochs=300, lr=1e-3, seed=0)
        m.fit({name: X_tr}, make_targets(train_idx))
        y_pred = (m.predict_proba({name: X_va})[:, 1] >= 0.50).astype(int)
        fitted_sigma = m.kernel_params()[name].sigma
        return {"macro_f1": f1_score(cohort.y_binary[val_idx], y_pred, average="macro", zero_division=0),
                "fitted_sigma": float(fitted_sigma)}
    return evaluate_fn
```

`run_mccv_grid` (`src/evaluation/protocol.py:46-49`) already aggregates every key `evaluate_fn` returns into
`mean_<key>`/`std_<key>` — `stage1_grid_search.csv` gets `mean_fitted_sigma`/`std_fitted_sigma` columns for
free, no `protocol.py` change.

`extract_stage1_best` drops the now-nonexistent `encoder`/`y_train` fields and adds `mean_sigma`:

```python
def extract_stage1_best(name, best_row):
    d = {
        "sigma_mult": float(best_row["sigma_mult"]),
        "mean_sigma": float(best_row["mean_fitted_sigma"]),
        "mean_macro_f1": float(best_row["mean_macro_f1"]),
        "std_macro_f1": float(best_row["std_macro_f1"]),
        "cfg_id": int(best_row["cfg_id"]),
    }
    rep = best_row.get("rep") if name != "tab" else None
    if isinstance(rep, (list, tuple)):
        rep = tuple(rep)
    d["rep"] = rep
    return d
```

`winner_val_probs` (fusion-weight search) drops `y_train`/`encoder` from its `MemKDM(...)` call the same
way, adds `w_train=False`; otherwise unchanged.

### 1.3 `build_joint_kernels_encoders` — frozen sigma, no per-fold `_sigma_from_knn` call

```python
def build_joint_kernels_encoders(subset):
    kernels = {mod: KernelSpec(sigma=STAGE1_BEST[mod]["mean_sigma"], trainable=False) for mod in subset}
    encoders = {mod: EncoderSpec("identity") for mod in subset}
    return kernels, encoders
```

No `X_tr` argument (sigma no longer derived from fold data), no `_sigma_from_knn` import needed in
`train.py` (Stage 1 still uses it internally via `MemKDM.fit()`'s existing `sigma_mult` path — unchanged).
Both call sites (`stage2_evaluate_factory`, `phase_b_condition`) drop the `X_tr` argument to this call and
the `cfg["sigma_scale"]`/`cfg["kernel_trainable"]` references.

### 1.4 Stage 2 `evaluate_fn` and Phase B — fixed flags, no cfg-driven branching

```python
def stage2_evaluate_factory(subset, label_smoothing=0.0):
    def evaluate_fn(cfg, train_idx, val_idx):
        X_tr, X_va = {}, {}
        for mod in subset:
            rep = STAGE1_BEST[mod]["rep"]
            X_tr[mod], X_va[mod] = bm(mod, rep, train_idx, val_idx)
        kernels, encoders = build_joint_kernels_encoders(subset)
        m = MemKDM(kernels=kernels, encoders=encoders, x_train=False, y_train=False, w_train=False,
                   label_smoothing=label_smoothing, epochs=300, lr=1e-3, seed=0)
        m.fit(X_tr, make_targets(train_idx))
        y_pred = (m.predict_proba(X_va)[:, 1] >= 0.50).astype(int)
        return {"macro_f1": f1_score(cohort.y_binary[val_idx], y_pred, average="macro", zero_division=0)}
    return evaluate_fn
```

`extract_stage2_best` drops `sigma_scale`/`x_train`/`y_train`/`kernel_trainable` fields (nothing left to
record besides the metric and `cfg_id=0`).

`phase_b_condition`'s `evaluate_fn` mirrors §1.3/§1.4 — `kernels, encoders = build_joint_kernels_encoders(subset)`,
`x_train=False, y_train=False, w_train=False`. The unimodal Phase B loop's hardcoded
`cfg = {"x_train": False, "y_train": STAGE1_BEST[mod]["y_train"], "sigma_scale": 1.0, "kernel_trainable": True}`
is deleted — unimodal Phase B now calls `build_joint_kernels_encoders([mod])` exactly like every joint
condition, no special-casing needed since there's no per-condition cfg left to carry.

`n_seeds=1` throughout (unchanged from exp_26 — no `linear` encoder, still fully deterministic; now doubly
so since nothing trains in Phase B at all).

### 1.5 Step 0 — unchanged

`repro_evaluate_fn` and the `check_roundtrip` loop keep exp_26's exact code (`KernelSpec(sigma_mult=2.0)`,
default `trainable=True`/`w_train=True`) — per `DESIGN.md` §4, this stays a divergent-by-design historical
oracle, not part of the redesign.

### 1.6 Everything else

Cohort loading, `iter_mccv_splits`, fusion-weight search structure, Phase B assembly (`ALL_CONDITIONS`,
`loocv_metrics`, McNemar tests), confidence task, `oof_particle_signals.csv`, figure generation copied from
`exp_26/scripts/train.py` verbatim, `[exp_26]` log-tag prefix changed to `[exp_27]`, docstrings/comments
updated to point at this file's own `DESIGN.md`/`IMPLEMENTATION.md`. Grid-search figure calls
(`plot_grid_search_curves`) drop the now-nonexistent `group_cols=["y_train"]` /
`group_cols=["kernel_trainable", "y_train"]` arguments (single-line curves now, no grouping dimension left).

## 2. Command Lines

```bash
cd /Users/fgonza/Documents/research/code/prostate-cancer-reasoning
conda activate pytorch   # per project memory: local dependencies (spacy, torch, etc.) live here

# smoke test first (5 MCCV splits, 6 LOOCV folds — bug-catching, not timing)
python experiments/exp_27/scripts/train.py --smoke

# full run
python experiments/exp_27/scripts/train.py
```

## 3. Post-Execution (after results exist and are reviewed)

- Confirm `results/reproduction_check.json` still passes (Step 0's historical-oracle path, §1.5 —
  independent of this experiment's own model).
- Compare `results/loocv_metrics.json["unimodal_tab"]` to `exp_26`'s 0.6694214876033058 — expected to
  diverge (`DESIGN.md` §4); report the actual number, don't assume it must match.
- Write `reports/summary.md` (ml-experiment-reporter conventions), framed against `exp_25`/`exp_26`'s H1
  verdict per `DESIGN.md` §1/§6.
- Add exp_27's row to `experiments/INDEX.md`.
- Append `.logbook.md`/`.discussion.md` entries once the user has reviewed and approved the results, per
  `CLAUDE.md`'s workflow — only record points the user has explicitly approved.
