# Implementation Plan: MemKDM without the noise-selected linear encoder (exp_26)
**Experiment**: experiments/exp_26/ · **Status**: Complete

See `DESIGN.md` for the motivation and scope. This is a targeted diff against
`experiments/exp_25/scripts/train.py` (§0's runtime measurement, cohort/split loading, Step 0 reproduction
gate wording, fusion-weight search structure, confidence task, and figure generation are all reused
verbatim from exp_25/IMPLEMENTATION.md — not re-specified here). Only the concrete code changes below.

## 1. `experiments/exp_26/scripts/train.py` — diff against `exp_25/scripts/train.py`

### 1.1 Grids
```python
TAB_GRID = [{"sigma_mult": sm, "encoder": "identity", "y_train": yt}
            for sm in [0.25, 0.5, 1.0, 2.0] for yt in [False, True]]
MRI_GRID = [{"rep": rep, "sigma_mult": sm, "encoder": "identity", "y_train": yt}
            for rep in ["raw_l2", "pca90_l2"] for sm in [0.5, 1.0, 2.0] for yt in [False, True]]
TXT_GRID = [{"rep": (mf, 0.90), "sigma_mult": sm, "encoder": "identity", "y_train": yt}
            for mf in [500, 2000, None] for sm in [0.5, 1.0, 2.0] for yt in [False, True]]
assert len(TAB_GRID) == 8 and len(MRI_GRID) == 12 and len(TXT_GRID) == 18
```
`STAGE2_GRID`, `CONDITIONS`, `JOINT_KEYS`, `LATE_FUSION_NAMES` unchanged from exp_25. `get_out_dim` is
deleted (no `linear` encoder anywhere to size).

### 1.2 Encoder construction
Every `EncoderSpec("identity") if cfg["encoder"] == "identity" else EncoderSpec("linear", out_dim=...)`
branch (in `stage1_evaluate_factory`, `winner_val_probs`, `build_joint_kernels_encoders`) collapses to a
bare `EncoderSpec("identity")`. `cfg["encoder"]`/`best["encoder"]` values remain in the grid/best_hparams
dicts for schema continuity (always `"identity"`) but are no longer branched on.

### 1.3 Sigma transfer — `build_joint_kernels_encoders`
```python
from kdm.init import _sigma_from_knn  # noqa: E402   # new import, top of file

def build_joint_kernels_encoders(subset, cfg, X_tr):
    kernels, encoders = {}, {}
    for mod in subset:
        base_mult = STAGE1_BEST[mod]["sigma_mult"]
        scale = 1.0 if mod == "tab" else cfg["sigma_scale"]
        sigma_val = _sigma_from_knn(np.asarray(X_tr[mod]), base_mult * scale)
        kernels[mod] = KernelSpec(sigma=sigma_val, trainable=cfg["kernel_trainable"])
        encoders[mod] = EncoderSpec("identity")
    return kernels, encoders
```
Both call sites (`stage2_evaluate_factory`'s `evaluate_fn` and `phase_b_condition`'s `evaluate_fn`)
already build `X_tr` (a `{modality: array}` dict) before calling this function — just thread it through
as the new third argument. No other change to either closure.

`Step 0`'s reproduction gate (`repro_evaluate_fn`) and roundtrip-check loop are untouched — they build
`KernelSpec(sigma_mult=...)` directly, not through `build_joint_kernels_encoders`, and must stay that way
since they're the independent oracle this experiment's Step 0/§3 check is validated against.

Stage 1's own `stage1_evaluate_factory` and `winner_val_probs` are untouched apart from §1.2 — they keep
using `KernelSpec(sigma_mult=cfg["sigma_mult"])` directly, per `DESIGN.md` §2.3 ("Stage 1's own grid
search is unchanged").

### 1.4 Seed count
All three `n_seeds = 10 if <encoder is linear> else 1` call sites (unimodal loop, joint loop,
confidence_arm) simplify to `n_seeds = 1` (hardcoded; the `if smoke: n_seeds = min(n_seeds, 3)` lines are
removed as a no-op once `n_seeds` is always 1). `phase_b_condition` itself is unchanged — its existing
`if n_seeds > 1: ... else: mode_vote_agreement, macro_f1_std = 1.0, 0.0` branch already handles
`n_seeds=1` correctly.

### 1.5 Everything else
Cohort loading, `iter_mccv_splits`, Step 0 (reproduction gate + roundtrip check), fusion-weight search,
Stage 2 grid/evaluate structure (besides §1.3), Phase B assembly (`ALL_CONDITIONS`, `loocv_metrics`,
McNemar tests against `late_fusion_optimal` and exp_23), confidence task, `oof_particle_signals.csv`, and
figure generation are copied from `exp_25/scripts/train.py` verbatim, with only the `[exp_25]` log-tag
prefix changed to `[exp_26]` and docstrings/comments referencing exp_25's file updated to point at this
file's own `DESIGN.md`/`IMPLEMENTATION.md`.

## 2. Command Lines

```bash
cd /Users/fgonza/Documents/research/code/prostate-cancer-reasoning
conda activate pytorch   # per project memory: local dependencies (spacy, torch, etc.) live here

# smoke test first (5 MCCV splits, 6 LOOCV folds — bug-catching, not timing)
python experiments/exp_26/scripts/train.py --smoke

# full run
python experiments/exp_26/scripts/train.py
```

## 3. Post-Execution (after results exist and are reviewed)

- Confirm §3's sanity check: `results/reproduction_check.json` passes, and
  `results/loocv_metrics.json["unimodal_tab"]["macro_f1"] == 0.6694214876033058` (bit-for-bit).
- Write `reports/summary.md` (ml-experiment-reporter conventions), framed as a correction/follow-up to
  exp_25's H1 verdict, not a fresh hypothesis writeup.
- Add exp_26's row to `experiments/INDEX.md`.
- Append `.logbook.md` / `.discussion.md` entries once the user has reviewed and approved the results,
  per `CLAUDE.md`'s workflow — only record points the user has explicitly approved.
