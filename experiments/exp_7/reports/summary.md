# exp_7 Report: Tuning + Skew-Aware Preprocessing for exp_6's KDM Backbone

**Experiment**: experiments/exp_7/ · **Project**: challenge_chimera_2 · **Date**: 2026-08-11 · **Status**: Complete

---

## 1. Summary

**Refuted — and the reason why is the most useful part of this experiment.** A 144-combination
hyperparameter search plus a skew-aware `log1p` transform found a configuration that looked like
a modest, genuine improvement under cross-validation (0.622 vs. `exp_6`'s 0.593, a repeated-CV
margin that cleared the pre-registered 0.02 "clear margin" bar) — but the same configuration
scored **worse than the unmodified `exp_6` backbone on the genuine held-out split**
(macro-F1 0.490 vs. 0.593, a −0.102 delta). This is exactly the failure mode `DESIGN.md`'s §9
risk section anticipated for a search this size, and exactly why that held-out check was built
into the plan rather than treated as optional. Confidence and weights, re-evaluated with `exp_6`'s
unchanged readout code on top of this backbone, showed no meaningful change either way.

| Target | exp_6 (unmodified) | exp_7 CV result | exp_7 held-out result | Verdict |
|---|---|---|---|---|
| Decision | 0.593 macro-F1 (CV) / 0.593 (held-out, n=19) | 0.622 macro-F1 (CV) | **0.490** macro-F1 (held-out, n=19) | ❌ Refuted — CV "win" doesn't survive held-out |
| Confidence (best) | 0.731 ord. dist. | 0.730 ord. dist. | — | ❌ No change |
| Weights (occlusion) | 0.405 mean ord. error | 0.404 mean ord. error | — | ❌ No change |

## 2. What Was Run

- **Search**: 144 combinations (`N_EPOCHS` × lr × `sigma_mult` × {Adam, AdamW×3 weight-decay
  values}), all evaluated **with** `log1p` applied to `cli_psa`/`cli_psad`/`cli_vol`, via 5-fold ×
  3-repeat CV. Full grid in `results/hyperparameter_search/grid.csv`.
- **Winner**: `n_epochs=300, lr=3e-3, sigma_mult=0.5, optimizer=adam, weight_decay=0` — 0.619
  search-phase macro-F1 (3-repeat), re-evaluated at the full 10-repeat protocol as
  `decision_kdm_v2`: **0.622**.
- **Ablations** (`run_ablations.py`, isolating which lever did the work, both full 10-repeat CV):
  - `decision_kdm_log1p_only` (log1p, exp_6's original hyperparameters): **0.615**
  - `decision_kdm_tuned_only` (winning hyperparameters, no log1p): **0.591**
- **Held-out check** (`holdout_eval_v2.py`, the same 19-case split `exp_3`'s original
  `holdout_eval.py` used, never touched by the search): exp_6 plain KDM = 0.593 macro-F1;
  exp_7 tuned+log1p KDM = **0.490** macro-F1.
- **Full readout re-evaluation** (`run_signals_v2.py`, `exp_6`'s unchanged confidence/weights
  code on the new backbone, full 10-repeat CV): all 5 confidence conditions and all 3 weights
  conditions came in within noise of `exp_6`'s original numbers — no meaningful change either
  direction. `probs_check_ok=True` across every fold.

## 3. The Central Finding: CV Said Yes, Held-Out Said No

**Attributing the CV-measured "improvement"**: comparing the two ablations against the 0.593
baseline, `log1p` alone accounts for essentially all of the CV-measured gain (0.615, +0.022) while
tuned hyperparameters alone contribute nothing (0.591, −0.002 — indistinguishable from noise).
The combined condition's 0.622 is consistent with `log1p` doing the work and the hyperparameter
search adding little beyond it. This is itself a clean, useful sub-finding: **of the two low-risk
levers this experiment tested, the skew-aware preprocessing was the one with any real signal
under CV; the 144-way hyperparameter search was not.**

**But the held-out check reverses the entire picture.** The exact configuration that beat
baseline under repeated CV (0.622 vs. 0.593) lost to it decisively on 19 cases neither the search
nor the model ever saw (0.490 vs. 0.593). Two things are true simultaneously and worth holding
onto separately:

1. **The CV margin was never large relative to `exp_6`'s own measured noise.** `exp_6`'s
   `decision_kdm_backbone` has a 10-repeat std of 0.045; exp_7's 0.622−0.593=0.029 improvement is
   well within that band. The 0.02 "clear margin" threshold this experiment pre-registered was a
   reasonable bar in isolation, but it was set without reference to `exp_6`'s own noise floor —
   worth a note for any future experiment's threshold-setting: compare against the *incumbent's*
   measured std, not an arbitrary constant.
2. **A 144-way search is exactly the setting most likely to find a fold-flattering configuration
   by chance.** `DESIGN.md` §9 predicted this outcome by name ("with 144 candidates, some will
   beat 0.593 by chance alone even if no real improvement exists") and built the held-out check
   specifically to catch it. It did. This is the mechanism working as designed, not a failure of
   the experiment.

The held-out split itself is small (n=19) and therefore also noisy — a single −0.102 delta on 19
cases shouldn't be over-read as definitive proof the tuned config is *worse* in general, only that
it provides no evidence of being better, and the CV signal that looked like a win does not survive
the one check specifically designed to stress-test it.

## 4. Confidence and Weights: No Change

Re-running `exp_6`'s exact, unmodified confidence/weights readout code on the new backbone
produced numbers within noise of `exp_6`'s originals in both directions:

| Condition | exp_6 | exp_7 (`_v2`) | Δ |
|---|---|---|---|
| `confidence_kdm_entropy_isotonic` | 0.731 | 0.730 | ~0 |
| `confidence_kdm_dispersion_isotonic` | 0.776 | 0.776 | 0 |
| `confidence_kdm_blend` | 0.754 | 0.748 | ~0 |
| `weights_kdm_occlusion` | 0.405 | 0.404 | ~0 |
| `weights_kdm_kernel_distance` | 0.526 | 0.548 | +0.022 (worse) |
| `weights_kdm_blend` | 0.742 | 0.802 | +0.060 (worse) |

`weights_kdm_occlusion`'s per-factor breakdown (`results/weights_kdm_occlusion_v2/metrics.json`)
replicates `exp_6`'s exact pattern — `pirads`/`bx`/`age`/`psa`/`dre` solid, `cspca`/`comorbidity`/
`vol`/`psad` still near-zero `decisive_set_f1`. Nothing about the backbone change touched the
solvable/unsolvable factor split established across two prior experiments now.

Per `DESIGN.md` §8's decision rules: this outcome is the **"backbone accuracy was not the
bottleneck"** branch, but with an added twist the rules didn't fully anticipate — the backbone's
apparent accuracy *gain* itself didn't hold up, so there was no genuine backbone improvement to
propagate downstream in the first place. Confidence/weights being flat is therefore not surprising
in retrospect: there was nothing better to inherit from.

## 5. Interpretation

1. **Neither low-risk lever produced a verified improvement.** `log1p` showed a real if modest CV
   signal that a combined hyperparameter search didn't meaningfully add to, but even that signal
   failed the held-out check. Hyperparameter tuning alone showed nothing at any stage.
2. **The experiment's infrastructure did its job.** The held-out check, the ablation isolating
   which lever contributed, and the pre-registered margin threshold all functioned exactly as
   `DESIGN.md` intended — catching a plausible-looking but spurious CV result before it could be
   reported as a genuine win. This is a successful application of the project's own stated
   discipline, not a wasted experiment.
3. **This closes the low-risk end of `exp_6`'s deferred lever menu.** Per `DESIGN.md` §8's last
   branch, the architecture-level levers explicitly set aside at the start of this experiment
   (ARD sigma, reduced-set prototypes, alternate kernels, `y_train=True`) are now the more clearly
   justified next questions if KDM's decision accuracy is still worth pursuing — incremental
   tuning of the current architecture has been tried and did not hold up.

## 6. Recommendation

- **Do not adopt the searched hyperparameters or `log1p` transform** as a new KDM default for
  any future experiment building on `exp_6` — the held-out evidence points against it, not for it.
- **Keep `exp_6`'s original backbone configuration** as the reference KDM setup.
- **If KDM's decision accuracy remains worth improving**, the next experiment should target one
  of the architecture-level levers deferred at the start of this one, not further hyperparameter
  search on the current architecture — this experiment is reasonably strong evidence that avenue
  is exhausted for this model.
- **Methodological note for future multi-way searches in this project**: set the "clear margin"
  threshold relative to the incumbent's own measured CV std, not a fixed constant, and treat a
  held-out check as mandatory before reporting any multi-way-search result as a genuine
  improvement — not optional verification, as this experiment's outcome demonstrates concretely.
