# Experiment Design: MemKDM without the noise-selected linear encoder (exp_26)
**Experiment**: experiments/exp_26/ · **Project**: pathology-reasoning · **Date**: 2026-08-19 · **Status**: Complete

---

## 1. Motivation — a corrective follow-up to exp_25, not a new hypothesis space

exp_25's Stage 1 grid search for `tab` selected `encoder=linear` (mean MCCV Macro-F1 0.6044) over
`encoder=identity` (0.6013) — a margin of 0.003, two orders of magnitude smaller than each config's own
noise (`std_macro_f1` ≈ 0.11). `src/evaluation/protocol.py`'s `select_best` docstring already flags
exactly this failure mode ("winners' std_macro_f1 (~0.11 in exp_13/23) dwarfs typical inter-config gaps
(<0.02)") but its tie-break (prefer lower std) only fires within `tol=1e-9` of the best mean, far tighter
than this 0.003 gap — so plain argmax picked the statistically-indistinguishable `linear` option.

`encoder=linear` (`src/methods/mem_kdm.py:118-136`) is a trainable `nn.Linear` with random (Kaiming)
init, trained jointly with the KDM. Unlike `identity` (no parameters, fully deterministic), this makes
every downstream model stochastic across seeds — exp_25's `unimodal_tab` LOOCV result
(`per_seed_macro_f1` over 10 seeds: 0.52–0.67, mean 0.5978) sits ~0.07 below exp_23 Arm B's deterministic
0.6694 (`sigma_mult=2.0, encoder=identity`) — a config exp_25 itself exactly reproduces elsewhere in the
same run (`results/reproduction_check.json`) as a sanity gate. Since `tab`'s Stage-1 winner selects the
encoder used in every Stage-2 joint condition too (`STAGE1_BEST["tab"]["encoder"]`), this one
noise-driven pick propagates into `joint_tab_mri`, `joint_tab_txt`, `joint_trimodal`, and `confidence_arm`
as well.

**H (single, corrective).** Restricting Stage 1/Stage 2 to `encoder=identity` only (dropping `linear`
from the grid entirely) removes this selection-noise-driven regression: `unimodal_tab`'s LOOCV Macro-F1
recovers to match exp_23 Arm B (0.6694, expected bit-for-bit — see §3), and re-evaluating exp_25's H1
(`joint_trimodal` vs. best unimodal vs. leak-free late fusion vs. exp_23 0.6694) under this fix shows
whether exp_25's "H1 refuted" verdict survives once the encoder-instability confound is removed.

## 2. Scope — diff against exp_25

Reuses exp_25's cohort loading, feature builders, MCCV/LOOCV harness (`src/evaluation/*`),
`MemKDM`/`LateFusionMemKDM` (`src/methods/mem_kdm.py`) unchanged — no `src/` code changes. Only
`experiments/exp_26/scripts/train.py` (built from `exp_25/scripts/train.py`) changes:

1. `TAB_GRID`/`MRI_GRID`/`TXT_GRID`: drop the `encoder` loop dimension; every config is `encoder="identity"`.
   Grid sizes: tab 16→8, mri 24→12, txt 36→18. `get_out_dim` (linear-encoder output dims) is removed —
   dead code once no config uses `linear`.
2. Every config in every stage is now deterministic (no trainable encoder), so all `n_seeds=10 if
   encoder=="linear" else 1` branching collapses to `n_seeds=1` everywhere. `loocv_metrics.json`'s
   schema (`per_seed_macro_f1`, `macro_f1_std_across_seeds`, `mode_vote_agreement`, `deterministic`,
   `n_seeds`) is unchanged — fields just take their trivial values (`n_seeds=1`, `deterministic=True`).
3. **Sigma transfer, Stage 1 → Stage 2/Phase B.** Stage 1's own grid search is unchanged — it still
   searches `sigma_mult` (a multiplier on the per-fold `_sigma_from_knn` heuristic), exactly as exp_25.
   What changes is how the *winning* `sigma_mult` is carried into Stage 2's joint kernel and into every
   Phase-B evaluate closure (unimodal and joint both route through the same
   `build_joint_kernels_encoders` helper): instead of re-passing `sigma_mult` forward and letting
   `MemKDM.fit()` internally re-derive `sigma = mult × _sigma_from_knn(encoded_block)`, the absolute
   `sigma` is resolved explicitly in `train.py`, fold-locally — using only that fold's own training
   features (`_sigma_from_knn(X_tr[mod], base_mult * scale)`) — and passed forward as
   `KernelSpec(sigma=...)`. Verified in `mem_kdm.py:333-342` that with `encoder=identity` the internal
   `block` used for this heuristic is exactly the raw fold-training features, so this is numerically a
   no-op relative to exp_25's approach — it only makes the transfer explicit/inspectable instead of
   implicit inside `.fit()`. Still computed fresh per fold from that fold's own training data only — no
   leakage introduced (this is the failure mode `.discussion.md`'s exp_25 entry already flagged for
   `MemKDM.from_unimodal`'s *fitted*-sigma transfer, and deliberately avoided here by resolving
   fold-locally rather than reusing one globally-fitted number).
4. Stage 2's `sigma_scale` grid dimension is unchanged (`[0.5, 1.0, 2.0]`) and keeps its existing
   semantics (multiplies the *tab*-modality's Stage-1 bandwidth by a fixed 1.0 always, and multiplies
   mri/txt's Stage-1 bandwidth by `cfg["sigma_scale"]`) — now applied to the resolved absolute sigma
   rather than to the mult, which (per §2.3) is the same number either way.

No change to: cohort/target construction, MCCV split design, LOOCV protocol, `select_best`'s tie-break
rule, grid values for `sigma_mult`/`rep`/`y_train`/`x_train`/`kernel_trainable`, confidence-arm
construction (`label_smoothing=0.10`), or any hypothesis/threshold from exp_25 not listed above.

## 3. Expected sanity check

Step 0's reproduction gate (exp_23 Arm B, tab-only, `sigma_mult=2.0, identity`, computed via the
untouched direct-`sigma_mult` code path) must still reproduce `0.6694214876033058` exactly — this path
is not touched by this experiment. Additionally, Stage 1's own `unimodal_tab` winner is expected to now
select `sigma_mult=2.0, encoder=identity` (the top identity-only row in exp_25's own Phase A ranking),
and Phase B's `unimodal_tab` — now going through the new explicit-sigma-transfer path — is expected to
match this same `0.6694` value bit-for-bit, since encoder is forced identity and the transferred sigma is
computed by the identical `_sigma_from_knn` heuristic on the identical fold data, just resolved outside
`.fit()` instead of inside it. This is the load-bearing check that the "transfer sigma, not sigma_mult"
refactor is truly a no-op for the unimodal case before trusting it for Stage 2's joint conditions.

## 4. Verdict criteria

Same three-way comparison as exp_25 H1 (`joint_trimodal` vs. best unimodal vs. leak-free late fusion vs.
exp_23 `0.6694`), re-run under the fixed grid. Reported as a correction/follow-up to exp_25 in
`reports/summary.md`, not a new hypothesis space — exp_25's H2/H3 are out of scope here unless Stage 2
numbers move enough to warrant flagging in the summary (not re-litigated from scratch).
