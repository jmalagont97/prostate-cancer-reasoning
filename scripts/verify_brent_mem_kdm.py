"""Standalone verification for `src/methods/brent_mem_kdm.py`. No test framework in this repo by
convention (see `mem_kdm.py`'s own `check_roundtrip=True` probe, `exp_23`'s `init_and_check`) — this
prints PASS/FAIL per check and exits non-zero if anything fails.

Checks:
  1. Fast-vs-torch exactness — the whole search runs on the fast (Nadaraya-Watson-reduction) path,
     so this is the load-bearing check: the search is only as correct as this agreement. Probed at
     the search's lower bound, sigma_ref (center), upper bound, AND a random interior point — the
     bounds are where KDMLayer's out_w clamp is most likely to fire (see brent_mem_kdm.py's module
     docstring), so a center-only probe would pass while the corners silently disagreed.
  2. Vectorized macro-F1 (`binary_macro_f1`) vs `sklearn.metrics.f1_score(average="macro",
     zero_division=0)`, on random inputs AND degenerate single-class folds (where sklearn's
     `labels=None` union-of-present-classes behavior is easy to get wrong).
  3. Unimodal `tab` sanity — Brent search over the full 100 MCCV splits should score >= exp_27's
     4-point `sigma_mult` grid winner on the SAME splits (identical objective, finer resolution).
  4. Nested 2-/3-modality smoke test — trace/recursion levels well-formed, `result.score` equals the
     tracked best leaf evaluation.
  5. Determinism — two identical runs return bit-identical sigma*.
  6. knn truncation (`knn_k`, brent_mem_kdm.py's k-NN section) — five sub-checks: (a) k=1 closed form
     (p1 = y_soft_train[nearest], independent of sigma — pins truncation, divisor, and clamp at once);
     (b) knn_k=None is bit-exact against the implicit default (guards the early-return branch); (c)
     k >= n_train agrees with knn_k=None (float32 gather-vs-direct-sum noise expected, not bit-exact);
     (d) fast-vs-torch agreement under truncation, via `_knn_submodel`, at the same lower/center/upper
     sigma regimes as check 1 (the lower bound is where the KDM_EPS clamp is live under truncation);
     (e) determinism.

Usage:
    conda activate pytorch                              # NOT histo-DL (see project memory)
    python scripts/verify_brent_mem_kdm.py               # all 6 checks
    python scripts/verify_brent_mem_kdm.py --quick        # checks 1, 2, 4, 6 only (skips the 100-split search)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.data import (  # noqa: E402
    OLD_SCHEMA, build_mri_features, build_tabular_features, build_targets, build_text_features,
    clean_texts_spacy, load_cohort, resolve_data_dir, CONFIDENCE_CERTAINTY_MAP,
)
from src.evaluation.protocol import iter_mccv_splits  # noqa: E402
from src.methods.base import Targets  # noqa: E402
from src.methods.brent_mem_kdm import (  # noqa: E402
    Fold, _FoldCache, _bounds_for_modality, _knn_submodel, _sigma_ref_per_modality,
    binary_macro_f1, run_brent_search,
)
from src.methods.mem_kdm import EncoderSpec, KernelSpec, MemKDM  # noqa: E402

# Tolerance for check 1 (fast-vs-torch exactness). Not 1e-5 everywhere: at small sigma, most training
# points' k^2 underflows to exact 0 in float32 well before KDMLayer's 1e-12 floor, so the prediction
# becomes a near-tie between the (tiny) true winner and the aggregate contribution of every OTHER point
# sitting at that floor. A ~1e-6-relative difference in the squared-distance matrix — normal cross-library
# float32 GEMM noise between numpy's BLAS and torch's backend — gets divided by sigma^2 and exponentiated,
# which can shift that near-tie enough to move p1 by ~1e-4-1e-3. Calibrated empirically (see the
# implementation notes) against 4 modality sets x ~20 random sigma draws each: every non-extreme regime
# agreed to <3e-5, and the worst observed (an extreme lower-bound draw) was 5.3e-4. This is a property of
# comparing two independent float32 implementations in an ill-conditioned (near-clamp) regime, not a
# defect in the Nadaraya-Watson reduction itself (verified analytically in brent_mem_kdm.py's module
# docstring, and empirically: non-extreme draws consistently agree far tighter than this bound).
FAST_VS_TORCH_TOL = 1e-3

RESULTS: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    RESULTS.append((name, ok))
    return ok


def build_modality(cohort, cleaned_texts, name, rep, train_idx, val_idx):
    if name == "tab":
        return build_tabular_features(cohort.df_tab, train_idx, val_idx, dre_categories=cohort.dre_categories)
    if name == "mri":
        pca_variance = None if rep == "raw_l2" else 0.90
        return build_mri_features(cohort.df_mri, train_idx, val_idx, pca_variance=pca_variance)
    if name == "txt":
        max_features, pca_variance = rep
        return build_text_features(cleaned_texts, train_idx, val_idx, max_features=max_features, pca_variance=pca_variance)
    raise ValueError(name)


def build_folds(cohort, cleaned_texts, y_soft, splits, mods, reps) -> list:
    folds = []
    for _, train_idx, val_idx in splits:
        X_tr, X_va = {}, {}
        for name in mods:
            X_tr[name], X_va[name] = build_modality(cohort, cleaned_texts, name, reps.get(name), train_idx, val_idx)
        folds.append(Fold(X_train=X_tr, y_soft_train=y_soft[train_idx], X_val=X_va, y_val=cohort.y_binary[val_idx]))
    return folds


# ---------------------------------------------------------------------------
# Check 1 — fast vs torch exactness
# ---------------------------------------------------------------------------
def _torch_probs(folds: list, mods: list, sigmas: dict) -> np.ndarray:
    out = []
    for f in folds:
        kernels = {m: KernelSpec(sigma=float(sigmas[m]), trainable=False) for m in mods}
        encoders = {m: EncoderSpec("identity") for m in mods}
        model = MemKDM(kernels=kernels, encoders=encoders, x_train=False, y_train=False, w_train=False, seed=0)
        model.fit(f.X_train, Targets(y_binary=np.zeros(len(f.y_soft_train), dtype=int), y_soft=f.y_soft_train))
        out.append(model.predict_proba(f.X_val)[:, 1])
    return np.concatenate(out)


def check_fast_vs_torch(folds: list, mods: list, rng: np.random.Generator) -> None:
    sigma_ref = _sigma_ref_per_modality(folds, mods)
    bounds = {m: _bounds_for_modality(sigma_ref[m], (1.0 / 32, 32.0)) for m in mods}
    cache = _FoldCache(folds, mods, label_smoothing=0.0)

    regimes = {
        "lower_bound": {m: bounds[m][0] for m in mods},
        "sigma_ref": {m: sigma_ref[m] for m in mods},
        "upper_bound": {m: bounds[m][1] for m in mods},
    }
    for k in range(3):
        regimes[f"random_interior_{k}"] = {
            m: float(np.exp(rng.uniform(np.log(bounds[m][0]), np.log(bounds[m][1])))) for m in mods
        }

    label = "+".join(mods)
    for regime_name, sigmas in regimes.items():
        p_fast = np.concatenate(cache.probs(sigmas))
        p_torch = _torch_probs(folds, mods, sigmas)
        max_err = float(np.abs(p_fast - p_torch).max())
        check(f"fast-vs-torch exactness [{label}] regime={regime_name}", max_err < FAST_VS_TORCH_TOL,
              f"max|Δp|={max_err:.2e} (tol={FAST_VS_TORCH_TOL:.0e})")


# ---------------------------------------------------------------------------
# Check 6 — knn truncation (brent_mem_kdm.py's k-NN section)
# ---------------------------------------------------------------------------
# Same regime as FAST_VS_TORCH_TOL and for the same reason: near KDM_EPS's clamp (live at small sigma,
# which is exactly where truncation's divisor change matters most) two independent float32
# implementations agree loosely, not bit-exactly.
KNN_FAST_VS_TORCH_TOL = 1e-3


def check_knn_truncation(folds: list, mods: list, rng: np.random.Generator) -> None:
    sigma_ref = _sigma_ref_per_modality(folds, mods)
    bounds = {m: _bounds_for_modality(sigma_ref[m], (1.0 / 32, 32.0)) for m in mods}
    cache = _FoldCache(folds, mods, label_smoothing=0.0)
    label = "+".join(mods)
    n_train = len(folds[0].y_soft_train)

    regimes = {
        "lower_bound": {m: bounds[m][0] for m in mods},
        "sigma_ref": {m: sigma_ref[m] for m in mods},
        "upper_bound": {m: bounds[m][1] for m in mods},
    }

    # (a) k=1 closed form: p1 = y_soft_train[nearest], independent of sigma — pins the truncation, the
    # divisor, and the clamp all at once (see brent_mem_kdm.py's module docstring).
    for regime_name, sigmas in regimes.items():
        p1_k1 = cache.probs(sigmas, knn_k=1)
        max_err = 0.0
        for i, f in enumerate(folds):
            expo = np.zeros((len(f.y_val), n_train), dtype=np.float64)
            for m in mods:
                d2 = ((np.asarray(f.X_val[m])[:, None, :] - np.asarray(f.X_train[m])[None, :, :]) ** 2).sum(-1)
                expo += d2 / (float(sigmas[m]) ** 2)
            nearest = np.argmin(expo, axis=-1)
            expected = f.y_soft_train[nearest]
            max_err = max(max_err, float(np.abs(p1_k1[i] - expected).max()))
        check(f"knn k=1 closed form [{label}] regime={regime_name}", max_err < 1e-5, f"max|Δ|={max_err:.2e}")

    # (b) knn_k=None is bit-exact against the implicit default (guards the early-return branch).
    sigmas_center = regimes["sigma_ref"]
    p1_default = cache.probs(sigmas_center)
    p1_explicit_none = cache.probs(sigmas_center, knn_k=None)
    same = all(np.array_equal(a, b) for a, b in zip(p1_default, p1_explicit_none))
    check(f"knn_k=None bit-exact vs default call [{label}]", same)

    # (c) k >= n_train agrees with knn_k=None. Not bit-exact: the truncated branch gathers columns
    # then sums (via einsum over a k-length axis) while the untruncated branch sums the full row
    # directly — a different float32 accumulation order, same as check (b)'s absence of this gap shows.
    p1_full_k = cache.probs(sigmas_center, knn_k=n_train * 10)
    max_err = max(float(np.abs(a - b).max()) for a, b in zip(p1_default, p1_full_k))
    check(f"knn k>=n_train equiv to knn_k=None [{label}]", max_err < 1e-4, f"max|Δ|={max_err:.2e}")

    # (d) fast-vs-torch agreement under truncation, via the exact per-query `_knn_submodel` path, at
    # the same lower/center/upper sigma regimes check 1 uses — the lower bound is where KDM_EPS's
    # clamp is live under truncation and where a wrong divisor would show up.
    f0 = folds[0]
    for k in (1, min(5, n_train)):
        for regime_name, sigmas in regimes.items():
            p_fast = cache.probs(sigmas, knn_k=k)[0]
            max_err = 0.0
            for v in range(len(f0.y_val)):
                x_row = {m: np.asarray(f0.X_val[m])[v:v + 1] for m in mods}
                sub_model, _nbr = _knn_submodel(f0.X_train, f0.y_soft_train, mods, sigmas, k, x_row,
                                                 label_smoothing=0.0, seed=0)
                p_exact = sub_model.predict_proba(x_row)[0, 1]
                max_err = max(max_err, float(abs(p_exact - p_fast[v])))
            check(f"knn fast-vs-torch [{label}] k={k} regime={regime_name}", max_err < KNN_FAST_VS_TORCH_TOL,
                  f"max|Δp|={max_err:.2e} (tol={KNN_FAST_VS_TORCH_TOL:.0e})")

    # (e) determinism — ties in argpartition/argsort resolve the same way every time.
    r1 = cache.probs(sigmas_center, knn_k=3)
    r2 = cache.probs(sigmas_center, knn_k=3)
    same = all(np.array_equal(a, b) for a, b in zip(r1, r2))
    check(f"knn determinism [{label}]", same)


# ---------------------------------------------------------------------------
# Check 2 — vectorized macro-F1 vs sklearn
# ---------------------------------------------------------------------------
def check_macro_f1_vs_sklearn(rng: np.random.Generator) -> None:
    max_err = 0.0
    n_cases = 500
    for _ in range(n_cases):
        n = int(rng.integers(1, 30))
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.integers(0, 2, size=n)
        ours = binary_macro_f1(y_true, y_pred)
        theirs = f1_score(y_true, y_pred, average="macro", zero_division=0)
        max_err = max(max_err, abs(ours - theirs))
    check("vectorized macro-F1 vs sklearn (random)", max_err < 1e-9, f"max|Δ|={max_err:.2e} over {n_cases} cases")

    degenerate = [
        (np.zeros(5, dtype=int), np.zeros(5, dtype=int)),   # all-0 true, all-0 pred
        (np.ones(5, dtype=int), np.ones(5, dtype=int)),     # all-1 true, all-1 pred
        (np.zeros(5, dtype=int), np.ones(5, dtype=int)),    # all-0 true, all-1 pred
        (np.ones(5, dtype=int), np.zeros(5, dtype=int)),    # all-1 true, all-0 pred
        (np.array([1]), np.array([1])),                     # n=1
        (np.array([0]), np.array([1])),                     # n=1, wrong
    ]
    max_err_deg = 0.0
    for y_true, y_pred in degenerate:
        ours = binary_macro_f1(y_true, y_pred)
        theirs = f1_score(y_true, y_pred, average="macro", zero_division=0)
        max_err_deg = max(max_err_deg, abs(ours - theirs))
    check("vectorized macro-F1 vs sklearn (degenerate single-class folds)", max_err_deg < 1e-9,
          f"max|Δ|={max_err_deg:.2e}")


# ---------------------------------------------------------------------------
# Check 3 — unimodal tab sanity vs exp_27's grid
# ---------------------------------------------------------------------------
def check_unimodal_tab_sanity(cohort, cleaned_texts, y_soft, splits) -> None:
    folds = build_folds(cohort, cleaned_texts, y_soft, splits, ["tab"], {})
    t0 = time.time()
    result = run_brent_search(folds, ["tab"], metric="macro_f1", strategy="nested", n_prescan=15, maxiter=20)
    elapsed = time.time() - t0
    print(f"    brent tab search: sigma*={result.sigmas['tab']:.5f} "
          f"(sigma_mult={result.sigma_mult['tab']:.3f}), score={result.score:.4f}, "
          f"n_evals={result.n_evals}, {elapsed:.1f}s")

    # Reproduce exp_27's 4-point sigma_mult grid with the SAME fast-path scorer for an apples-to-apples
    # comparison (both evaluate the identical frozen-sigma Nadaraya-Watson reduction on the same splits).
    cache = _FoldCache(folds, ["tab"], label_smoothing=0.0)
    sigma_ref = _sigma_ref_per_modality(folds, ["tab"])["tab"]
    grid_scores = {}
    for sm in [0.25, 0.5, 1.0, 2.0]:
        score, _ = cache.score({"tab": sigma_ref * sm}, "macro_f1", 0.50, "mean")
        grid_scores[sm] = score
    best_grid_sm = max(grid_scores, key=grid_scores.get)
    best_grid_score = grid_scores[best_grid_sm]
    print(f"    exp_27-style grid: {grid_scores}, best sigma_mult={best_grid_sm} -> {best_grid_score:.4f}")

    check("unimodal tab: Brent score >= exp_27-style grid winner", result.score >= best_grid_score - 1e-9,
          f"brent={result.score:.4f} vs grid={best_grid_score:.4f}")


# ---------------------------------------------------------------------------
# Check 4 — nested smoke test
# ---------------------------------------------------------------------------
def check_nested_smoke(cohort, cleaned_texts, y_soft, splits) -> None:
    for mods, reps in [
        (["tab", "mri"], {"mri": "pca90_l2"}),
        (["tab", "mri", "txt"], {"mri": "pca90_l2", "txt": (2000, 0.90)}),
    ]:
        folds = build_folds(cohort, cleaned_texts, y_soft, splits[:5], mods, reps)
        result = run_brent_search(folds, mods, metric="macro_f1", strategy="nested", n_prescan=5, maxiter=8)
        label = "+".join(mods)

        levels_seen = {e["level"] for e in result.trace}
        well_formed = levels_seen == {len(mods) - 1} and all(set(mods) == set(e["sigmas"]) for e in result.trace)
        check(f"nested smoke [{label}]: trace well-formed", well_formed,
              f"levels={sorted(levels_seen)}, n_evals={result.n_evals}")

        best_leaf_score = max(e["score"] for e in result.trace)
        check(f"nested smoke [{label}]: result.score matches tracked best leaf",
              abs(result.score - best_leaf_score) < 1e-12,
              f"result.score={result.score:.6f}, best_leaf={best_leaf_score:.6f}")


# ---------------------------------------------------------------------------
# Check 5 — determinism
# ---------------------------------------------------------------------------
def check_determinism(cohort, cleaned_texts, y_soft, splits) -> None:
    folds = build_folds(cohort, cleaned_texts, y_soft, splits[:10], ["tab", "mri"], {"mri": "pca90_l2"})
    r1 = run_brent_search(folds, ["tab", "mri"], metric="macro_f1", strategy="nested", n_prescan=7, maxiter=10)
    r2 = run_brent_search(folds, ["tab", "mri"], metric="macro_f1", strategy="nested", n_prescan=7, maxiter=10)
    same = r1.sigmas == r2.sigmas and r1.score == r2.score
    check("determinism: repeated identical runs", same,
          f"run1={r1.sigmas}, run2={r2.sigmas}")


def main(quick: bool = False) -> int:
    rng = np.random.default_rng(0)

    print("[verify] loading cohort...")
    data_dir = resolve_data_dir(PROJECT_ROOT)
    cohort = load_cohort(data_dir, PROJECT_ROOT, OLD_SCHEMA, load_mri=True, load_text=True)
    targets = build_targets(cohort.y_binary, cohort.confidence, certainty_map=CONFIDENCE_CERTAINTY_MAP)
    print("[verify] spaCy cleaning text corpus (once)...")
    cleaned_texts = clean_texts_spacy(cohort.df_text["clinical_prompt_text"].values)
    all_splits = list(iter_mccv_splits(cohort.df_design, n_splits=100))

    print("\n=== Check 1: fast-vs-torch exactness ===")
    small_splits = all_splits[:3]
    check_fast_vs_torch(build_folds(cohort, cleaned_texts, targets.y_soft, small_splits, ["tab"], {}), ["tab"], rng)
    check_fast_vs_torch(build_folds(cohort, cleaned_texts, targets.y_soft, small_splits, ["mri"], {"mri": "pca90_l2"}),
                         ["mri"], rng)
    check_fast_vs_torch(build_folds(cohort, cleaned_texts, targets.y_soft, small_splits, ["txt"], {"txt": (2000, 0.90)}),
                         ["txt"], rng)
    check_fast_vs_torch(
        build_folds(cohort, cleaned_texts, targets.y_soft, small_splits, ["tab", "mri", "txt"],
                    {"mri": "pca90_l2", "txt": (2000, 0.90)}),
        ["tab", "mri", "txt"], rng,
    )

    print("\n=== Check 2: vectorized macro-F1 vs sklearn ===")
    check_macro_f1_vs_sklearn(rng)

    print("\n=== Check 4: nested smoke test ===")
    check_nested_smoke(cohort, cleaned_texts, targets.y_soft, all_splits)

    print("\n=== Check 6: knn truncation ===")
    check_knn_truncation(build_folds(cohort, cleaned_texts, targets.y_soft, small_splits, ["tab"], {}), ["tab"], rng)
    check_knn_truncation(
        build_folds(cohort, cleaned_texts, targets.y_soft, small_splits, ["tab", "mri"], {"mri": "pca90_l2"}),
        ["tab", "mri"], rng,
    )

    if not quick:
        print("\n=== Check 3: unimodal tab sanity vs exp_27-style grid ===")
        check_unimodal_tab_sanity(cohort, cleaned_texts, targets.y_soft, all_splits)

        print("\n=== Check 5: determinism ===")
        check_determinism(cohort, cleaned_texts, targets.y_soft, all_splits)
    else:
        print("\n[verify] --quick: skipping checks 3 and 5 (100-split search)")

    n_pass = sum(1 for _, ok in RESULTS if ok)
    n_total = len(RESULTS)
    print(f"\n[verify] {n_pass}/{n_total} checks passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    sys.exit(main(quick=args.quick))
