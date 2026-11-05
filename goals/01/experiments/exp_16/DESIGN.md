# exp_16: ICI Regression Tree — Corrected (v2)

**Status:** Pending execution
**Date:** 2026-08-18
**Objective:** Predict `target_confidence` using `DecisionTreeRegressor` on ICI with ordinal encoding (uncertain=0, borderline=1, clear=2).

### Critical corrections from v1

1. **Base models use `ConfidenceWeightedKNN`** (not `StandardKNN`), matching exp_4–exp_11 winners.
2. **KNNs are trained on `target_biopsy_decision_binary`**, NOT on `target_confidence`.
3. Full 2×2 factorial: `criterion` × `sample_weight`.

---

## Conditions

| Condition | `criterion` | `sample_weight` |
|-----------|------------|-----------------|
| `reg_l1_none` | absolute_error | None |
| `reg_l1_balanced` | absolute_error | Balanced by class |
| `reg_l2_none` | squared_error | None |
| `reg_l2_balanced` | squared_error | Balanced by class |

Weights: $w_c = N / (3 \cdot n_c)$ computed on training fold.

---

## Output Conversion

$\hat{y} = \text{clip}(\lfloor \hat{s} + 0.5 \rfloor, 0, 2)$

---

## Frozen Multimodal Ensemble

Same as exp_14: ConfidenceWeightedKNN trained on `target_biopsy_decision_binary`.

## Evaluation

Same protocol as exp_14: MOE_abs primary, F1_macro tiebreaker.
