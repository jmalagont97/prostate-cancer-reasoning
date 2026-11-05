# exp_15: 4D Vector [ICI, p_T, p_M, p_X] — Corrected (v2)

**Status:** Pending execution
**Date:** 2026-08-18
**Objective:** Evaluate whether adding p_T, p_M, p_X as features improves over ICI-only (exp_14) for predicting `target_confidence`.

### Critical corrections from v1

1. **Base models use `ConfidenceWeightedKNN`** (not `StandardKNN`), matching exp_4–exp_11 winners.
2. **KNNs are trained on `target_biopsy_decision_binary`**, NOT on `target_confidence`.
3. **No circularity:** The evaluated sample's own confidence is never used in its prediction.

---

## Feature Families

| Family | Input | Dimensions |
|--------|-------|------------|
| `ici_only` | [ICI] | 1D |
| `p_only` | [p_T, p_M, p_X] | 3D |
| `ici_plus_p` | [ICI, p_T, p_M, p_X] | 4D |

---

## Configurations

| Config | Family | `class_weight` |
|--------|--------|---------------|
| `ici_only_none` | ici_only | None |
| `ici_only_balanced` | ici_only | "balanced" |
| `p_only_none` | p_only | None |
| `p_only_balanced` | p_only | "balanced" |
| `ici_plus_p_none` | ici_plus_p | None |
| `ici_plus_p_balanced` | ici_plus_p | "balanced" |

Selection within each family by MOE_abs. LOO for each family winner.

---

## Frozen Multimodal Ensemble

Same as exp_14: ConfidenceWeightedKNN trained on `target_biopsy_decision_binary`.

---

## Evaluation

Same protocol as exp_14: MOE_abs primary, F1_macro tiebreaker, MCCV selection, LOO sanity check.
