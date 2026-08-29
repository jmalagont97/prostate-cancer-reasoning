# EXPERIMENT 7 — corrected BX rerun

**Protocol:** MCCV-30 with corrected canonical tabular data; `bx` included as an input feature.
**Author:** Juan Sebastian Malagón Torres
**Status:** Complete

| Condition | Macro-F1 | Key metrics |
|---|---:|---|
| `C0_Baseline_Majority` | 0.2529 ± 0.0000 | mean_accuracy=0.6111 |
| `C1_Canonical_Fuzzy_ICI` | 0.2792 ± 0.0861 | mean_accuracy=0.3352 |
| `C1_clf_Fuzzy_ICI` | 0.3002 ± 0.1227 | mean_accuracy=0.3389 |
| `C2_Optimal_Weighted_ICI` | 0.2516 ± 0.0926 | mean_accuracy=0.3167 |
| `C3_Unweighted_UniformTree` | 0.2540 ± 0.0187 | mean_accuracy=0.5926 |

La ejecución se remanufacturó después de corregir la fuente de BX. Las cifras no deben compararse como equivalentes con ejecuciones históricas que excluían esta variable.
