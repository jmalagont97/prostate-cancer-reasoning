# EXPERIMENT 9 — corrected BX rerun

**Protocol:** MCCV-30 with corrected canonical tabular data; `bx` included as an input feature.
**Author:** Juan Sebastian Malagón Torres
**Status:** Complete

| Condition | Macro-F1 | Key metrics |
|---|---:|---|
| `C1_SHAP_Only_13D` | 0.2453 ± 0.0318 |  |
| `C2_Raw_Only_10D` | 0.2798 ± 0.0236 |  |
| `C3_Joint_Forest_23D` | 0.2746 ± 0.0257 |  |
| `C4_Joint_ExtraTrees_23D` | 0.2779 ± 0.0268 |  |
| `C5_Targeted_Hybrid_23D` | 0.2578 ± 0.0246 |  |

La ejecución se remanufacturó después de corregir la fuente de BX. Las cifras no deben compararse como equivalentes con ejecuciones históricas que excluían esta variable.
