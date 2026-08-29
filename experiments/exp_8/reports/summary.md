# EXPERIMENT 8 — corrected BX rerun

**Protocol:** MCCV-30 with corrected canonical tabular data; `bx` included as an input feature.
**Author:** Juan Sebastian Malagón Torres
**Status:** Complete

| Condition | Macro-F1 | Key metrics |
|---|---:|---|
| `C1_10D_Raw_Tree` | 0.2324 ± 0.0193 | mean_s_weights=0.8452; mean_decisive_set_f1=0.6817; mean_psa_rho=0.0121; mean_pirads_rho=0.1408 |
| `C2_10D_Balanced_Tree` | 0.2198 ± 0.0315 | mean_s_weights=0.7751; mean_decisive_set_f1=0.6086; mean_psa_rho=0.0904; mean_pirads_rho=0.1538 |
| `C3_Coalesced_Balanced_Tree` | 0.2226 ± 0.0283 | mean_s_weights=0.7798; mean_decisive_set_f1=0.6145; mean_psa_rho=0.0277; mean_pirads_rho=0.1599 |
| `C4_Coalesced_Balanced_Forest` | 0.2444 ± 0.0235 | mean_s_weights=0.7978; mean_decisive_set_f1=0.6273; mean_psa_rho=0.0794; mean_pirads_rho=0.1823 |
| `C5_Hybrid_Regularized` | 0.2327 ± 0.0283 | mean_s_weights=0.7902; mean_decisive_set_f1=0.6658; mean_psa_rho=0.0277; mean_pirads_rho=0.1599 |

La ejecución se remanufacturó después de corregir la fuente de BX. Las cifras no deben compararse como equivalentes con ejecuciones históricas que excluían esta variable.
