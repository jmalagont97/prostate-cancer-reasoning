# exp_5 — BrentMemKDM multimodal (MCCV-30)

## Estado
Ejecución completa regenerada con la variable basal canónica `bx` incluida obligatoriamente en la rama tabular. `Positive=1`, `Negative=0` y ausencia tratada mediante imputación e indicador de missingness ajustados dentro de cada fold. No se utilizaron campos categóricos históricos ni campos histopatológicos post-biopsia.

## Protocolo conservado
- 195 casos y 30 particiones MCCV estandarizadas.
- Arquitecturas, búsquedas de `sigma`, fusión, semillas, evaluador `chimera_metrics`, métricas y pruebas estadísticas sin cambios.
- Brazos originales `hard` y `soft`; no se ejecutaron ablaciones.

## Resultados agregados
| Arm | Condición | Macro-F1 (media ± DE) | S_gate |
|---|---|---:|---:|
| hard | C0_Baseline | 0.5007 ± 0.0853 | 0.6481 |
| hard | C1_Tabular_KDM | 0.6002 ± 0.0956 | 0.6370 |
| hard | C2_MRI_KDM | 0.6615 ± 0.1056 | 0.7056 |
| hard | C3_Text_KDM | 0.3885 ± 0.0352 | 0.6148 |
| hard | C4_EqualFusion | 0.5998 ± 0.1265 | 0.6648 |
| hard | C5_OptimalFusion | 0.6655 ± 0.1011 | 0.6981 |
| soft | C0_Baseline | 0.5007 ± 0.0853 | 0.6481 |
| soft | C1_Tabular_KDM | 0.6427 ± 0.0845 | 0.6704 |
| soft | C2_MRI_KDM | 0.5969 ± 0.1068 | 0.6519 |
| soft | C3_Text_KDM | 0.4667 ± 0.1061 | 0.6463 |
| soft | C4_EqualFusion | 0.6116 ± 0.1150 | 0.6815 |
| soft | C5_OptimalFusion | 0.6687 ± 0.1101 | 0.6833 |

## Artefactos
Los resultados por split, predicciones, configuraciones optimizadas, pruebas Wilcoxon, logs y versiones se encuentran en `results/hard/`, `results/soft/` y `results/execution.log`.

## Validación
Smoke test de preprocesamiento: correcto. La ejecución completa terminó con código 0 en `tmux:0`.
