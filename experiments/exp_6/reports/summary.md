# Experiment Report: Scalar Direct-KDM with corrected BX input (exp_6)
**Experiment**: `experiments/exp_6/`  
**Report date**: 2026-08-29  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete

---

## 1. Summary
Se reejecutó `exp_6` bajo MCCV-30 con la variable basal canónica `bx` incluida en la matriz tabular junto con su indicador explícito de ausencia. Se conservaron las tres condiciones planificadas y el baseline de mayoría; no se modificaron las métricas ni el protocolo de evaluación. La mejor Macro-F1 observada fue la de `C1_Scalar_KDM_23col_clean`.

## 2. Hipótesis y veredicto
**Hipótesis:** la replicación Scalar Direct-KDM mantiene un rendimiento útil para la predicción ordinal de confianza cuando se ejecuta con entradas clínicas pre-biopsia corregidas y MCCV-30.

**Veredicto:** ⚠️ **Actualización empírica completada; comparación histórica no concluyente.**

La reejecución es válida y reproduce el protocolo, pero la inclusión de BX cambia el espacio de entrada respecto de la ejecución histórica; por tanto, sus métricas deben tratarse como una nueva referencia corregida y no como una comparación aislada de arquitectura.

## 3. Configuración ejecutada
- **Datos:** cohortes procesadas de Task 1, 91 casos etiquetados para confianza clínica.
- **Protocolo:** MCCV-30, con particiones estandarizadas.
- **Modelo:** Scalar Direct-KDM / Parzen KDM con condiciones C1, C2 y C3.
- **Corrección:** `bx` numérica incluida y `bx_missing` conservado explícitamente; imputación ajustada dentro de cada fold.
- **Artefactos:** métricas y predicciones por condición bajo `results/<condition>/`.

## 4. Resultados
| Condición | Macro-F1 | Ordinal distance | MOE_abs | Accuracy | Folds |
|---|---:|---:|---:|---:|---:|
| `C1_Scalar_KDM_23col_clean` | 0.5259 ± 0.1445 | 0.2602 ± 0.0853 | 0.3060 | 0.6185 | 30 |
| `C2_Scalar_KDM_19col_clean` | 0.5256 ± 0.1124 | 0.2824 ± 0.0756 | 0.3241 | 0.6204 | 30 |
| `C3_Scalar_KDM_23col_ordinal_weighted` | 0.5231 ± 0.1437 | 0.2611 ± 0.0883 | 0.3083 | 0.6167 | 30 |
| `Baseline_Majority_Clear` | 0.2529 ± 0.0000 | 0.2778 ± 0.0000 | 0.5000 | 0.6111 | 30 |

La condición con mayor Macro-F1 es `C1_Scalar_KDM_23col_clean` (0.5259). El baseline de mayoría se conserva como referencia no paramétrica y no representa un modelo clínico útil.

## 5. Integridad y reproducibilidad
- [x] MCCV-30 completado.
- [x] Predicciones y métricas guardadas por condición.
- [x] `version.txt` presente en cada condición.
- [x] BX verificada en el espacio de entrada mediante smoke test.
- [x] No se modificaron las métricas, el número de folds ni las reglas de evaluación.
- [ ] Checkpoints: no aplican a este experimento.

## 6. Limitaciones
La ejecución actual no debe compararse directamente con los valores históricos de `exp_6` sin considerar el cambio de representación inducido por BX. Las diferencias observadas pueden deberse a la nueva información admisible, a la imputación/missingness o a la interacción con el KDM.

## 7. Conclusión
`exp_6` corregido queda cerrado como nueva referencia MCCV-30 con BX. El análisis comparativo definitivo debe realizarse conjuntamente con las reejecuciones corregidas de `exp_7`, `exp_8` y `exp_9`, manteniendo la misma versión de datos y la misma convención de BX.
