# Experiment Report: exp_4 — MCCV-30 Unimodal Sweep and Late Fusion

**Experiment**: `experiments/exp_4/`  
**Project**: `prostate-cancer-reasoning`  
**Report date**: 2026-08-29  
**Author**: Juan Sebastian Malagón Torres & AI Co-Researcher  
**Status**: Complete ✅

---

## 1. Summary

Se reejecutó completamente `exp_4` con la variable basal corregida `bx` procedente de `clinical_tabular.csv`. La corrección sustituyó la referencia obsoleta `cli_bx` sin modificar el protocolo MCCV-30, los modelos, la búsqueda de hiperparámetros, el evaluador ni las métricas. La ejecución terminó correctamente en `tmux:0` y produjo todos los artefactos estructurados.

---

## 2. Experimental Setup

- **Dataset**: 195 casos; 91 etiquetados (56 yes, 35 no); 30 particiones MCCV estandarizadas.
- **Corrección BX**: `Positive → 1`, `Negative → 0`, ausencia conservada mediante imputación e indicador fold-safe.
- **Preprocesamiento**: ajustado exclusivamente dentro del fold de entrenamiento.
- **Modelos**: configuraciones y espacios de búsqueda originales.
- **Ejecución**: `scripts/run_exp4_mccv_evaluation.py` en `tmux:0`.
- **Evaluador**: `chimera_metrics.task1.compute_biopsy_decision_gate`.
- **Desviaciones**: ninguna en métricas, splits, semillas, arquitectura o protocolo.

## 3. Configuraciones óptimas observadas

- **C1_Tabular_Optimal**: `{'spearman_tau': 0.8, 'model_type': 'fuzzy_knn_reg', 'k': 5, 'metric': 'manhattan', 'weights': 'uniform'}`
- **C2_MRI_Optimal**: `{'pca_components': 16, 'model_type': 'standard_knn_clf', 'k': 1, 'metric': 'euclidean', 'weights': 'uniform'}`
- **C3_Text_Optimal**: `{'max_features': 1000, 'svd_components': None, 'model_type': 'fuzzy_knn_reg', 'k': 3, 'metric': 'cosine', 'weights': 'uniform'}`
- **C4_LateFusion_Optimal**: `{'w_tab': 0.15, 'w_mri': 0.05, 'w_txt': 0.8, 'threshold': 0.56, 'mean_macro_f1': 0.7051036402463242, 'std_macro_f1': 0.12163536523654717}`

## 4. Resultados MCCV-30

| Condición | Macro-F1 | S_gate | Sensibilidad | Especificidad |
|---|---:|---:|---:|---:|
| Baseline | 0.5007 ± 0.0853 | 0.6481 | 0.9667 | 0.1476 |
| Tabular | 0.6456 ± 0.0895 | 0.6611 | 0.6879 | 0.6190 |
| MRI | 0.6240 ± 0.1101 | 0.6574 | 0.7515 | 0.5095 |
| Texto | 0.6770 ± 0.1140 | 0.6944 | 0.7485 | 0.6095 |
| Fusión tardía | 0.7051 ± 0.1237 | 0.7185 | 0.7576 | 0.6571 |

## 5. Pruebas estadísticas

Se conservaron las pruebas pareadas de Wilcoxon frente a `C0_Baseline`, sin cambios en su definición. Los valores ejecutados se encuentran en `results/wilcoxon_tests.json`.

## 6. Artefactos y reproducibilidad

- `results/aggregate_metrics.json`
- `results/split_metrics.json`
- `results/split_predictions.json`
- `results/optimized_configs.json`
- `results/wilcoxon_tests.json`
- `results/execution.log`
- `results/version.txt`
- `scripts/models.py`
- `scripts/run_exp4_mccv_evaluation.py`

## 7. Conclusión

`exp_4` fue remanufacturado y ejecutado con BX incorporada obligatoriamente. La ejecución finalizó con código de retorno 0 y el runner reportó verificación exitosa. Las métricas se calcularon con el protocolo original y no se añadieron estudios de ablación.
