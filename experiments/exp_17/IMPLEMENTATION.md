# Implementation Plan: Búsqueda Sistemática de Hiperparámetros para la Decisión Terapéutica (exp_17)
**Experiment**: `experiments/exp_17/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Task 2)  
**Date**: 2026-08-30  
**Status**: In Progress

---

## Componentes de Implementación

1. **Script Principal de Búsqueda**: `experiments/exp_17/scripts/run_exp17_hyperparameter_search.py`
   - Carga de datasets procesados desde `data/processed/task2/` y particiones desde `data/processed/task2/mccv_splits_30.csv`.
   - Búsqueda en grilla de hiperparámetros para clasificadores Hard-SVC y regresores Fuzzy-SVR en cada una de las 3 modalidades.
   - Búsqueda exhaustiva en el simplex $\Delta^2$ (paso 0.05, 231 puntos) sobre las 30 particiones.
   - Evaluación obligatoria con `chimera_metrics.task2.decision.compute_treatment_decision_gate`.
2. **Artefactos de Resultados**:
   - `experiments/exp_17/results/best_hyperparams.json`: Mejores parámetros unimodales.
   - `experiments/exp_17/results/best_fusion_weights.json`: Ponderaciones óptimas en $\Delta^2$.
   - `experiments/exp_17/results/aggregate_metrics.json`: Métricas globales agregadas (mean ± std).
   - `experiments/exp_17/results/version.txt`: Hash y estado de git.
3. **Reporte Final**:
   - `experiments/exp_17/reports/summary.md`: Análisis comparativo frente a `exp_16`.
