# Implementation Plan: Replicación de Fusión Trimodal Fuzzy-SVR para Decisión Terapéutica (Subtarea 2.1) bajo MCCV-30
**Experiment**: `experiments/exp_16/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-30  
**Status**: Approved & Executed  

---

## 1. Plan Overview

Implementar y ejecutar el runner estandarizado de benchmark para la Subtarea 2.1 (Decisión del Tratamiento) evaluando:
1. **Condición C1 (Hard-SVC Baseline)**: Clasificadores lineales (`SVC(kernel='linear', probability=True)`) por modalidad + soft voting con pesos óptimos ($0.85, 0.15, 0.00$).
2. **Condición C2 (Fuzzy-SVR Champion)**: Regresión multi-salida (`MultiOutputRegressor(LinearSVR)`) sobre targets difusos derivados de la confianza clínica ($y_k = 0.25 + 0.75c$) con pesos óptimos ($0.55, 0.20, 0.25$).
3. **Validación MCCV-30**: 30 splits multiobjetivo extraídos de `data/processed/task2/mccv_splits_30.csv`.
4. **Gobernanza de Métricas**: Gobernado exclusivamente a través de `utils/pkg/chimera_metrics` (`compute_treatment_decision_gate` y `norm_treatment_decision`).

## 2. Scripts and Artifacts

- Runner: `experiments/exp_16/scripts/run_exp16_decision_mccv30.py`
- Salidas de Métricas:
  - `experiments/exp_16/results/aggregate_metrics.json`
  - `experiments/exp_16/results/split_metrics.json`
  - `experiments/exp_16/results/version.txt`
- Modelos serializados:
  - `experiments/exp_16/models/` (modelos entrenados sobre la cohorte completa N=70 para despliegue downstream)
- Reporte:
  - `experiments/exp_16/reports/summary.md`
