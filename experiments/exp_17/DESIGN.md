# Experiment Design: Búsqueda Sistemática de Hiperparámetros para la Decisión Terapéutica (Subtarea 2.1) bajo MCCV-30
**Experiment**: `experiments/exp_17/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Task 2)  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: In Progress

---

## 1. Hypothesis

Una búsqueda sistemática y no sesgada de hiperparámetros (parámetro de regularización $C$, margen $\epsilon$, balance de clases y pesos de mezcla tardía en el simplex $\Delta^2$) adaptada al protocolo estandarizado **MCCV-30 multi-subtarea** mejorará el rendimiento de la decisión médica (Macro-F1 $\ge 0.640$ y Brier $\le 0.380$) frente a la transferencia estática de hiperparámetros de `exp_16`.

## 2. Experimental Setup

- **Dataset**: Cohorte de Task 2 ($N=70$, `data/processed/task2/`).
- **Splits**: `data/processed/task2/mccv_splits_30.csv` (30 particiones estratificadas multi-subtarea).
- **Modalidades**:
  1. Tabular: `clinical_tabular_task2.csv` (preprocesamiento canónico con imputación y `StandardScaler` ajustado por split).
  2. MRI: `mri_embeddings_task2.csv` (1024D con `StandardScaler` ajustado por split).
  3. Texto: `clinical_texts_task2.csv` (`TfidfVectorizer` ajustado por split sobre `full_encounter_narrative`).
- **Control de Fuga**: Todos los transformadores e hiperparámetros se optimizan dentro de las particiones de entrenamiento de cada split.

## 3. Espacio de Búsqueda de Hiperparámetros

### Fase 1: Modelos Unimodales
- **Hard-SVC (`SVC(kernel='linear', probability=True)`)**:
  - $C \in \{0.1, 1.0, 10.0\}$
  - `class_weight` $\in \{\text{None}, \text{'balanced'}\}$
  - 6 configuraciones por modalidad (Tabular, MRI, Texto).

- **Fuzzy-SVR (`MultiOutputRegressor(LinearSVR())`)**:
  - $C \in \{0.1, 1.0, 10.0\}$
  - $\epsilon \in \{0.05, 0.10\}$
  - 6 configuraciones por modalidad sobre targets difusos $y_k = 0.25 + 0.75c$.

### Fase 2: Simplex de Fusión Tardía $\Delta^2$
- Grilla con paso $0.05$ (231 combinaciones de pesos $(w_{\text{tab}}, w_{\text{mri}}, w_{\text{txt}})$ con $\sum w_i = 1$).
- Selección orientada a maximizar Macro-F1 evaluado vía `chimera_metrics`.

## 4. Evaluation Protocol

Uso exclusivo de **`utils/pkg/chimera_metrics`**:
- `chimera_metrics.task2.decision.compute_treatment_decision_gate`
- `chimera_metrics.core.base.norm_treatment_decision`
- Métricas: Macro-F1, Micro-F1 / Gate Accuracy, Brier Score, AUROC (OVR), y F1 por clase.

## 5. Next Steps

1. Registrar `experiments/exp_17/IMPLEMENTATION.md`.
2. Ejecutar `experiments/exp_17/scripts/run_exp17_hyperparameter_search.py`.
3. Consolidar resultados en `results/` y generar `reports/summary.md`.
