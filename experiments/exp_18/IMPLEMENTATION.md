# Implementation Plan: Búsqueda Sistemática de Hiperparámetros y Fusión para Confianza Clínica (exp_18)
**Experiment**: `experiments/exp_18/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Task 2)  
**Date**: 2026-08-30  
**Status**: In Progress  

---

## 1. Overview & Architecture

Implementar el script de búsqueda de hiperparámetros y optimización en el simplex de fusión tardía para la **Subtarea 2.2 (Confianza Clínica)** bajo el protocolo oficial MCCV-30 (`data/processed/task2/mccv_splits_30.csv`).

---

## 2. Componentes y Código a Crear

- **Script principal**: `experiments/exp_18/scripts/run_exp18_confidence_mccv30.py`
  1. Carga de datos de `data/processed/task2/` (`clinical_tabular_task2.csv`, `mri_embeddings_task2.csv`, `clinical_texts_task2.csv`, `ground_truth_task2.csv`).
  2. Ajuste fold-isolated de preprocesadores (`MinMaxScaler` y `TfidfVectorizer`).
  3. Exploración de hiperparámetros de Fuzzy-KNN por modalidad sobre soft-targets ordinales (`clear=1.0`, `borderline=0.5`, `uncertain=0.25`).
  4. Optimización de pesos de fusión en el simplex $\Delta^2$ (paso 0.05, 231 ponderaciones) maximizando $S_{\text{conf}}$ medio en validación.
  5. Evaluación exclusiva mediante `chimera_metrics.task2.reasoning.compute_confidence_score`.
- **Artefactos de salida**:
  - `experiments/exp_18/results/version.txt`
  - `experiments/exp_18/results/best_hyperparams.json`
  - `experiments/exp_18/results/best_fusion_weights.json`
  - `experiments/exp_18/results/split_metrics.json`
  - `experiments/exp_18/results/aggregate_metrics.json`
  - `experiments/exp_18/reports/summary.md`
