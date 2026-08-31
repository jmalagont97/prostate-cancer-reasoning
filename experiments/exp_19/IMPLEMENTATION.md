# Implementation Plan: Búsqueda Sistemática de Hiperparámetros y Meta-Modelos por Variable para Relevancia Clínica (exp_19)

**Experiment**: `experiments/exp_19/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Task 2)  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres (PI) & AI Co-Researcher  

---

## 1. Scope & Objective
Implementar y ejecutar el runner canónico `run_exp19_weights_mccv30.py` para la Subtarea 2.3 (Ponderación de 11 Variables Clínicas de Relevancia), evaluando un ensamble multimodal (Tabular 32D + MRI PCA 16D + Texto TF-IDF 50D) desacoplado por variable con búsqueda exhaustiva de algoritmos (`RandomForest` vs `ExtraTrees`) e hiperparámetros ($n_{\text{estimators}}$, `criterion`, `class_weight`) bajo el protocolo canónico **MCCV-30**, gobernado por `chimera_metrics`.

---

## 2. Archivos e Implementación
- **Runner Script**: `experiments/exp_19/scripts/run_exp19_weights_mccv30.py`
  - Carga de datos canónicos: `clinical_tabular_task2.csv`, `mri_embeddings_task2.csv`, `clinical_texts_task2.csv`, `ground_truth_task2.csv`.
  - Carga de ground truth de 11 variables desde `prostate-treatment-decision-reasoning.json`.
  - Extracción de características multimodales:
    - Tabular (32D): Imputación por mediana + One-Hot Encoding + `MinMaxScaler(0,1)`.
    - MRI (16D): `PCA(n_components=16)` ajustado estrictamente en train fold.
    - Texto (50D): `TfidfVectorizer(max_features=50)` ajustado estrictamente en train fold.
  - Condiciones:
    - **C1 (Baseline Estático `exp_12`)**: Meta-modelos fijos con $n=200$, `max_features='sqrt'`, `class_weight='balanced'` según selección per-variable de exp_12.
    - **C2 (Per-Variable Tuned Ensemble)**: Grid search por variable optimizando el Macro-F1 de 4 clases (`not_used`, `noted`, `important`, `decisive`).
  - Evaluación canónica con `chimera_metrics`:
    - `chimera_metrics.task1.reasoning.compute_variable_weights_score` ($S_{\text{weights}}$).
    - `chimera_metrics.task1.reasoning.compute_important_decisive_f1` (Set-$F_1$).
- **Artefactos generados**:
  - `results/best_hyperparams.json`: Configuración y algoritmo ganador por variable clínica.
  - `results/split_metrics.json`: Métricas en cada uno de los 30 splits para C1 y C2.
  - `results/aggregate_metrics.json`: Medias, desviaciones estándar y pruebas pareadas de Wilcoxon.
  - `results/version.txt`: SHA de git y metadatos de ejecución.
  - `reports/summary.md`: Informe final estructurado.
