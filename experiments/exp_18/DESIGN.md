# Experiment Design: Búsqueda Sistemática de Hiperparámetros y Fusión para Confianza Clínica (Subtarea 2.2) bajo MCCV-30
**Experiment**: `experiments/exp_18/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Task 2)  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Draft  

---

## 1. Hypothesis

La optimización sistemática de hiperparámetros unimodales ($k$, ponderación por distancia y vocabulario TF-IDF) junto con la búsqueda en el simplex de fusión $\Delta^2$ para un modelo **Fuzzy-KNN Ordinal Trimodal**, evaluado rigurosamente sobre las 30 particiones de **MCCV-30** (`data/processed/task2/mccv_splits_30.csv`) mediante la métrica canónica `chimera_metrics.task2.reasoning.compute_confidence_score`, alcanzará una distancia ordinal normalizada media $S_{\text{conf}} \ge 0.85$ y un Macro-F1 $\ge 0.40$ (a pesar del desbalance estructural donde la clase `uncertain` tiene $n=0$ casos en el Ground Truth de Task 2).

---

## 2. Experimental Setup

- **Datos de Entrada** (`data/processed/task2/`):
  - `clinical_tabular_task2.csv` ($70 \times 33$)
  - `mri_embeddings_task2.csv` ($70 \times 1025$)
  - `clinical_texts_task2.csv` ($70 \times 2$, columna `full_encounter_narrative`)
  - `ground_truth_task2.csv` ($70 \times 14$, columna `confidence`)
- **Particiones de Validación**:
  - Matriz oficial `data/processed/task2/mccv_splits_30.csv` (30 particiones estratificadas multi-subtarea).
- **Target Ordinal Soft-Target Formulation**:
  - $\text{clear} \to 1.0$
  - $\text{borderline} \to 0.5$
  - $\text{uncertain} \to 0.25$
- **Control Estricto de Fuga**:
  - `MinMaxScaler` (Tabular) y `TfidfVectorizer` (Texto) se ajustan exclusivamente en los folds de entrenamiento de cada partición.

---

## 3. Espacio de Búsqueda de Hiperparámetros (Replicando `exp_12`)

### Fase 1: Grillas Unimodales
- **Tabular (Fuzzy-KNN)**:
  - $k_{\text{tab}} \in \{1, 3, 5, 7\}$
  - `weights` $\in \{\text{'uniform'}, \text{'distance'}\}$
- **MRI (Fuzzy-KNN)**:
  - $k_{\text{mri}} \in \{1, 3, 5\}$
  - `weights` $\in \{\text{'uniform'}, \text{'distance'}\}$
- **Texto (TF-IDF + Fuzzy-KNN)**:
  - $k_{\text{txt}} \in \{1, 3, 5\}$
  - `max_features` $\in \{500, \text{None}\}$ (vocabulario completo)
  - `weights` $\in \{\text{'uniform'}, \text{'distance'}\}$

### Fase 2: Simplex de Fusión Tardía $\Delta^2$
- Exploración de la mezcla de predicciones continuas:
  $$\hat{y}_{\text{ens}} = w_{\text{tab}} \hat{y}_{\text{tab}} + w_{\text{mri}} \hat{y}_{\text{mri}} + w_{\text{txt}} \hat{y}_{\text{txt}}$$
  con $\mathbf{w} \in \Delta^2$, $\sum w_i = 1$, $w_i \ge 0$, paso $0.05$ (231 combinaciones).
- Mapeo de decisión ordinal por umbrales calibrados:
  $$\hat{c} = \begin{cases} \text{'uncertain'} & \text{si } \hat{y} < 0.375 \\ \text{'borderline'} & \text{si } 0.375 \le \hat{y} < 0.75 \\ \text{'clear'} & \text{si } \hat{y} \ge 0.75 \end{cases}$$

---

## 4. Condiciones Experimentales

| ID | Condición | Estimador | Búsqueda Modalidades | Optimización de Fusión |
|:---|:---|:---|:---|:---|
| **C1** | **Baseline Estático (`exp_12`)** | Fuzzy-KNN | $k_{\text{tab}}=1, k_{\text{mri}}=1, k_{\text{txt}}=3$ | Pesos fijos $w=(0.00, 0.55, 0.45)$ |
| **C2** | **Fuzzy-KNN Tuned MCCV-30** | Fuzzy-KNN | Búsqueda en grilla por modalidad | Simplex $\mathbf{w} \in \Delta^2$ (paso 0.05) maximizando $S_{\text{conf}}$ |

---

## 5. Gobernanza y Protocolo de Evaluación

Queda terminantemente prohibido calcular métricas ad-hoc de confianza en el código del experimento. Se consumirá directamente:
- `chimera_metrics.task2.reasoning.compute_confidence_score(gt_dict, pred_dict)`
- Métricas complementarias estándar: Macro-F1 (3 clases), Micro-F1 / Accuracy, F1 por clase (`clear`, `borderline`, `uncertain`).

---

## 6. Estructura de Artefactos

```text
experiments/exp_18/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── run_exp18_confidence_mccv30.py
├── results/
│   ├── version.txt
│   ├── best_hyperparams.json
│   ├── best_fusion_weights.json
│   ├── split_metrics.json
│   └── aggregate_metrics.json
└── reports/
    └── summary.md
```
