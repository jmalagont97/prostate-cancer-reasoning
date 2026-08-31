# Experiment Design: Búsqueda Sistemática de Hiperparámetros y Meta-Modelos por Variable para Relevancia Clínica (Subtarea 2.3) bajo MCCV-30
**Experiment**: `experiments/exp_19/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-30  
**Author**: Juan Sebastian Malagón Torres (PI) & AI Co-Researcher  
**Status**: Draft  

---

## 1. Hypothesis
La optimización sistemática desacoplada de meta-clasificadores de árboles de decisión (Random Forest vs. Extra-Trees) sobre un espacio de representación conjunto multimodal (Tabular 32D + MRI PCA 16D + Texto TF-IDF 50D) para cada una de las 11 variables clínicas, evaluada sobre las 30 particiones estratificadas de **MCCV-30** usando exclusivamente el paquete canónico `chimera_metrics`, superará la asignación estática no balanceada en exactitud ordinal ponderada ($S_{\text{weights}}$) y en el $F_1$ de factores decisivos e importantes ($F_{1,\text{decisive/important}}$).

---

## 2. Experimental Setup & Data Governance
- **Cohorte & Datos**:
  - `data/processed/task2/clinical_tabular_task2.csv` (covariables clínicas legítimas con exclusión estricta de variables post-biopsia).
  - `data/processed/task2/mri_embeddings_task2.csv` (1024D latentes).
  - `data/processed/task2/clinical_texts_task2.csv` (narrativas clínicas).
  - `data/processed/task2/ground_truth_task2.csv` ($N=70$ casos rotulados).
  - Ground truth multi-variable extraído directamente de `prostate-treatment-decision-reasoning.json` ($V=11$ variables: `age`, `bx_gl_prim`, `bx_gl_sec`, `bx_isup`, `comorbidity`, `cspca`, `ct`, `fh`, `pirads`, `psa`, `psad`).
- **Protocolo de Validación**:
  - `data/processed/task2/mccv_splits_30.csv` (30 particiones multi-objetivo con balance estricto).
- **Espacio de Búsqueda de Hiperparámetros por Variable**:
  - **Algoritmos**: `RandomForestClassifier` vs `ExtraTreesClassifier`.
  - **Estimadores ($n_{\text{estimators}}$)**: $\{100, 200, 300\}$.
  - **Criterio de División**: `{'gini', 'entropy'}`.
  - **Estrategia de Balance**: `{'balanced', 'balanced_subsample', None}`.
  - **Extracción de Características por Modalidad**:
    - Tabular (32D): Imputación por mediana + One-Hot Encoding + `MinMaxScaler(0, 1)`.
    - MRI (16D): `PCA(n_components=16)` ajustado estrictamente en cada split de train.
    - Texto (50D): `TfidfVectorizer(max_features=50)` ajustado estrictamente en cada train fold.

---

## 3. Condiciones Experimentales

| Condición | Descripción | Espacio de Representación | Modelo por Variable ($j \in \{1 \dots 11\}$) |
|---|---|---|---|
| **C1 (Baseline Estático `exp_12`)** | Meta-modelos fijos con $n=200$, `max_features='sqrt'`, `class_weight='balanced'` | Joint 98D (32 Tab + 16 MRI + 50 Txt) | RF vs ET predeterminado |
| **C2 (Per-Variable Tuned Ensemble)** | Búsqueda exhaustiva de hiperparámetros por variable ($j$) seleccionando la mejor combinación en CV interno de cada split | Joint 98D (32 Tab + 16 MRI + 50 Txt) | Mejor configuración $\{ \text{Alg}, n, \text{crit}, \text{weight} \}_j$ |

---

## 4. Métricas de Evaluación Oficial (`chimera_metrics`)
La evaluación **NO** contendrá código ad-hoc de métricas. Se importará y consumirá directamente de `chimera_metrics`:
1. **$S_{\text{weights}}$ (Score Ordinal Canónico MAE)**:
   $$S_{\text{weights}} = 1.0 - \frac{1}{3|V|} \sum_{v \in V} |\text{rank}(\hat{w}_v) - \text{rank}(w_v)|$$
   Vía `chimera_metrics.task1.reasoning.compute_variable_weights_score` / `chimera_metrics.task2.reasoning.compute_variable_weights_score`.
2. **Decisive & Important Factors Set-$F_1$**:
   Vía `chimera_metrics.task1.reasoning.compute_important_decisive_f1` / `chimera_metrics.task2.reasoning.compute_important_decisive_f1`.
3. **Macro-$F_1$ por Variable Clínica y Global**:
   Promedio de Macro-$F_1$ en los 4 niveles ordinales (`not_used`, `noted`, `important`, `decisive`).

---

## 5. Reglas de Decisión y Criterio de Éxito
- **Éxito**: La condición optimizada $C_2$ debe lograr un $\overline{S_{\text{weights}}} \ge 0.80$ y superar al baseline $C_1$ con significancia estadística o paridad cualitativa sin generar sobreajuste en clases raras.
- **Trazabilidad**: Guardar `results/best_hyperparams.json`, `results/split_metrics.json`, `results/aggregate_metrics.json` y `results/version.txt`.
