# Implementation Plan: Línea Base CART Clásico sin Restricciones (Task 1.1)

**Experiment**: `experiments/exp_4/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Ready for Approval  

---

## 1. Overview & Objectives

Este plan detalla la implementación técnica para ejecutar **`exp_4`** conforme a lo establecido en [`DESIGN.md`](DESIGN.md):
1. Construir un pipeline reproducible y libre de fuga de datos (*data leakage*) para entrenar un árbol de decisión clásico CART voraz no podado (`DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)`).
2. Entrenar el modelo estrictamente sobre la partición Train ($N=45$) utilizando las 10 covariables canónicas de Ground Truth de CHIMERA Task 1.
3. Evaluar de forma ciega y out-of-sample sobre la partición Test ($N=46$) y auditar en simultáneo la partición Train ($N=45$) con la misma batería métrica de `exp_3` (Accuracy, ROC-AUC, Macro-F1, $F1_{\text{yes}}$, $F1_{\text{no}}$, Sensibilidad/TPR, Especificidad/TNR, Matriz de Confusión).
4. Cuantificar formalmente la brecha de sobreajuste empírico ($\Delta \text{Acc} = \text{Accuracy}_{\text{train}} - \text{Accuracy}_{\text{test}}$) y la fragmentación en variables continuas secundarias.
5. Contrastar los resultados cuantitativos frente al árbol neurosimbólico guiado por guías EAU/ESMO de `exp_3` ($AUC = 0.8571$, $\text{Macro-}F1 = 0.8175$).

---

## 2. Directory & Component Structure

Toda la implementación quedará confinada en `goals/02/experiments/exp_4/`:

```
goals/02/experiments/exp_4/
├── DESIGN.md                          ← Documento de diseño de investigación
├── IMPLEMENTATION.md                  ← Este archivo (plan de implementación)
├── scripts/
│   └── run_cart_baseline.py           ← Pipeline autocontenido: carga, preprocesamiento, ajuste, inferencia y métricas
├── results/
│   ├── artifacts/
│   │   ├── cart_model.joblib          ← Checkpoint con pipeline ajustado (OneHotEncoder + DecisionTreeClassifier)
│   │   └── tree_rules.txt             ← Representación textual jerárquica de todas las bifurcaciones y hojas
│   ├── train_metrics.json             ← Métricas en Train (N=45) para certificar memorización
│   ├── test_metrics.json              ← Métricas out-of-sample en Test (N=46)
│   └── version.txt                    ← Git SHA, fecha y especificación del entorno
└── reports/
    ├── figures/
    │   ├── roc_curve.png              ← Curva ROC en Test (escalonada por hojas puras)
    │   ├── confusion_matrix.png       ← Matriz de confusión en Test (TN, FP, FN, TP)
    │   ├── decision_tree_structure.png← Visualización gráfica completa de la topología del árbol
    │   └── train_vs_test_overfitting.png ← Gráfico comparativo de brecha Train vs Test
    └── summary.md                     ← Reporte consolidado de hallazgos
```

---

## 3. Detailed Component Specifications

### 3.1. Script de Entrenamiento y Evaluación: `scripts/run_cart_baseline.py`

El script implementará un flujo secuencial, determinista y modular:

#### A. Carga y Aislamiento de Datos
- Carga `../../data/inputs.csv`, `../../data/outputs.csv` y `../../data/experimental-design.csv`.
- Filtra estrictamente por la columna `split`:
  - `Train`: 45 casos ($28$ `yes`, $17$ `no`).
  - `Test`: 46 casos ($28$ `yes`, $18$ `no`).
- Selecciona las 10 covariables canónicas de Ground Truth:
  ```python
  FEATURES = [
      "radiology_report__pirads",
      "radiology_report__psa_density",
      "structured_prompt__bx",
      "laboratory_results__psa",
      "structured_prompt__age",
      "structured_prompt__dre",
      "clinical_data__family_history",
      "clinical_data__cspca",
      "radiology_report__prostate_volume",
      "previous_notes__prior_cancer_confirmed",
  ]
  ```

#### B. Preprocesamiento Libre de Data Leakage
- Tratamiento de valores faltantes en categóricas (`structured_prompt__bx`, `structured_prompt__dre`, `clinical_data__family_history`): imputar como `'Missing'`.
- Codificación mediante `ColumnTransformer`:
  - `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` para categóricas.
  - `passthrough` para variables numéricas continuas/ordinales.
  - El transformador se ajusta **estrictamente sobre Train** (`preprocessor.fit(X_train)`) y transforma `X_train` y `X_test` por separado.

#### C. Ajuste del Modelo CART
- Configuración estándar sin restricciones:
  ```python
  clf = DecisionTreeClassifier(
      criterion="gini",
      splitter="best",
      max_depth=None,
      min_samples_split=2,
      min_samples_leaf=1,
      random_state=42,
  )
  ```
- Ajuste: `clf.fit(X_train_trans, y_train)`.

#### D. Extracción de Métricas y Diagnósticos
- Para ambas particiones (Train y Test):
  - Accuracy (`accuracy_score`).
  - AUC-ROC (`roc_auc_score` sobre `predict_proba[:, 1]`).
  - Macro-F1 (`f1_score(..., average='macro')`).
  - $F1_{\text{yes}}$ y $F1_{\text{no}}$ (`f1_score` pos_label=1 y pos_label=0).
  - Sensibilidad / TPR (`recall_score` pos_label=1).
  - Especificidad / TNR (`recall_score` pos_label=0).
  - Matriz de confusión ($TN, FP, FN, TP$).
- Métricas topológicas:
  - Profundidad máxima (`clf.get_depth()`).
  - Total de hojas terminales (`clf.get_n_leaves()`).
  - Importancia de características (`feature_importances_`).

#### E. Generación de Figuras y Artefactos
1. `roc_curve.png`: Curva ROC en Test con comparación de la diagonal aleatoria.
2. `confusion_matrix.png`: Matriz de confusión anotada con conteos y porcentajes.
3. `decision_tree_structure.png`: Renderizado gráfico del árbol completo con `plot_tree(filled=True)`.
4. `train_vs_test_overfitting.png`: Gráfico de barras paralelas comparando las 5 métricas primarias (Acc, AUC, Macro-F1, Sensibilidad, Especificidad) entre Train y Test para ilustrar la magnitud del sobreajuste.
5. `cart_model.joblib`: Serialización del pipeline (`joblib.dump`).
6. `tree_rules.txt`: Volcado textual de las reglas con `export_text`.
7. `train_metrics.json` y `test_metrics.json`.
8. `version.txt`: Hash SHA de Git y versión del paquete `scikit-learn`.

---

## 4. Execution Workflow (Step-by-Step)

Dado que este modelo entrena y evalúa en menos de 5 segundos, se ejecutará en primer plano respetando las normas de ejecución trivial (<60s):

```bash
# 1. Ejecutar pipeline completo de entrenamiento, evaluación y generación de figuras
source /home/jmalagont/miniconda3/etc/profile.d/conda.sh && conda activate ml-harness && \
python goals/02/experiments/exp_4/scripts/run_cart_baseline.py
```

---

## 5. Verification & Acceptance Criteria

Para considerar satisfactoria la ejecución técnica y validar la hipótesis formulada en `DESIGN.md`:
1. **Verificación de Integridad:**
   - 45 casos en Train y 46 casos en Test evaluados.
   - Cero NaNs en predicciones o probabilidades.
2. **Criterios Cuantitativos:**
   - $\text{Accuracy}_{\text{train}} = 100\%$ (confirmación de memorización pura).
   - $\text{Accuracy}_{\text{test}} \le 75\%$ (brecha $\Delta \text{Acc} \ge 25\%$).
   - $\text{AUC}_{\text{test}} \le 0.72$ y $\text{Macro-}F1_{\text{test}} \le 0.72$.
   - Especificidad en Test severamente degradada frente al árbol neurosimbólico ($\le 60\%$ vs $77.78\%$).
3. **Consolidación:**
   - Generación de `reports/summary.md` documentando la comparativa frente a `exp_3`.
   - Registro en `goals/02/.logbook.md` y actualización de estado a `completed` en `goals/02/experiments/index.md`.
