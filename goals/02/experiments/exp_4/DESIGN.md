# Experiment Design: Línea Base CART Clásico sin Restricciones (Task 1.1)

**Experiment**: `experiments/exp_4/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Complete  

---

## 1. Hypothesis
Un árbol de decisión estadístico estándar entrenado mediante el algoritmo clásico CART voraz sin restricciones de profundidad ni poda (`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`) sobre las 10 covariables clínicas de Ground Truth en la partición congelada de Train ($N=45$) alcanzará memorización completa del entrenamiento ($\text{Accuracy}_{\text{train}} = 100\%$), pero exhibirá una brecha de generalización severa en la partición ciega de Test ($N=46$) caracterizada por **$\text{Accuracy}_{\text{test}} \le 75\%$**, **$AUC \le 0.72$** y **$\text{Macro-}F1 \le 0.72$**, con un colapso en la Especificidad ($\le 60\%$) debido a la fragmentación espuria en variables continuas secundarias, resultando cuantitativa y cualitativamente inferior al árbol neurosimbólico guiado por literatura urológica de `exp_3` ($AUC = 0.8571$, $\text{Macro-}F1 = 0.8175$).

---

## 2. Experimental Setup
- **Dataset Canónico y Partición Congelada**:
  - `../../data/inputs.csv`: 195 casos con 21 covariables clínicas.
  - `../../data/outputs.csv`: 91 casos con etiquetas Ground Truth (decisión binaria `yes`/`no`).
  - `../../data/experimental-design.csv`: Asignación oficial inmutable y balanceada ($SMD = 0.0275$):
    * **Partición de Entrenamiento (Train)**: $N=45$ casos (28 `yes`, 17 `no`).
    * **Partición de Evaluación Ciega (Test)**: $N=46$ casos (28 `yes`, 18 `no`).
    * **Casos No Anotados**: 104 casos (estrictamente aislados contra fuga de datos).
- **Espacio de Características (10 Covariables Canónicas de Ground Truth)**:
  1. `radiology_report__pirads` (Numérico / Ordinal: 1 a 5)
  2. `radiology_report__psa_density` (Numérico continuo: ng/ml/cc)
  3. `structured_prompt__bx` (Categórico: 'Positive', 'Negative', o 'Missing'/biopsia no realizada previa)
  4. `laboratory_results__psa` (Numérico continuo: ng/ml)
  5. `structured_prompt__age` (Numérico discreto: años)
  6. `structured_prompt__dre` (Categórico: 'Normal', 'Nodus', 'Suspicious', 'Not done')
  7. `clinical_data__family_history` (Categórico: 'No', 'Yes', 'Unknown')
  8. `clinical_data__cspca` (Numérico continuo: probabilidad [0.0 - 1.0])
  9. `radiology_report__prostate_volume` (Numérico continuo: cc)
  10. `previous_notes__prior_cancer_confirmed` (Binario: 0 ó 1)
- **Preprocesamiento Libre de Data Leakage**:
  - Imputación de faltantes en variables categóricas asignando una categoría explícita `'Missing'`.
  - Codificación de variables categóricas mediante `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` ajustado **estrictamente sobre la partición Train**.
- **Modelo y Algoritmo**:
  - `sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)`.
- **Entorno de Ejecución**:
  - Entorno Conda: `ml-harness`.
  - Ejecución en primer plano (validación determinista rápida, $<60$ segundos).

---

## 3. File Layout for This Experiment
```
goals/02/experiments/exp_4/
├── DESIGN.md                          ← Este documento (diseño experimental y contrastes)
├── IMPLEMENTATION.md                  ← Plan de implementación técnica (scripts y guardado de artefactos)
├── scripts/
│   └── run_cart_baseline.py           ← Entrenamiento en Train, inferencia ciega en Test y métricas
├── results/
│   ├── artifacts/
│   │   ├── cart_model.joblib          ← Objeto serializado del modelo y preprocesador
│   │   └── tree_rules.txt             ← Representación textual de las reglas y cortes inducidos
│   ├── train_metrics.json             ← Métricas en Train (N=45) para auditar sobreajuste
│   ├── test_metrics.json              ← Métricas en Test (N=46) para contrastar con exp_3
│   └── version.txt                    ← Hash git del código y estado del entorno
└── reports/
    ├── figures/
    │   ├── roc_curve.png              ← Curva ROC de CART en Test
    │   ├── confusion_matrix.png       ← Matriz de confusión en Test
    │   ├── decision_tree_structure.png← Visualización gráfica del árbol CART no podado
    │   └── train_vs_test_overfitting.png ← Gráfico comparativo de brecha Train vs Test
    └── summary.md                     ← Reporte consolidado de hallazgos
```

---

## 4. Baselines & Contrasts

| Modelo / Enfoque | Tipo de Aprendizaje | Restricciones de Poda | Fuente de Inducción | Objetivo Comparativo |
| :--- | :--- | :--- | :--- | :--- |
| **Condición Propuesta (exp_4)** | CART Clásico (`DecisionTreeClassifier`) | Ninguna (`max_depth=None`, hojas puras) | Optimización empírica voraz (Gini en Train $N=45$) | Cuantificar sobreajuste voraz y límite inferior out-of-sample |
| **Referencia Neurosimbólica (exp_3)** | Árbol Agéntico en 2 Fases (Oráculo + Evaluador) | Guiado por literatura urológica | Directrices clínicas EAU/ESMO (ChromaDB) + Train $N=45$ | Medir el beneficio del sesgo inductivo clínico ($AUC=0.8571$) |

---

## 5. Proposed Conditions

Se evaluará una condición experimental primaria orientada a establecer la línea base clásica:
- **Condición C0 (Unconstrained CART)**:
  - Criterio de división: Impureza de Gini.
  - Criterio de parada: Pureza total de hojas ($\text{Gini} = 0.0$) o $N < 2$.
  - Semilla fija: `random_state=42`.

---

## 6. Evaluation Protocol & Metrics

La evaluación contrastará el desempeño empírico tanto en la muestra de inducción como en la partición out-of-sample:

### Batería de Métricas:
1. **Accuracy ($\text{Acc}$)**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Área bajo la curva ROC (AUC-ROC)**:
   Calculado a partir de las probabilidades a posteriori estimadas en las hojas (`predict_proba`).
3. **Macro-F1**:
   $$\text{Macro-}F1 = \frac{F1_{\text{yes}} + F1_{\text{no}}}{2}$$
4. **$F1_{\text{yes}}$ y $F1_{\text{no}}$**:
   Métricas de balance específico para las clases positiva y negativa.
5. **Sensibilidad / Recall Positivo ($TPR$)**:
   $$\text{Sensibilidad} = \frac{TP}{TP + FN}$$
6. **Especificidad / Recall Negativo ($TNR$)**:
   $$\text{Especificidad} = \frac{TN}{TN + FP}$$
7. **Matriz de Confusión**:
   Conteo exhaustivo de $TN$, $FP$, $FN$, $TP$.
8. **Métricas de Complejidad del Árbol**:
   - Profundidad máxima alcanzada (`clf.get_depth()`).
   - Número total de hojas (`clf.get_n_leaves()`).
   - Brecha de generalización: $\Delta \text{Acc} = \text{Accuracy}_{\text{train}} - \text{Accuracy}_{\text{test}}$.

---

## 7. Expected Results & Decision Rules

### Criterios de Validación de la Hipótesis:
- **Confirmación de Hipótesis (Success)**:
  - $\text{Accuracy}_{\text{train}} = 100\%$ (memorización completa del conjunto de entrenamiento).
  - En Test ($N=46$): $\text{Accuracy}_{\text{test}} \le 75\%$, $AUC \le 0.72$ y $\text{Macro-}F1 \le 0.72$.
  - El árbol genera bifurcaciones en variables continuas secundarias (como cortes en `prostate_volume` o `age`) para separar submuestras individuales ($N=1$), confirmando fragmentación empírica voraz.
  - La brecha frente al árbol neurosimbólico de `exp_3` en Test es superior a **10 puntos porcentuales** en Accuracy y más de **0.12 puntos** en AUC.
- **Refutación de Hipótesis (Alternative Outcome)**:
  - Si el árbol CART sin podar generaliza con $AUC \ge 0.80$ y $\text{Macro-}F1 \ge 0.78$ en Test, indicaría que las 10 covariables seleccionadas poseen una separabilidad geométrica natural tan favorable que la optimización voraz es suficiente sin necesidad de guiado urológico.

---

## 8. Risks & Mitigations

1. **Riesgo de Fuga de Datos (*Data Leakage*) en Preprocesamiento**:
   - *Mitigación*: El codificador (`OneHotEncoder`) se ajusta exclusivamente con `fit(X_train)` y posteriormente transforma `transform(X_test)` de forma completamente aislada.
2. **Riesgo de Inconsistencia de Columnas Categóricas entre Folds**:
   - *Mitigación*: Se utiliza `handle_unknown='ignore'` y se verifica la alineación exacta de nombres de columnas en la matriz de diseño.
3. **Sensibilidad a Empates Numéricos**:
   - *Mitigación*: Fijación obligatoria de `random_state=42` para garantizar reproducibilidad exacta y determinista.

---

## 9. Reproducibility Checklist
- [ ] Semilla aleatoria fijada de forma determinista (`seed = 42`).
- [ ] Partición Train/Test inmutable y documentada en `../../data/experimental-design.csv`.
- [ ] Sin preprocesamiento global compartido; transformaciones ajustadas solo en Train.
- [ ] Checkpoint y reglas del modelo serializados en `results/artifacts/`.
- [ ] Hash de commit y versión registrados en `results/version.txt`.
- [ ] Todas las rutas relativas al proyecto.

---

## 10. Next Steps
1. Revisión y aprobación de este `DESIGN.md` por parte del Investigador Principal.
2. Elaboración del plan de implementación técnica (`IMPLEMENTATION.md`).
3. Construcción y ejecución del script de entrenamiento y evaluación en `experiments/exp_4/scripts/run_cart_baseline.py`.
4. Consolidación de resultados y contraste formal en `reports/summary.md` y `.logbook.md`.
