# Experiment Design: Líneas Base Heurísticas Clínicas Deterministas (Task 1.1)

**Experiment**: `experiments/exp_5/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Complete  

---

## 1. Hypothesis
Las reglas heurísticas urológicas deterministas estándar exhiben un compromiso rígido insuperable entre sensibilidad y especificidad: la regla naïve (H0: $PIRADS \ge 4 \lor PSAD \ge 0.15$) alcanza alta sensibilidad ($\ge 95\%$) pero colapsa en especificidad ($\le 20\%$) debido a sobrebiopsias fútiles en pacientes con cáncer ya confirmado, mientras que la regla con filtro binario estático de antecedentes tumorales (H1/H2) eleva la especificidad ($>80\%$) a costa de un colapso severo en sensibilidad ($\le 62\%$) por omitir casos en vigilancia activa o progresión, resultando ambas significativamente inferiores en balance diagnóstico global ($\text{Macro-}F1 \le 0.70$, $\text{Accuracy} \le 70\%$) al árbol de decisión neurosimbólico de `exp_3` ($\text{Accuracy} = 82.61\%$, $\text{Macro-}F1 = 0.8175$, $\text{Sensibilidad} = 85.71\%$, $\text{Especificidad} = 77.78\%$) en la partición out-of-sample congelada de Test ($N=46$).

---

## 2. Experimental Setup
- **Dataset Canónico y Partición Congelada**:
  - `../../data/inputs.csv`: 195 casos clínicos con 21 covariables.
  - `../../data/outputs.csv`: 91 casos con etiquetas Ground Truth (decisión binaria `yes`/`no`).
  - `../../data/experimental-design.csv`: Asignación oficial inmutable y balanceada ($SMD = 0.0275$):
    * **Partición de Inducción (Train)**: $N=45$ casos (28 `yes`, 17 `no`).
    * **Partición de Evaluación Ciega (Test)**: $N=46$ casos (28 `yes`, 18 `no`).
    * **Casos No Anotados**: 104 casos (estrictamente aislados).
- **Condiciones Heurísticas Evaluadas (Modelos Deterministas)**:
  1. **Condición H0 (Heurística Naïve Estándar - Guía PI-RADS/EAU Básica)**:
     $$\text{Biopsia} = \text{'yes'} \iff (PIRADS \ge 4) \lor (PSAD \ge 0.15)$$
     Representa el protocolo estándar de sospecha radiológica y bioquímica sin cruce de antecedentes oncológicos.
  2. **Condición H1 (Heurística Filtrada por Antecedente Histológico Previo)**:
     $$\text{Biopsia} = \text{'yes'} \iff [(PIRADS \ge 4) \lor (PSAD \ge 0.15)] \land (bx \ne \text{'Positive'} \land prior\_cancer \ne 1)$$
     Modela el criterio clínico conservador donde pacientes con neoplasia confirmada no son re-biopsiados diagnósticamente.
  3. **Condición H2 (Guía Clínica Estratificada EAU Formal)**:
     - Si $PIRADS \le 2 \implies \text{'no'}$.
     - Si $PIRADS = 3 \implies \text{'yes'} \iff PSAD \ge 0.15$, de lo contrario $\text{'no'}$.
     - Si $PIRADS \ge 4 \implies \text{'no'} \iff (bx == \text{'Positive'} \lor prior\_cancer == 1)$, de lo contrario $\text{'yes'}$.
- **Entorno de Ejecución**:
  - Entorno Conda: `ml-harness`.
  - Ejecución en primer plano (<1 segundo de cómputo determinista).

---

## 3. File Layout for This Experiment
```
goals/02/experiments/exp_5/
├── DESIGN.md                          ← Este documento (diseño experimental)
├── IMPLEMENTATION.md                  ← Plan de implementación técnica (scripts y guardado de artefactos)
├── scripts/
│   └── run_heuristics_baseline.py     ← Script de evaluación determinista de las 3 heurísticas en Train y Test
├── results/
│   ├── heuristics_train_metrics.json  ← Métricas detalladas de H0, H1 y H2 en Train (N=45)
│   ├── heuristics_test_metrics.json   ← Métricas detalladas de H0, H1 y H2 en Test (N=46)
│   └── version.txt                    ← Git SHA, fecha y metadatos de reproducibilidad
└── reports/
    ├── figures/
    │   ├── heuristics_comparison_test.png    ← Comparativa de barras: H0 vs H1 vs H2 en Test
    │   ├── confusion_matrices_heuristics.png ← Matrices de confusión de las 3 heurísticas
    │   └── triangular_benchmark_tradeoff.png ← Mapa de dispersión Sensibilidad vs Especificidad (Heurísticas vs CART vs Neurosimbólico)
    └── summary.md                     ← Reporte consolidado de hallazgos
```

---

## 4. Baselines & Contrasting Paradigms

Este experimento permite completar la **tríada comparativa metodológica** del proyecto en Task 1.1:

| Paradigma Metodológico | Experimento | Modelo Evaluado | Mecanismo de Decisión | Limitación Teórica |
| :--- | :--- | :--- | :--- | :--- |
| **Heurística Clínica Naïve** | `exp_5` (H0) | $PIRADS \ge 4 \lor PSAD \ge 0.15$ | Regla médica fija no condicionada | Colapso de Especificidad ($\le 20\%$) |
| **Heurística Clínica Filtrada** | `exp_5` (H1/H2) | Regla urológica con filtro binario | Regla médica fija con descarte estático | Colapso de Sensibilidad ($\le 62\%$) |
| **Machine Learning Clásico** | `exp_4` | CART Voraz (`max_depth=None`) | Minimización estadística de Gini | Sobreajuste y fragmentación espuria |
| **IA Agéntica Neurosimbólica** | `exp_3` | Árbol inducido por LLM + RAG EAU | Razonamiento clínico contextualizado | Requiere inducción iterativa |

---

## 5. Proposed Conditions

- **Condición H0**: Regla Naïve estándar ($PIRADS \ge 4 \lor PSAD \ge 0.15$).
- **Condición H1**: Regla con filtro de biopsia previa negativa/ausente.
- **Condición H2**: Guía EAU estratificada por niveles de PI-RADS.

---

## 6. Evaluation Protocol & Metrics

Las tres heurísticas se evaluarán estrictamente sobre la partición congelada out-of-sample **Test ($N=46$)** y sobre **Train ($N=45$)**:

1. **Accuracy ($\text{Acc}$)**:
   $$\text{Accuracy} = \frac{TP + TN}{N}$$
2. **Macro-F1**:
   $$\text{Macro-}F1 = \frac{F1_{\text{yes}} + F1_{\text{no}}}{2}$$
3. **$F1_{\text{yes}}$ y $F1_{\text{no}}$**:
   Desempeño desglosado para clase de biopsia y diferimiento.
4. **Sensibilidad (TPR)**:
   $$\text{TPR} = \frac{TP}{TP + FN}$$
5. **Especificidad (TNR)**:
   $$\text{TNR} = \frac{TN}{TN + FP}$$
6. **Balanced Accuracy**:
   $$\text{Balanced Acc} = \frac{\text{TPR} + \text{TNR}}{2}$$
   (Equivalente exacto del área bajo la curva ROC para clasificadores binarios deterministas rígidos sin umbral variable).
7. **Matriz de Confusión ($TN, FP, FN, TP$)**.

---

## 7. Expected Results & Decision Rules

### Criterios de Confirmación de la Hipótesis:
- **Confirmación de Compromiso Rígido (Success)**:
  - H0 supera el 90% en Sensibilidad pero colapsa en Especificidad ($\le 25\%$), incurriendo en más de 12 sobrebiopsias falsas positivas en Test.
  - H1 y H2 logran Especificidad $\ge 80\%$, pero su Sensibilidad colapsa a $\le 65\%$, omitiendo más de 10 casos que requerían biopsia diagnóstica (falsos negativos).
  - Ninguna heurística supera el **70% de Accuracy** ni **0.70 de Macro-F1** en Test.
  - El árbol neurosimbólico de `exp_3` supera a todas las heurísticas en al menos **+12 puntos de Accuracy** ($82.61\%$ vs $\le 70\%$) y **+12 puntos de Macro-F1** ($0.8175$ vs $\le 0.70$), demostrando la necesidad de ramificación contextualizada en pacientes con cáncer previo.
- **Refutación de la Hipótesis (Alternative Outcome)**:
  - Si alguna heurística estática simple logra simultáneamente $\text{Sensibilidad} \ge 80\%$ y $\text{Especificidad} \ge 75\%$, se refutaría la necesidad de inducción agéntica para la decisión de biopsia.

---

## 8. Risks & Mitigations

1. **Riesgo de Inconsistencias Semánticas en Cáncer Previo**:
   - *Mitigación*: Considerar tanto la variable histológica estructurada `structured_prompt__bx == 'Positive'` como la variable clínica `previous_notes__prior_cancer_confirmed == 1` para asegurar que el filtro capture todos los pacientes con diagnóstico tumoral previo.
2. **Riesgo de Comparaciones Injustas por Probabilidades Continuas**:
   - *Mitigación*: Las heurísticas son deterministas (salidas binarias 0/1); el contraste se evalúa rigurosamente sobre métricas fijas de clasificación (Accuracy, Macro-F1, Balanced Accuracy, Sensibilidad y Especificidad).

---

## 9. Reproducibility Checklist
- [ ] Partición Train/Test inmutable de `../../data/experimental-design.csv`.
- [ ] Lógica heurística explícita y transparente sin hiperparámetros ocultos.
- [ ] Métricas guardadas en JSON en `results/`.
- [ ] Hash de commit registrado en `results/version.txt`.
- [ ] Todas las rutas relativas.

---

## 10. Next Steps
1. Revisión y aprobación del presente `DESIGN.md`.
2. Generación del plan de implementación técnica (`IMPLEMENTATION.md`).
3. Construcción y ejecución de `scripts/run_heuristics_baseline.py`.
4. Consolidación de resultados y contraste tripartito en `reports/summary.md` y `.logbook.md`.
