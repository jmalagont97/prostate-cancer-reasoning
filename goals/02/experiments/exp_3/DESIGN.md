# Experiment Design: Inducción y Evaluación Ciega del Árbol de Decisión Neurosimbólico Agéntico (Task 1.1)

**Experiment**: `experiments/exp_3/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Complete  

---

## 1. Hypothesis
El árbol de decisión neurosimbólico inducido mediante la tríada multiagente urológica (Oráculo clínico consultando guías EAU/ESMO vía ChromaDB, Evaluador de impureza Gini sobre la partición Train $N=45$, y Consolidador arquitectónico) gobernada por la directiva clínica RCT acordada, alcanza en la partición out-of-sample congelada de Test ($N=46$) un desempeño diagnóstico discriminante sólido caracterizado por **$AUC \ge 0.85$**, **$\text{Macro-}F1 \ge 0.80$**, **$F1_{yes} \ge 0.85$**, **$F1_{no} \ge 0.75$** y **$\text{Accuracy} \ge 85\%$**, demostrando que las reglas causales urológicas descubiertas de forma autónoma a partir de la literatura clínica generalizan de manera insesgada y robusta a pacientes no vistos sin requerir optimización voraz sobreajustada.

---

## 2. Experimental Setup
- **Dataset Canónico**:
  - `goals/02/data/inputs.csv`: 195 casos con 21 covariables clínicas completas.
  - `goals/02/data/outputs.csv`: 91 casos con etiquetas Ground Truth (decisión binaria `yes`/`no` y explicabilidad).
  - `goals/02/data/experimental-design.csv`: Asignación oficial inmutable y balanceada (SMD = 0.0275):
    * **Partición de Inducción (Train)**: $N=45$ casos (28 `yes`, 17 `no`).
    * **Partición de Evaluación Ciega (Test)**: $N=46$ casos (28 `yes`, 18 `no`).
    * **Casos No Anotados**: 104 casos (estrictamente aislados contra fuga de datos).
- **Base de Conocimiento Clínico (RAG Vectorial)**:
  - ChromaDB local persistente en `goals/02/data/chroma_db/` con colección `prostate_clinical_guidelines`.
  - Módulo de consulta urológica: `goals/02/pkg/query_clinical_literature.py` sobre guías internacionales EAU y ESMO.
- **Directiva Humana de Entrada**:
  - Esquema **RCT (Rol, Contexto con las 21 variables clínicas semánticas, Tarea centrada en severidad, valor pronóstico y riesgo invasivo)** formalizado en `.discussion.md`.
  - Libre de restricciones técnicas o hiperparámetros de ingeniería en el prompt.
- **Entorno de Ejecución**:
  - Conda Environment: `ml-harness`.
  - Ejecución desacoplada en sesión persistente de tmux.

---

## 3. File Layout for This Experiment
```
goals/02/experiments/exp_3/
├── DESIGN.md                          ← Este documento de diseño experimental exclusivo del árbol
├── IMPLEMENTATION.md                  ← Plan de implementación técnica (scripts, herramientas y runners)
├── scripts/
│   ├── run_agentic_induction.py       ← Orquestador del ciclo cerrado de inducción en Train (N=45)
│   └── evaluate_test.py               ← Script de inferencia ciega y cálculo de métricas en Test (N=46)
├── results/
│   ├── artifacts/
│   │   ├── decision_tree.json         ← Topología serializada del árbol aprendido (reglas y probabilidades)
│   │   └── model.py                   ← Rutina determinista en Python puro para inferencia offline
│   ├── blackboard.md                  ← Registro trazable del diálogo iterativo y rechazos
│   ├── test_metrics.json              ← Métricas detalladas en Test (Acc, AUC, F1s, Sensibilidad, Especificidad)
│   └── version.txt                    ← Hash git del código y estado del entorno
└── reports/
    ├── figures/
    │   ├── roc_curve.png              ← Curva ROC del árbol neurosimbólico en Test
    │   ├── confusion_matrix.png       ← Matriz de confusión detallada en Test
    │   └── decision_tree_topology.png ← Diagrama visual del árbol clínico inducido
    └── summary.md                     ← Reporte consolidado de hallazgos
```

---

## 4. Scope & Baseline Decoupling
**Exclusividad del Experimento:**
Este experimento se enfoca **exclusivamente** en la inducción, serialización determinista y evaluación out-of-sample del árbol de decisión neurosimbólico agéntico. 

Los contrastes y comparativas frente a modelos de referencia (heurísticas urológicas simples y árboles clásicos voraces CART de scikit-learn) quedan formalmente desacoplados y se reservan para un experimento posterior de benchmarking comparativo (`exp_4`), asegurando que este hito certifique primero la solidez, reproducibilidad y generalización out-of-sample del modelo neurosimbólico como entidad autónoma.

---

## 5. Metodología de Inducción Multiagente
El arnés opera en ciclo cerrado mediante tres roles especializados desacoplados en `blackboard.md`:

1. **Oráculo Clínico (`oracle_agent`)**:
   - Ciego a los registros individuales de pacientes.
   - Recibe el contexto del nodo activo (pacientes pendientes de resolver) e interroga ChromaDB (`query_clinical_literature.py`) para recuperar directrices EAU/ESMO.
   - Propone una regla urológica fundamentada: variable (de entre las 21 disponibles), operador (`<=`, `<`, `>=`, `>`, `==`), umbral y justificación clínica con cita formal.
2. **Evaluador Empírico (`evaluator_agent`)**:
   - Ciego a la literatura médica.
   - Ejecuta la herramienta de impureza `compute_gini_split.py` **estrictamente sobre los 45 casos de Train**.
   - Calcula: Gini del nodo padre, Gini de los nodos hijos, ganancia de impureza ($\Delta Gini$) y balance de muestras.
   - Emite veredicto: `ACCEPTED` si $\Delta Gini \ge 0.03$ y ninguna partición queda vacía; de lo contrario `REJECTED` con retroalimentación cuantitativa.
3. **Consolidador Arquitectónico (`consolidator_agent`)**:
   - Administra el estado global del árbol en `blackboard.md`.
   - Si la regla es aceptada, bifurca el nodo y encola las ramas hijas impuras.
   - Si es rechazada, incrementa el contador de reintentos del nodo y solicita una reformulación al Oráculo (presupuesto de $k=3$ intentos antes de aplicar parada temprana y sellar con la clase mayoritaria).
   - Al finalizar, genera `artifacts/decision_tree.json` y el código ejecutable `artifacts/model.py` (`predict_biopsy(patient: dict) -> str`).

---

## 6. Evaluation Protocol & Metrics

La evaluación del modelo entrenado se ejecutará de forma **estrictamente ciega sobre la partición out-of-sample de Test ($N=46$)** (28 `yes`, 18 `no`), sin que ninguna de estas muestras haya sido observada durante la inducción.

### Batería de Métricas de Clasificación Solicitadas:
1. **Accuracy Global ($\text{Acc}$)**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Curva ROC y AUC-ROC**:
   - Curva ROC trazando Sensibilidad ($TPR$) frente a Tasa de Falsos Positivos ($FPR = 1 - TNR$) a través de las probabilidades a priori estimadas en cada hoja terminal.
   - Área bajo la curva ($AUC$), midiendo la capacidad discriminativa global del árbol independiente del umbral operativo.
3. **F1-Score Clase Positiva ($F1_{yes}$)**:
   $$F1_{yes} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
   Mide la precisión y exhaustividad al indicar biopsias verdaderamente necesarias.
4. **F1-Score Clase Negativa ($F1_{no}$)**:
   $$F1_{no} = \frac{2 \cdot TN}{2 \cdot TN + FP + FN}$$
   Mide la precisión y exhaustividad al diferir o evitar biopsias en pacientes que no las requieren.
5. **Macro-F1 ($\text{Macro-}F1$)**:
   $$\text{Macro-}F1 = \frac{F1_{yes} + F1_{no}}{2}$$
   Métrica de síntesis balanceada no afectada por el desbalance relativo de clases.
6. **Sensibilidad / Recall Positivo ($TPR$)**:
   $$TPR = \frac{TP}{TP + FN}$$
   Evalúa la seguridad oncológica (minimizar falsos negativos de cáncer clínicamente significativo).
7. **Especificidad / Recall Negativo ($TNR$)**:
   $$TNR = \frac{TN}{TN + FP}$$
   Evalúa la protección contra procedimientos invasivos innecesarios (minimizar sobrebiopsias fútiles).

---

## 7. Expected Results & Decision Rules

### Criterios de Veredicto (Success / Failure):
- **Success (Éxito Pleno)**:
  - El árbol neurosimbólico alcanza en Test ($N=46$): **$AUC \ge 0.85$**, **$\text{Macro-}F1 \ge 0.80$**, **$F1_{yes} \ge 0.85$**, **$F1_{no} \ge 0.75$** y **$\text{Accuracy} \ge 85\%$**.
  - La topología resultante es compacta ($\le 3$ niveles de decisión) y cada regla está explícitamente justificada por directrices urológicas EAU/ESMO.
- **Partial Success (Éxito Parcial)**:
  - $0.75 \le \text{Macro-}F1 < 0.80$ o $0.80 \le AUC < 0.85$, manteniendo una alta sensibilidad ($TPR \ge 0.85$) y reglas clínicamente coherentes, señalando necesidad de afinar la zona gris de PI-RADS 3.
- **Failure (Fallo)**:
  - $\text{Macro-}F1 < 0.75$, $AUC < 0.80$ o sensibilidad $TPR < 0.80$ (omisión peligrosa de casos de cáncer), lo que requerirá una auditoría postmortem de los rechazos del Evaluador en `blackboard.md`.

---

## 8. Risks & Mitigations

1. **Riesgo de Fuga de Información (*Data Leakage*)**:
   - *Mitigación*: La herramienta `compute_gini_split.py` filtra obligatoriamente por `case_id` en `split == 'train'`. El archivo de test no se lee en ningún momento de la inducción.
2. **Riesgo de Errores de Sintaxis en `model.py`**:
   - *Mitigación*: Validación estricta con `ast.parse()` y prueba unitaria de ejecución determinista sobre un caso sintético antes de exportar el script final.
3. **Dependencias Externas en Inferencia**:
   - *Mitigación*: `model.py` se genera como código Python estándar puro (solo sentencias `if/elif/else` nativas), compatible con entornos desconectados de red (`--network none`).

---

## 9. Reproducibility Checklist
- [x] Semilla aleatoria fijada de forma determinista (`seed = 42`).
- [x] Partición Train/Test inmutable y documentada en `goals/02/data/experimental-design.csv`.
- [x] Base vectorial ChromaDB indexada y persistente en `goals/02/data/chroma_db/`.
- [x] Prompt de inducción estandarizado en formato RCT sin contaminación técnica.
- [x] Traza completa del diálogo multiagente registrada en `results/blackboard.md`.
- [x] Hash de commit y versión del software registrados en `results/version.txt`.

---

## 10. Next Steps
1. Revisión y aprobación formal del presente `DESIGN.md` corregido.
2. Elaboración del plan de implementación técnica (`IMPLEMENTATION.md`).
3. Construcción y verificación de los scripts ejecutables en `experiments/exp_3/scripts/`.
4. Ejecución del entrenamiento agéntico en sesión desacoplada de `tmux`.
