# Implementation Plan: Inducción y Evaluación Ciega del Árbol de Decisión Neurosimbólico Agéntico (Task 1.1)

**Experiment**: `experiments/exp_3/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Ready for Approval  

---

## 1. Overview & Objectives

Este plan detalla la implementación técnica para ejecutar **`exp_3`** conforme a lo estipulado en [`DESIGN.md`](DESIGN.md):
1. Adaptar el motor de inducción multiagente en ciclo cerrado (`oracle_agent`, `evaluator_agent`, `consolidator_agent`) de `task-driven-harness` al espacio clínico de Goal 02.
2. Suministrar la directiva clínica humana en formato **RCT** (Rol, Contexto de 21 variables, Tarea urológica basada en severidad, pronóstico y riesgo) formalizada en `.discussion.md`.
3. Conectar al Oráculo Clínico con la base de conocimiento vectorial persistente de guías urológicas EAU/ESMO en `goals/02/data/chroma_db/`.
4. Blindar el cálculo de impureza de Gini para que opere **exclusivamente sobre la partición Train ($N=45$)** mediante `compute_gini_split.py`.
5. Serializar el árbol inducido en artefactos canónicos (`decision_tree.json` y `model.py` en Python puro determinista).
6. Evaluar ciegamente el modelo resultante sobre la partición congelada out-of-sample **Test ($N=46$)**, computando Accuracy, curva ROC, AUC-ROC, $F1_{yes}$, $F1_{no}$, Macro-F1, Sensibilidad y Especificidad.

---

## 2. Directory & Component Structure

Toda la implementación quedará confinada en `goals/02/experiments/exp_3/`:

```
goals/02/experiments/exp_3/
├── DESIGN.md
├── IMPLEMENTATION.md                  ← Este archivo
├── scripts/
│   ├── declarations/
│   │   ├── agents/
│   │   │   ├── oracle_agent.json      ← Oráculo urológico (RAG ChromaDB, ciego a pacientes)
│   │   │   ├── evaluator_agent.json   ← Evaluador de Gini (ciego a literatura, opera en Train)
│   │   │   └── consolidator_agent.json← Árbitro de blackboard y arquitecto del árbol
│   │   ├── tools/
│   │   │   ├── query_clinical_literature.json
│   │   │   └── compute_gini_split.json
│   │   └── flows/
│   │       └── induction_flow.json    ← Grafo declarativo de interacción iterativa
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── compute_gini_split.py      ← Herramienta de cálculo Gini sobre Train (21 variables)
│   │   └── query_clinical_literature.py ← Wrapper hacia goals/02/pkg/
│   ├── run_agentic_induction.py       ← Orquestador principal de inducción en Train (N=45)
│   └── evaluate_test.py               ← Script de inferencia ciega y cálculo de métricas en Test (N=46)
├── results/
│   ├── artifacts/
│   │   ├── decision_tree.json         ← Topología serializada con nodos, reglas y probabilidades
│   │   └── model.py                   ← Rutina determinista predict_biopsy() en Python puro
│   ├── blackboard.md                  ← Traza completa de diálogo, splits y rechazos
│   ├── test_metrics.json              ← Métricas completas en Test
│   └── version.txt                    ← Git SHA, fecha y entorno de ejecución
└── reports/
    ├── figures/
    │   ├── roc_curve.png              ← Gráfico de la curva ROC con área AUC
    │   ├── confusion_matrix.png       ← Matriz de confusión en Test
    │   └── decision_tree_topology.png ← Diagrama visual jerárquico del árbol clínico
    └── summary.md                     ← Reporte final consolidado
```

---

## 3. Detailed Component Specifications

### 3.1. Herramienta Empírica: `scripts/tools/compute_gini_split.py`
- **Ubicación de datos:** Lee `goals/02/data/inputs.csv` y `goals/02/data/outputs.csv`.
- **Blindaje Anti-Leakage Mandatorio:**
  - Carga `goals/02/data/experimental-design.csv`.
  - Filtra estrictamente las filas donde `split == 'train'` ($N=45$).
  - Rechaza cualquier evaluación si se detectan `case_id` pertenecientes a `test` o `unannotated`.
- **Soporte de Atributos Mixtos (21 Variables):**
  - Numéricas continuas/ordinales (`pirads`, `psa_density`, `psa`, `age`, `psav`, `free_psa_percent`): operadores `<=`, `<`, `>=`, `>`.
  - Categóricas y booleanas (`bx`, `prior_cancer_confirmed`, `dre`, `active_treatment_flag`): operadores `==`, `!=`, `in`.
- **Cálculo Matemático:**
  - Impureza del padre: $Gini_{parent} = 1 - (p_{yes}^2 + p_{no}^2)$.
  - Impureza de hijos: $Gini_{left}$ y $Gini_{right}$.
  - Ganancia: $\Delta Gini = Gini_{parent} - \left(\frac{N_L}{N} Gini_L + \frac{N_R}{N} Gini_R\right)$.
- **Regla de Validación:**
  - `REJECTED`: Si $N_L = 0$ o $N_R = 0$ (partición vacía), o si $\Delta Gini < 0.03$. Provee retroalimentación detallada con rangos en Train y distribución resultante.
  - `ACCEPTED`: Si $\Delta Gini \ge 0.03$ y ambas particiones tienen $\ge 1$ muestra (preferente $\ge 3$ para hojas terminales).

### 3.2. Declaraciones de Agentes (`scripts/declarations/agents/`)
- **`oracle_agent.json`**:
  - Rol: Especialista en Urología Oncológica guiado por literatura EAU/ESMO.
  - Herramienta asignada: `query_clinical_literature` (conecta al ChromaDB local).
  - Ceguera: Estricta prohibición de acceder a los datos de pacientes.
  - Contrato de salida: JSON con `target_leaf_id`, `feature_expression`, `operator`, `threshold`, `target_class_if_true`, `literature_citation`, `clinical_rationale` y `markdown_summary`.
- **`evaluator_agent.json`**:
  - Rol: Evaluador empírico cuantitativo de impureza en Train.
  - Herramienta asignada: `compute_gini_split`.
  - Ceguera: Estricta prohibición de consultar literatura médica.
  - Contrato de salida: JSON con `decision` (`ACCEPTED`/`REJECTED`), `delta_gini`, `samples_left`, `samples_right` y `feedback`.
- **`consolidator_agent.json`**:
  - Rol: Árbitro de `blackboard.md` y arquitecto del árbol.
  - Lógica: Encola nodos impuros; administra el presupuesto de reintentos ($k=3$ por nodo); si se agotan reintentos, sella el nodo como terminal con la clase mayoritaria.

### 3.3. Orquestador de Inducción: `scripts/run_agentic_induction.py`
- Lee la directiva humana RCT formalizada en `.discussion.md` e inicializa la sección `problem_spec` en `results/blackboard.md`.
- Inicializa la raíz del árbol con los 45 casos de Train ($28$ `yes`, $17$ `no`, $Gini = 0.4691$).
- Ejecuta el bucle cerrado de turnos:
  1. Identifica la hoja impura activa de mayor tamaño.
  2. Invoca al Oráculo para interrogar ChromaDB y emitir hipótesis.
  3. Invoca al Evaluador para computar `compute_gini_split`.
  4. Invoca al Consolidador para arbitrar y actualizar la estructura del árbol.
- Al converger (todas las hojas puras o detenidas por paciencia):
  - Serializa la topología a `results/artifacts/decision_tree.json`.
  - Genera `results/artifacts/model.py` con una función `predict_biopsy(row: dict) -> tuple[str, float]`:
    - Código en Python puro sin dependencias externas.
    - Valida sintaxis formal mediante `ast.parse()`.
    - Realiza una prueba unitaria sintética en memoria antes de persistir.

### 3.4. Script de Evaluación Ciega: `scripts/evaluate_test.py`
- Importa dinámicamente `results/artifacts/model.py`.
- Carga `inputs.csv` y `outputs.csv` **exclusivamente para los 46 casos de Test** definidos en `experimental-design.csv`.
- Genera predicciones binarias y probabilidades a priori por caso.
- Calcula y exporta a `results/test_metrics.json`:
  - `accuracy`: Accuracy general.
  - `auc_roc`: Área bajo la curva ROC.
  - `f1_yes`: F1 de la clase positiva (`yes`).
  - `f1_no`: F1 de la clase negativa (`no`).
  - `macro_f1`: Promedio Macro-F1.
  - `sensitivity`: Recall de la clase positiva (detección de cáncer).
  - `specificity`: Recall de la clase negativa (evitar biopsias fútiles).
  - `confusion_matrix`: Conteo $[[TN, FP], [FN, TP]]$.
- Genera visualizaciones diagnósticas en `reports/figures/`:
  - `roc_curve.png`: Curva ROC con umbrales y valor de AUC.
  - `confusion_matrix.png`: Mapa de calor de la matriz de confusión.
  - `decision_tree_topology.png`: Visualización jerárquica con reglas y proporciones de muestra.

---

## 4. Execution Workflow (Step-by-Step)

### Paso 1: Verificación de Herramientas y Datos
Validar en primer plano que `compute_gini_split.py` lee correctamente `inputs.csv` y `experimental-design.csv`, y que una llamada de prueba sobre `radiology_report__pirads <= 2` arrolle métricas válidas en Train sin tocar Test.

### Paso 2: Verificación de Agentes y RAG
Validar que `query_clinical_literature.py` responde consultas semánticas desde `goals/02/data/chroma_db/` con citas de EAU/ESMO.

### Paso 3: Ejecución de la Inducción Agéntica en tmux
Lanzar la inducción multiagente en una sesión persistente de `tmux`:
```bash
tmux new-session -d -s 02_exp_3 \
  "source /home/jmalagont/miniconda3/etc/profile.d/conda.sh && conda activate ml-harness && \
   python -u goals/02/experiments/exp_3/scripts/run_agentic_induction.py; bash"
```

### Paso 4: Evaluación Ciega y Generación de Reportes
Una vez generado `results/artifacts/model.py`, ejecutar la evaluación out-of-sample:
```bash
source /home/jmalagont/miniconda3/etc/profile.d/conda.sh && conda activate ml-harness && \
python goals/02/experiments/exp_3/scripts/evaluate_test.py
```

### Paso 5: Consolidación de Resultados
Inspeccionar `results/test_metrics.json` frente a los criterios de éxito estipulados en `DESIGN.md`, generar `reports/summary.md` y actualizar `goals/02/experiments/index.md` a `completed`.

---

## 5. Verification & Acceptance Criteria

Para dar por concluido el experimento con veredicto de **Success**, los resultados en Test ($N=46$) deben cumplir:
- $\text{AUC} \ge 0.85$
- $\text{Macro-}F1 \ge 0.80$
- $F1_{yes} \ge 0.85$
- $F1_{no} \ge 0.75$
- $\text{Accuracy} \ge 85\%$
- Cero casos de data leakage certificados.
- Artefacto `model.py` determinista y ejecutable sin conexión a red.
