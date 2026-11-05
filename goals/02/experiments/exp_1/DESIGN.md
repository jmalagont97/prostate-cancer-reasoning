# Experiment Design: exp_1 — Transformación Tabular Compartimentada y EDA para CHIMERA Task 1

**Experiment**: `experiments/exp_1/`  
**Project**: CHIMERA-Agent Phase 2 / Goal 02  
**Date**: 2026-11-05  
**Author**: Co-Investigador (Experto en IA Agéntica)  
**Status**: Complete  

---

## 1. Hypothesis & Research Questions

### 1.1 Falsifiable Hypothesis
La transformación estructurada y compartimentada de los registros JSON no estructurados de Task 1 en dos archivos tabulares normalizados (`inputs.csv` prefijado por secciones del EHR y `outputs.csv` con decisión y razonamiento completo) recuperará con $\ge 98\%$ de completitud las variables clínicas canónicas (PI-RADS, volumen, PSAD, PSA, antecedentes de biopsia, DRE y probabilidad de deep learning), permitiendo aislar empíricamente las reglas clínicas deterministas y las excepciones históricas que explican la decisión binaria de biopsia (`"yes"` vs `"no"`).

### 1.2 Research Questions
1. **Completitud e Integridad:** ¿Es posible mapear de forma determinista el 100% de las variables clínicas de los 195 casos (91 con Ground Truth de decisión y razonamiento, 104 de prueba ciega) asignando a cada columna el prefijo canónico de su herramienta de origen?
2. **Distribución y Discrepancias Clínicas:** ¿Cuáles son los puntos de corte fisiopatológicos y las excepciones lógicas (ej. casos con PI-RADS 5 pero decisión `"no"`, o PI-RADS 2-3 con decisión `"yes"`) presentes en los datos reales?
3. **Mapeo de Herramientas y Razonamiento:** ¿Qué patrones de co-ocurrencia existen en `reveal_sequence` y cómo se relacionan con los 10 pesos de variables asignados por los especialistas?

---

## 2. Experimental Setup

- **Dataset de Origen:**
  - Ubicación: `../../data/raw/latests/task1/` (195 pacientes en formato JSON: `structured-prompt.json`, `prostate-biopsy-decision-clinical-data.json`, `prostate-biopsy-decision.json`, `prostate-biopsy-decision-reasoning.json`).
- **Datasets de Destino:**
  - `../../data/inputs.csv` (o `./data/inputs.csv` a nivel de goal): 195 filas con atributos clínicos prefijados por herramienta (`radiology_report__*`, `psa_trend__*`, `laboratory_results__*`, `previous_notes__*`, `clinical_data__*`, `structured_prompt__*`).
  - `../../data/outputs.csv` (o `./data/outputs.csv` a nivel de goal): 91 filas con Ground Truth completo (`case_id`, `biopsy_decision`, `confidence`, matriz de 10 `weight__*`, `reveal_sequence`, `free_text`).
- **Entorno de Ejecución:**
  - Conda Environment: `ml-harness`.
  - Cómputo: Local CPU (ejecución determinista de parsing y análisis estadístico en Pandas / NumPy / Scipy / Seaborn).

---

## 3. File Layout for This Experiment

```
goals/02/experiments/exp_1/
├── DESIGN.md                  ← Este archivo (diseño de investigación y EDA)
├── IMPLEMENTATION.md          ← Plan de implementación (scripts de parsing y notebooks/scripts de EDA)
├── scripts/
│   ├── parse_task1_tabular.py ← Extracción y validación de esquema tabular
│   └── run_task1_eda.py       ← Análisis exploratorio univariado, bivariado y multivariado
├── results/
│   ├── dataset_audit.json     ← Métricas de completitud, conteo de nulos y verificación de tipos
│   ├── statistical_summary.json ← Medidas de tendencia central y dispersión por clase
│   └── version.txt            ← Commit hash y metadata del estado del código
└── reports/
    ├── figures/               ← Histogramas, distribuciones, correlaciones y co-ocurrencias
    └── summary.md             ← Reporte final consolidado del experimento
```

---

## 4. Proposed Conditions & Experimental Steps

| Condición | Descripción | Insumo Principal | Artefacto de Salida |
|---|---|---|---|
| **C1: Ingestion & Verification** | Parseo exhaustivo de los 195 casos, extracción con regex robustos de PI-RADS, volumen y PSAD, y estructuración de `inputs.csv` y `outputs.csv`. | `data/raw/latests/task1/` | `goals/02/data/inputs.csv`, `goals/02/data/outputs.csv`, `results/dataset_audit.json` |
| **C2: Univariate & Bivariate EDA** | Análisis de distribución de cada variable clínica frente a la etiqueta `biopsy_decision` (`yes` vs `no`). Identificación de valores atípicos y balance de clases. | `inputs.csv`, `outputs.csv` | `reports/figures/distributions_by_class.png`, `results/statistical_summary.json` |
| **C3: Clinical Exceptions & Discrepancies** | Auditoría focalizada de casos de discordancia (ej. PI-RADS $\ge 4$ con decisión `"no"`, o PI-RADS $\le 2$ con decisión `"yes"`) cruzados con antecedentes (`bx = Positive`) y notas previas. | `inputs.csv`, `outputs.csv` | `reports/figures/confusion_pirads_vs_decision.png`, tabla de casos discordantes |
| **C4: Reasoning & Tool Usage Profiling** | Análisis de frecuencias de `reveal_sequence`, co-ocurrencias de apertura de herramientas y correlación entre pesos asignados (`decisive`, `important`, `noted`, `not_used`) y la decisión. | `outputs.csv` | `reports/figures/tool_usage_frequency.png`, `reports/figures/weights_distribution.png` |

---

## 5. Evaluation Protocol & Acceptance Criteria

### 5.1 Criterios de Aceptación Cuantitativos
1. **Completitud del Dataset Tabular:**
   - $\text{Completitud de } inputs.csv \ge 98\%$ en variables clínicas clave (`pirads`, `prostate_volume`, `psa_density`, `cspca`, `age`, `bx`).
   - Cero valores nulos en `outputs.csv` para los 91 casos con ground truth en `case_id`, `biopsy_decision`, `confidence` y las 10 columnas de `weight__*`.
2. **Consistencia de Tipos y Rangos:**
   - `radiology_report__pirads` estrictamente entero en $\{1, 2, 3, 4, 5\}$.
   - `clinical_data__cspca` flotante acotado en $[0.0, 1.0]$.
   - `biopsy_decision` estrictamente categórico binario en `{"yes", "no"}`.
3. **Mapeo de Discordancias:**
   - Identificación del 100% de los casos donde $PIRADS \ge 4$ con decisión `"no"` y caracterización del mecanismo de exclusión clínica subyacente.

---

## 6. Expected Results & Decision Rules

- **Si la hipótesis se confirma ($\ge 98\%$ completitud y explicación clara de discordancias):**
  - Se formaliza `inputs.csv` y `outputs.csv` como el benchmark canónico para Goal 02.
  - Se procede inmediatamente al **Experimento 2 (`exp_2`)**: Entrenamiento e inducción del árbol neurosimbólico agéntico (Triada Blackboard) y benchmarking de baselines frente a CART clásico sobre 5-Fold Stratified Cross-Validation.
- **Si se detectan sesgos o anomalías en la extracción ($< 98\%$ o inconsistencia en regex):**
  - Se ajustan los analizadores textuales o se incorpora imputación determinista fundamentada en la relación urológica canónica ($PSAD = PSA / Volume$).

---

## 7. Risks & Mitigations

1. **Variabilidad textual en reportes radiológicos:**
   - *Riesgo:* Distintos hospitales usan formatos heterogéneos (ej. comillas `PI-RADS "5"`, dos puntos `PI-RADS: 4`, o minúsculas `prostate volume: 32 mL`).
   - *Mitigación:* Expresiones regulares compuestas multi-patrón con respaldo de búsqueda de proximidad y normalización previa a minúsculas.
2. **Valores ausentes en PSAD explícito:**
   - *Riesgo:* Algunos informes omiten el valor precalculado de PSA density.
   - *Mitigación:* Si `radiology_report__psa_density` no aparece en texto, se computa algebraicamente a partir de `psa_trend__latest_psa / radiology_report__prostate_volume`.
3. **Fuga de datos (Data Leakage):**
   - *Riesgo:* Incluir información de razonamiento o etiquetas en `inputs.csv`.
   - *Mitigación:* Estricta separación física: `inputs.csv` solo recibe datos clínicos diagnósticos pre-decisión; `outputs.csv` aloja exclusivamente las decisiones y justificaciones de Ground Truth.

---

## 8. Reproducibility Checklist

- [ ] Semilla aleatoria fijada para visualizaciones y particiones reproducibles (`seed=42`).
- [ ] Versión del código y estado de Git guardado en `results/version.txt`.
- [ ] Esquema de columnas de `inputs.csv` y `outputs.csv` congelado y documentado con tipos explícitos.
- [ ] Registro de salida unificado en `results/dataset_audit.json`.
- [ ] Reporte final estructurado generado en `reports/summary.md` mediante `experiment-reporter`.

---

## 9. Next Steps

1. Revisar y ratificar este diseño experimental (`exp_1/DESIGN.md`).
2. Elaborar el plan de implementación (`exp_1/IMPLEMENTATION.md`) detallando los scripts `parse_task1_tabular.py` y `run_task1_eda.py`.
3. Ejecutar la transformación de datos y el pipeline de análisis exploratorio para producir los artefactos y el reporte de síntesis.
