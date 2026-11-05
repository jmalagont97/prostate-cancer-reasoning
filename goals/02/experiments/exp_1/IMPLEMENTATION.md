# Implementation Plan: exp_1 — Transformación Tabular Compartimentada y EDA Task 1

**Experiment**: `experiments/exp_1/`  
**Project**: CHIMERA-Agent Phase 2 / Goal 02  
**Date**: 2026-11-05  
**Status**: Complete  

---

## 1. Overview & Execution Workflow

Este plan implementa las cuatro condiciones de diseño definidas en [`DESIGN.md`](DESIGN.md):
1. **C1 (Ingestión y Extracción Tabular):** Script `scripts/parse_task1_tabular.py` para parsear los 195 casos crudos de `../../data/raw/latests/task1/` y generar `./data/inputs.csv` (195 filas) y `./data/outputs.csv` (91 filas), auditando la integridad en `results/dataset_audit.json`.
2. **C2, C3 y C4 (Análisis Exploratorio y Profiling de Razonamiento):** Script `scripts/run_task1_eda.py` para calcular estadísticas univariadas/bivariadas, mapear discrepancias clínicas (ej. $PIRADS=5$ con decisión `"no"`), perfilar `reveal_sequence` y exportar figuras en `reports/figures/`.

---

## 2. Archivos y Módulos a Crear

### A. Scripts del Experimento (`experiments/exp_1/scripts/`)

#### 1. `scripts/parse_task1_tabular.py`
- **Propósito:** Parseador robusto multi-regex y ensamblador de esquemas tabulares.
- **Rutas Relativas:**
  - Entrada: `../../data/raw/latests/task1/`
  - Salida: `../../data/inputs.csv` y `./data/inputs.csv`
  - Salida GT: `../../data/outputs.csv` y `./data/outputs.csv`
  - Auditoría: `results/dataset_audit.json`
- **Lógica de Extracción:**
  - Regex tolerante a formatos de informe radiológico (`PI-RADS 5`, `PI-RADS: 4`, `PI-RADS "5"`).
  - Normalización de volumen (`prostate volume: X mL`, `measures X`, `registers at X`).
  - Extracción de PSA reciente de `psa_trend` y `laboratory_results`.
  - Cálculo automático de densidad si no está en texto: $PSAD = \frac{PSA}{Volumen}$.
  - Extracción de las 10 variables oficiales para `outputs.csv` (`weight__*`).
  - Serialización de `reveal_sequence` como lista JSON.

#### 2. `scripts/run_task1_eda.py`
- **Propósito:** Pipeline estadístico y generador de gráficos de síntesis diagnóstica.
- **Rutas Relativas:**
  - Insumos: `./data/inputs.csv` y `./data/outputs.csv`
  - Métricas: `results/statistical_summary.json`
  - Figuras:
    - `reports/figures/distributions_by_class.png` (Boxplots de PSA, PSAD, csPCa por clase).
    - `reports/figures/pirads_vs_decision.png` (Matriz de contingencia PIRADS vs Biopsia).
    - `reports/figures/tool_usage_and_weights.png` (Frecuencia de secciones y mapa de calor de pesos).
  - Versionado: Guarda SHA de Git y metadata en `results/version.txt`.

---

## 3. Estructura de Salida de Datos

### Columnas de `inputs.csv` (195 filas)
```csv
case_id,structured_prompt__age,structured_prompt__dre,structured_prompt__bx,structured_prompt__ct,structured_prompt__psav,structured_prompt__ipss,structured_prompt__active_treatment_flag,radiology_report__pirads,radiology_report__prostate_volume,radiology_report__psa_density,clinical_data__cspca,clinical_data__family_history,psa_trend__latest_psa,psa_trend__earliest_psa,psa_trend__psa_delta,psa_trend__readings_count,laboratory_results__psa,laboratory_results__free_psa,laboratory_results__free_psa_percent,previous_notes__prior_cancer_confirmed,previous_notes__notes_count
```

### Columnas de `outputs.csv` (91 filas)
```csv
case_id,biopsy_decision,confidence,weight__age,weight__fh,weight__cspca,weight__pirads,weight__vol,weight__psa,weight__comorbidity,weight__psad,weight__dre,weight__bx,reveal_sequence,free_text
```

---

## 4. Comandos de Ejecución

Dado que ambos scripts son ligeros y toman $< 10$ segundos de cómputo en CPU, se ejecutan en primer plano dentro del entorno conda `ml-harness`:

```bash
# 1. Parseo y generación de tablas
cd goals/02/experiments/exp_1
python -u scripts/parse_task1_tabular.py

# 2. Análisis exploratorio y generación de reportes
python -u scripts/run_task1_eda.py
```

---

## 5. Criterios de Validación de la Implementación
1. `inputs.csv` contiene exactamente 195 filas y 0 nulos en `radiology_report__pirads`, `structured_prompt__age` y `clinical_data__cspca`.
2. `outputs.csv` contiene exactamente 91 filas sin registros vacíos en las 10 columnas de pesos ni en `biopsy_decision`.
3. `results/dataset_audit.json` y `results/statistical_summary.json` se escriben con éxito.
4. Las tres figuras analíticas se renderizan en `reports/figures/`.
