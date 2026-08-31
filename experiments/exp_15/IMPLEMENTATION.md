# Implementation Plan: MCCV-30 estratificado multiobjetivo para Task 2
**Experiment**: `experiments/exp_15/`  
**Project**: `prostate-cancer-reasoning`  
**Date**: 2026-08-30  
**Status**: In Progress

---

## 1. Resumen y Objetivos

Este plan implementa la infraestructura experimental definida en `experiments/exp_15/DESIGN.md`:
1. Extracción e integración canónica de las 11 etiquetas de relevancia de variables clínicas (`weight_<var>`) en `data/processed/ground_truth.csv` y `data/processed/task2/ground_truth_task2.csv`.
2. Generación del artefacto de validación cruzada `data/processed/task2/mccv_splits_30.csv` con 30 particiones deterministas optimizadas mediante estratificación multiobjetivo conjunta (decisión de 4 clases, confianza ordinal de 3 niveles y distribución multivariada de los pesos de las 11 variables).
3. Script de verificación y auditoría matemática de particiones, fuga de datos y balance distributivo utilizando exclusivamente `utils/pkg/chimera_metrics`.

---

## 2. Archivos e Infraestructura

```text
experiments/exp_15/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── extract_ground_truth_weights.py
│   ├── generate_mccv_splits_task2.py
│   └── audit_splits_task2.py
├── results/
│   ├── splits_diagnostics.json
│   └── version.txt
└── reports/
```

---

## 3. Especificación Técnica de Componentes

### 3.1 `extract_ground_truth_weights.py`
- Lee los 70 archivos `prostate-treatment-decision-reasoning.json` correspondientes a los casos válidos de `data/processed/input.csv`.
- Extrae el diccionario `variable_weights` compuesto por exactamente 11 variables:
  `['age', 'bx_gl_prim', 'bx_gl_sec', 'bx_isup', 'comorbidity', 'cspca', 'ct', 'fh', 'pirads', 'psa', 'psad']`.
- Actualiza `data/processed/ground_truth.csv` y `data/processed/task2/ground_truth_task2.csv` preservando `pid`, `treatment_decision`, `confidence` y añadiendo las 11 columnas con prefijo `weight_`.

### 3.2 `generate_mccv_splits_task2.py`
- Lee `ground_truth.csv` (70 casos).
- Fija la proporción train/val en 80/20 (56 entrenamiento / 14 validación por partición).
- Estratificación primaria por decisión (exacta en proporciones para las 4 clases: `active_treatment: 6/25`, `active_surveillance: 5/20`, `continued_surveillance: 2/10`, `watchful_waiting: 1/1` en val/train para máxima preservación).
- Búsqueda determinista por semilla que minimiza la función de pérdida conjunta:
  $$\mathcal{L} = \text{EMD}_{\text{confianza}} + \frac{1}{11} \sum_{v=1}^{11} \text{EMD}_{\text{peso}_v}$$
  donde $\text{EMD}$ es la distancia Earth Mover's Distance sobre distribuciones ordinales discretas.
- Guarda `data/processed/task2/mccv_splits_30.csv`.

### 3.3 `audit_splits_task2.py`
- Valida dimensiones, ausencia de NaNs, ortogonalidad train/val en cada partición y métricas distributivas vía `chimera_metrics`.
- Genera `results/splits_diagnostics.json`.
