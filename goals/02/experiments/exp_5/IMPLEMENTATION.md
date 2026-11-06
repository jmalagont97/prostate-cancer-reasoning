# Implementation Plan: Líneas Base Heurísticas Clínicas Deterministas (Task 1.1)

**Experiment**: `experiments/exp_5/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Ready for Approval  

---

## 1. Overview & Objectives

Este plan detalla la implementación técnica para ejecutar **`exp_5`** conforme a lo establecido en [`DESIGN.md`](DESIGN.md):
1. Implementar de forma determinista y transparente las 3 reglas heurísticas urológicas clásicas empleadas en la práctica clínica para la recomendación de biopsia prostática:
   - **H0 (Heurística Naïve):** Criterio de sospecha estándar ($PIRADS \ge 4 \lor PSAD \ge 0.15$).
   - **H1 (Heurística Filtrada):** Criterio de sospecha condicionado a la ausencia de antecedentes neoplásicos confirmados previa ($bx \ne \text{'Positive'} \land prior\_cancer \ne 1$).
   - **H2 (Guía EAU Formal Estratificada):** Decisión trilateral basada en tramos de resonancia (PI-RADS $\le 2 \to$ No, PI-RADS $3 \to$ según PSAD $\ge 0.15$, PI-RADS $\ge 4 \to$ según antecedente de cáncer).
2. Evaluar de forma ciega y out-of-sample las 3 condiciones sobre la partición congelada Test ($N=46$) y auditar en simultáneo Train ($N=45$) a partir de `../../data/experimental-design.csv`.
3. Computar la batería de métricas estándar: Accuracy, Macro-F1, $F1_{\text{yes}}$, $F1_{\text{no}}$, Sensibilidad (TPR), Especificidad (TNR), Balanced Accuracy y Matrices de Confusión.
4. Generar la visualización diagnóstica de la **Tríada Comparativa Metodológica** (Sensibilidad vs. Especificidad enfrentando Heurísticas vs. CART de `exp_4` vs. Árbol Neurosimbólico de `exp_3`).
5. Persistir métricas estructuradas en JSON y registrar el reporte consolidado.

---

## 2. Directory & Component Structure

Toda la implementación quedará confinada en `goals/02/experiments/exp_5/`:

```
goals/02/experiments/exp_5/
├── DESIGN.md                          ← Documento de diseño experimental
├── IMPLEMENTATION.md                  ← Este archivo (plan de implementación)
├── scripts/
│   └── run_heuristics_baseline.py     ← Script autocontenido de ejecución de las 3 heurísticas
├── results/
│   ├── heuristics_train_metrics.json  ← Métricas de H0, H1 y H2 en Train (N=45)
│   ├── heuristics_test_metrics.json   ← Métricas out-of-sample en Test (N=46)
│   └── version.txt                    ← Git SHA, fecha y metadatos del entorno
└── reports/
    ├── figures/
    │   ├── heuristics_comparison_test.png    ← Comparativa de métricas en Test (H0 vs H1 vs H2)
    │   ├── confusion_matrices_heuristics.png ← Panel 1x3 con las matrices de confusión
    │   └── triangular_benchmark_tradeoff.png ← Trade-off Sensibilidad vs Especificidad (Tríada completa)
    └── summary.md                     ← Reporte consolidado de hallazgos
```

---

## 3. Detailed Component Specifications

### 3.1. Script de Evaluación Heurística: `scripts/run_heuristics_baseline.py`

El script implementará funciones deterministas puras para cada regla:

#### A. Definición de Heurísticas Clínicas
```python
def predict_h0(row: dict) -> str:
    """H0: Naive clinical suspicion (PIRADS >= 4 or PSAD >= 0.15)."""
    pirads = float(row.get("radiology_report__pirads", 0))
    psad = float(row.get("radiology_report__psa_density", 0.0))
    return "yes" if (pirads >= 4.0 or psad >= 0.15) else "no"

def predict_h1(row: dict) -> str:
    """H1: Filtered suspicion (excludes patients with confirmed prior cancer)."""
    pirads = float(row.get("radiology_report__pirads", 0))
    psad = float(row.get("radiology_report__psa_density", 0.0))
    bx = str(row.get("structured_prompt__bx", "")).strip().lower()
    prior_cancer = int(row.get("previous_notes__prior_cancer_confirmed", 0))
    has_prior_cancer = (bx == "positive" or prior_cancer == 1)
    
    suspicion = (pirads >= 4.0 or psad >= 0.15)
    return "yes" if (suspicion and not has_prior_cancer) else "no"

def predict_h2(row: dict) -> str:
    """H2: EAU stratified clinical guideline rule."""
    pirads = float(row.get("radiology_report__pirads", 0))
    psad = float(row.get("radiology_report__psa_density", 0.0))
    bx = str(row.get("structured_prompt__bx", "")).strip().lower()
    prior_cancer = int(row.get("previous_notes__prior_cancer_confirmed", 0))
    has_prior_cancer = (bx == "positive" or prior_cancer == 1)

    if pirads <= 2.0:
        return "no"
    elif pirads == 3.0:
        return "yes" if psad >= 0.15 else "no"
    else:  # pirads >= 4.0
        return "no" if has_prior_cancer else "yes"
```

#### B. Flujo de Datos
- Carga `../../data/inputs.csv`, `../../data/outputs.csv` y `../../data/experimental-design.csv`.
- Aísla estrictamente Train ($N=45$) y Test ($N=46$).
- Ejecuta las 3 funciones sobre cada cohorte.

#### C. Extracción de Métricas
- Para cada heurística y cada partición:
  - Accuracy (`accuracy_score`).
  - Macro-F1 (`f1_score(average='macro')`).
  - $F1_{\text{yes}}$ y $F1_{\text{no}}$.
  - Sensibilidad / TPR (`recall_score` pos_label='yes').
  - Especificidad / TNR (`recall_score` pos_label='no').
  - Balanced Accuracy: $(\text{TPR} + \text{TNR}) / 2$.
  - Matriz de Confusión ($TN, FP, FN, TP$).

#### D. Visualizaciones Diagnósticas
1. **`heuristics_comparison_test.png`**: Gráfico de barras agrupadas comparando Accuracy, Macro-F1, Sensibilidad y Especificidad entre H0, H1 y H2 en Test.
2. **`confusion_matrices_heuristics.png`**: Panel de 1x3 con las matrices de confusión anotadas para H0, H1 y H2.
3. **`triangular_benchmark_tradeoff.png`**: Gráfico de dispersión bidimensional (Sensibilidad en eje Y vs. Especificidad en eje X):
   - Mapeo de H0, H1, H2 (`exp_5`).
   - Mapeo de CART Clásico (`exp_4`).
   - Mapeo del Árbol Neurosimbólico (`exp_3`).
   - Ilustra visualmente la frontera de compromiso clínico y la superioridad del enfoque neurosimbólico (óptimo de Pareto en el cuadrante superior derecho).

---

## 4. Execution Workflow (Step-by-Step)

Dado que la inferencia de reglas lógicas sobre 91 casos toma menos de 1 segundo, la ejecución se realizará en primer plano:

```bash
# Ejecutar evaluación de líneas base heurísticas y generación de figuras diagnósticas
source /home/jmalagont/miniconda3/etc/profile.d/conda.sh && conda activate ml-harness && \
python goals/02/experiments/exp_5/scripts/run_heuristics_baseline.py
```

---

## 5. Verification & Acceptance Criteria

Para considerar satisfactoria la ejecución técnica y validar la hipótesis:
1. **Integridad:** 45 casos en Train y 46 en Test evaluados en las 3 condiciones sin nulos ni fallos de tipo.
2. **Confirmación de Hipótesis:**
   - H0 exhibe alta sensibilidad ($\ge 95\%$) pero especificidad colapsada ($\le 20\%$).
   - H1 y H2 logran especificidad alta ($\ge 80\%$) pero sensibilidad deficiente ($\le 62\%$).
   - Ninguna regla heurística estática supera el **70% de Accuracy** ni **0.70 de Macro-F1**.
   - El árbol neurosimbólico de `exp_3` ($Acc = 82.61\%$, $Macro\text{-}F1 = 0.8175$) supera contundentemente a todas las heurísticas clínicas tradicionales.
3. **Consolidación:** Generación de `summary.md`, actualización de `index.md` a `completed` y registro inmediato en `.logbook.md`.
