# Implementation Plan: Generación de experimental-design.csv y Batería de EDA Comparativo

**Experiment**: `experiments/exp_2/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Parent Design**: [`experiments/exp_2/DESIGN.md`](DESIGN.md)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Ready for Execution  

---

## 1. Overview & High-Level Architecture

Este plan de implementación traduce formalmente las especificaciones de investigación de `DESIGN.md` en código ejecutable modular, reproducible y trazable. El objetivo principal es generar el artefacto canónico `goals/02/data/experimental-design.csv` y ejecutar la auditoría distributiva exhaustiva (EDA comparativo) que certifique la ausencia de sesgo de selección ($SMD < 0.10$ y $p > 0.05$) entre `train` y `test`.

```
goals/02/
├── data/
│   ├── inputs.csv  (195 filas)
│   ├── outputs.csv (91 filas)
│   └── experimental-design.csv      <-- [Artefacto Canónico Generado]
└── experiments/exp_2/
    ├── DESIGN.md
    ├── IMPLEMENTATION.md            <-- [Este Documento]
    ├── scripts/
    │   ├── generate_design.py       <-- [Generador y Validador de Partición]
    │   └── comparative_eda.py       <-- [EDA Comparativo, Tests y Gráficos]
    ├── results/
    │   ├── split_audit.json         <-- [Conteo y Balance por Estrato]
    │   ├── comparative_eda.json     <-- [SMDs y P-values Estadísticos]
    │   └── version.txt              <-- [Hash Git de Reproducibilidad]
    └── reports/
        └── figures/
            ├── stratum_distribution_by_fold.png
            ├── comparative_distributions_folds.png
            ├── categorical_proportions_folds.png
            └── qq_plots_folds.png
```

---

## 2. Definición del Esquema del Artefacto Objetivo (`data/experimental-design.csv`)

El archivo contendrá **195 filas exactas** (una por cada `case_id` de `inputs.csv`) con el siguiente esquema estricto:

| Nombre de Columna | Tipo de Dato | Cardinalidad / Regla | Ejemplo |
| :--- | :--- | :--- | :--- |
| `case_id` | `str` | 195 identificadores únicos sin nulos | `PT-pseudo_0020cfca66c8` |
| `split` | `str` | `'train'` ($N=45$), `'test'` ($N=46$), `'unannotated'` ($N=104$) | `'train'` |
| `fold_a` | `str` | Rol en Fase 1 CV: `'train'` ($45$), `'test'` ($46$), `'unannotated'` ($104$) | `'train'` |
| `fold_b` | `str` | Rol en Fase 2 CV: `'test'` ($45$), `'train'` ($46$), `'unannotated'` ($104$) | `'test'` |
| `status` | `str` | `'annotated'` ($N=91$), `'unannotated'` ($N=104$) | `'annotated'` |
| `is_annotated` | `bool` | `True` para los 91 anotados, `False` para los 104 sin Ground Truth | `True` |
| `stratum_id` | `str` | `stratum_0` a `stratum_6`, o `unannotated` | `stratum_3` |
| `biopsy_decision` | `str` | `'yes'` (56 casos), `'no'` (35 casos), `NaN` (104 casos) | `'no'` |

---

## 3. Especificación Técnica de Scripts

### Script 1: `goals/02/experiments/exp_2/scripts/generate_design.py`

- **Responsabilidad Única**: Particionamiento estratificado determinista, validación de integridad referencial y exportación de `goals/02/data/experimental-design.csv`.
- **Lógica Algorítmica Paso a Paso**:
  1. **Carga y Verificación**:
     - Carga `goals/02/data/inputs.csv` ($N=195$) y `goals/02/data/outputs.csv` ($N=91$).
     - Valida que todos los `case_id` de `outputs.csv` existan en `inputs.csv`.
  2. **Asignación de Estratos Multivariables**:
     - Combina `radiology_report__pirads`, `structured_prompt__bx` y `biopsy_decision`.
     - Clasifica cada caso anotado en uno de los 7 estratos formalizados en `DESIGN.md`:
       - `stratum_0`: $PIRADS \le 2$ ($N=3$, todos `no`).
       - `stratum_1`: $PIRADS = 3$ y `biopsy_decision == no` ($N=8$).
       - `stratum_2`: $PIRADS = 3$ y `biopsy_decision == yes` ($N=2$).
       - `stratum_3`: $PIRADS \ge 4$, `bx == Positive` y `biopsy_decision == no` ($N=22$).
       - `stratum_4`: $PIRADS \ge 4$, `bx == Positive` y `biopsy_decision == yes` ($N=21$).
       - `stratum_5`: $PIRADS \ge 4$, `bx != Positive` y `biopsy_decision == no` ($N=2$).
       - `stratum_6`: $PIRADS \ge 4$, `bx != Positive` y `biopsy_decision == yes` ($N=33$).
  3. **Partición Determinista (`seed=42`)**:
     - Para cada estrato:
       - Ordena por `case_id` para garantizar determinismo cross-platform.
       - Aplica permutación aleatoria con generador `np.random.RandomState(42)`.
       - Asigna alternadamente las muestras a Fold A (`train`) y Fold B (`test`).
     - Para `stratum_2` ($N=2$): garantiza explícitamente 1 caso en `train` y 1 caso en `test`.
  4. **Aserciones Booleanas Mandatorias (Sanity Gates)**:
     - Longitud total de filas == 195.
     - Casos anotados == 91 (45 en `train`, 46 en `test`).
     - Casos sin anotar == 104 (todos con `split == 'unannotated'`).
     - Intersección de identificadores entre `train` y `test` == conjunto vacío ($\emptyset$).
     - Paridad de decisión global: $|\frac{28}{45} - \frac{28}{46}| = |0.6222 - 0.6087| = 0.0135 \le 0.02$ (1.35% de diferencia).
  5. **Persistencia**:
     - Exporta `goals/02/data/experimental-design.csv`.
     - Guarda `experiments/exp_2/results/split_audit.json` con conteos detallados por estrato.

---

### Script 2: `goals/02/experiments/exp_2/scripts/comparative_eda.py`

- **Responsabilidad Única**: Batería estadística exhaustiva y generación de figuras de diagnóstico para certificar la ausencia de sesgo de selección entre `train` (Fold A) y `test` (Fold B).
- **Lógica Analítica y Batería de Tests**:
  1. **Cálculo de Diferencia Media Estandarizada ($SMD$)**:
     - Para covariables continuas: Edad, PSA total, Densidad de PSA (PSAD), Volumen Prostático, Free PSA %.
     - Fórmula:
       $$SMD = \frac{|\bar{X}_{\text{train}} - \bar{X}_{\text{test}}|}{\sqrt{\frac{s_{\text{train}}^2 + s_{\text{test}}^2}{2}}}$$
     - Umbral de parada: Se verifica que $SMD < 0.10$ en todas las covariables continuas principales.
  2. **Pruebas Estadísticas de Hipótesis**:
     - **Kolmogorov-Smirnov (2-sample)**: `scipy.stats.ks_2samp` para verificar identidad en la función de distribución acumulada ($p > 0.05$).
     - **Mann-Whitney U**: `scipy.stats.mannwhitneyu` para evaluar igualdad de medianas no paramétricas ($p > 0.05$).
     - **Chi-cuadrado / Fisher Exact**: `scipy.stats.chi2_contingency` y `scipy.stats.fisher_exact` sobre PI-RADS (2, 3, 4, 5), estado de biopsia previa y DRE ($p > 0.05$).
  3. **Generación de Figuras de Alta Resolución (300 DPI)**:
     - `reports/figures/stratum_distribution_by_fold.png`: Barras agrupadas comparando conteos de cada uno de los 7 estratos entre `train` y `test`.
     - `reports/figures/comparative_distributions_folds.png`: Grid 2x2 de violin plots con scatter de puntos superpuesto para Edad, PSA, PSAD y Volumen Prostático.
     - `reports/figures/categorical_proportions_folds.png`: Barras apiladas normalizadas al 100% comparando proporciones de PI-RADS y antecedentes de biopsia.
     - `reports/figures/qq_plots_folds.png`: Gráficos Q-Q comparando cuantiles empíricos de Train vs Test para PSA y PSAD.
  4. **Persistencia**:
     - Exporta `experiments/exp_2/results/comparative_eda.json` con todas las métricas, estadísticos y p-values tabulados.

---

## 4. Protocolo de Ejecución y Telemetría

- **Entorno de Ejecución**: Entorno Conda `ml-harness`.
- **Naturaleza del Cómputo**: Validación trivial en primer plano (<~10 segundos).
- **Comando de Ejecución Unificado**:
  ```bash
  python goals/02/experiments/exp_2/scripts/generate_design.py && \
  python goals/02/experiments/exp_2/scripts/comparative_eda.py && \
  git rev-parse HEAD > goals/02/experiments/exp_2/results/version.txt
  ```
- **Línea de Telemetría Condensada Esperada**:
  `[EXP-02][Setup: 195 rows (Train: 45 | Test: 46 | Unannotated: 104)][Max SMD: <0.10 | KS p-values: >0.05][Status: Success]`

---

## 5. Criterios de Aceptación y Gatekeeping

El experimento `exp_2` se considerará completado exitosamente cuando:
1. `goals/02/data/experimental-design.csv` exista y pase el 100% de las aserciones de integridad.
2. Todas las figuras diagnósticas existan en `experiments/exp_2/reports/figures/`.
3. `comparative_eda.json` certifique que no hay sesgo de selección ($SMD < 0.10$).
4. Se ejecute la consolidación inmediata en `goals/02/experiments/exp_2/reports/summary.md` y en `goals/02/.logbook.md`.

---

**Estado:** Plan redactado y listo para ejecución.  
¿Das tu visto bueno para ejecutar los scripts y generar `experimental-design.csv` junto con el EDA comparativo?
