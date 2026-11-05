# Experiment Design: Particionamiento Estratificado 50/50, experimental-design.csv y EDA Comparativo (Task 1)

**Experiment**: `experiments/exp_2/`  
**Project**: `prostate-cancer-reasoning` (Goal 02)  
**Date**: 2026-11-05  
**Author**: Experto en IA Agéntica & Investigador Principal  
**Status**: Complete  

---

## 1. Hypothesis
Un esquema de particionamiento estratificado compuesto por la interacción multivariable de fenotipos clínicos diagnósticos $(\text{biopsy\_decision} \times \text{pirads\_binned} \times \text{prior\_bx})$ sobre los 91 casos anotados de Task 1 garantiza la paridad distributiva de los subgrupos críticos entre los subconjuntos de `train` y `test` con una discrepancia máxima inferior al 5% ($MSD \le 5\%$), y una ausencia demostrable de sesgo de selección ($SMD < 0.10$ y tests de equivalencia distributiva $p > 0.05$ en todas las covariables clínicas), consolidando en `goals/02/data/experimental-design.csv` la asignación inmutable para el protocolo de evaluación cruzada simétrica out-of-sample sin fuga de datos (*data leakage*) respecto a los 104 casos sin anotar.

---

## 2. Experimental Setup
- **Dataset de Entrada**: 
  - `goals/02/data/inputs.csv`: 195 casos con 21 variables estructuradas.
  - `goals/02/data/outputs.csv`: 91 casos con Ground Truth clínico (15 columnas).
- **Artefacto Objetivo Obligatorio en `goals/02/data/`**:
  - **`goals/02/data/experimental-design.csv`**: Tabla canónica de 195 filas indexadas por `case_id`, señalando explícitamente para cada paciente su pertenencia a `train`, `test` o `unannotated`.
  - **Estructura de Columnas del Artefacto**:
    1. `case_id`: Identificador único (`PT-pseudo_...`).
    2. `split`: Asignación canónica primaria (`'train'` para los 45 casos de Fold A, `'test'` para los 46 casos de Fold B, `'unannotated'` para los 104 casos sin anotar).
    3. `fold_a`: Rol en Fase 1 de evaluación cruzada (`'train'` [45], `'test'` [46], `'unannotated'` [104]).
    4. `fold_b`: Rol en Fase 2 de evaluación cruzada (`'test'` [45], `'train'` [46], `'unannotated'` [104]).
    5. `status`: Estado de anotación (`'annotated'` [91 casos], `'unannotated'` [104 casos]).
    6. `is_annotated`: Indicador booleano (`True` [91], `False` [104]).
    7. `stratum_id`: Identificador del estrato clínico multivariable (`stratum_0` a `stratum_6`, o `unannotated`).
    8. `biopsy_decision`: Etiqueta Ground Truth de biopsia (`'yes'`, `'no'`, `NaN` si no anotado).
- **Estratificación Compuesta Multivariable (7 Estratos Clínicos)**:
  - `stratum_0` ($N=3$): $PIRADS \le 2$ (todos `biopsy_decision == no`).
  - `stratum_1` ($N=8$): $PIRADS = 3$ y `biopsy_decision == no`.
  - `stratum_2` ($N=2$): $PIRADS = 3$ y `biopsy_decision == yes` (zona gris de alta sospecha).
  - `stratum_3` ($N=22$): $PIRADS \ge 4$, `bx == Positive` y `biopsy_decision == no` (inversión terapéutica por cáncer confirmado).
  - `stratum_4` ($N=21$): $PIRADS \ge 4$, `bx == Positive` y `biopsy_decision == yes` (progresión/re-estadificación).
  - `stratum_5` ($N=2$): $PIRADS \ge 4$, `bx != Positive` y `biopsy_decision == no` (discordancia extrema/comorbilidad).
  - `stratum_6` ($N=33$): $PIRADS \ge 4$, `bx != Positive` y `biopsy_decision == yes` (sospecha estándar de novo).
  - `unannotated` ($N=104$): Casos de prueba sin Ground Truth.
- **Asignación Simétrica 2-Fold (Double-Blind 50/50)**:
  - **Train / Fold A** ($N=45$): 28 `yes` / 17 `no`.
  - **Test / Fold B** ($N=46$): 28 `yes` / 18 `no`.
  - **Fase 1**: Train = Fold A (inducción agéntica) $\to$ Test = Fold B (evaluación ciega pura).
  - **Fase 2**: Train = Fold B (inducción agéntica) $\to$ Test = Fold A (evaluación ciega pura).
- **Semilla Global**: `seed = 42` determinista.

---

## 3. File Layout for This Experiment
```
goals/02/
├── data/
│   ├── inputs.csv
│   ├── outputs.csv
│   └── experimental-design.csv       ← ARTEFACTO CLAVE GENERADO POR ESTE EXPERIMENTO
└── experiments/exp_2/
    ├── DESIGN.md                     ← Este documento (diseño metodológico y EDA comparativo)
    ├── IMPLEMENTATION.md             ← Plan de implementación
    ├── scripts/
    │   ├── generate_design.py        ← Script que genera data/experimental-design.csv
    │   └── comparative_eda.py        ← Auditoría distributiva, tests de sesgo y generación de figuras
    ├── results/
    │   ├── split_audit.json          ← Distribución exacta por estrato y fold
    │   ├── comparative_eda.json      ← Métricas estadísticas de balance (SMD, Kolmogorov-Smirnov, Chi-cuadrado, Mann-Whitney)
    │   └── version.txt               ← Hash git del commit ejecutor
    └── reports/
        ├── figures/
        │   ├── stratum_distribution_by_fold.png    ← Comparativa de estratos Train vs Test
        │   ├── comparative_distributions_folds.png ← Violin/Boxplots de variables continuas (Edad, PSA, PSAD, Volumen)
        │   ├── categorical_proportions_folds.png   ← Proporciones de variables categóricas (PI-RADS, Biopsia previa, DRE)
        │   └── qq_plots_folds.png                  ← Gráficos Q-Q de alineación de cuantiles entre Folds
        └── summary.md                ← Consolidación final del experimento
```

---

## 4. Baselines
Para contrastar la efectividad de la estratificación compuesta multivariable, se comparan 3 métodos de particionamiento:

| Condición | Estrategia de Split | Criterio de Partición | Riesgo Evaluado |
| :--- | :--- | :--- | :--- |
| **Baseline 0 (Naive)** | Random Shuffle 50/50 | Asignación aleatoria simple sin estratificar | Desbalance severo en subgrupos minoritarios ($PIRADS=3$, $bx=\text{Positive}$) |
| **Baseline 1 (Monovariable)** | `StratifiedKFold(n_splits=2)` | Estratificado exclusivamente por `biopsy_decision` | Sesgo en variables clínicas secundarias ($PSAD$, $PIRADS$, sesgo de selección) |
| **Condición 2 (Propuesta)** | Compuesta Multivariable | Interacción $(\text{decision} \times \text{PIRADS} \times \text{bx})$ | Balance simultáneo de decisión y fenotipo urológico con sesgo despreciable |

---

## 5. Protocolo de EDA Comparativo y Certificación de Ausencia de Sesgo de Selección

Para garantizar formalmente que las dos particiones (`train` / Fold A y `test` / Fold B) son epidemiológica y clínicamente intercambiables (evitando sesgos de selección inadvertidos), se establece el siguiente protocolo analítico:

### A. Balance de Covariables Continuas (Diferencia Media Estandarizada - SMD)
Para cada covariable continua $j$ ($\text{edad}, \text{PSA total}, \text{PSAD}, \text{volumen prostático}, \text{free PSA \%}$), se calcula la Diferencia Media Estandarizada (*Standardized Mean Difference*):
$$SMD_j = \frac{|\bar{X}_{j, \text{train}} - \bar{X}_{j, \text{test}}|}{\sqrt{\frac{s_{j, \text{train}}^2 + s_{j, \text{test}}^2}{2}}}$$
- **Criterio Epidemiológico Riguroso**: $SMD_j < \mathbf{0.10}$ (indica balance óptimo y sesgo de selección clínicamente despreciable).
- Cualquier variable con $SMD \ge 0.20$ constituye causal de rechazo de la partición.

### B. Pruebas de Hipótesis de Equivalencia Distributiva
1. **Variables Continuas**:
   - **Test de Kolmogorov-Smirnov (2 muestras)**: $H_0$: Las distribuciones empíricas acumuladas de Train y Test son idénticas ($F_{\text{train}}(x) = F_{\text{test}}(x)$). Umbral requerido: $p > \mathbf{0.05}$.
   - **Test U de Mann-Whitney (No paramétrico)**: $H_0$: Las medianas y rangos entre Train y Test provienen de la misma población. Umbral requerido: $p > \mathbf{0.05}$.
2. **Variables Categóricas y Ordinales** ($\text{PI-RADS}, \text{bx previa}, \text{DRE}, \text{csPCa score}$):
   - **Test $\chi^2$ de Homogeneidad / Test Exacto de Fisher** (para celdas con $N < 5$): $H_0$: La proporción de cada categoría es equivalente entre ambos subconjuntos. Umbral requerido: $p > \mathbf{0.05}$.

### C. Visualizaciones Diagnósticas Comparativas
1. **Violin / Boxplots Paralelos** (`comparative_distributions_folds.png`): Comparación de la forma, mediana y dispersión intercuartílica de Edad, PSA, PSAD y Volumen Prostático entre Train y Test.
2. **Diagramas de Barras Apiladas Normalizadas** (`categorical_proportions_folds.png`): Distribución porcentual de cada nivel de PI-RADS (2, 3, 4, 5) y estado de biopsia previa entre Train y Test.
3. **Gráficos Cuantil-Cuantil (Q-Q Plots)** (`qq_plots_folds.png`): Evaluación de alineación lineal estricta sobre la diagonal identidad ($y = x$) para cuantiles de PSA y PSAD.

---

## 6. Evaluation Protocol & Decision Rules

### Métricas Primarias y Umbrales de Aceptación
1. **Discrepancia Máxima de Estrato ($MSD$)**:
   $$MSD = \max_{s \in \text{Estratos}} \left| \frac{N_{s, \text{train}}}{N_{\text{train}}} - \frac{N_{s, \text{test}}}{N_{\text{test}}} \right| \times 100 \le \mathbf{5.0\%}$$
2. **Paridad de Decisión Global**:
   $$\Delta P(\text{yes}) = |P(\text{yes} \mid \text{train}) - P(\text{yes} \mid \text{test})| \le \mathbf{2.0\%}$$
3. **Preservación de Subgrupos Críticos Raros**:
   - Para el estrato crítico `stratum_2` ($PIRADS=3, \text{decision}=\text{yes}, N=2$): **exactamente 1 caso en Train y 1 caso en Test**.
   - Para `stratum_0` ($PIRADS \le 2, N=3$): **al menos 1 caso en cada subconjunto** (1 en uno, 2 en el otro).
4. **Certificación de No-Sesgo de Selección (EDA Comparativo)**:
   - $SMD < \mathbf{0.10}$ en el 100% de las covariables clínicas continuas.
   - P-value de Kolmogorov-Smirnov $p > \mathbf{0.05}$ en Edad, PSA y PSAD.
5. **Blindaje contra Data Leakage y Exhaustividad**:
   - $\text{Train} \cap \text{Test} = \emptyset$ (cero solapamiento de identificadores).
   - $\text{Train} \cup \text{Test} = \text{Casos Anotados}$ ($N = 91$).
   - $\text{Casos Sin Anotar} = 104$ estrictamente etiquetados como `unannotated`.
   - Longitud total de `goals/02/data/experimental-design.csv` = exactamente 195 filas.

### Reglas de Decisión
- **Éxito (Aprobación y Congelación)**: Si se cumplen simultáneamente las 5 condiciones anteriores, el archivo `goals/02/data/experimental-design.csv` y el reporte de balance quedan **certificados y congelados** como la verdad fundamental de particionamiento para el desarrollo del árbol neurosimbólico en `exp_3`.
- **Fallo**: Si se detecta $SMD \ge 0.15$ o $p \le 0.05$ en alguna covariable primaria (indicando sesgo de selección), se reajustará la semilla o se incluirá un criterio de minimización de discrepancia multivariada antes de exportar el diseño.

---

## 7. Risks & Mitigations
- **Riesgo 1: Estratos con cardinalidad impar indivisible (ej. $N=3$)**:  
  *Mitigación*: La asignación alternada determinista reparte 2 muestras en un subconjunto y 1 en el otro; el EDA comparativo audita que esta asimetría local no desvíe las distribuciones de variables continuas ($SMD < 0.10$).
- **Riesgo 2: Fuga inadvertida de muestras no etiquetadas**:  
  *Mitigación*: Validación de integridad con aserciones booleanas estrictas sobre la lista de `case_id` ($N_{\text{annotated}} = 91$, $N_{\text{unannotated}} = 104$).

---

## 8. Reproducibility Checklist
- [ ] Semillas fijadas deterministamente (`seed=42`).
- [ ] Verificación de cardinalidades: 91 casos anotados (45 Train / Fold A, 46 Test / Fold B), 104 sin anotar.
- [ ] Archivo canónico generado y verificado en `goals/02/data/experimental-design.csv`.
- [ ] Métricas de balance y EDA exportadas en `experiments/exp_2/results/comparative_eda.json`.
- [ ] Gráficos diagnósticos generados en `experiments/exp_2/reports/figures/`.
- [ ] Commit git registrado en `experiments/exp_2/results/version.txt`.

---

## 9. Next Steps
1. Revisión y aprobación final del presente `DESIGN.md` corregido.
2. Generación del plan de implementación (`experiments/exp_2/IMPLEMENTATION.md`) detallando la estructura de scripts y tests de aserción.
3. Ejecución del particionamiento, generación del EDA comparativo y consolidación de resultados en `.logbook.md`.
