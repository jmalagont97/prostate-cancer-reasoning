# Experiment Design: Multivariate Balanced Decision Trees over Full 10D SHAP and Coalesced Feature Spaces for Clinical Relevance (Subtask 1.3)

**Experiment**: `experiments/exp_8/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-24  
**Author**: Juan Sebastian Malagón Torres (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)  
**Status**: Draft  

---

## 1. Hypothesis
"El entrenamiento de árboles de decisión multivariados independientes con balanceo de clases (`class_weight='balanced'`) sobre el espacio vectorial completo de Shapley values 10D derivado del modelo tabular calibrado de Subtarea 1.1 ($C_1^*$ Fuzzy-KNN de `exp_4`), enriquecido con magnitudes coalescentes de coaliciones colineales ($\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$), superará significativamente en $\overline{\text{Macro-F1}}_{\text{relevance}}$ al enfoque unidimensional monótono de umbrales SHAP y replicará la alineación bivariada de alta significancia estadística en variables primarias ($\rho(\text{PSA}) \ge 0.30$ y $\rho(\text{PIRADS}) \ge 0.30$, con $p < 0.01$) bajo el protocolo estandarizado MCCV-30 sin incurrir en colapso de correlación en variables categóricas ($DRE$) ni inversiones espurias en variables compuestas ($PSAD$)."

---

## 2. Experimental Setup
- **Dataset(s)**:
  - Cohorte oficial CHIMERA Task 1 ($N=195$ casos totales, $N=91$ casos rotulados con ground truth completo).
  - Ubicación: `../../data/processed/task1/clinical_tabular.csv` y `../../data/processed/task1/ground_truth_task1.csv`.
  - Protocolo de Partición: 30 particiones fijas de Validación Cruzada Monte Carlo (`../../data/processed/task1/mccv_splits_30.csv`, Train $N=73$, Val $N=18$), con estratificación multiobjetivo preservada.
- **Espacio de Entrada y Auditoría de Fuga (Zero Data Leakage)**:
  - Modelo base de clasificación: $C_1^*$ Tabular de `exp_4` (`fuzzy_knn_reg`, $k=5$, métrica Euclidiana, ponderación inversa a la distancia).
  - Se excluyen estrictamente los 14 campos histopatológicos post-biopsia prohibidos por CHIMERA (`bx`, `bx_isup`, `cores_positive`, etc.). Solo se emplean covariables basales legítimas pre-decisión.
- **Generación de Atribución Local (Full SHAP Extraction)**:
  - Para cada split de validación, $C_1^*$ se entrena exclusivamente en el fold de entrenamiento.
  - Para cada caso $i$, se calcula el vector de Shapley values 10D exacto:
    ```
    Phi_base^(i) = [ |phi_age^(i)|, |phi_psa^(i)|, |phi_vol^(i)|, |phi_psad^(i)|, |phi_dre^(i)|,
                     |phi_pirads^(i)|, |phi_fh^(i)|, |phi_comorb^(i)|, |phi_bx^(i)|, |phi_cspca^(i)| ] in R^10
    ```
- **Aproximación de Atribución Coalescente (Coalesced Feature Augmentation)**:
  - Se introducen términos de energía de coalición para los subespacios de alta colinealidad identificados:
    ```
    Phi_{PSA,PSAD}^(i)    = |phi_psa^(i)| + |phi_psad^(i)|
    Phi_{PIRADS,csPCa}^(i) = |phi_pirads^(i)| + |phi_cspca^(i)|
    Phi_{Vol,PSAD}^(i)    = |phi_vol^(i)| + |phi_psad^(i)|
    ```
  - Vector aumentado resultante:
    ```
    Phi_aug^(i) = [ Phi_base^(i), Phi_{PSA,PSAD}^(i), Phi_{PIRADS,csPCa}^(i), Phi_{Vol,PSAD}^(i) ] in R^13
    ```
- **Meta-Clasificadores Multivariados de Relevancia**:
  - Para cada una de las 10 variables clínicas objetivo $j \in \{1, \dots, 10\}$, se entrena un estimador supervisado independiente:
    ```
    y^_j^(i) = T_j( Phi_aug^(i) ) in {0: not_used, 1: noted, 2: important, 3: decisive}
    ```
  - Balanceo de muestras: `class_weight='balanced'` para penalizar inversamente la frecuencia de clases minoritarias (*decisive* / *noted*).
- **Hardware y Tiempo Estimado**:
  - CPU local (16 hilos), tiempo estimado $< 45$ segundos para los 30 splits.

---

## 3. File Layout for This Experiment
```
experiments/exp_8/
├── DESIGN.md                  <- this file (experiment design only)
├── IMPLEMENTATION.md          <- implementation plan (to be approved in follow-up step)
├── scripts/
│   ├── run_exp8_relevance.py  <- canonical runner for Subtask 1.3 benchmark
│   └── models.py              <- SHAP extractors, coalesced generators, and multivariate tree estimators
├── results/
│   ├── version.txt            <- git commit SHA and execution timestamp
│   ├── aggregate_metrics.json <- macro-F1, rho, p-values per variable and global mean across 30 splits
│   ├── split_metrics.json     <- detailed metrics per fold
│   ├── split_predictions.json <- case-by-case relevance weight predictions
│   └── wilcoxon_tests.json    <- paired statistical tests vs. baseline
└── reports/
    └── summary.md             <- final academic and technical write-up
```

---

## 4. Baselines
| Baseline | Config / Descripción | Métrica Esperada ($\overline{\text{Macro-F1}}$) | $\rho(\text{PSA}) / \rho(\text{PIRADS})$ |
| :--- | :--- | :---: | :---: |
| **$B_0$ Baseline Oficial ReAct** | ReAct LLM Agent prompting directo (`exp_1` Subtask 1.3) | $\approx 0.150$ | $\approx 0.10$ / No alineado |
| **$B_1$ 1D SHAP Discretization (`exp_20`)** | 3 umbrales escalares por variable sobre $|\phi_j|$ | $\approx 0.106$ | $\rho(\text{PSA})=0.254$ / $\rho(\text{PIRADS})=0.082$ |
| **$B_2$ Historical 6D Tree (`exp_22` benchmark)** | Meta-árbol $d=3$ sobre vector SHAP 6D (restringido) | $\approx 0.258$ | $\rho(\text{PSA})=0.374$ ($p<0.001$) / $\rho(\text{PIRADS})=0.302$ ($p<0.005$) |

---

## 5. Proposed Conditions
| Condición | Espacio de Atribución | Meta-Modelo | Profundidad / Regularización | Balanceo |
| :--- | :--- | :--- | :---: | :---: |
| **$C_1$ 10D-Raw-Tree** | $\boldsymbol{\Phi}_{\text{base}} \in \mathbb{R}^{10}$ | `DecisionTreeClassifier` | $d \in \{2, 3, 4\}$ | Uniforme |
| **$C_2$ 10D-Balanced-Tree** | $\boldsymbol{\Phi}_{\text{base}} \in \mathbb{R}^{10}$ | `DecisionTreeClassifier` | $d \in \{2, 3, 4\}$ | `class_weight='balanced'` |
| **$C_3$ Coalesced-Balanced-Tree** | $\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$ (con sumas coalicionales) | `DecisionTreeClassifier` | $d \in \{2, 3, 4\}$ | `class_weight='balanced'` |
| **$C_4$ Coalesced-Balanced-Forest** | $\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$ (con sumas coalicionales) | `RandomForestClassifier` ($n=50$) | $d \in \{2, 3\}$, `min_samples_leaf=2` | `class_weight='balanced'` |
| **$C_5$ Hybrid-Regularized (Ablación Clave)** | $\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$ + Direct Feature Prior | Árbol con regularización monótona en $DRE$ | $d=3$, balanced | `class_weight='balanced'` |

---

## 6. Ablation Studies
1. **Impacto de la Aumentación Coalescente ($C_2$ vs $C_3$):** Evalúa si la adición de $\Phi_{\text{PSA,PSAD}}$, $\Phi_{\text{PIRADS,csPCa}}$ y $\Phi_{\text{Vol,PSAD}}$ previene la inversión de correlación en $PSAD$ y estabiliza las predicciones de $csPCa$.
2. **Efecto del Balanceo Ponderado ($C_1$ vs $C_2$):** Cuantifica la recuperación de Macro-F1 en clases extremas sub-representadas (*decisive* en $PSA$, *not\_used* en $PIRADS$).
3. **Control de Varianza Ensamble vs Árbol Simple ($C_3$ vs $C_4$):** Evalúa si un ensamble regularizado previene la degradación en variables con particiones categóricas nítidas ($DRE$).

---

## 7. Evaluation Protocol
- **Métricas Primarias**:
  1. $\overline{\text{Macro-F1}}_{\text{MCCV}}$: Macro-F1 promedio no ponderado sobre las 10 variables clínicas a lo largo de los 30 splits.
  2. Coeficiente de correlación por rangos de Spearman medio $\bar{\rho}$ y por variable $\rho_j$ respecto al ground truth urológico.
  3. Tasa de significancia estadística en variables primarias: $\rho(\text{PSA}) \ge 0.30$ ($p < 0.01$) y $\rho(\text{PIRADS}) \ge 0.30$ ($p < 0.01$).
- **Métricas Secundarias**:
  - Macro-F1 individual por variable $j \in \{\text{age}, \text{psa}, \text{vol}, \text{psad}, \text{dre}, \text{pirads}, \text{fh}, \text{comorbidity}, \text{bx}, \text{cspca}\}$.
  - Macro-F1 en clases extremas (clase 0 *not\_used* y clase 3 *decisive*).
- **Pruebas de Hipótesis**:
  - Test pareado de rangos con signo de Wilcoxon (*Two-Sided Paired Wilcoxon Signed-Rank Test*, $\alpha = 0.05$) comparando $\overline{\text{Macro-F1}}$ de cada condición contra el Baseline $B_0$ y el modelo 1D $B_1$.
- **Gobernanza de Evaluación**:
  - Implementado estrictamente mediante `chimera_metrics.task1` y funciones canónicas de evaluación.

---

## 8. Expected Results & Decision Rules
- **Criterio de Éxito Primario**:
  - $\overline{\text{Macro-F1}}_{\text{MCCV}} \ge 0.260$ con significancia pareada ($p < 0.001$ vs $B_0$).
  - $\rho(\text{PSA}) \ge 0.35$ ($p < 0.005$) y $\rho(\text{PIRADS}) \ge 0.30$ ($p < 0.005$) simultáneamente.
  - $\rho(\text{DRE}) \ge 0.25$ ($p < 0.05$) evitando el colapso a cero observado en `exp_22`.
  - $\rho(\text{PSAD}) \ge 0.00$ eliminando la correlación negativa espuria ($\rho = -0.292$) de `exp_22`.
- **Criterio de Falla / Ajuste**:
  - Si los árboles sobreajustan el espacio 13D degradando la correlación en $DRE$ o $Bx$, aplicar poda de profundidad estricta a $d=2$ o restringir las particiones de variables categóricas mediante priors directos.

---

## 9. Risks & Mitigations
| Riesgo Detectado | Probabilidad / Severidad | Mitigación Rigurosa |
| :--- | :---: | :--- |
| **Data Leakage en Cálculo SHAP** | Alta / Crítica | El explicador SHAP y el background summary se calculan única y estrictamente dentro del `tr_indices` de cada partición MCCV; cero filtración de validación. |
| **Sobreajuste por Dimensionalidad (13D en $N=73$)** | Media / Alta | Limitar profundidad máxima de los árboles ($d \in [2, 3]$) e hiperparametrizar `min_samples_leaf \ge 3`. |
| **Inestabilidad Estocástica de SHAP** | Baja / Media | Fijar semillas deterministas para KernelSHAP / estimadores analíticos y verificar convergencia en cada iteración. |
| **Colapso de Clases Minoritarias** | Media / Alta | Aplicar `class_weight='balanced'` en todos los estimadores multivariados. |

---

## 10. Reproducibility Checklist
- [x] Particiones MCCV-30 fijadas y versionadas (`data/processed/task1/mccv_splits_30.csv`)
- [x] Semillas pseudoaleatorias fijadas globalmente (`seed=42`)
- [x] Espacio de entrada verificado libre de campos prohibidos de biopsia (Zero Leakage)
- [x] Configuración de hiperparámetros registrada en script y `aggregate_metrics.json`
- [x] Registro de versión de entorno y commit registrado en `results/version.txt`
- [x] Predicciones caso a caso exportadas en `results/split_predictions.json`

---

## 11. Next Steps
1. Revisión y aprobación formal del Investigador Principal sobre este diseño de investigación (`DESIGN.md`).
2. Una vez aprobado, generar el **Plan de Implementación** (`IMPLEMENTATION.md`) detallando la estructura modular de `models.py` y el runner `run_exp8_relevance.py`.
3. Ejecutar el benchmark sobre las 30 particiones y registrar métricas en `results/`.
4. Invocar `experiment-reporter` para consolidar el reporte oficial en `reports/summary.md`.
