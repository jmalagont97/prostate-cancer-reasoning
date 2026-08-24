# Experiment Design: Replication of Post-Hoc Composite Fuzzy ICI for Clinical Confidence Prediction (exp_7)

**Experiment**: `experiments/exp_7/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent Challenge / MICCAI 2026)  
**Task**: Subtask 1.2 — Clinical Confidence Prediction ($c \in \{\text{uncertain}, \text{borderline}, \text{clear}\}$)  
**Origin of Replication**: branch `main` → `exp_17` (Composite Fuzzy ICI + Meta-thresholds)  
**Protocol Governance**: `utils/pkg/chimera_metrics/PROTOCOL.md` (NORMATIVE & INVIOLABLE)  
**Evaluation Framework**: Centralized via `chimera_metrics.task1`  
**Author**: Juan Sebastian Malagón Torres (PI) & AI Co-Researcher (Deep Learning & Clinical Reasoning Specialist)  
**Date**: 2026-08-24  
**Status**: Draft  

---

## ⚠️ Governance & Leak-Free Directives (Non-Negotiable)

1. **Strict Metric Governance:** All case-level and cohort-level metrics ($S_{\text{conf}}$, $\text{MOE}_{\text{abs}}$, Macro-F1, Accuracy) must be evaluated via `chimera_metrics.task1` or standard scipy/sklearn wrappers without modifying the canonical metric definitions.
2. **Pre-Biopsy Feature Space (Zero Leakage):** All post-biopsy features (`bx`, `bx_isup`, `cores_positive`, `path_hist_*`, etc.) and tool-gated EHR files remain strictly excluded. Only the pre-biopsy multimodal features from canonical `exp_2` are used.
3. **No Unimodal Hyperparameter Search (Frozen exp_4):** Per experimental mandate, we DO NOT perform any grid/Bayesian sweep over unimodal architectures or hyperparameters. The unimodal models are strictly frozen to the optimal configurations discovered in `exp_4`:
   - **Tabular ($T$):** `fuzzy_knn_reg`, $k=5$, metric=`euclidean`, weights=`distance`, no pruning.
   - **MRI ($M$):** `standard_knn_clf`, $k=1$, metric=`euclidean`, weights=`uniform`, PCA components = 16.
   - **Text ($X$):** `fuzzy_knn_reg`, $k=3$, metric=`cosine`, weights=`uniform`, TF-IDF max_features = 1000.
4. **Partition Invariance:** Evaluation is executed across the standardized 30-split stratified MCCV partitions (`data/processed/task1/mccv_splits_30.csv`, Train $N=73$, Val $N=18$).

---

## 1. Hypothesis & Research Objective

### Falsifiable Hypothesis
Computing a continuous **Composite Fuzzy Inconsistency/Confidence Index (ICI)** post-hoc from the predictive probability distributions of the frozen multimodal models from `exp_4` ($p_T, p_M, p_X$), and discretizing it via leak-free 1D Decision Tree meta-thresholds with balanced class weighting (`class_weight='balanced'`), will:
1. Achieve a statistically significant positive Spearman rank correlation with ground-truth clinical confidence annotations ($\rho > 0.20$, $p < 0.05$).
2. Outperform the trivial majority-class baseline (`clear`, Macro-F1 $\approx 0.2600$, $\text{MOE}_{\text{abs}} = 0.5000$) by achieving $\overline{\text{Macro-F1}}_{\text{MCCV}} \ge 0.44$ and $\overline{\text{MOE}}_{\text{abs}} < 0.40$ on MCCV-30 without suffering zero-recall collapse on the minority classes (`uncertain`, `borderline`).

---

## 2. Mathematical Formulation of Composite Fuzzy ICI

```
       p_T (Tabular Fuzzy KNN) ──┐
       p_M (MRI Standard KNN)  ──┼──> [ Consenso p_bar + Dispersión sigma ] ──> ICI continuo ──> [ Meta-umbrales 1D Tree (balanced) ] ──> {0, 1, 2}
       p_X (Text Fuzzy KNN)    ──┘
```

Given the out-of-fold or test predictions from the 3 frozen modalities $p_T, p_M, p_X \in [0, 1]$:

### 2.1 Consenso Intermodal y Dispersión
- **Probabilidad Media ($\bar{p}$):**
  $$\bar{p} = \frac{p_T + p_M + p_X}{3}$$
- **Dispersión Estándar Intermodal ($\sigma$):**
  $$\sigma = \sqrt{\frac{1}{3} \sum_{m \in \{T, M, X\}} (p_m - \bar{p})^2} \quad (\text{ddof} = 0)$$

### 2.2 Índice de Confianza Compuesto (ICI)
$$\text{ICI} = 2 \cdot |\bar{p} - 0.5| \cdot (1 - 2\sigma) \quad \in [0, 1]$$
- **Término de Certeza ($2|\bar{p} - 0.5|$):** Maximiza el índice cuando la probabilidad promedio se acerca a los extremos de certidumbre ($0.0$ o $1.0$).
- **Factor de Penalización por Conflicto ($1 - 2\sigma$):** Degrada la confianza cuando las distintas modalidades divergen entre sí.

### 2.3 Meta-Umbralización sin Fuga (1D Decision Tree)
En cada partición de entrenamiento de MCCV-30:
1. Se obtienen los vectores de probabilidad de entrenamiento $\mathbf{p}_{T, \text{tr}}, \mathbf{p}_{M, \text{tr}}, \mathbf{p}_{X, \text{tr}}$ mediante un esquema interno libre de fuga (inner-split OOF o cross-fitting en $N_{\text{tr}}=73$).
2. Se computa $\text{ICI}_{\text{tr}}$.
3. Se entrena un `DecisionTreeClassifier(max_depth=2, max_leaf_nodes=3, min_samples_leaf=5, class_weight='balanced', random_state=42)` sobre $(\text{ICI}_{\text{tr}}, y_{\text{conf, tr}})$.
4. El árbol extrae dos puntos de corte ordenados $\tau_1 < \tau_2$, definiendo la regla de decisión:
   $$\hat{c} = \begin{cases} 0 \text{ (uncertain)}, & \text{si } \text{ICI} < \tau_1 \\ 1 \text{ (borderline)}, & \text{si } \tau_1 \le \text{ICI} < \tau_2 \\ 2 \text{ (clear)}, & \text{si } \text{ICI} \ge \tau_2 \end{cases}$$

---

## 3. File Layout for This Experiment

```
experiments/exp_7/
├── DESIGN.md                  ← This research design document
├── IMPLEMENTATION.md          ← Implementation plan (to be generated in next step)
├── scripts/
│   ├── models.py              ← Base model definitions & ICI computation module
│   └── run_exp7_fuzzy_ici.py  ← Main execution and evaluation runner
├── results/
│   ├── C0_Majority_Clear/     ← Baseline results
│   ├── C1_Unweighted_Fuzzy_ICI/← Canonical exp_17 replication (equal modality weights)
│   ├── C2_Optimal_Weighted_ICI/← Ablation: exp_4 optimal late fusion weights (0.55/0.20/0.25)
│   ├── summary_metrics.csv    ← Consolidated cross-condition metrics
│   ├── predictions.csv        ← Case-level predictions across all 30 splits
│   └── execution.log          ← Console execution log
└── reports/
    ├── summary.md             ← Structured final report
    └── figures/               ← Confusion matrices & ICI distributions
```

---

## 4. Baselines & Reference Points

| Modelo / Referencia | Configuración / Descripción | Macro-F1 | Accuracy | Spearman $\rho$ | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$C_0$: Majority Baseline** | Asigna siempre la clase modal (`clear`) | 0.2588 | 63.74% | N/A | N/A |
| **Historical $exp\_17$ (LOOCV)** | Composite Fuzzy ICI sobre Fuzzy-KNN | 0.4470 | 57.95% | 0.2790 | 0.0085 ($\checkmark$) |
| **$exp\_6$ Direct-KDM (MCCV-30)** | KDM directo escalar 3 clases | 0.4836 | 62.04% | N/A | N/A |

---

## 5. Proposed Conditions & Ablation Arms

| ID | Condición | Formulación de $\bar{p}$ | Meta-Umbrales | Objetivo / Hipótesis Específica |
| :--- | :--- | :--- | :--- | :--- |
| **$C_0$** | `Baseline_Majority` | N/A | N/A | Línea base trivial siempre-`clear`. |
| **$C_1$** | `Canonical_Fuzzy_ICI` | $\bar{p} = \frac{p_T + p_M + p_X}{3}$ | 1D Tree (`class_weight='balanced'`) | Replicación fiel y canónica de $exp\_17$ bajo MCCV-30. |
| **$C_2$** | `Weighted_Fuzzy_ICI` | $\bar{p}_w = 0.55 p_T + 0.20 p_M + 0.25 p_X$ | 1D Tree (`class_weight='balanced'`) | Evalúa si la ponderación multimodal óptima de $exp\_4$ mejora la calibración del ICI frente al promedio simple. |
| **$C_3$** | `Unweighted_ICI_UniformTree` | $\bar{p} = \frac{p_T + p_M + p_X}{3}$ | 1D Tree (`class_weight=None`) | Ablación del impacto del balanceo de clases en los meta-umbrales. |

---

## 6. Evaluation Protocol & Rigor

1. **Protocolo:** 30 splits independientes de MCCV ($N=91$, $N_{\text{tr}}=73$, $N_{\text{val}}=18$).
2. **Métricas Principales:**
   - **Macro-F1 (3 clases):** Media y desviación estándar en 30 splits.
   - **Balanced Mean Ordinal Error ($\text{MOE}_{\text{abs}}$):** Gobernado por `chimera_metrics.task1`.
   - **Ordinal Agreement Score ($S_{\text{conf}}$):** Métrica oficial de proximidad ordinal.
   - **Spearman Rank Correlation ($\rho, p$):** Correlación de ranking entre ICI continuo y confianza real (evaluado sobre predicciones out-of-fold agrupadas).
   - **Accuracy global y Recall por clase:** Verificación de no-colapso ($\text{Recall}_k > 0$ para $k \in \{0, 1, 2\}$).
3. **Pruebas de Significancia Estadística:** Test pareado de rangos con signo de Wilcoxon ($\alpha = 0.05$) sobre los 30 splits frente a $C_0$.

---

## 7. Decision Rules

- **Si $C_1$ alcanza Macro-F1 $\ge 0.44$, $\text{MOE}_{\text{abs}} < 0.40$ y $\rho$ significativo ($p < 0.05$):**  
  Se confirma la replicabilidad y validez metodológica del enfoque post-hoc Composite Fuzzy ICI bajo el protocolo estricto MCCV-30 sin fuga de datos.
- **Si $C_2$ supera a $C_1$:**  
  Se valida que transferir los pesos óptimos de late-fusion de la Subtarea 1.1 al cálculo de certidumbre en la Subtarea 1.2 aporta un beneficio medible en la coherencia intermodal.
- **Si el árbol no-balanceado ($C_3$) colapsa en la clase mayoritaria:**  
  Se demuestra formalmente la necesidad matemática de la ponderación de clases inversa para prevenir el colapso del recall en datos clínicos asimétricos.

---

## 8. Reproducibility Checklist

- [x] Repositorio fijado en particiones fijas `data/processed/task1/mccv_splits_30.csv`.
- [x] Hiperparámetros base congelados desde `exp_4/results/optimized_configs.json`.
- [x] Semillas aleatorias fijadas (`random_state=42`).
- [x] Código de evaluación desacoplado bajo `chimera_metrics`.
- [x] Versión del código y hash de entorno a registrar en `results/version.txt`.

---

## 9. Next Steps

1. Revisar y aprobar este documento de diseño de investigación (`DESIGN.md`).
2. Generar el plan de implementación detallado (`IMPLEMENTATION.md`) especificando la estructura de scripts y módulos de ejecución.
