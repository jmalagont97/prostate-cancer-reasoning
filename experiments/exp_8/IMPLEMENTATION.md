# Implementation Plan: Multivariate Balanced Decision Trees over Full 10D SHAP and Coalesced Feature Spaces (Subtask 1.3)

**Experiment**: `experiments/exp_8/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-24  
**Author**: Juan Sebastian Malagón Torres (PI) & AI Co-Researcher  
**Status**: Approved & Ready for Execution  

---

## 1. Modular Architecture Overview

El objetivo es implementar la extracción de atribuciones locales conjuntas y meta-árboles balanceados para predecir los 10 pesos urológicos ordinales ($y \in \{0, 1, 2, 3\}$), bajo el protocolo estricto MCCV-30 (73 train / 18 val).

```
<experiments/exp_8/>
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── models.py              <- Modelos tabulares, estimador KernelSHAP exacto por fold, generador coalescente y meta-árboles
│   └── run_exp8_relevance.py  <- Runner MCCV-30 con visualización compacta en tiempo real vía rich/ANSI y evaluación canónica
└── results/
    ├── version.txt
    ├── aggregate_metrics.json
    ├── split_metrics.json
    ├── split_predictions.json
    └── wilcoxon_tests.json
```

---

## 2. Component Specifications

### A. Base Tabular Model ($C_1^*$ Frozen from `exp_4`)
- **Estimador:** `FuzzyKNNRegressor(k=5, metric='euclidean', weights='distance')`.
- **Preprocesamiento:** Imputación por mediana/moda sobre el fold de entrenamiento, codificación One-Hot de covariables categóricas legítimas (`dre`, `enc_type`), escalado `MinMaxScaler` ajustado en train.
- **Zero Leakage:** Exclusión estricta de todos los campos histopatológicos post-biopsia.

### B. Local SHAP Attribution Extractor (`ShapAttributionExtractor`)
- Para cada split $k \in \{1, \dots, 30\}$:
  - Se entrena $C_1^*$ con `X_train_fold`.
  - Se construye un `shap.KernelExplainer` o estimador de perturbación analítico sobre el `X_train_fold`.
  - Se obtienen las importancias marginales $|\phi_j^{(i)}|$ para las 10 covariables clínicas:
    ```
    Phi_base^(i) = [ |phi_age|, |phi_psa|, |phi_vol|, |phi_psad|, |phi_dre|,
                     |phi_pirads|, |phi_fh|, |phi_comorbidity|, |phi_bx|, |phi_cspca| ] in R^10
    ```

### C. Coalesced Feature Augmentation (`CoalescedAugmenter`)
- Cálculo de términos de coalición para mitigar colinealidades:
  ```
  Phi_{PSA,PSAD}    = |phi_psa| + |phi_psad|
  Phi_{PIRADS,csPCa} = |phi_pirads| + |phi_cspca|
  Phi_{Vol,PSAD}    = |phi_vol| + |phi_psad|
  ```
- Vector aumentado resultante: $\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$.

### D. Meta-Clasificadores Multivariados
- Para cada variable $j \in \{1, \dots, 10\}$, se evalúan 5 condiciones:
  1. **$C_1$ (10D-Raw-Tree):** `DecisionTreeClassifier(max_depth=d)` sobre $\boldsymbol{\Phi}_{\text{base}}$.
  2. **$C_2$ (10D-Balanced-Tree):** `DecisionTreeClassifier(max_depth=d, class_weight='balanced')` sobre $\boldsymbol{\Phi}_{\text{base}}$.
  3. **$C_3$ (Coalesced-Balanced-Tree):** `DecisionTreeClassifier(max_depth=d, class_weight='balanced')` sobre $\boldsymbol{\Phi}_{\text{aug}}$.
  4. **$C_4$ (Coalesced-Balanced-Forest):** `RandomForestClassifier(n_estimators=50, max_depth=d, class_weight='balanced', random_state=42)` sobre $\boldsymbol{\Phi}_{\text{aug}}$.
  5. **$C_5$ (Hybrid-Regularized):** Árbol balanceado sobre $\boldsymbol{\Phi}_{\text{aug}}$ combinando atribución y valor crudo para $DRE$ y antecedentes de biopsia.

---

## 3. Real-Time Terminal Visualization Protocol

El runner imprimirá en consola (diseñado para la sesión `tmux :0`):
1. **Encabezado Compacto:** Resumen del experimento, configuración de particiones (MCCV-30) y variables.
2. **Tabla de Progreso en Tiempo Real:** Fila por cada fold (1 al 30) mostrando $\overline{\text{Macro-F1}}$, $\rho(\text{PSA})$, $\rho(\text{PIRADS})$, $\rho(\text{DRE})$, $S_{\text{weights}}$ instantáneos y acumulados.
3. **Resumen Final de Comparación:** Tabla estructurada con medias, desviaciones estándar, $p$-values de Wilcoxon y verificación de hipótesis.

---

## 4. Execution Command via tmux
```bash
tmux send-keys -t 0 "python3 experiments/exp_8/scripts/run_exp8_relevance.py" C-m
```
