# Experiment Design: Systematic MCCV-30 Unimodal Sweep and Trimodal Late Fusion for Biopsy Decision (exp_4)

**Experiment**: `experiments/exp_4/`  
**Project**: prostate-cancer-reasoning (CHIMERA-Agent Challenge / MICCAI 2026)  
**Task**: Subtask 1.1 — Binary Prostate Biopsy Indication Decision  
**Protocol Governance**: `utils/pkg/chimera_metrics/PROTOCOL.md`  
**Evaluation Framework**: Centralized via `chimera_metrics` (`chimera_metrics.task1.compute_biopsy_decision_gate`)  
**Date**: 2026-08-23  
**Status**: Complete ✅  

---

## 1. Scientific Context & Purpose

The CHIMERA Subtask 1.1 objective is the accurate, robust, and calibrated binary decision for prostate biopsy recommendation ($\hat{y} \in \{\text{yes}, \text{no}\}$).

The purpose of **exp_4** is to:
1. Conduct a rigorous, fold-isolated **30-fold Monte Carlo Cross-Validation (MCCV-30) Grid Search** for each unimodal representation (Tabular, MRI, Narrative Text) with compact neighbor spaces ($k \in \{1, 3, 5\}$) and focused dimensionality reductions ($\ge 4$ components) to determine optimal hyperparameter configurations $C_1^*, C_2^*, C_3^*$.
2. Execute a systematic **Late Fusion Grid Search** over the probability simplex $\mathbf{w} \in \Delta^2$ and binarization threshold $\theta \in [0.20, 0.80]$ using the optimal frozen unimodal models across the 30 MCCV splits.
3. Quantify statistical superiority and effect size through paired two-sided Wilcoxon signed-rank tests against the **Official ReAct Baseline ($C_0$, `exp_1`)**.

---

## 2. Evaluation Protocol & Governance Directives

In strict accordance with `utils/pkg/chimera_metrics/PROTOCOL.md`:

1. **Centralized Evaluation:** All evaluation metrics must be derived exclusively via `chimera_metrics.task1.compute_biopsy_decision_gate`.
2. **Standardized Partition Invariance:** All models are evaluated across the exact 30 deterministic partitions specified in:
   ```
   data/processed/task1/mccv_splits_30.csv
   ```
   - **Per-split cohort:** $N_{\text{train}} = 73$, $N_{\text{val}} = 18$ ($11\text{ yes} / 7\text{ no}$), $N_{\text{unlabeled}} = 104$.
3. **Metric Space for Subtask 1.1:**
   - $\text{Macro-F1} = \frac{1}{2}(F_{1\text{-yes}} + F_{1\text{-no}})$ (Primary selection criterion across 30 splits)
   - $F_{1\text{-yes}}$ (Official biopsy-positive harmonic score)
   - $F_{1\text{-no}}$ (Official biopsy-negative harmonic score)
   - $S_{\text{gate}}$ / Exact Match Accuracy
   - Sensitivity / Recall$_{\text{yes}}$
   - Specificity / Recall$_{\text{no}}$

---

## 3. Fuzzy Confidence Regression Formulation

The **Fuzzy KNN** formulation converts binary classification into non-parametric clinical label smoothing by mapping the binary decision and the ordinal pathological confidence to a continuous risk target $y^* \in [0.0, 1.0]$:

$$\begin{aligned}
(\text{"yes"}, \text{"clear"})       &\longrightarrow 1.00 \\
(\text{"yes"}, \text{"borderline"})  &\longrightarrow 0.75 \\
(\text{"yes"}, \text{"uncertain"})   &\longrightarrow 0.50 \\
(\text{"no"}, \text{"uncertain"})    &\longrightarrow 0.50 \\
(\text{"no"}, \text{"borderline"})   &\longrightarrow 0.25 \\
(\text{"no"}, \text{"clear"})        &\longrightarrow 0.00
\end{aligned}$$

The continuous predicted probability for sample $x_i$ is:
$$\hat{p}(x_i) = \frac{\sum_{j \in \mathcal{N}_k(x_i)} \alpha_j \cdot y_j^*}{\sum_{j \in \mathcal{N}_k(x_i)} \alpha_j}$$
where $\alpha_j = 1.0$ (uniform) or $\alpha_j = \frac{1}{d(x_i, x_j) + \epsilon}$ (distance-weighted).

Binary prediction with threshold $\theta$:
$$\hat{y}_i = \begin{cases} \text{"yes"} & \text{si } \hat{p}(x_i) \ge \theta \\ \text{"no"} & \text{si } \hat{p}(x_i) < \theta \end{cases}$$

---

## 4. Unimodal Hyperparameter Search Spaces (Phase 1: MCCV-30)

| Dimensión | 1. Modalidad Tabular ($C_1$) | 2. Modalidad MRI ($C_2$) | 3. Modalidad Texto Clínico ($C_3$) |
| :--- | :--- | :--- | :--- |
| **Fuente de Datos** | `clinical_tabular.csv` ($195 \times 28$) | `mri_features.csv` / Embeddings ($195 \times 1024$) | `narratives.csv` / Texto narrativo clínico ($195$) |
| **Protocolo de Preprocesamiento** | Missingness indicators (`col__is_missing`) + imputación mediana/moda + MinMax scaler fit en train. | Verificación dimensional ($73 \times 1024$) y finitud numérica. | Preservación de negaciones clínicas (`no`, `not`, `negative`, `without`, `absence`, `denies`) + lematización + TF-IDF. |
| **Reducción / Selección** | **Poda Jerárquica Spearman:**<br>$\tau \in \{0.50, 0.60, 0.70, 0.80, \text{no\_prune}\}$ | **PCA:**<br>$n_{\text{comp}} \in \{4, 8, 16, 32, 64, \text{no\_pca}\}$ | **TruncatedSVD:**<br>$n_{\text{comp}} \in \{4, 10, 20, 40, \text{no\_svd}\}$ |
| **Hiperparámetros de Clasificador** | • Tipo: $\{\text{standard\_knn\_clf}, \text{fuzzy\_knn\_reg}\}$<br>• $k \in \{1, 3, 5\}$<br>• Métrica: $\{\text{cosine}, \text{euclidean}, \text{manhattan}\}$<br>• Pesos kernel: $\{\text{uniform}, \text{distance}\}$ | • Tipo: $\{\text{standard\_knn\_clf}, \text{fuzzy\_knn\_reg}\}$<br>• $k \in \{1, 3, 5\}$<br>• Métrica: $\{\text{euclidean}, \text{cosine}\}$<br>• Pesos kernel: $\{\text{uniform}, \text{distance}\}$ | • Tipo: $\{\text{standard\_knn\_clf}, \text{fuzzy\_knn\_reg}\}$<br>• $max\_features \in \{500, 1000, 2000, \text{None}\}$<br>• $k \in \{1, 3, 5\}$<br>• Métrica: $\{\text{cosine}, \text{euclidean}\}$<br>• Pesos kernel: $\{\text{uniform}, \text{distance}\}$ |
| **Configuraciones en Grilla** | $5_{\tau} \times 2_{\text{tipo}} \times 3_k \times 3_{\text{dist}} \times 2_{\text{peso}} = \mathbf{180}$ | $6_{\text{pca}} \times 2_{\text{tipo}} \times 3_k \times 2_{\text{dist}} \times 2_{\text{peso}} = \mathbf{144}$ | $4_{\text{feat}} \times 5_{\text{svd}} \times 2_{\text{tipo}} \times 3_k \times 2_{\text{dist}} \times 2_{\text{peso}} = \mathbf{480}$ |
| **Evaluaciones Totales (30 Splits)** | $180 \times 30 = \mathbf{5.400}$ | $144 \times 30 = \mathbf{4.320}$ | $480 \times 30 = \mathbf{14.400}$ |
| **Criterio de Selección** | $C_1^* = \arg\max \overline{\text{Macro-F1}}_{\text{MCCV}}$ | $C_2^* = \arg\max \overline{\text{Macro-F1}}_{\text{MCCV}}$ | $C_3^* = \arg\max \overline{\text{Macro-F1}}_{\text{MCCV}}$ |

---

## 5. Fusión Tardía y Contraste Estadístico (Fases 2 y 3)

* **Fase 2 (Late Fusion):** Con $C_1^*, C_2^*, C_3^*$ fijos:
  $$\hat{p}_{\text{fusion}} = w_{\text{tab}} p_{C_1^*} + w_{\text{mri}} p_{C_2^*} + w_{\text{txt}} p_{C_3^*}$$
  Búsqueda sobre $\mathbf{w} \in \Delta^2$ (paso $0.05$, 231 combinaciones) y umbral $\theta \in [0.20, 0.80]$ (paso $0.02$, 31 umbrales) $\implies C_4^* = (\mathbf{w}^*, \theta^*)$.
* **Fase 3 (Wilcoxon Test):** Prueba pareada de Wilcoxon signed-rank ($\alpha=0.05$) contra el Baseline Oficial ReAct ($C_0$, `exp_1`) sobre las 30 particiones.
