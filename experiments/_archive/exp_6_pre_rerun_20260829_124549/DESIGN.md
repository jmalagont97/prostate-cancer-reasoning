# Experiment Design: Replication of exp_12 Scalar Direct-KDM for Clinical Confidence (Leak-Free, Multimodal) (exp_6)

**Experiment**: `experiments/exp_6/`
**Project**: prostate-cancer-reasoning (CHIMERA-Agent Challenge / MICCAI 2026)
**Task**: Subtask 1.2 — Clinical Confidence Prediction ($c \in \{\text{uncertain}, \text{borderline}, \text{clear}\}$)
**Origin of Replication**: branch `rdacostav_dev` → `experiments/exp_12` (Scalar Direct-KDM, 3-class direct training)
**Protocol Governance**: `utils/pkg/chimera_metrics/PROTOCOL.md` (NORMATIVE & INVIOLABLE)
**Evaluation Framework**: Centralized via `chimera_metrics.task1`
**Author**: Juan Sebastian Malagón Torres
**Date**: 2026-08-23
**Status**: Complete (Verified ✅)

---

## ⚠️ Governance Warnings (Non-Negotiable)

1. **NO evaluation code inside the execution path.** All case-level and cohort-level metrics MUST be computed exclusively by importing `chimera_metrics`. Any local scoring code invalidates the run per PROTOCOL.md §1.
2. **Data Leakage Exclusion (Task 1 pre-biopsy setting).** The following biopsy/pathology-derived fields are REMOVED from ALL model inputs (tabular frames, structured prompts, narrative text) to prevent post-intervention leakage into the biopsy-decision/confidence task:
   `bx`, `bx_isup`, `bx_gl_prim`, `bx_gl_sec`, `bx_gl_tert`, `bx_isup_pred`, `cores_positive`, `cores_total`, `max_core_pct`, `growth_pattern`, `high_risk_patterns`, `pni`, `lvi`, `tumor_location`, plus every `path_hist_*` column. In particular, the historical `exp_12` features `cli_bx_positive`, `cli_bx_missing` and `cli_isup` (= `path_hist_bx_isup`) are **strictly banned** in this replication.
   Additionally: global exclusion of tool-gated `prostate-biopsy-decision-clinical-data.json` (EHR sections: `radiology_report`, `previous_notes`, `laboratory_results`, `psa_trend`, `family_history`) per exp_2 canonical pipeline.
3. **Split invariance.** Only the 30 deterministic MCCV partitions from `data/processed/task1/mccv_splits_30.csv` are admissible.
4. **No information beyond the canonical release.** The replication uses ONLY the canonical exp_2 processed artifacts (`data/processed/task1/`). No external data, no hidden EHR, no post-biopsy signals.

---

## 1. Hypothesis

A 3-class direct-trained Kernel Density Matrix model with a single scalar bandwidth $\sigma \in \mathbb{R}^+$ (replicating the architecture of source `exp_12`), trained on the **leak-free multimodal feature space** of our canonical pipeline (exp_2 preprocessing + exp_4 representation structure), achieves Macro-F1 $\geq 0.45$ and balanced ordinal error below the trivial baseline ($\overline{\text{MOE}}_{\text{abs}} < 0.50$) on MCCV-30, without collapsing any confidence class.

---

## 2. Experimental Setup & Representation Spaces

- **Dataset(s)**: `data/processed/task1/clinical_tabular.csv` (195 cases; 91 supervised with confidence labels), `data/processed/task1/mri_embeddings.csv` (1024-d), `data/processed/task1/clinical_texts.csv`; splits exclusively from `data/processed/task1/mccv_splits_30.csv`. Confidence label distribution over the 91 labeled cases: `clear`=58, `borderline`=18, `uncertain`=15.
- **Preprocessing (identical structure to exp_4 / canonical pipeline)**:
  - `tab`: median imputation (numeric) / mode (categorical) fit on train only, missingness indicator flags, One-Hot Encoding, `MinMaxScaler` $[0, 1]$. Strict filtering of all post-biopsy/pathology variables.
  - `mri`: imputation with train-mean vector, then `PCA(n_components=16, random_state=42)` fit on train only.
  - `txt`: `TfidfVectorizer(max_features=1000, stop_words="english", lowercase=True, token_pattern=r"(?u)\b\w+(?:-\w+)*\b")` fit on train only.
- **Model / Algorithm**: Direct Scalar-KDM (memory-based kernel density matrix generalizing fuzzy k-NN):
  ```
  k_{j,m}(x) = exp( - ||x_m - c_{j,m}||^2 / (2 * sigma^2) )      # scalar sigma shared across modalities
  clamped_j  = max( (1/N_tr) * prod_m k_{j,m} , eps )             # eps = 1e-12
  W_j(x)     = clamped_j / sum_l clamped_l
  P(c = k|x) = sum_j W_j(x) * Y~_{j,k}                            # Y~ one-hot 3-class
  c_hat      = argmax_k P(c = k|x)
  ```
- **Optimization target (hyperparameter search preserved)**: continuous 1-D search over the scalar bandwidth $\sigma$ via Brent's method maximizing mean validation score across folds, bounded to $[\sigma_{\text{ref}}/32,\; 32\,\sigma_{\text{ref}}]$ with $\sigma_{\text{ref}}$ = median pairwise train distance (same search machinery as exp_5).
- **Process**: two-phase protocol — Phase A: Brent bandwidth optimization per condition on MCCV out-of-fold predictions; Phase B: frozen-$\sigma^*$ blind evaluation across all 30 partitions.
- **Hardware**: local CPU (N_train = 73 per fold; memory operations vectorizable).

## 3. File Layout for This Experiment

```
experiments/exp_6/
├── DESIGN.md                  ← this file
├── IMPLEMENTATION.md          ← to be written after plan approval
├── scripts/                   ← run scripts/configs (no scoring code allowed here)
├── results/
│   └── <condition>/           ← predictions + chimera_metrics outputs + version.txt
└── reports/
    └── summary.md             ← written later by experiment-reporter
```

Scripts reference project-root-relative paths (`../../utils/pkg/chimera_metrics/`, `../../data/processed/task1/`). Specific script names/configs are decided in the implementation plan.

## 4. Baselines

| Baseline | Reference | Expected range (Macro-F1) |
|----------|-----------|---------------------------|
| Trivial always-`clear` | analytical | 0.260 |
| Historical incumbent SVM (confidence_svm) | docs/old_tries rdacostav_dev | ~0.40 |
| Source exp_12 Scalar Direct-KDM (**contaminated**, 23 cols incl. bx/isup) | docs/old_tries rdacostav_dev | 0.589 (LOO) — NOT comparable directly; upper-bound reference only |
| Best current pipeline context (exp_4 fusion, decision task) | exp_4 | — |

## 5. Proposed Conditions

| ID | Condition | Description |
|----|-----------|-------------|
| C1 | `tab_direct_scalar_kdm` | Direct-KDM on leak-free tabular frame only; Brent-optimized scalar $\sigma$ |
| C2 | `mri_direct_scalar_kdm` | Direct-KDM on MRI PCA-16 embeddings; scalar $\sigma$ |
| C3 | `txt_direct_scalar_kdm` | Direct-KDM on TF-IDF narratives; scalar $\sigma$ |
| C4 | `fusion_equal` | Equal-weight late fusion (1/3 each) of C1–C3 probability vectors |
| C5 | `fusion_optimal_leakfree` | Late fusion weights tuned ONLY on out-of-fold MCCV probabilities (simplex grid, as exp_4) |
| Supervision arms | `hard` / `ordinal-weighted` | Hard one-hot targets vs. ordinal-distance-weighted soft targets ($Y~_{j,k} \propto 1 - |\text{rank}(y_j)-k|/2$, row-normalized) as an exploratory arm |

⚠️ All conditions use identical preprocessing fitted strictly on train rows per fold.

## 6. Ablation Studies

- **Scalar vs. ARD bandwidths**: replace scalar $\sigma$ with per-dimension ARD $\sigma_j$ (source exp_11 variant) on the clean tabular frame, isolating the regularization benefit claimed by exp_12 under leak-free inputs.
- **Soft-target ablation**: hard one-hot vs. ordinal-weighted targets within C1 (isolates the contribution of ordinal supervision).
- **Modality contribution**: C1 vs. C4/C5 quantifies genuine multimodal gain on confidence prediction.

## 7. Evaluation Protocol & Explicit Metric Definitions

All metrics are evaluated over the exact 30 MCCV-30 partitions ($N_{\text{tr}}=73$, $N_{\text{val}}=18$) and computed through centralized governance via `chimera_metrics.task1`.

### 7.1 Mapeo Ordinal
Se establece el orden discreto con paso unitario y rango $K_{\text{conf}} = 2$:
```
rank(c) = 0 si c = "uncertain",  1 si c = "borderline",  2 si c = "clear"
```

### 7.2 Métrica Oficial del Reto ($S_{\text{conf}}$ — Ordinal Agreement Score)
Puntaje normalizado de proximidad ordinal por caso:
```
S_conf(c_hat, c) = 1.0 - (|rank(c_hat) - rank(c)| / K_conf)
```
- Parámetros: $c, \hat{c} \in \{\text{uncertain}, \text{borderline}, \text{clear}\}$, $K_{\text{conf}} = 2$, rango $S_{\text{conf}} \in [0.0, 1.0]$.
- Concordancia exacta ($\Delta = 0$): $S_{\text{conf}} = 1.0$; error adyacente ($\Delta = 1$): $S_{\text{conf}} = 0.5$; error extremo ($\Delta = 2$): $S_{\text{conf}} = 0.0$.

### 7.3 Métrica Canónica de Selección de Modelos ($\text{MOE}_{\text{abs}}$ — Balanced Mean Ordinal Error)
Error medio ordinal balanceado por clase para evitar el sesgo de la clase modal (`clear` = 63.7%):
```
MOE_abs = (1 / 3) * ∑_{k=0}^2 ( (1 / N_k) * ∑_{i: c_i = k} (|rank(c_hat_i) - k| / K_conf) )
```
- Parámetros: $N_k$ es el soporte de la clase $k \in \{0, 1, 2\}$, $K_{\text{conf}} = 2$, rango $\text{MOE}_{\text{abs}} \in [0.0, 1.0]$ (menor es mejor).

### 7.4 Métricas de Clasificación Multiclase y No-Colapso
- **Macro-F1 (3 clases):**
  ```
  Macro-F1 = (1 / 3) * (F1_uncertain + F1_borderline + F1_clear)
  ```
- **Recall por Clase ($\text{Recall}_k$):**
  ```
  Recall_k = TP_k / (TP_k + FN_k),  para k in {uncertain, borderline, clear}
  ```
  *Condición estricta:* $\text{Recall}_k > 0 \quad \forall k$ (filtro de no-colapso).
- **Kappa Ponderado Cuadrático ($\kappa_w$ — Quadratic Weighted Kappa):**
  ```
  κ_w = 1.0 - (∑_{i,j} w_{i,j} * O_{i,j}) / (∑_{i,j} w_{i,j} * E_{i,j}),   w_{i,j} = (i - j)^2 / 4
  ```
  *Variables:* $O_{i,j}$ es la matriz de confusión normalizada observada y $E_{i,j}$ la matriz esperada por azar.

### 7.5 Estadísticos y Agregación
- Promedio y desviación estándar a través de los 30 folds: $\overline{M} \pm \text{std}(M)$ e intervalo de confianza empírico al 95% ($\overline{M} \pm 1.96 \cdot \text{std} / \sqrt{30}$).
- Test de significancia estadística: Wilcoxon signed-rank bilateral ($p < 0.05$) entre la mejor fusión multimodal (C4/C5) vs. modelo unimodal tabular (C1).
- Matrices de confusión agregadas (*pooled*) sobre las 30 particiones.

### 7.6 Compuertas de Aceptación en Cascada (Decision Cascade)
```
┌───────────────────────────────────┐
│ 1. Validez Estructural (100%)     │ Salidas estrictas en {"uncertain", "borderline", "clear"}
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────────┐
│ 2. Superación del Trivial         │ MOE_abs < 0.5000 (supera el baseline trivial "always clear")
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────────┐
│ 3. Filtro de No-Colapso           │ Recall(k) > 0 para todas las clases k in {0, 1, 2}
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────────┐
│ 4. Criterio de Selección Primario │ argmin MOE_abs en validación MCCV-30
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────────┐
│ 5. Desempate / Benchmark          │ argmax Macro-F1 en validación MCCV-30
└───────────────────────────────────┘
```

- **Ubicación de Resultados:** `results/<condition>/` conteniendo predicciones por fold, métricas consolidadas `metrics.json` y `version.txt`.
- **Presupuesto de Cómputo:** 5 condiciones $\times$ 2 brazos de supervisión $\times$ 30 splits $\approx$ 300 evaluaciones analíticas ligeras (segundos/minutos de CPU).

## 8. Expected Results & Decision Rules

- If best condition reaches Macro-F1 ≥ 0.45 AND $\overline{\text{MOE}}_{\text{abs}}$ < 0.50 with no collapse → hypothesis holds; adopt the winning configuration as the Subtask 1.2 head and proceed to joint Task-1 scoring integration (decision gate × reasoning stage).
- If Macro-F1 ∈ [0.35, 0.45) → partial support: signal exists but is weaker than contaminated estimate suggested; investigate modality-specific contributions before architectural conclusions.
- If Macro-F1 < 0.35 or any class collapses → the direct approach fails without leakage; pivot to derived-uncertainty approaches (margin/entropy from the Subtask 1.1 decision model) as follow-up experiment.
- Stopping criterion: if equal-weight fusion (C4) matches C5 within CI, freeze uniform weights (parsimony rule).

## 9. Risks & Mitigations

- ⚠️ **Residual cohort mismatch**: source exp_12 used an older cohort/preprocessing; numbers serve only as an upper-bound reference. Mitigation: record dataset hashes in `results/<condition>/version.txt`; treat source values as non-comparable.
- ⚠️ **Class imbalance** (58/18/15): mitigated by balanced MOE_abs selection criterion and no-collapse gate rather than plain accuracy/F1 micro.
- ⚠️ **Leakage via fusion tuning**: restricted to out-of-fold MCCV predictions only.
- ⚠️ **Small validation folds** (18 cases, min class ≈ 1–2): high fold variance; mitigated by reporting CI over 30 folds and pooled confusion matrices.
- ⚠️ **Metric governance**: forbidden to reimplement scoring locally — all metrics flow through `chimera_metrics`.

## 10. Reproducibility Checklist

- [ ] Random seeds fixed and logged (`random_state=42`; Brent search is deterministic given bounds)
- [ ] Config saved in `scripts/configs/`
- [ ] Dataset version / hash recorded in `results/<condition>/version.txt`
- [ ] No checkpoints needed (memory-based, non-parametric model)
- [ ] Environment pinned (`requirements.txt`)
- [ ] Evaluation exclusively via `chimera_metrics` imports

## 11. Next Steps

1. Review and accept this experiment plan (hypothesis, conditions, metrics, decision rules).
2. Once accepted, produce an **implementation plan** covering scripts, configs, and code changes for Sections 5–6. Do not begin editing implementation files until that plan is approved.
