# Implementation Plan: Replication of exp_12 Scalar Direct-KDM for Clinical Confidence (exp_6)

**Experiment**: `experiments/exp_6/`
**Design Document**: `experiments/exp_6/DESIGN.md`
**Status**: Ready for Execution

---

## 1. Scope & Execution Flow

This implementation replicates the **Direct-KDM 3-Class Architecture** (historic record holder `exp_12`) under the strict **leak-free multimodal canonical representation** (established in `exp_2`, `exp_4`) and the standardized **MCCV-30 protocol** (`utils/pkg/chimera_metrics/PROTOCOL.md`).

### Pipeline Workflow:
1. **Data Ingestion (Canonical & Leak-Free)**: Load `clinical_tabular.csv`, `clinical_texts.csv`, `mri_embeddings.csv`, and `ground_truth_task1.csv` (labeled subset $N=91$). Include the canonical baseline `bx`; exclude all post-biopsy pathology fields (`bx_isup`, `path_hist_*`, `clinical-data.json`).
2. **Leak-Free Preprocessing (Fit on Train Folds Only)**:
   - **Tabular ($\mathbf{x}_{\text{tab}}$)**: Median imputation for numerical features + One-Hot encoding for categorical clinical variables + MinMax scaling.
   - **MRI ($\mathbf{x}_{\text{mri}}$)**: 16-component PCA fit strictly on train embeddings per fold.
   - **Text ($\mathbf{x}_{\text{txt}}$)**: TF-IDF (1000 features) + TruncatedSVD (16 components) fit on `full_encounter_narrative`.
3. **Model Training & Continuous Bandwidth Search**:
   - `ScalarDirectKDM`: Memory of prototypes $(\mathbf{x}_i, \mathbf{y}_i)$ where $\mathbf{y}_i \in \mathbb{R}^3$ (hard one-hot or ordinal-weighted soft target).
   - Bandwidth $\sigma$: Optimized via bounded Brent search minimizing $\text{MOE}_{\text{abs}}$ or Negative Log-Likelihood on out-of-fold predictions.
4. **Conditions Evaluated (MCCV-30)**:
   - `C1_tab_direct_scalar_kdm`: Tabular unimodal Direct-KDM.
   - `C2_mri_direct_scalar_kdm`: MRI unimodal Direct-KDM.
   - `C3_txt_direct_scalar_kdm`: Text unimodal Direct-KDM.
   - `C4_fusion_equal`: Equal-weight late fusion of probability distributions.
   - `C5_fusion_optimal_leakfree`: Simplex grid-searched late fusion ($\Delta^2$) tuned strictly on out-of-fold training probabilities.
   - **Supervision Arms**: Evaluated under both `hard` one-hot targets and `ordinal-weighted` soft targets.
   - **Ablation**: Scalar $\sigma$ vs. Multi-dimensional ARD bandwidth $\mathbf{\sigma} \in \mathbb{R}^D$ (replicating `exp_11`).
5. **Centralized Evaluation & Output Generation**:
   - Compute official $S_{\text{conf}}$, $\text{MOE}_{\text{abs}}$, $\text{Macro-F1}$, per-class recalls, and Quadratic Weighted Kappa ($\kappa_w$) via `chimera_metrics`.
   - Write per-condition `predictions.csv`, `metrics.json`, and `version.txt`.

---

## 2. Directory & File Manifest

```
experiments/exp_6/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── models.py                   # Preprocessors, ScalarDirectKDM, ARD-KDM, MultimodalFusion
│   └── run_exp6_confidence_mccv.py  # Full MCCV-30 execution pipeline & chimera_metrics evaluator
├── results/
│   ├── C1_tab_direct_scalar_kdm/
│   ├── C2_mri_direct_scalar_kdm/
│   ├── C3_txt_direct_scalar_kdm/
│   ├── C4_fusion_equal/
│   ├── C5_fusion_optimal_leakfree/
│   ├── ablation_ard_tabular/
│   ├── ablation_ordinal_weighted/
│   └── execution.log
└── reports/
    └── summary.md
```

---

## 3. Implementation Details: `models.py`

### 3.1 Preprocessing Pipeline
- **Tabular Features**: Numeric columns (`age`, `psa`, `vol`, `psad`, `pirads`, `psav`, `psap`, `months`, `ipss_score`, `vitals_*`) with median imputation and missingness indicators. Categorical columns (`dre`, `active_treatment_flag`, comorbidity flags).
- **Zero-Leakage Assurance**: Include baseline `bx`; exclude `isup`, `bx_isup`, `cores`, and all post-biopsy findings.
- **Dimensionality Reduction**: MRI (PCA-16) and Text (TF-IDF + SVD-16) fitted per fold.

### 3.2 `ScalarDirectKDM` Classifier
- **Forward Inference**:
  $$W_i(\mathbf{x}) = \frac{\exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_i\|^2}{2\sigma^2}\right)}{\sum_{j=1}^{N_{\text{tr}}} \exp\left(-\frac{\|\mathbf{x} - \mathbf{c}_j\|^2}{2\sigma^2}\right)}, \qquad \mathbf{p}(\mathbf{x}) = \sum_{i=1}^{N_{\text{tr}}} W_i(\mathbf{x}) \cdot \mathbf{y}_i$$
- **Decision Rule**:
  $$\hat{c} = \arg\max_{k \in \{0, 1, 2\}} p_k(\mathbf{x})$$

---

## 4. Execution Protocol & Verification

- **Split Set**: `data/processed/task1/mccv_splits_30.csv`
- **Seeds**: Deterministic `random_state=42`.
- **Metrics Governance**: `chimera_metrics.task1.compute_confidence_score`.
