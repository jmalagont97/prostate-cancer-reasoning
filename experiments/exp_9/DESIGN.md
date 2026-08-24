# Experiment Design: Joint Feature-Attribution Fusion (Raw Clinical Features + Coalesced SHAP Vectors 23D) for Subtask 1.3

**Experiment**: `experiments/exp_9/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-24  
**Author**: AI Co-Researcher & Juan Sebastian Malagón Torres  
**Status**: Complete  
**Verdict**: **Validated / Champion**. Joint ExtraTrees 23D ($C_4$) beats SHAP-only models ($p = 1.86 \times 10^{-9}$) and attains Macro-F1 of $0.2934 \pm 0.0224$, $S_{\text{weights}} = 0.8278$, and peak clinical rank alignments $\rho(\text{PIRADS}) = +0.4486$, $\rho(\text{Bx}) = +0.4768$, $\rho(\text{DRE}) = +0.3547$, and $\rho(\text{PSA}) = +0.1935$.  

---

## 1. Hypothesis
Concatenating raw baseline clinical phenotype features ($\mathbf{x} \in \mathbb{R}^{10}$) with local model attribution values and coalesced coalition energies ($\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$) into a joint 23D representation $\mathbf{z} = [\mathbf{x}, |\boldsymbol{\Phi}|, \boldsymbol{\Phi}_{\text{coalesced}}] \in \mathbb{R}^{23}$, trained under class-balanced regularized ensemble forests, resolves the clinical polarity ambiguity inherent to pure SHAP magnitudes ($|\phi_j|$) and significantly outperforms both pure-SHAP models ($C_4$ of `exp_8`) and raw-feature-only baselines, achieving $\overline{\text{Macro-F1}} \ge 0.280$ while maintaining strictly positive rank correlations across primary clinical drivers ($\rho(\text{PSA}) \ge 0.15, \rho(\text{PIRADS}) \ge 0.20, \rho(\text{DRE}) \ge 0.20$) under the standardized 30-split MCCV protocol without post-biopsy data leakage.

---

## 2. Experimental Setup
- **Dataset**: Canonical pre-biopsy tabular features `clinical_tabular.csv` and ground truth `ground_truth_task1.csv` ($N=91$ labeled cases).
- **Leakage Prevention**: Complete exclusion of the 14 post-biopsy and pathology-derived fields (`bx`, `bx_isup`, `cores_positive`, etc.). All input representations are derived strictly from pre-decision variables.
- **Cross-Validation**: Standardized MCCV-30 protocol (`data/processed/task1/mccv_splits_30.csv`, $N=73$ train / $18$ val per split).
- **Base Attribution Generator**: Unimodal Tabular Champion of Task 1.1 ($C_1^*$ from `exp_4`, `fuzzy_knn_reg`, $k=5$, Euclidean metric, distance-weighted), fitted strictly within the training fold of each MCCV split.
- **Joint Input Space ($\mathbf{z} \in \mathbb{R}^{23}$)**:
  1. **Direct Phenotypic Features ($\mathbf{x} \in \mathbb{R}^{10}$)**: `[age, psa, vol, psad, dre_num, pirads, fh_num, comorb_score, bx_prior_num, cspca]`.
  2. **Model Sensitivity Features ($|\boldsymbol{\Phi}| \in \mathbb{R}^{10}$)**: Local Shapley absolute values for the 10 features extracted from $C_1^*$.
  3. **Coalesced Coalition Energies ($\boldsymbol{\Phi}_{\text{coalesced}} \in \mathbb{R}^3$)**:
     ```
     Phi_{PSA,PSAD}    = |phi_PSA| + |phi_PSAD|
     Phi_{PIRADS,csPCa}= |phi_PIRADS| + |phi_csPCa|
     Phi_{Vol,PSAD}    = |phi_Vol| + |phi_PSAD|
     ```
- **Compute Environment**: Local Linux workstation, executed within background tmux session `:0` with real-time compact progress logging.

---

## 3. File Layout for This Experiment
```
experiments/exp_9/
├── DESIGN.md                  <- this research-design document
├── IMPLEMENTATION.md          <- implementation plan (approved prior to execution)
├── scripts/
│   ├── models.py              <- feature extractors, joint augmenters, and forest classifiers
│   └── run_exp9_joint.py      <- 30-split MCCV runner with real-time terminal reporting
├── results/
│   ├── aggregate_metrics.json <- 30-split summary metrics (mean, std, CIs)
│   ├── split_metrics.json     <- per-split detailed metrics
│   ├── split_predictions.json <- test-fold prediction logs
│   ├── wilcoxon_tests.json    <- paired hypothesis significance tests
│   └── version.txt            <- git commit / execution hash
└── reports/
    └── summary.md             <- final academic and empirical synthesis
```

---

## 4. Baselines
| Baseline | Description | Expected $\overline{\text{Macro-F1}}$ |
| :--- | :--- | :---: |
| **$B_0$ (Official ReAct Baseline)** | Zero-shot ReAct agent (`exp_1`) | $0.2312 \pm 0.0245$ |
| **$B_1$ (10D SHAP Raw Tree)** | Condition $C_1$ of `exp_8` ($10\text{D}$ SHAP without class weighting) | $0.2348 \pm 0.0234$ |
| **$B_2$ (Coalesced Balanced Forest)** | Champion $C_4$ of `exp_8` ($13\text{D}$ Coalesced SHAP Forest) | $0.2523 \pm 0.0299$ |

---

## 5. Proposed Conditions
1. **$C_1$ (SHAP-Only Control / $13\text{D}$)**: Random Forest ($n=50$, $d=3$, `class_weight='balanced'`) trained exclusively on $\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$.
2. **$C_2$ (Raw-Features-Only Control / $10\text{D}$)**: Random Forest ($n=50$, $d=3$, `class_weight='balanced'`) trained exclusively on raw clinical features $\mathbf{x} \in \mathbb{R}^{10}$.
3. **$C_3$ (Joint Concatenated Forest / $23\text{D}$)**: Random Forest ($n=50$, $d=3$, `max_features='sqrt'`, `class_weight='balanced'`, `min_samples_leaf=2`) trained on $\mathbf{z} = [\mathbf{x}, |\boldsymbol{\Phi}|, \boldsymbol{\Phi}_{\text{coalesced}}] \in \mathbb{R}^{23}$.
4. **$C_4$ (Joint Extra-Trees / $23\text{D}$)**: Extremely Randomized Trees ensemble ($n=50$, $d=3$, `class_weight='balanced'`) on $\mathbf{z} \in \mathbb{R}^{23}$ with randomized split thresholds to suppress collinearity noise.
5. **$C_5$ (Targeted Dual-Path Hybrid / $23\text{D}$)**: Variable-specific targeted estimator: assigns discrete monotonic trees for deterministic categories ($DRE$, $Bx$), while using joint $23\text{D}$ forests for non-linear biomarkers ($PSA$, $PIRADS$, $PSAD$, $Vol$).

---

## 6. Evaluation Protocol
- **Primary Metric**: Mean Macro-F1 across the 10 clinical variables ($\overline{\text{Macro-F1}}_{\text{relevance}} = \frac{1}{10}\sum_{j=1}^{10} \text{Macro-F1}_j$).
- **Secondary Metrics**:
  - Reasoning Quality: $S_{\text{weights}} = 1.0 - \text{mean}(|\text{rank}_{\text{pred}} - \text{rank}_{\text{gt}}| / 3.0)$.
  - Critical Variable Set-F1: `Decisive Factors Set-F1` for `{important, decisive}` factor sets.
  - Clinical Alignment: Spearman rank correlation $\rho_j$ and $p$-values for $PSA, PIRADS, DRE, Bx, PSAD$.
- **Statistical Significance**: Two-sided paired Wilcoxon signed-rank test against $B_2$ ($C_4$ of `exp_8`) across all 30 MCCV splits ($\alpha = 0.05$).

---

## 7. Decision Rules
- **Acceptance Threshold**: $C_3$ or $C_4$ achieves $\overline{\text{Macro-F1}} \ge 0.280$ with $p < 0.05$ vs. $B_2$, maintaining $\rho(\text{PSA}) > 0.15$ and $\rho(\text{PIRADS}) > 0.20$.
- **Partial Pass**: $\overline{\text{Macro-F1}} \in [0.260, 0.280)$ with significant gains in $S_{\text{weights}}$ ($>0.82$) and clinical correlations.
- **Rejection**: $\overline{\text{Macro-F1}} \le 0.2523$ (indicating that 23D feature concatenation induces severe overfitting on $N=73$).

---

## 8. Reproducibility Checklist
- [x] Standardized MCCV-30 split partition matrix locked (`data/processed/task1/mccv_splits_30.csv`)
- [x] Preprocessing and attribution pipeline strictly decoupled (no test leakage)
- [x] Real-time terminal reporting integrated into `tmux:0`
- [x] JSON artifact serialization configured (`aggregate_metrics.json`, `split_metrics.json`, `split_predictions.json`, `wilcoxon_tests.json`)
- [x] Git execution commit and timestamp tracked in `results/version.txt`
