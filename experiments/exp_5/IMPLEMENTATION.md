# Implementation Plan: Replication of BrentMemKDM Multimodal for Biopsy Decision (exp_5)

## 1. Overview & Architecture
This plan implements the experimental replication of **BrentMemKDM Multimodal** (density-matrix associative memory with continuous 1-D Brent search bandwidth alignment) on CHIMERA Subtask 1.1 (Biopsy Decision).

The implementation strictly maintains:
1. **Decoupled Evaluation Governance:** Zero evaluation or metric logic implemented inside execution scripts; all evaluation is imported from `utils/pkg/chimera_metrics`.
2. **Identical Preprocessing Pipelines (`exp_4` $\rightarrow$ `exp_5`):**
   - Tabular: Canonical baseline `bx` plus missing-indicator flags, median/mode imputation, One-Hot Encoding and `MinMaxScaler` $[0, 1]$ (purging all post-biopsy fields).
   - MRI: 1024-d visual embeddings, train-mean imputation, `PCA(n_components=16)` fit on training splits.
   - Text: `TfidfVectorizer(max_features=1000, stop_words="english", lowercase=True)` fit on training splits.
3. **Continuous Bandwidth Kernel Density Matrix Memory (`BrentMemKDM`):**
   - RBF Kernel: $k_{j,m}(x) = \exp(-\|x_m - c_{j,m}\|^2 / 2\sigma_m^2)$.
   - Continuous 1-D Brent Optimization: $\sigma_m^* \in [\sigma_{\text{ref}, m}/32, 32\,\sigma_{\text{ref}, m}]$ maximizing mean out-of-fold Macro-F1 across MCCV splits.
   - Multimodal Simplex Search: $\mathbf{w}^* = (w_{\text{tab}}, w_{\text{mri}}, w_{\text{txt}}) \in \Delta^2$ with decision threshold $\theta^* \in [0.3, 0.7]$.
4. **Supervision Arms:**
   - Hard binary targets: $y \in \{0, 1\}$.
   - Soft clinical confidence targets: $\tilde{y} \in [0, 1]$ (`clear`=1.0/0.0, `borderline`=0.75/0.25, `uncertain`=0.50).

## 2. File Organization
```
experiments/exp_5/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── models.py                   # Preprocessors (exp_4) + BrentMemKDM engine
│   └── run_exp5_brent_kdm.py       # Two-phase MCCV execution script with live logging
├── results/
│   ├── version.txt
│   ├── execution.log
│   ├── optimized_configs.json
│   ├── split_predictions.json
│   ├── split_metrics.json
│   ├── aggregate_metrics.json
│   └── wilcoxon_tests.json
└── reports/
    └── summary.md
```

## 3. Protocol & Execution Steps
- **Phase A (Bandwidth & Weight Optimization):** 
  - Estimate unsupervised reference anchor $\sigma_{\text{ref}, m}$ via training fold 3-NN mean distances.
  - Optimize $\sigma_m^*$ via bounded Brent search on cross-validated predictions across MCCV-30.
  - Optimize late fusion simplex $\mathbf{w}^*$ and threshold $\theta^*$.
- **Phase B (Blind Evaluation & Metric Logging):**
  - Run all 30 splits with frozen $\sigma_m^*$ and $\mathbf{w}^*$.
  - Record predictions to `results/split_predictions.json`.
  - Pass predictions to `chimera_metrics` for official metric computation.
  - Run two-sided Wilcoxon signed-rank paired tests against `exp_1` ReAct baseline and `exp_4` $k$-NN baseline.
