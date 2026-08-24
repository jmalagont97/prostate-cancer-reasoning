# Implementation Plan: Joint Feature-Attribution Fusion (Raw Features + Coalesced SHAP 23D) for Subtask 1.3

**Experiment**: `experiments/exp_9/`  
**Project**: `prostate-cancer-reasoning` (CHIMERA-Agent)  
**Date**: 2026-08-24  
**Author**: AI Co-Researcher & Juan Sebastian Malagón Torres  
**Status**: In Progress  

---

## 1. Overview & Architecture
This implementation realizes the joint fusion paradigm proposed in `DESIGN.md`. For each patient case $i$, it extracts:
1. **Raw Phenotypic Features ($\mathbf{x}^{(i)} \in \mathbb{R}^{10}$)**: Canonical pre-biopsy variables (`age`, `psa`, `vol`, `psad`, `dre_num`, `pirads`, `fh_num`, `comorb_score`, `bx_prior_num`, `cspca`).
2. **Local Model Sensitivities ($|\boldsymbol{\Phi}^{(i)}| \in \mathbb{R}^{10}$)**: Absolute Shapley values computed with respect to the optimal frozen tabular classifier ($C_1^*$ Fuzzy-KNN, $k=5$, Euclidean metric from `exp_4`).
3. **Coalesced Energies ($\boldsymbol{\Phi}_{\text{coalesced}}^{(i)} \in \mathbb{R}^3$)**:
   - $\Phi_{\text{PSA,PSAD}}^{(i)} = |\phi_{\text{PSA}}^{(i)}| + |\phi_{\text{PSAD}}^{(i)}|$
   - $\Phi_{\text{PIRADS,csPCa}}^{(i)} = |\phi_{\text{PIRADS}}^{(i)}| + |\phi_{\text{csPCa}}^{(i)}|$
   - $\Phi_{\text{Vol,PSAD}}^{(i)} = |\phi_{\text{Vol}}^{(i)}| + |\phi_{\text{PSAD}}^{(i)}|$
4. **Joint Representation ($\mathbf{z}^{(i)} \in \mathbb{R}^{23}$)**: $\mathbf{z}^{(i)} = [\mathbf{x}^{(i)}, |\boldsymbol{\Phi}^{(i)}|, \boldsymbol{\Phi}_{\text{coalesced}}^{(i)}]$.

---

## 2. Experimental Conditions & Meta-Models
1. **$C_1$ (SHAP-Only Control / $13\text{D}$)**: `RandomForestClassifier(n_estimators=50, max_depth=3, max_features='sqrt', class_weight='balanced', random_state=42)` fitted on $\boldsymbol{\Phi}_{\text{aug}} \in \mathbb{R}^{13}$.
2. **$C_2$ (Raw-Features-Only Control / $10\text{D}$)**: `RandomForestClassifier(n_estimators=50, max_depth=3, max_features='sqrt', class_weight='balanced', random_state=42)` fitted on $\mathbf{x} \in \mathbb{R}^{10}$.
3. **$C_3$ (Joint Concatenated Forest / $23\text{D}$)**: `RandomForestClassifier(n_estimators=50, max_depth=3, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=42)` fitted on $\mathbf{z} \in \mathbb{R}^{23}$.
4. **$C_4$ (Joint Extra-Trees / $23\text{D}$)**: `ExtraTreesClassifier(n_estimators=50, max_depth=3, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=42)` fitted on $\mathbf{z} \in \mathbb{R}^{23}$.
5. **$C_5$ (Targeted Dual-Path Hybrid / $23\text{D}$)**:
   - For deterministic/discrete variables ($DRE$, $Bx$, $Age$, $FH$, $Comorbidity$): Uses specialized balanced tree/forest focusing on $[\mathbf{x}, |\phi_j|]$.
   - For complex continuous/radiological biomarkers ($PSA$, $PIRADS$, $PSAD$, $Vol$, $csPCa$): Uses the joint 23D Extra-Trees ensemble.

---

## 3. Directory & Script Layout
```
experiments/exp_9/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   ├── models.py              <- Data preprocessors, SHAP extractors, and meta-classifiers
│   └── run_exp9_joint.py      <- 30-split MCCV runner with real-time logging
└── results/
    ├── aggregate_metrics.json
    ├── split_metrics.json
    ├── split_predictions.json
    ├── wilcoxon_tests.json
    └── version.txt
```

---

## 4. Execution & Governance Protocol
- Evaluated strictly across all 30 splits of `data/processed/task1/mccv_splits_30.csv`.
- Training and validation sets completely decoupled. Zero post-biopsy leakage.
- Live progress rendered in `tmux:0` terminal pane.
