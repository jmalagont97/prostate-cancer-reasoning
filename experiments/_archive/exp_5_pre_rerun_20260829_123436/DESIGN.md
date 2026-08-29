# Experiment Design: Replication of BrentMemKDM Multimodal (Continuous σ Bandwidth Alignment) for Biopsy Decision (exp_5)

**Experiment**: `experiments/exp_5/`
**Project**: prostate-cancer-reasoning (CHIMERA-Agent Challenge / MICCAI 2026)
**Task**: Subtask 1.1 — Binary Prostate Biopsy Indication Decision
**Origin of Replication**: branch `fagonzalezo` → `experiments/exp_28` (BrentMemKDM Multimodal, continuous bandwidth alignment)
**Protocol Governance**: `utils/pkg/chimera_metrics/PROTOCOL.md` (NORMATIVE & INVIOLABLE)
**Evaluation Framework**: Centralized via `chimera_metrics.task1.compute_biopsy_decision_gate`
**Author**: Juan Sebastian Malagón Torres
**Date**: 2026-08-23
**Status**: Draft

---

## ⚠️ Governance Warnings (Non-Negotiable)

1. **NO evaluation code inside the execution path.** Under no circumstances may any script under `experiments/exp_5/scripts/` implement scoring, metric approximation, ranking, or ad-hoc evaluation logic. All case-level and cohort-level metrics MUST be computed exclusively by importing `chimera_metrics`. Any result produced with local scoring code is scientifically invalid per PROTOCOL.md §1.
2. **Data Leakage Exclusion (Task 1 pre-biopsy setting).** The following biopsy/pathology-derived fields were REMOVED from all model inputs (`structured-prompt.json`, tabular matrices, narrative text) to prevent post-intervention information leaking into the biopsy-decision task:
   `bx`, `bx_isup`, `bx_gl_prim`, `bx_gl_sec`, `bx_gl_tert`, `bx_isup_pred`, `cores_positive`, `cores_total`, `max_core_pct`, `growth_pattern`, `high_risk_patterns`, `pni`, `lvi`, `tumor_location`, plus every `path_hist_*` column.
3. **Split invariance.** Only the 30 deterministic MCCV partitions from `data/processed/task1/mccv_splits_30.csv` are admissible.

---

## 1. Hypothesis

A unified memory-based multimodal Kernel Density Matrix architecture with **continuous bandwidth optimization** (1-D Brent search over $\sigma_m$ per modality, instead of discrete $k$-NN neighbor grids) reproduces or improves upon the source experiment's Macro-F1 = 0.7068 on the CHIMERA Task 1 biopsy decision, evaluated under our centralized MCCV-30 protocol with strict leak-free inputs.

---

## 2. Experimental Setup & Representation Spaces (Aligned with `exp_4`)

- **Dataset(s)**: `data/processed/task1/clinical_tabular.csv` (leak-free pre-biopsy features only), `data/processed/task1/mri_embeddings.csv` (1024-d visual embeddings), `data/processed/task1/clinical_texts.csv` (narrative text, negation-preserving preprocessing), splits from `data/processed/task1/mccv_splits_30.csv` (73 train / 18 val / 104 unlabeled; 11 yes / 7 no per val fold).
- **Representation Pipelines (identical to `exp_4`)**:
  - `tab`: Imputación mediana (numéricas) y moda (categóricas) fit en train, variables indicadoras de missingness, One-Hot Encoding, `MinMaxScaler` $[0, 1]$. Filtrado estricto de variables patológicas post-biopsia.
  - `mri`: Normalización / Imputación con vector medio de train y reducción de dimensionalidad con `PCA(n_components=16, random_state=42)` fit en train.
  - `txt`: `TfidfVectorizer(max_features=1000, stop_words="english", lowercase=True, token_pattern=r"(?u)\b\w+(?:-\w+)*\b")` fit en train.
- **Model / Algorithm**: BrentMemKDM Multimodal — a memory of kernel density matrices generalizing Fuzzy k-NN:
  ```
  k_{j,m}(x) = exp( - ||x_m - c_{j,m}||^2 / (2 * sigma_m^2) )
  clamped_j  = max( (1/N_tr) * prod_m k_{j,m}^2 , eps )      # eps = 1e-12
  w_j(x)     = clamped_j / sum_l clamped_l
  p(y=1|x)   = sum_j w_j(x) * y~_j
  ```
  where $y~_j \in [0,1]$ is either hard ($y \in \{0,1\}$) or soft confidence-weighted target (following exp_4 mapping: clear=1.00/0.00, borderline=0.75/0.25, uncertain=0.50).
- **Optimization target**: continuous bandwidths $\sigma_m^*$ per modality via 1-D Brent search maximizing mean Macro-F1 across MCCV folds, bounded to $[\sigma_{\text{ref}}/32,\; 32\,\sigma_{\text{ref}}]$; late-fusion weights $\mathbf{w}^* = (w_{\text{tab}}, w_{\text{mri}}, w_{\text{txt}})$ optimized over the simplex using out-of-fold MCCV probabilities.
- **Process**: two-phase protocol — Phase A: hyperparameter/bandwidth optimization on MCCV folds; Phase B: frozen-parameter blind evaluation + paired Wilcoxon vs. exp_4 fusion and exp_1 ReAct baseline.
- **Hardware**: local CPU (kernel-memory operations are vectorizable; N_train = 73 makes this lightweight).

## 3. File Layout for This Experiment

```
experiments/exp_5/
├── DESIGN.md                  ← this file
├── IMPLEMENTATION.md          ← to be written after plan approval
├── scripts/                   ← run scripts/configs (no scoring code allowed here)
├── results/
│   └── <condition>/           ← predictions + chimera_metrics outputs
└── reports/
    └── summary.md             ← written later by experiment-reporter
```

Scripts reference project-root-relative paths (`../../utils/pkg/chimera_metrics/`, `../../data/processed/task1/`). Specific script names/configs are decided in the implementation plan.

## 4. Baselines

| Baseline | Reference | Expected Macro-F1 range |
|----------|-----------|------------------------|
| Official ReAct baseline | exp_1 | ~0.61 (ranking score R_1 = 0.6103) |
| Best unimodal fuzzy-KNN | exp_4 | 0.63 – 0.67 |
| Late fusion grid search | exp_4 | 0.70 – 0.72 |
| Source exp_28 (fagonzalezo) | docs/old_tries | **0.7068** |

## 5. Proposed Conditions

| ID | Condition | Description |
|----|-----------|-------------|
| C1 | `tab_brentmemkdm` | BrentMemKDM on leak-free tabular features; Brent-optimized $\sigma_{\text{tab}}$ |
| C2 | `mri_brentmemkdm` | BrentMemKDM on MRI embeddings (raw L2 and PCA variants); $\sigma_{\text{mri}}$ |
| C3 | `txt_brentmemkdm` | BrentMemKDM on TF-IDF/SVD clinical narratives; $\sigma_{\text{txt}}$ |
| C4 | `fusion_equal` | Equal-weight late fusion (1/3, 1/3, 1/3) of C1–C3 probabilities |
| C5 | `fusion_optimal_leakfree` | Late fusion with $(\mathbf{w}^*, \theta^*)$ tuned ONLY on out-of-fold MCCV predictions |
| Supervision arms | `hard` / `soft` | Hard binary targets vs. soft confidence-weighted regression targets |

## 6. Ablation Studies

- **Discrete-$\sigma$ ablation**: replace Brent continuous search with the historical discrete grid $\sigma_{\text{mult}} \in \{0.25, 0.5, 1.0, 2.0\}$ to isolate the contribution of continuous alignment.
- **Product-of-kernels exponent ablation**: squared kernels ($k^2$) vs. linear ($k^1$) inside the clamped product.

## 7. Evaluation Protocol

- Primary metric: mean **Macro-F1** across the exact 30 MCCV partitions (via `chimera_metrics.task1.compute_biopsy_decision_gate`; NO local metric code).
- Secondary: $F_{1\text{-yes}}$, $F_{1\text{-no}}$, gate accuracy, sensitivity/recall_yes, specificity/recall_no, AUROC, Brier score.
- Statistics: mean ± empirical 95% CI ($\overline{M} \pm 1.96\sigma/\sqrt{30}$); two-sided Wilcoxon signed-rank ($p<0.05$) vs. exp_4 best fusion and exp_1 ReAct baseline.
- Results written to `results/<condition>/`; version recorded per run in `results/<condition>/version.txt`.
- Compute budget: 5 conditions × 2 supervision arms × 30 splits ≈ 300 lightweight evaluations (CPU minutes).

## 8. Expected Results & Decision Rules

- If mean Macro-F1 ≥ 0.70 with CI overlapping the source 0.7068 → replication successful; proceed to uncertainty-signal extensions (source exp_29/exp_30 follow-ups).
- If Macro-F1 drops > 0.03 below source → investigate dataset-version drift (our canonical 195-case cohort vs. old_tries cohort) before architectural conclusions.
- Stopping criterion: if equal-weight fusion (C4) already matches C5 within CI, freeze weights at uniform (parsimony rule).

## 9. Risks & Mitigations

- ⚠️ **Cohort mismatch**: source exp_28 used an older preprocessing (docs/old_tries); we replicate on the canonical exp_2 pipeline. Mitigation: record dataset hashes in results and treat source numbers as reference, not ground truth.
- ⚠️ **Leakage via fusion-weight tuning**: mitigated by restricting $(\mathbf{w}^*, \theta^*)$ selection to out-of-fold MCCV predictions only.
- ⚠️ **Metric mismatch**: forbidden by governance — all metrics flow exclusively through `chimera_metrics`.

## 10. Reproducibility Checklist

- [ ] Random seeds fixed and logged (only affects Brent initialization bounds; search itself is deterministic)
- [ ] Config saved in `scripts/configs/`
- [ ] Dataset version / hash recorded in `results/<condition>/version.txt`
- [ ] No checkpoints needed (memory-based, non-parametric model)
- [ ] Environment pinned (`requirements.txt`)
- [ ] Evaluation exclusively via `chimera_metrics` imports

## 11. Next Steps

1. Review and accept this experiment plan (hypothesis, conditions, metrics, decision rules).
2. Once accepted, produce an **implementation plan** covering scripts, configs, and code changes (model code lives in `utils/`, NOT inside `experiments/exp_5/`). Do not begin editing implementation files until that plan is approved and saved as `experiments/exp_5/IMPLEMENTATION.md`.
