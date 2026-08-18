# Tabular KDM Biopsy Decision Prediction (exp_23) Summary Report

**Date**: 2026-08-18  
**Model**: `KDMClassModel` (Kernel Density Matrix classifier), `n_comp = n_train`, two target arms (hard / uncertainty-guided soft)  
**Dataset**: Labeled Complete-Case Tabular Clinical Data ($N_{\mathrm{labeled}} = 88$, 54 yes / 34 no)  

## Phase A: 100-Split MCCV Grid Search (32 configs x 2 arms)
- **Arm `hard`** best config: `sigma_mult=2.0, x_train=False, y_train=False, encoder=linear` — Mean Validation Macro-F1: **0.5965** (std=0.1218)  
- **Arm `soft`** best config: `sigma_mult=2.0, x_train=False, y_train=False, encoder=identity` — Mean Validation Macro-F1: **0.6013** (std=0.1074)  
- **Fuzzy KNN reference** best config: `{'k': 1, 'weights': 'uniform', 'metric': 'euclidean'}` — recomputed inline on identical rows/splits/folds.  

## Phase B: LOOCV (88 folds), R=10 seeds
| Model | Macro-F1 | Accuracy | Sensitivity | Specificity | AUROC | Brier | Deterministic |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| KDM `hard` | **0.5636** | 62.50% | 0.8148 | 0.3235 | 0.6334 | 0.2494 | False |
| KDM `soft` | **0.6694** | 71.59% | 0.8889 | 0.4412 | 0.6498 | 0.2263 | True |
| Fuzzy KNN reference (recomputed) | 0.6364 | 65.91% | 0.7407 | 0.5294 | 0.6304 | 0.2908 | — |
| exp_13 published (other checkout, reference only) | 0.6364 | 65.91% | 0.7407 | 0.5294 | 0.6304 | 0.2908 | — |

### 2x2 Confusion Matrix (best KDM arm: `soft`)
| Ground Truth \ Predicted | No Biopsy | Biopsy |
|:---|:---:|:---:|
| **No Biopsy** ($N=34$) | **15** | 19 |
| **Biopsy** ($N=54$) | 6 | **48** |

### McNemar's Exact Test (KDM vs. recomputed Fuzzy KNN reference — both arms reported, two comparisons)
| Arm | Discordant b (KDM right/KNN wrong) | Discordant c (KDM wrong/KNN right) | p-value |
|:---|:---:|:---:|:---:|
| `hard` | 7 | 10 | 0.6291 |
| `soft` | 11 | 6 | 0.3323 |

**Note on the soft-target ceiling**: only the 32 non-`clear` patients can differ between the hard and soft arms (ỹ takes 6 distinct values, mostly 0/1) — a small Arm A/B gap should not be over-read.

## Secondary Objective: Diagnostic Confidence from Native Uncertainty
| Signal | Target-informed | Macro-F1 | Accuracy | Spearman rho | p-value | Direction | Fallback/100 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `entropy_hard` | False | 0.2532 | 44.32% | 0.1231 | 2.5328e-01 | +1 | 0 |
| `log_marginal_hard` | False | 0.3340 | 32.95% | 0.3368 | 1.3334e-03 | -1 | 0 |
| `entropy_soft` | True | 0.4164 | 45.45% | 0.2160 | 4.3269e-02 | +1 | 0 |
| `log_marginal_soft` | True | 0.3681 | 40.91% | 0.1340 | 2.1308e-01 | -1 | 0 |
| `joint_2d_hard` (local-fit, exp_12/22 pattern) | False | 0.3140 | 35.23% | 0.2844 | 7.2336e-03 | — | — |
| **`exp_17` baseline (Composite Fuzzy ICI)** | — | **0.4470** | **57.95%** | **0.2790** | 0.0085 | — | — |

## Known-Pitfall Checklist (see DESIGN.md Sec.6)
- Class totals reported as 54 yes / 34 no throughout (not exp_13's incorrect 56/32).
- `results/git_commit.txt` written before this run.
- Amplitude round-trip assertion passed during Phase A (checked once per arm, split 0 / config 0).
