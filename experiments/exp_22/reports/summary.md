# Multivariate SHAP Vector Input Decision Tree Clinical Relevance Attribution (exp_22) Summary Report

**Date**: 2026-08-06  
**Model**: Multivariate 6D SHAP Vector `DecisionTreeClassifier(max_depth=3, class_weight='balanced')`  
**Validation Protocol**: Direct Out-of-Fold LOOCV ($N = 88$)  

## 1. Frozen LOOCV Out-of-Fold Performance Summary per Feature
| Clinical Feature | Spearman Rank $\rho$ | p-value | 4-Class Macro-F1 | Accuracy | Mean \|SHAP\| |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`age`** | `0.0741` | `4.9263e-01` | `0.2609` | `42.05%` | `0.1058` |
| **`psa`** | **`0.3738`** | `3.3345e-04` | `0.3446` | `45.45%` | `0.0322` |
| **`vol`** | `-0.0141` | `8.9643e-01` | `0.2211` | `35.23%` | `0.0844` |
| **`pirads`** | **`0.3019`** | `4.2523e-03` | `0.3991` | `61.36%` | `0.1233` |
| **`psad`** | **`-0.2918`** | `5.8004e-03` | `0.0233` | `2.27%` | `0.0275` |
| **`dre`** | `0.0023` | `9.8293e-01` | `0.2463` | `59.09%` | `0.0868` |

## 2. Independent 4x4 Confusion Matrices per Clinical Feature (LOOCV)

### Feature: `age` (Spearman $\rho = 0.0741$, Macro-F1 = `0.2609`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 0 | 1 | 0 | 1 |
| **noted (1)** | 0 | **21** | 4 | 3 | 28 |
| **important (2)** | 2 | 32 | **15** | 7 | 56 |
| **decisive (3)** | 0 | 2 | 0 | **1** | 3 |

### Feature: `psa` (Spearman $\rho = 0.3738$, Macro-F1 = `0.3446`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 1 | 0 | 0 | 1 |
| **noted (1)** | 2 | **22** | 1 | 4 | 29 |
| **important (2)** | 0 | 31 | **12** | 5 | 48 |
| **decisive (3)** | 0 | 2 | 2 | **6** | 10 |

### Feature: `vol` (Spearman $\rho = -0.0141$, Macro-F1 = `0.2211`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 3 | 0 | 0 | 4 |
| **noted (1)** | 4 | **23** | 38 | 2 | 67 |
| **important (2)** | 0 | 8 | **7** | 0 | 15 |
| **decisive (3)** | 1 | 1 | 0 | **0** | 2 |

### Feature: `pirads` (Spearman $\rho = 0.3019$, Macro-F1 = `0.3991`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 0 | 0 | 1 | 1 |
| **noted (1)** | 0 | **0** | 0 | 0 | 0 |
| **important (2)** | 2 | 0 | **38** | 4 | 44 |
| **decisive (3)** | 0 | 0 | 27 | **16** | 43 |

### Feature: `psad` (Spearman $\rho = -0.2918$, Macro-F1 = `0.0233`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **2** | 3 | 3 | 1 | 9 |
| **noted (1)** | 19 | **0** | 30 | 2 | 51 |
| **important (2)** | 12 | 12 | **0** | 0 | 24 |
| **decisive (3)** | 1 | 1 | 2 | **0** | 4 |

### Feature: `dre` (Spearman $\rho = 0.0023$, Macro-F1 = `0.2463`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 1 | 1 | 0 | 2 |
| **noted (1)** | 3 | **47** | 11 | 2 | 63 |
| **important (2)** | 0 | 17 | **5** | 0 | 22 |
| **decisive (3)** | 0 | 1 | 0 | **0** | 1 |

