# Clinical Feature Relevance Attribution via SHAP Shapley Values (exp_21) Summary Report

**Date**: 2026-08-05  
**Model**: SHAP `KernelExplainer` on Distance-Weighted Tabular Fuzzy KNN (`exp_13`)  
**Cohort**: Complete-Case Labeled Cohort ($N = 88$)  

## 1. SHAP Feature-Independent Meta-Thresholds (100 MCCV Splits)
| Clinical Feature | Meta-Threshold 1 ($ar{\tau}_1$) | Meta-Threshold 2 ($ar{\tau}_2$) | Meta-Threshold 3 ($ar{\tau}_3$) |
|:---|:---:|:---:|:---:|
| **`age`** | `0.0061` | `0.0352` | `0.0720` |
| **`psa`** | `0.0019` | `0.0037` | `0.0137` |
| **`vol`** | `0.0209` | `0.0509` | `0.0925` |
| **`pirads`** | `0.0093` | `0.0204` | `0.0323` |
| **`psad`** | `0.0037` | `0.0108` | `0.0157` |
| **`dre`** | `0.0110` | `0.0285` | `0.1201` |

## 2. Frozen LOOCV Out-of-Fold Performance Summary
| Clinical Feature | Spearman Rank $\rho$ | p-value | 4-Class Macro-F1 | Accuracy | Mean \|SHAP\| |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`age`** | `-0.0252` | `8.1582e-01` | `0.1405` | `19.32%` | `0.1058` |
| **`psa`** | **`0.3593`** | `5.8638e-04` | `0.1728` | `26.14%` | `0.0322` |
| **`vol`** | `0.0800` | `4.5858e-01` | `0.1906` | `30.68%` | `0.0844` |
| **`pirads`** | `0.0733` | `4.9750e-01` | `0.1987` | `45.45%` | `0.1233` |
| **`psad`** | `-0.0995` | `3.5618e-01` | `0.2182` | `28.41%` | `0.0275` |
| **`dre`** | `0.2047` | `5.5739e-02` | `0.2032` | `21.59%` | `0.0868` |

## 3. Independent 4x4 Confusion Matrices per Clinical Feature (LOOCV)

### Feature: `age` (Spearman $\rho = -0.0252$, Macro-F1 = `0.1405`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 0 | 0 | 1 | 1 |
| **noted (1)** | 0 | **5** | 10 | 13 | 28 |
| **important (2)** | 3 | 13 | **10** | 30 | 56 |
| **decisive (3)** | 1 | 0 | 0 | **2** | 3 |

### Feature: `psa` (Spearman $\rho = 0.3593$, Macro-F1 = `0.1728`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 1 | 0 | 0 | 1 |
| **noted (1)** | 3 | **1** | 10 | 15 | 29 |
| **important (2)** | 0 | 1 | **12** | 35 | 48 |
| **decisive (3)** | 0 | 0 | 0 | **10** | 10 |

### Feature: `vol` (Spearman $\rho = 0.0800$, Macro-F1 = `0.1906`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 1 | 0 | 2 | 4 |
| **noted (1)** | 7 | **24** | 13 | 23 | 67 |
| **important (2)** | 2 | 4 | **1** | 8 | 15 |
| **decisive (3)** | 0 | 1 | 0 | **1** | 2 |

### Feature: `pirads` (Spearman $\rho = 0.0733$, Macro-F1 = `0.1987`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 1 | 0 | 0 | 1 |
| **noted (1)** | 0 | **0** | 0 | 0 | 0 |
| **important (2)** | 2 | 3 | **4** | 35 | 44 |
| **decisive (3)** | 5 | 1 | 1 | **36** | 43 |

### Feature: `psad` (Spearman $\rho = -0.0995$, Macro-F1 = `0.2182`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 0 | 3 | 5 | 9 |
| **noted (1)** | 11 | **18** | 5 | 17 | 51 |
| **important (2)** | 6 | 9 | **3** | 6 | 24 |
| **decisive (3)** | 0 | 1 | 0 | **3** | 4 |

### Feature: `dre` (Spearman $\rho = 0.2047$, Macro-F1 = `0.2032`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 0 | 1 | 0 | 2 |
| **noted (1)** | 4 | **6** | 45 | 8 | 63 |
| **important (2)** | 1 | 3 | **11** | 7 | 22 |
| **decisive (3)** | 0 | 0 | 0 | **1** | 1 |

