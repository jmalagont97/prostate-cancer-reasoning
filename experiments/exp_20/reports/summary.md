# Clinical Feature Relevance Attribution via Mode/Median Perturbation (exp_20) Summary Report

**Date**: 2026-08-05  
**Model**: Feature-Independent Class-Weighted Decision Trees on Fuzzy KNN Probability Displacements  
**Cohort**: Labeled Complete-Case Cohort ($N_{labeled} = 88$)  

## 1. Feature-Independent Meta-Thresholds (100 MCCV Splits)
| Clinical Feature | Meta-Threshold 1 ($\bar{\tau}_1$) | Meta-Threshold 2 ($\bar{\tau}_2$) | Meta-Threshold 3 ($\bar{\tau}_3$) |
|:---|:---:|:---:|:---:|
| **`age`** | `0.0737` | `0.2450` | `0.4031` |
| **`psa`** | `0.3383` | `0.3850` | `0.4515` |
| **`vol`** | `0.0650` | `0.2069` | `0.3887` |
| **`pirads`** | `0.1356` | `0.4169` | `0.7041` |
| **`psad`** | `0.1955` | `0.5216` | `0.5767` |
| **`psav`** | `0.3728` | `0.6206` | `0.6759` |
| **`psap`** | `0.3971` | `0.4450` | `0.5055` |
| **`dre`** | `0.1269` | `0.3275` | `0.5575` |

## 2. Frozen LOOCV Out-of-Fold Performance Summary
| Clinical Feature | Spearman Rank $\rho$ | p-value | 4-Class Macro-F1 | Accuracy | Mean Displacement $\Delta p$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`age`** | `0.1965` | `6.6549e-02` | `0.1261` | `18.18%` | `0.2443` |
| **`psa`** | `0.1840` | `8.6202e-02` | `0.0511` | `2.27%` | `0.0114` |
| **`vol`** | `0.1694` | `1.1467e-01` | `0.1020` | `9.09%` | `0.1932` |
| **`pirads`** | `-0.1660` | `1.2217e-01` | `0.1735` | `22.73%` | `0.3565` |
| **`psad`** | `-0.0033` | `9.7581e-01` | `0.0578` | `11.36%` | `0.0298` |
| **`psav`** | **`0.2617`** | `1.3788e-02` | `0.0891` | `3.41%` | `0.0227` |
| **`psap`** | `-0.2052` | `5.5131e-02` | `0.0000` | `0.00%` | `0.0085` |
| **`dre`** | **`0.4290`** | `3.0427e-05` | `0.1272` | `7.95%` | `0.1293` |

## 3. Independent 4x4 Confusion Matrices per Clinical Feature (LOOCV)

### Feature: `age` (Spearman $\rho = 0.1965$, Macro-F1 = `0.1261`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 0 | 0 | 0 | 1 |
| **noted (1)** | 19 | **0** | 6 | 3 | 28 |
| **important (2)** | 26 | 2 | **14** | 14 | 56 |
| **decisive (3)** | 2 | 0 | 0 | **1** | 3 |

### Feature: `psa` (Spearman $\rho = 0.1840$, Macro-F1 = `0.0511`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 0 | 0 | 0 | 1 |
| **noted (1)** | 29 | **0** | 0 | 0 | 29 |
| **important (2)** | 48 | 0 | **0** | 0 | 48 |
| **decisive (3)** | 9 | 0 | 0 | **1** | 10 |

### Feature: `vol` (Spearman $\rho = 0.1694$, Macro-F1 = `0.1020`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **2** | 1 | 0 | 1 | 4 |
| **noted (1)** | 45 | **3** | 11 | 8 | 67 |
| **important (2)** | 8 | 0 | **2** | 5 | 15 |
| **decisive (3)** | 0 | 0 | 1 | **1** | 2 |

### Feature: `pirads` (Spearman $\rho = -0.1660$, Macro-F1 = `0.1735`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 0 | 0 | 0 | 1 |
| **noted (1)** | 0 | **0** | 0 | 0 | 0 |
| **important (2)** | 11 | 13 | **10** | 10 | 44 |
| **decisive (3)** | 21 | 9 | 4 | **9** | 43 |

### Feature: `psad` (Spearman $\rho = -0.0033$, Macro-F1 = `0.0578`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **9** | 0 | 0 | 0 | 9 |
| **noted (1)** | 48 | **1** | 0 | 2 | 51 |
| **important (2)** | 23 | 1 | **0** | 0 | 24 |
| **decisive (3)** | 4 | 0 | 0 | **0** | 4 |

### Feature: `psav` (Spearman $\rho = 0.2617$, Macro-F1 = `0.0891`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **1** | 0 | 0 | 0 | 1 |
| **noted (1)** | 29 | **0** | 0 | 0 | 29 |
| **important (2)** | 48 | 0 | **0** | 0 | 48 |
| **decisive (3)** | 8 | 0 | 0 | **2** | 10 |

### Feature: `psap` (Spearman $\rho = -0.2052$, Macro-F1 = `0.0000`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **0** | 0 | 0 | 1 | 1 |
| **noted (1)** | 29 | **0** | 0 | 0 | 29 |
| **important (2)** | 48 | 0 | **0** | 0 | 48 |
| **decisive (3)** | 10 | 0 | 0 | **0** | 10 |

### Feature: `dre` (Spearman $\rho = 0.4290$, Macro-F1 = `0.1272`)
| Ground Truth \ Predicted | not_used (0) | noted (1) | important (2) | decisive (3) | Total Real |
|:---|:---:|:---:|:---:|:---:|:---:|
| **not_used (0)** | **2** | 0 | 0 | 0 | 2 |
| **noted (1)** | 56 | **0** | 2 | 5 | 63 |
| **important (2)** | 11 | 2 | **4** | 5 | 22 |
| **decisive (3)** | 0 | 0 | 0 | **1** | 1 |

