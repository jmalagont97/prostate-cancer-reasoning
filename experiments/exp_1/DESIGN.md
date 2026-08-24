# Experiment Design: Baseline Benchmarking on Pre-Biopsy Decision (Task 1)

**Experiment**: experiments/exp_1/  
**Project**: prostate-cancer-reasoning  
**Date**: 2026-08-22  
**Author**: Juan Sebastian Malagon Torres  
**Status**: Complete

---

## 1. Hypothesis
The official CHIMERA-Agent baseline evaluated on Task 1 (pre-biopsy decision) against the released training ground-truth cohort provides a measurable lower-bound benchmark on decision accuracy ($S_{\text{gate}}$), reasoning alignment ($S_{\text{conf}}, S_{\text{weights}}, F_{1\text{-vars}}$), tool efficiency ($S_{\text{tool}}$), and official ranking score ($R_1$).

## 2. Experimental Setup
- **Dataset(s)**: `data/raw/train_release/task1/` (91 paired ground-truth cases out of 195 raw cases).
- **Model/Algorithm**: Official ReAct Baseline (`data/base-line/task1/`).
- **Evaluation Package**: `chimera_metrics` v0.1.0 in `../../utils/pkg/chimera_metrics/`.
- **Hardware**: CPU / Local Linux environment.

## 3. File Layout for This Experiment
```
experiments/exp_1/
├── DESIGN.md                  ← This design document (Status: Complete)
├── IMPLEMENTATION.md          ← Implementation plan
├── scripts/                   ← Evaluation script (evaluate_baseline_task1.py)
├── results/
│   └── baseline/              ← Artifacts (case_metrics.json, aggregate_metrics.json, version.txt)
└── reports/
    └── summary.md             ← Final scientific benchmark report
```

## 4. Baselines & Experimental Conditions
| Condition ID | Description | Input Source | Expected Role |
|---|---|---|---|
| `cond_0_oracle` | Ground Truth vs. Ground Truth | `data/raw/train_release/task1/` | Upper Bound (1.0000) |
| `cond_1_baseline` | ReAct Baseline vs. Ground Truth | `data/base-line/task1/` | Baseline Benchmark |

## 5. Evaluation Protocol
- **Primary Metric**: Task 1 Official Ranking Score: $R_1 = 0.5 \cdot \overline{S}_{\text{case}} + 0.5 \cdot F_{1\text{-yes}}$.
- **Secondary Metrics**: $S_{\text{gate}}$, $S_{\text{conf}}$, $S_{\text{weights}}$, $F_{1\text{-vars}}$, $S_{\text{tool}}$, $S_{\text{grounding}}$.
- **Results Output**: `experiments/exp_1/results/baseline/`.

## 6. Reproducibility Checklist
- [x] Evaluation code modularized and version-pinned in `utils/pkg/chimera_metrics`
- [x] Evaluation run script committed in `experiments/exp_1/scripts/`
- [x] Raw case metrics and aggregate metrics logged in `results/baseline/`
- [x] Version metadata recorded in `results/baseline/version.txt`
- [x] Final report generated in `reports/summary.md`
