# Implementation Plan: Task 1 Baseline Benchmarking (`exp_1`)

**Experiment**: experiments/exp_1/  
**Project**: prostate-cancer-reasoning  
**Author**: Juan Sebastian Malagon Torres  
**Date**: 2026-08-22  

---

## 1. Scope & Implementation Overview
This plan specifies the data loading, metric computation pipeline, and persistence mechanisms for benchmarking the official CHIMERA-Agent baseline on Task 1 against the supervised ground-truth cohort in `data/raw/train_release/task1/`.

---

## 2. Directory Layout & Artifacts
```
experiments/exp_1/
├── DESIGN.md
├── IMPLEMENTATION.md
├── scripts/
│   └── evaluate_baseline_task1.py   ← Python execution script
├── results/
│   └── baseline/
│       ├── case_metrics.json         ← Full per-case evaluation outputs
│       ├── aggregate_metrics.json    ← Dataset-level statistics & ranking scores
│       ├── oracle_metrics.json       ← Ground truth upper-bound verification
│       └── version.txt               ← Environment & code provenance
└── reports/
    └── summary.md                    ← Generated research report
```

---

## 3. Data Flow & Execution Steps
1. **Discovery & Pairing:**
   - Scan `data/raw/train_release/task1/` for cases containing both `prostate-biopsy-decision.json` and `prostate-biopsy-decision-reasoning.json` (91 cases).
   - Match each case to `data/base-line/task1/<case_id>/prediction.json`.
2. **Evaluation Execution (`chimera_metrics`):**
   - For each paired case:
     - Load ground-truth payload: `{ "biopsy_decision": gt_dec, **gt_reas }`.
     - Load baseline prediction: `pred_dict`.
     - Evaluate `Decision Gate`, `Confidence Score`, `Variable Weights MAE`, `Important/Decisive Factors Set-F1`, `Cost-Aware Tool Efficiency`, and `Section Grounding`.
     - Compute individual case composite score ($S_{\text{case}}$).
3. **Aggregate Scoring & Ranking:**
   - Calculate Mean Case Score ($\overline{S}_{\text{case}}$).
   - Calculate Binary Decision Accuracy and $F_{1\text{-yes}}$.
   - Compute Task 1 Official Ranking Score: $R_1 = 0.5 \times \overline{S}_{\text{case}} + 0.5 \times F_{1\text{-yes}}$.
   - Compute Decision Gate Pass Rate and Quadratic Cohen's Kappa ($\kappa_{\text{conf}}, \kappa_{\text{weights}}$).
4. **Oracle Upper-Bound Control:**
   - Run identical evaluation pairing ground truth with itself (`cond_0_oracle`) to verify metric normalization and maximum bounds ($R_1 = 1.0$).
5. **Persistence:**
   - Save JSON outputs to `results/baseline/` and generate report in `reports/summary.md`.
