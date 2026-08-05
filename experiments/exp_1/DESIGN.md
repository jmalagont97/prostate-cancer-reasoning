# Experiment Design: Task 1 Cohort Composition Analysis (Simplified EDA)

**Experiment**: experiments/exp_1/  
**Project**: pathology-reasoning  
**Date**: 2026-07-20  
**Author**: Co-Investigator (Gemini Expert on Digital Pathology & Deep Learning)  
**Status**: Draft

---

## 1. Hypothesis
Quantifying the file presence and active modality distributions across the Task 1 cohort will establish data completeness rates and confirm the exact subset of patients possessing each source of information (structured clinical text, tabular data, labels, and MRI embeddings).

## 2. Experimental Setup
*   **Dataset**: Chimera26 Task 1 raw dataset located at `data/chimera26/raw/task1/`.
*   **Analysis Code**: Python statistics script to be implemented in `experiments/exp_1/scripts/eda.py`.
*   **Hardware**: CPU for scanning directories and parsing JSON keys.

## 3. File Layout for This Experiment
```
experiments/exp_1/
├── DESIGN.md                  ← this file (experiment design only)
├── scripts/
│   └── eda.py                 ← Analysis script (decided in implementation plan)
├── results/
│   └── eda_metrics.json       ← output metrics and stats (file presence, counts per modality)
└── reports/
    ├── figures/
    │   ├── file_presence.png  ← Bar chart showing the absolute count and percentage of files present
    │   └── active_sources.png ← Bar chart showing the presence of active information sources
    └── summary.md             ← final data composition summary report
```

## 4. Baselines
*   **Baseline**: N/A (Exploratory Data Analysis).

## 5. Proposed Conditions (Analyses)
We will execute a simplified cohort characterization across the 195 patient folders in Task 1:

*   **Condition 1: File Existence & Ingestion Audit**
    Scan all patient subdirectories under `task1/` and log the existence (True/False) of the key files:
    1.  `prostate-biopsy-decision.json` (Target label)
    2.  `prostate-biopsy-decision-reasoning.json` (Rationales)
    3.  `prostate-biopsy-decision-clinical-data.json` (Raw reports)
    4.  `prostate-modality-level-neural-representations.json` (Embeddings)
    5.  `structured-prompt.json` (Tabularized features)
*   **Condition 2: Modality & Source Presence Counts**
    Count the number of patients that possess active representations for:
    1.  MRI image embeddings (non-null entries in representations JSON)
    2.  Clinical narrative text (note sections present in structured prompt)
    3.  Clinical lab data (lab entries present in clinical data JSON)
    4.  Decision labels (yes/no targets present)
*   **Condition 3: Visual Reporting (Bar Charts)**
    Generate bar charts using Python (`matplotlib`/`seaborn`) to visually present the counts:
    1.  A bar chart for overall file completeness and presence rates.
    2.  A bar chart for active input sources (MRI, structured prompt, lab data, etc.).

## 6. Evaluation Protocol
*   **Primary Metrics**:
    *   **File Presence Counts**: Absolute counts and percentages of folders containing each of the five target files.
    *   **Modality Completeness Rate**: Absolute counts of patients with active MRI embeddings, labels, and text data.
*   **Output Files**:
    *   `results/eda_metrics.json`
    *   `reports/figures/file_presence.png`
    *   `reports/figures/active_sources.png`
    *   `reports/summary.md`

## 7. Expected Results & Decision Rules
*   **Decision Rule**: If the completeness rate for core inputs (MRI embeddings, structured prompt clinical text) is $<100\%$ on the training split, we must design a filter to exclude incomplete cases or implement fallback features in our model training implementation plan.

## 8. Risks & Mitigations
*   **Risk**: Minor script crashes on missing or corrupted JSON files.  
    *   **Mitigation**: Implement robust try-except blocks during directory scanning to log any corrupted files without stopping execution.

## 9. Reproducibility Checklist
- [ ] Working tree clean at run time
- [ ] **Git commit hash recorded** to `results/eda/git_commit.txt` before launching

## 10. Next Steps
1.  Review and accept this simplified experiment design plan.
2.  Once approved, produce an **implementation plan** (in plan mode) to create `experiments/exp_1/scripts/eda.py` to count file existence, modality presence, and generate bar charts.
