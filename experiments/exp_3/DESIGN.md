# Experiment Design: Task 1 Cohort Deep Exploratory Data Analysis (Deep EDA)

**Experiment**: experiments/exp_3/  
**Project**: pathology-reasoning  
**Date**: 2026-07-20  
**Author**: Co-Investigator (Gemini Expert on Digital Pathology & Deep Learning)  
**Status**: Draft

---

## 1. Hypothesis
Deep analysis of text narrative lengths, tabular feature missingness rates, and t-SNE projections of MRI embeddings on the preprocessed tabular data will reveal structural properties and visual class separability, validating the feasibility of a similarity-based classifier.

## 2. Experimental Setup
*   **Dataset**: Preprocessed Task 1 CSV files in `data/chimera26/preprocessed/task1/`.
*   **Analysis Code**: Python script to be implemented in `experiments/exp_3/scripts/deep_eda.py`.
*   **Hardware**: CPU for general data loading, text tokenization, and missingness counting; GPU/CPU for t-SNE projection.
*   **Seeds**: Random seed 42 fixed for t-SNE visualization reproducibility.

## 3. File Layout for This Experiment
```
experiments/exp_3/
├── DESIGN.md                  ← this file (experiment design only)
├── scripts/
│   └── deep_eda.py            ← Deep EDA analysis script (decided in implementation plan)
├── results/
│   └── deep_eda_metrics.json  ← output stats (missingness rates, word count ranges, Silhouette score)
└── reports/
    ├── figures/
    │   ├── text_length_dist.png   ← Histogram showing word count distributions of prompts
    │   ├── missingness_rates.png  ← Bar chart showing missingness percentage per tabular feature
    │   └── tsne_mri.png           ← 2D t-SNE scatter plot of MRI embeddings colored by biopsy decision
    └── summary.md             ← final deep EDA summary report
```

All analysis scripts and outputs reference paths relative to the experiment root.

## 4. Baselines
*   **Baseline**: N/A (Diagnostic Analysis).

## 5. Proposed Conditions (Analyses)
We will execute four diagnostic checks on the preprocessed CSV files:

*   **Condition 1: Target Balance Audit**
    Analyze `biopsy_decision.csv` to count target class labels (`yes` vs. `no`) and verify that unlabeled test cases are correctly set to `NONE`.
*   **Condition 2: Text Length Distribution Analysis**
    Parse `clinical_prompts.csv` and `clinical_reasoning.csv`, split text columns by whitespace, and compute length statistics (min, max, mean, median, standard deviation) for `clinical_prompt_text` and `reasoning_text`. Generate histograms showing the distributions.
*   **Condition 3: Feature Missingness (Absenteeism) Analysis**
    Parse `clinical_data_tabular.csv` and `clinical_reasoning.csv` to count `'NONE'` values for each variable (columns). Compute and plot missingness percentages per field.
*   **Condition 4: t-SNE Representation Clustering Visualization**
    Load `mri_embeddings.csv` (1024 float columns). Project the embeddings to a 2D space using t-SNE (`random_state=42`, `perplexity=30`). Color the points by target labels from `biopsy_decision.csv`:
    *   `yes` -> Distinct color (e.g. Orange)
    *   `no` -> Distinct color (e.g. Blue)
    *   `NONE` (Test partition) -> Gray

## 6. Evaluation Protocol
*   **Primary Metrics**:
    *   **Word Count Statistics**: Min, max, mean, median, and std of prompt/reasoning texts.
    *   **Missingness Rates**: Percentage of `'NONE'` values per column.
    *   **Clustering Quality**: Silhouette score of the 2D t-SNE coordinates for the labeled cases to quantify visual separation.
*   **Output Files**:
    *   `results/deep_eda_metrics.json`
    *   `reports/figures/text_length_dist.png`
    *   `reports/figures/missingness_rates.png`
    *   `reports/figures/tsne_mri.png`
    *   `reports/summary.md`

## 7. Expected Results & Decision Rules
*   **Imputation Decision Rule**: If missingness in critical clinical variables (e.g. `pirads`, `psa`) is $>10\%$, we must implement imputation models (like KNNImputer) in the classification pipeline.
*   **Similarity Classifier Decision Rule**: If the t-SNE projection shows clear separation between target classes (`yes` vs. `no`) with a Silhouette score $>0.05$, a simple similarity-based classifier is mathematically viable. If the score is $\le 0$, multimodal fusion is strictly required.

## 8. Risks & Mitigations
*   **Risk**: Data leakage from test (unlabeled) cases during t-SNE.  
    *   **Mitigation**: t-SNE will be run in an unsupervised fashion without target labels. During visualization, test cases will be plotted in a neutral color (`gray`) to prevent leakage of any diagnostic target, and clustering metrics (Silhouette score) will be computed strictly on the labeled training partition.

## 9. Reproducibility Checklist
- [ ] Random seeds fixed for t-SNE (`random_state=42`)
- [ ] Working tree clean at run time
- [ ] **Git commit hash recorded** to `results/eda/git_commit.txt` before execution

## 10. Next Steps
1.  Review and accept this experiment design plan.
2.  Once approved, produce an **implementation plan** (in plan mode) to create `experiments/exp_3/scripts/deep_eda.py` to process variables and generate plots.
