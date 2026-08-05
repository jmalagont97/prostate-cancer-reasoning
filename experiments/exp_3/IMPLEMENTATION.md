# Implementation Plan: Task 1 Cohort Deep Exploratory Data Analysis (exp_3)

This document details the build plan for the deep exploratory data analysis (Deep EDA) script and reports of **exp_3**.

---

## 1. Scope & Script Location

We will create a Python script at `experiments/exp_3/scripts/deep_eda.py` which:
1.  Loads the 5 preprocessed CSV files from `data/chimera26/preprocessed/task1/`.
2.  Computes class distributions, prompt/reasoning word count ranges, and missingness rates.
3.  Executes unsupervised t-SNE on MRI embeddings (ignoring `patient_id` and cases with missing embeddings) and colors cases by label.
4.  Calculates the Silhouette score of the 2D t-SNE coordinates for labeled cases.
5.  Saves metrics JSON to `experiments/exp_3/results/deep_eda_metrics.json`.
6.  Generates three matplotlib plots in technical English, saved to `experiments/exp_3/reports/figures/`.
7.  Generates a summary Markdown report saved to `experiments/exp_3/reports/summary.md`.

---

## 2. Directory & Files to Create

```
experiments/exp_3/
├── scripts/
│   └── deep_eda.py            ← Main Deep EDA analysis script
├── results/
│   └── deep_eda_metrics.json  ← Extracted metrics (written by deep_eda.py)
└── reports/
    ├── figures/
    │   ├── text_length_dist.png   ← Histogram showing word counts
    │   ├── missingness_rates.png  ← Bar chart showing missingness rates
    │   └── tsne_mri.png           ← t-SNE scatter plot
    └── summary.md             ← final deep EDA summary report (written by deep_eda.py)
```

---

## 3. Detailed Logic of `deep_eda.py`

### A. Data Loading
*   Read the five CSV files into Pandas DataFrames:
    *   `mri_embeddings.csv`
    *   `clinical_prompts.csv`
    *   `clinical_data_tabular.csv`
    *   `clinical_reasoning.csv`
    *   `biopsy_decision.csv`

### B. Analytical Logic
*   **Target Distribution:** Analyze column `biopsy_decision` in `biopsy_decision.csv`.
*   **Text Token Lengths:**
    *   For `clinical_prompt_text` (in `clinical_prompts.csv`) and `reasoning_text` (in `clinical_reasoning.csv`), count words by splitting on whitespace (`split()`). Ignore cases that are `'NONE'`.
    *   Compute: min, max, mean, median, standard deviation.
*   **Tabular Missingness:**
    *   For each column in `clinical_data_tabular.csv` and `clinical_reasoning.csv`, count occurrences of the string `'NONE'`.
    *   Compute missingness percentage = `(none_count / 195) * 100`.
*   **t-SNE Representation Visualization:**
    *   Extract feature columns (`mri_feat_0` to `mri_feat_1023`) from `mri_embeddings.csv`.
    *   Identify and exclude cases where features are `'NONE'` (4 cases).
    *   Run `sklearn.manifold.TSNE` with parameters `n_components=2`, `perplexity=30`, `random_state=42`, `init='pca'`.
    *   Compute the Silhouette score of the 2D coordinates for the labeled cases (excluding `'NONE'` targets) using `sklearn.metrics.silhouette_score`.

### C. Plotting with Matplotlib
All plots will be styled simply with a white background and technical English labels:
1.  **Text Length Distribution (`text_length_dist.png`):**
    *   Two subplots: Histogram of word count for clinical prompts, and histogram of word count for reasoning text (labeled cases only).
2.  **Missingness Rates (`missingness_rates.png`):**
    *   Horizontal or vertical bar chart showing percentage of `'NONE'` values for each tabular variable.
3.  **t-SNE Embeddings Plot (`tsne_mri.png`):**
    *   2D scatter plot colored by `yes` (e.g. orange), `no` (e.g. blue), and `NONE` (e.g. gray). Include the Silhouette score in the title or legend.

---

## 4. Run Command

Execute in the active Conda environment:
```bash
python3 experiments/exp_3/scripts/deep_eda.py \
    --data_dir data/chimera26/preprocessed/task1 \
    --results_dir experiments/exp_3/results \
    --reports_dir experiments/exp_3/reports
```
