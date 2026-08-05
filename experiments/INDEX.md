# Experiments Index — Pathology Reasoning Project

Layout: each `exp_<n>/` holds `DESIGN.md` (research design) → `IMPLEMENTATION.md` (build plan)
→ `results/<condition>/` (runs) → `reports/summary.md` (write-up). See any `DESIGN.md` for detail.

| Exp | Title | Status | Hypothesis (1 line) | Verdict | Date |
|-----|-------|--------|---------------------|---------|------|
| [exp_1](exp_1/DESIGN.md) | EDA on Task 1 Cohort | Draft | Characterizing Task 1 data distributions, missingness, and embeddings will reveal key cohort properties for training. | — | 2026-07-20 |
| [exp_2](exp_2/DESIGN.md) | Tabular Data Preprocessing | Draft | Preprocessing and tabularizing all Task 1 multimodal sources to five synchronized CSV files with 195 cases will enable model training. | — | 2026-07-20 |
| [exp_3](exp_3/DESIGN.md) | Deep EDA on Preprocessed Data | Draft | Analyzing text lengths, tabular missingness, and visual t-SNE clustering on preprocessed Task 1 data will validate modeling viability. | — | 2026-07-20 |
| [exp_4](exp_4/DESIGN.md) | Monte Carlo CV Split Design | Complete | A 100-split stratified MCCV design will provide a more stable, lower-variance validation estimate than 5-Fold CV. | Success | 2026-08-04 |
| [exp_5](exp_5/DESIGN.md) | Tabular KNN Sweep & LOOCV | Complete | Grid sweeping KNN parameters over 100 MCCV splits and final LOOCV will optimize unimodal tabular performance. | Success | 2026-08-04 |
| [exp_6](exp_6/DESIGN.md) | MRI KNN Representation Sweep & LOOCV | Complete | Evaluating raw, PCA, EmbedKit, and correlation pruning on MRI embeddings over 100 MCCV splits and LOOCV will find the optimal representation. | Success | 2026-08-04 |
| [exp_7](exp_7/DESIGN.md) | Text TF-IDF KNN Representation Sweep & LOOCV | Complete | Sweeping TF-IDF vocabulary size and dimensionality reduction methods on clinical prompts over 100 MCCV splits and LOOCV will find the optimal text representation. | Success | 2026-08-04 |
| [exp_8](exp_8/DESIGN.md) | Multimodal Late Fusion Soft-Voting LOOCV | Complete | Combining predicted probabilities from optimal Tabular, MRI, and Text KNN models via Soft Voting LOOCV will determine if trimodal integration outperforms unimodal baselines. | Success | 2026-08-04 |
| [exp_9](exp_9/DESIGN.md) | Diagnostic Confidence Prediction via ICI & Meta-Thresholds | Complete | Learning ICI decision boundaries over 100 MCCV splits and evaluating frozen meta-thresholds in LOOCV will accurately predict medical diagnostic confidence without data leakage. | Success | 2026-08-04 |
| [exp_10](exp_10/DESIGN.md) | Balanced Diagnostic Confidence Prediction via Class-Weighted ICI | Complete | Incorporating balanced class weighting during 1D Decision Tree training in Phase A (100 MCCV splits) will adjust ICI meta-thresholds to improve 3-class out-of-fold Macro-F1 under frozen LOOCV. | Success | 2026-08-05 |
| [exp_11](exp_11/DESIGN.md) | Dynamic LOOCV Diagnostic Confidence Prediction via Local Tree Thresholds | Complete | Dynamically fitting decision tree cut-points locally within each LOOCV fold will eliminate oversmoothing caused by static MCCV meta-averaging, improving 3-class out-of-fold Macro-F1. | Success | 2026-08-05 |
| [exp_12](exp_12/DESIGN.md) | Diagnostic Confidence Prediction via 3D Probability State Vector | Complete | Using the full 3D probability vector p = [p_tab, p_mri, p_text] with class-weighted Decision Trees in LOOCV will preserve modal interactions and significantly improve out-of-fold Macro-F1. | Success | 2026-08-05 |
