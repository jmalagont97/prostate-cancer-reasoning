# Experiment Design: Multimodal Late Fusion (Soft-Voting Ensemble) LOOCV Evaluation
**Experiment**: experiments/exp_8/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Complete

---

## 1. Hypothesis
Combining predicted probability distributions from the optimal unimodal models across all three modalities (Tabular, MRI Embeddings, and spaCy-lemmatized Text Prompts) via a Soft-Voting Late Fusion ensemble will achieve higher out-of-fold generalization performance (LOOCV Macro-F1 $\ge 0.72$) and higher specificity than any single unimodal model (best unimodal: Text LOOCV Macro-F1 = 0.6988) or bimodal combination.

## 2. Experimental Setup
- **Dataset**:
  - Tabular Data: `data/chimera26/preprocessed/task1/tabular_imputed.csv`
  - MRI Embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`
  - Clinical Prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`
  - Targets: `data/chimera26/preprocessed/task1/biopsy_decision.csv`
  - Cohort: 88 complete-case labeled patients ($N=88$, 56 biopsy positive, 32 biopsy negative).
- **Validation Harness**:
  - Leave-One-Out Cross-Validation (LOOCV, 88 folds) over the 88 complete cases.
- **Unimodal Base Classifiers (Frozen Optimal Configurations)**:
  1. **Tabular KNN Model (`exp_5`)**:
     - Preprocessing: `MinMaxScaler` (numerical) + `OneHotEncoder` (categorical `dre`).
     - Representation: Raw.
     - Classifier: KNN ($k=3$, `weights='uniform'`, `metric='euclidean'`).
  2. **MRI Embeddings Model (`exp_6`)**:
     - Preprocessing: `MinMaxScaler`.
     - Representation: `EmbedKit Supervised` (frozen `target_dim=384`).
     - Classifier: KNN ($k=3$, `weights='uniform'`, `metric='euclidean'`).
  3. **Text Prompts Model (`exp_7`)**:
     - Preprocessing: spaCy NLP pipeline (`en_core_web_sm`: lowercasing, stop words removal, punctuation removal, lemmatization) + `TfidfVectorizer(max_features=500, norm='l2')`.
     - Representation: `PCA` (90% cumulative explained variance).
     - Classifier: KNN ($k=1$, `weights='uniform'`, `metric='cosine'`).
- **Late Fusion Strategy (Soft Voting)**:
  - For each fold, obtain predicted class probabilities $P(Y=1 | M)$ for each modality $M \in \{\text{Tabular}, \text{MRI}, \text{Text}\}$.
  - Soft-Voting formula:
    $$P_{\text{fusion}} = \sum_{M} w_M \cdot P(Y=1 | M), \quad \text{where } \sum w_M = 1$$
  - Threshold final decision: $\hat{Y} = 1$ if $P_{\text{fusion}} \ge 0.50$ else $0$.

## 3. File Layout for This Experiment
```
experiments/exp_8/
├── DESIGN.md                  ← this file (experiment design)
├── IMPLEMENTATION.md          ← build plan (added in plan mode)
├── scripts/
│   └── train.py               ← Late Fusion soft-voting LOOCV script
├── results/
│   ├── loocv_metrics.json      ← final out-of-fold metrics of trimodal fusion & ablations
│   └── loocv_predictions.csv   ← final out-of-fold predictions & probabilities per modality
└── reports/
    ├── figures/
    │   ├── roc_curves.png          ← ROC curves comparing unimodal vs bimodal vs trimodal fusion
    │   └── confusion_matrix.png     ← confusion matrix of final trimodal Late Fusion
    └── summary.md             ← write-up of results and multimodal comparative analysis
```

## 4. Baselines & Unimodal Inputs
| Modality / Source | Optimal Config | LOOCV Macro-F1 | LOOCV Accuracy | LOOCV Sensitivity | LOOCV Specificity |
|:---|:---|:---:|:---:|:---:|:---:|
| **Tabular (`exp_5`)** | MinMaxScaler + OHE, KNN ($k=3$, uniform, euclidean) | 0.6333 | 0.6818 | 0.8519 | 0.4118 |
| **MRI Embeddings (`exp_6`)** | EmbedKit Sup (dim=384), KNN ($k=3$, uniform, euclidean) | 0.5335 | 0.5682 | 0.6852 | 0.3824 |
| **Text Prompts (`exp_7`)** | spaCy + TF-IDF (500) + PCA (90%), KNN ($k=1$, uniform, cosine) | **0.6988** | **0.7159** | **0.7778** | **0.6176** |

## 5. Proposed Conditions & Fusion Ablations
| Condition ID | Included Modalities | Weights Strategy ($w_{\text{Tab}}, w_{\text{MRI}}, w_{\text{Text}}$) | Decision Rule |
|:---|:---|:---|:---|
| **COND-01-Equal-Trimodal** | Tabular + MRI + Text | Equal weights ($0.333, 0.333, 0.333$) | Soft-Voting ($P \ge 0.5$) |
| **COND-02-Weighted-Trimodal** | Tabular + MRI + Text | Weighted by LOOCV Macro-F1 ($0.33, 0.28, 0.39$) | Soft-Voting ($P \ge 0.5$) |
| **COND-03-Bimodal-Tabular-Text** | Tabular + Text | Equal weights ($0.50, 0.00, 0.50$) | Soft-Voting ($P \ge 0.5$) |
| **COND-04-Bimodal-Tabular-MRI** | Tabular + MRI | Equal weights ($0.50, 0.50, 0.00$) | Soft-Voting ($P \ge 0.5$) |
| **COND-05-Bimodal-Text-MRI** | Text + MRI | Equal weights ($0.00, 0.50, 0.50$) | Soft-Voting ($P \ge 0.5$) |
| **COND-06-Grid-Weighted-Sweep** | Tabular + MRI + Text | Grid sweep $w_{\text{Tab}}, w_{\text{MRI}}, w_{\text{Text}} \in [0, 1]$ | Soft-Voting ($P \ge 0.5$) |

## 6. Evaluation Protocol
- **Primary Metric**: Macro-F1 score on Leave-One-Out Cross-Validation (88 folds).
- **Secondary Metrics**: Accuracy, Sensitivity, Specificity, AUROC.
- **Protocol**:
  1. For each of the 88 LOOCV folds, fit each of the 3 unimodal pipelines independently on 87 training cases.
  2. Transform the validation case and predict out-of-fold probability $P(Y=1|M)$ for each modality.
  3. Store out-of-fold probability matrices.
  4. Compute Late Fusion Soft Voting across all conditions (Equal Trimodal, Weighted Trimodal, Bimodal Ablations, Grid Sweep).
  5. Compute confusion matrices, ROC curves, and save metrics.

## 7. Expected Results & Decision Rules
- **If Trimodal Fusion Macro-F1 > Best Unimodal (0.6988)**:
  - Multimodal integration is mathematically validated as beneficial.
- **If Bimodal (Tabular + Text) outperforms Trimodal (Tabular + Text + MRI)**:
  - Visual MRI embeddings are identified as a noisy modality that degrades Late Fusion ensemble performance.
- **If Unimodal Text remains superior**:
  - Standalone clinical narrative text is confirmed as the single dominant modality for biopsy prediction.

## 8. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42` for all pipelines)
- [ ] Training script placed under `scripts/`
- [ ] Output metrics and plots saved to `results/` and `reports/`
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt` before execution

## 9. Next Steps
1. Review and accept this experiment plan (hypothesis, fusion strategies, LOOCV evaluation).
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/train.py` and run it to execute Late Fusion LOOCV evaluation.
