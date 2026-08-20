# Experiment Design: Monte Carlo Cross-Validation (MCCV) 100-Split Validation Design
**Experiment**: experiments/exp_4/ · **Project**: pathology-reasoning · **Date**: 2026-08-04 · **Status**: Complete

---

## 1. Hypothesis
A 100-split Monte Carlo Cross-Validation (MCCV) partition strategy (specifically splitting the 88 complete labeled cases into 70 train and 18 validation samples per split) will provide a more stable, lower-variance estimate of classifier performance compared to standard 5-Fold Cross-Validation, while ensuring complete isolation of the 102 blind test cases.

## 2. Experimental Setup & Data Audit
- **Multimodal Data Sources**:
  - Text prompts: `data/chimera26/preprocessed/task1/clinical_prompts.csv`
  - Biopsy decisions (targets): `data/chimera26/preprocessed/task1/biopsy_decision.csv`
  - Tabular features: `data/chimera26/preprocessed/task1/clinical_data_tabular.csv`
  - MRI embeddings: `data/chimera26/preprocessed/task1/mri_embeddings.csv`
- **Audit Rule**:
  - A patient is eligible for modeling ONLY if they have complete records across all four modalities: text prompts $\neq$ `'NONE'`, biopsy decision $\neq$ `'NONE'` (for training/validation split), tabular data $\neq$ `'NONE'`, and MRI embeddings $\neq$ `'NONE'`.
  - Patients missing any modality (e.g. missing MRI embeddings) must be excluded from the active training/validation set.
- **Partitioning Strategy (Monte Carlo Cross-Validation)**:
  - Total labeled samples $N_{labeled} = 88$.
  - Generates $B = 100$ independent random train/validation splits (columns in `mccv_design.csv`).
  - Each split $b \in \{0, \dots, 99\}$ will partition the labeled samples into:
    - **Train split**: 70 cases (assigned index $0$, corresponding to training).
    - **Validation split**: 18 cases (assigned index $1$, corresponding to validation).
  - Class stratification must be preserved in both train and validation splits to maintain target class ratio.
  - **Blind Test Isolation**:
    - Unlabeled test cases ($N_{test} = 102$, identified by `biopsy_decision` = `'NONE'` or fold = `-1` in original design) must be assigned index `-1` in all 100 columns, ensuring they are completely untouched during training and validation loops.

## 3. File Layout for This Experiment
```
experiments/exp_4/
├── DESIGN.md                  ← this file (experiment design)
├── scripts/
│   └── generate_folds.py     ← partition generation script (decided in implementation plan)
├── results/
│   └── mccv_design.csv       ← output MCCV split file (190 rows, patient_id + 100 columns)
└── reports/
    ├── figures/
    │   └── split_distributions.png  ← class and size distribution charts
    └── summary.md             ← write-up of the partition audit
```

## 4. Proposed Conditions
| Condition | Stratification | Number of Splits (B) | Train/Val Ratio | Missing Case Handing | Output File |
|:---|:---:|:---:|:---:|:---:|:---|
| **MCCV-100-Stratified** | Yes | 100 | 70 / 18 (79.5% / 20.5%) | Excluded (5 cases) | `results/mccv_design.csv` |

## 5. Evaluation Protocol
- **Auditing Checks**:
  - The script must verify that the 190 complete patients are exactly aligned across all modalities.
  - Programmatically assert that all 5 problematic patients with missing values are excluded from training splits.
  - Programmatically assert that the output CSV contains exactly 190 rows (aligned with preprocessed patient IDs).
- **Stratification Validation**:
  - Compare the class ratio (Biopsy Yes / No) of the training splits ($70$ samples) and validation splits ($18$ samples) against the overall cohort ratio ($56/34 \approx 1.647$) to ensure low variance across all 100 runs.
- **Verification of Output**:
  - Validate that the output CSV is shaped `(190, 101)` (1 ID column + 100 split columns).
  - Verify that train indices are `0`, validation indices are `1`, and test indices are `-1`.

## 6. Risks & Mitigations
- **Risk: Patient Misalignment**: If patient IDs are shuffled during concatenation or vector matching, data leakage or target misalignment occurs.
  - *Mitigation*: The script must strictly check alignment using patient IDs before performing the split.
- **Risk: Random Seed Variance**: Without fixing the seed, regenerating the partitions will yield different splits, violating reproducibility.
  - *Mitigation*: Fix random state seed globally at `42` for all random generation operations.

## 7. Reproducibility Checklist
- [x] Random seeds fixed (`random_state=42`)
- [ ] Split generation script placed under `scripts/`
- [ ] Output CSV saved to `results/mccv_design.csv`
- [ ] Working tree clean before generating partition
- [ ] **Git commit hash recorded** — run `git log -1 --format="%H %s" > results/git_commit.txt`

## 8. Next Steps
1. Review and accept this experiment plan (auditing rules, split parameters, design output format).
2. Once accepted, produce an **implementation plan** (in plan mode) to write `scripts/generate_folds.py` and run it to produce `results/mccv_design.csv`.
