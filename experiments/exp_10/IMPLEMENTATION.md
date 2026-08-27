# Implementation Details: Trimodal Hard-KNN vs Fuzzy-KNN Late Fusion for Task 2 Treatment Recommendation (exp_10)

**Experiment**: `experiments/exp_10/`
**Protocol Governance**: `utils/pkg/chimera_metrics/chimera_metrics/task2/decision.py`
**Evaluation Standard**: `compute_treatment_decision_gate` (4-class exact match) + `sklearn.metrics` homologado para Macro/Micro/Weighted-F1, AUROC, Brier
**Date**: 2026-08-24
**Status**: Approved (DESIGN.md) — pending implementation

---

## 1. Architecture & Code Structure

```
experiments/exp_10/
├── DESIGN.md                  <- research design (approved)
├── IMPLEMENTATION.md          <- this file (implementation plan)
├── scripts/
│   ├── build_package.py       <- (no-op) guard to ensure scripts run with repo context
│   ├── preprocess_task2.py    <- JSON → canonical CSVs + cohort audit
│   ├── generate_mccv_splits.py<- 100 stratified 4-class MCCV splits
│   └── run_exp10_mccv.py      <- Phase A (grids + fusion) + Phase B (LOOCV) + metrics
├── results/
│   ├── version.txt            <- git commit SHA + execution timestamp
│   ├── aggregate_metrics.json <- means ± std per condition (Macro-F1, Micro-F1, Weighted-F1, AUROC, Brier, per-class F1)
│   ├── split_metrics.json     <- per-split metrics (100 rows)
│   ├── wilcoxon_tests.json    <- paired H1 vs H2 tests
│   └── best_hyperparams.json  <- optimal config per condition (Phase A)
└── reports/
    └── summary.md             <- final write-up (via experiment-reporter)
```

---

## 2. Script 1: `preprocess_task2.py`

**Input** (raw JSON, one folder per case under `data/raw/train_release/task2/`):

| File | Content extracted |
| :--- | :--- |
| `prostate-treatment-decision.json` | GT 4-class label (string, e.g. `"active_surveillance"`) |
| `prostate-treatment-decision-reasoning.json` | `confidence` for the GT (clear/borderline) |
| `structured-prompt.json` | Tabular features (`age, psa, months, pirads, psad, psav, psap, vol, dre, bx, cspca, bx_isup, bx_gl_prim, bx_gl_sec, ct`, `medhx` → comorbidity flags, `pmhx`, `vitals` parsed numeric fields, `ipss` parsed score) |
| `prostate-modality-level-neural-representations.json` | `"MRI image"` → 1024-d vector |
| `prostate-treatment-decision-clinical-data.json` | Free text: `radiology_report` + `previous_notes` (concatenated) + `psa_trend` stringified + `laboratory_results` stringified + `family_history` + `pathology_report` |

**Outputs** (canonical CSVs in `data/processed/task2/`):

```
data/processed/task2/
├── clinical_tabular_task2.csv   <- N=70, pid + engineered tabular columns
├── mri_embeddings_task2.csv     <- N=70, pid + emb_0..emb_1023
├── clinical_texts_task2.csv     <- N=70, pid + full_encounter_narrative
└── ground_truth_task2.csv       <- N=70, pid + treatment_decision + confidence
```

**Leakage audit (mandatory, before any training)**:
- Exclude `active_treatment_flag` from tabular features (it is the reveal target).
- Include `bx_isup`, `bx_gl_prim`, `bx_gl_sec`, `ct` — post-biopsy fields are **legitimate inputs** for Task 2 (the challenge releases them in the structured prompt).
- Log the exclusion in `results/version.txt`.

**Cohort audit log**: print per-case modal availability; assert N=70 with all 3 modalities; raise if `T2-042`/`T2-044` are not the only exclusions.

---

## 3. Script 2: `generate_mccv_splits.py`

- **100 stratified Monte Carlo Cross-Validation splits**, 70/30 (train/val) per split.
- Stratification key: 4-class `treatment_decision`.
- Guarantee per split:
  - `≥ 1` case of `watchful_waiting` in **train** (if feasible with the 2 available cases; if a random split cannot satisfy this, apply deterministic rejection sampling with `seed=42`).
  - At most 1 `watchful_waiting` in **val** per split.
- Output: `data/processed/task2/mccv_splits_100.csv` with columns `pid, split_0..split_99` using values `train`/`val`.

---

## 4. Script 3: `run_exp10_mccv.py`

### 4.1 Reused components from `experiments/exp_4/scripts/models.py`

| Class | Reuse | Adaptation for Task 2 |
| :--- | :--- | :--- |
| `MRIPreprocessor` | **Direct reuse** | `n_components=None` (H1) or `n_components=64` (grid option) |
| `TextPreprocessor` | **Direct reuse** | `max_features=500` + `n_components=None`/PCA for H1; `max_features=None` + `n_components=None` for H2 |
| `TabularPreprocessor` | **Direct reuse** (structure) | New `NUMERIC_COLS`/`CATEGORICAL_COLS` for the Task 2 structured prompt; no `spearman_tau` by default (kept as grid option `None`) |
| `CustomKNNPredictor` | **Adapted** | Must support **multiclass 4-class** targets, not binary ones |

### 4.2 New multiclass extension of `CustomKNNPredictor`

- `fit(X_train, y_multiclass: List[str], y_confidence: Optional[List[str]])`:
  - **Hard mode** (H1): one-hot encode `y_multiclass` into $K=4$ binary columns.
  - **Fuzzy mode** (H2): build 4-d soft target for each sample using the consensus formula:
    ```
    for GT class k, confidence c:
        y_k       = 0.25 + 0.75 * c        # clear → 1.00, borderline → 0.625
        y_j ≠ k   = (1 - y_k) / 3          # remaining probability mass
    ```
- `predict_proba(X_val) → np.ndarray (n_val, 4)`:
  - Compute pairwise distances to train, select $k$-neighbors.
  - Weighted (uniform or 1/d) mean of the **4-d target rows** of the neighbors.
  - L1-normalize each row to sum to 1.
- `predict(X_val) → List[str]`: `argmax` over the 4 columns.

### 4.3 Phase A: MCCV-100 unimodal grid + fusion search

| Modality | Grid | ~configs |
| :--- | :--- | :---: |
| Tabular (hard) | `k ∈ {1,3,5,7}`, weights `{uniform, distance}`, metric `{euclidean}` | 8 |
| Tabular (fuzzy) | `k ∈ {1,3,5}`, weights `{uniform, distance}`, metric `{euclidean}` | 6 |
| MRI (hard) | `k ∈ {1,3,5}`, PCA `{None}`, weights `{uniform, distance}`, metric `{euclidean}` | 6 |
| MRI (fuzzy) | `k ∈ {1,3,5}`, PCA `{None}`, weights `{uniform, distance}`, metric `{euclidean}` | 6 |
| Text (hard) | `max_features {500}`, SVD `{None}`, `k ∈ {1,3}`, weights `{uniform}`, metric `{cosine}` | 2 |
| Text (fuzzy) | `max_features {None}`, SVD `{None}`, `k ∈ {1,3,5}`, weights `{uniform}`, metric `{cosine}` | 3 |

- **Total**: 31 configs × 100 splits = 3,100 unimodal evaluations per family (hard/fuzzy).
- **Fusion grid**: $\mathbf{w} \in \Delta^2$ with step 0.05 → 231 weight vectors per family (hard, fuzzy). 231 × 100 splits = 23,100 fusion evaluations per family; 46,200 total. Trivial in memory (4-d probability arrays only).
- Selection rule: best unimodal config per modality per family by **mean val Macro-F1** over 100 splits.
- Fusion weights selected by **mean val Macro-F1** over 100 splits; Brier stored as secondary selection fallback.
- Run in `tmux:0` with compact progress logging every 10 splits.

### 4.4 Phase B: Frozen LOOCV

- Once Phase A hyperparameters are fixed (`best_hyperparams.json`), run Leave-One-Out CV over the N=70 cohort.
- For each of the 70 folds: fit preprocessors + KNN on 69 train cases, predict on the held-out case.
- Report honest Macro-F1 / Micro-F1 / Weighted-F1 / AUROC / Brier / per-class F1.
- Expected runtime: KNN on N≤70 is instantaneous; 70 folds × 6 condition fits ≈ seconds.

### 4.5 Metrics computation (governance-compliant)

- **Decision gate**: for every case, call `compute_treatment_decision_gate(gt_decision, pred_decision)` from `chimera_metrics.task2.decision`. This returns 1.0/0.0 per case; macro accuracy = mean over the split.
- **Macro-F1 / Micro-F1 / Weighted-F1 / per-class F1 / AUROC**: `sklearn.metrics` (`f1_score(average=...)`, `roc_auc_score(multi_class='ovr')`). This is the homologized route because `chimera_metrics` only exposes the per-case binary gate for Task 2, not cohort-level multiclass F1.
- **Brier score (multiclass)**: `sklearn.metrics.brier_score_loss` is binary-only; implement as:
  $$
  \text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{4} (\hat{p}_{ij} - y_{ij})^2
  $$
  where $y_{ij}$ is the one-hot (hard) or soft (fuzzy) target.

### 4.6 Statistical tests

- **H1 vs H2 (paired)**: two-sided Wilcoxon signed-rank on the 100 split-level Macro-F1 arrays; and two-sided Wilcoxon on the 100 split-level Brier arrays.
- **vs majority baseline ($B_0$)**: Wilcoxon paired on Macro-F1 (same splits).
- **McNemar exact** (optional, only if Wilcoxon is marginal, $p \in [0.05, 0.10]$): on the pooled 4×4 contingency of correct/incorrect decisions.

---

## 5. Storage Footprint Controls

- **No full prediction CSVs**. Only in-memory 4-d arrays.
- Persisted artifacts (all JSON, compact):
  - `aggregate_metrics.json` (~5 KB), `split_metrics.json` (~100 rows ~20 KB), `wilcoxon_tests.json` (~2 KB), `best_hyperparams.json` (~3 KB), `version.txt`.
- `reports/summary.md` generated by `experiment-reporter` after execution.

---

## 6. Governance Compliance

1. **Decision gate via `chimera_metrics`**: every per-case treatment decision is validated with `compute_treatment_decision_gate`.
2. **Fixed deterministic partitions**: 100 MCCV splits persist to `data/processed/task2/mccv_splits_100.csv` (seeded).
3. **Zero leakage**: `active_treatment_flag` excluded. Preprocessors (`MinMaxScaler`, `StandardScaler`, `TfidfVectorizer`, PCA) fit **only** on the train fold, exactly as in `exp_4`.
4. **Traceability**: all outputs with configs + git SHA in `results/version.txt`.
5. **No ad-hoc metrics**: any deviation from `chimera_metrics` is explicitly homologized (documented mapping to `sklearn.metrics`).

---

## 7. Execution Order

```bash
# from repo root, in tmux:0
python experiments/exp_10/scripts/preprocess_task2.py
python experiments/exp_10/scripts/generate_mccv_splits.py
python experiments/exp_10/scripts/run_exp10_mccv.py
```

Estimated total wall-time: **≤ 15 min** CPU (16 threads) for Phase A; Phase B is seconds.

---

## 8. Acceptance Criteria

- [ ] `preprocess_task2.py` outputs exactly N=70 canonical rows with the 3 modalities + GT + confidence; leakage audit passes.
- [ ] `mccv_splits_100.csv` has 100 columns and preserves per-split class stratification (at least 1 `watchful_waiting` in train).
- [ ] `run_exp10_mccv.py` completes Phase A and Phase B without OOM, in < 15 min.
- [ ] Aggregate metrics JSON contains all 6 metrics per condition for both families.
- [ ] Wilcoxon tests computed and persisted.
- [ ] No CSVs of predictions left behind (storage control).

---

## 9. Next Steps

1. **PI approval of this IMPLEMENTATION.md.**
2. Build `scripts/preprocess_task2.py` and run it (data audit verification).
3. Build `scripts/generate_mccv_splits.py` and verify stratification.
4. Build `scripts/run_exp10_mccv.py` and execute in `tmux:0`.
5. Run `experiment-reporter` and update `.logbook.md` + `.discussion.md`.

---