# Confidence-KDM Model Card

**Model**: `confidence_kdm_direct_scalar_23col`, from `experiments/exp_12/`
**File**: `confidence_kdm_23col.pkl`
**Task**: predicts the diagnostic-confidence label (`uncertain` / `borderline` / `clear`) a clinician
would assign to a prostate-biopsy decision, from the same structured clinical/MRI variables used for
the decision itself.
**Trained on**: all 91 labeled cases in `data/ground_truth.csv` (not a train/test split — see
"Important caveats" below before trusting the numbers a fresh call against training rows produces).
**Exported**: 2026-08-16, by `export_model.py` in this directory.

---

## 1. What this model is

A memory-based **Kernel Density Matrix (KDM)** classifier (`kdm-torch`), trained *directly* on the
3-class confidence label — not derived from a decision-trained backbone the way most of this
project's earlier confidence attempts were. Every training row becomes a frozen "prototype"; the
only thing gradient descent learns is a single shared RBF-kernel bandwidth (`sigma`). At inference,
a new case's predicted confidence is a similarity-weighted vote over all 91 prototypes' own
confidence labels.

This is the best confidence result produced across 12 experiments in this project
(`experiments/exp_12/reports/summary.md` has the full writeup; `experiments/INDEX.md` has the
project-wide comparison table).

## 2. Validated performance

| Method | Ordinal distance (official metric, lower=better) | Macro-F1 | Accuracy |
|---|---|---|---|
| 5×10 cross-validation | 0.491 | 0.544 | 0.642 |
| Leave-one-out (91-fold) | **0.440** | **0.589** | **0.681** |
| Repeated held-out (10 seeds, mean±std) | 0.447 ± 0.175 | 0.555 ± 0.129 | — |
| *(reference)* `confidence_svm` incumbent | 0.468 | 0.404 | — |
| *(reference)* Naive baseline (majority class) | 0.527 | 0.260 | — |

Two of three independent evaluation protocols (LOO, repeated-holdout) numerically beat the
`confidence_svm` incumbent on the official ordinal-distance metric; CV alone doesn't quite. Treat
**0.44–0.49 ordinal distance / 0.54–0.59 macro-F1** as the realistic expectation for new cases, not
the in-sample check `export_model.py` prints (which is expected to look near-perfect — see below).

## 3. Confusion matrix & classification report (leave-one-out)

Computed on the exact deployed configuration (direct scalar-KDM, 23-column frame), scored via
leave-one-out (91 folds, one case held out and refit each time, predictions pooled across all 91
before scoring — see `loo_confusion_matrix.py` in this directory). Reproduces `experiments/
exp_12/reports/summary.md`'s already-published LOO accuracy (0.681) and macro-F1 (0.589) exactly,
confirming this is the same validated fit, not a new one.

**Confusion matrix** (rows = true label, columns = predicted label):

| True \ Predicted | uncertain | borderline | clear |
|---|---|---|---|
| **uncertain** | 6 | 4 | 5 |
| **borderline** | 1 | 11 | 6 |
| **clear** | 6 | 7 | **45** |

**Classification report**:

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| uncertain | 0.462 | 0.400 | 0.429 | 15 |
| borderline | 0.500 | 0.611 | 0.550 | 18 |
| clear | 0.804 | 0.776 | 0.789 | 58 |
| **accuracy** | | | **0.681** | 91 |
| macro avg | 0.588 | 0.596 | 0.589 | 91 |
| weighted avg | 0.687 | 0.681 | 0.683 | 91 |

**Reading it**:
- The model is strongest on `clear` (recall 0.776, f1 0.789) — the majority class (58/91 cases,
  64% of the labeled set).
- `uncertain` is the weakest class (f1 0.429): of 15 true `uncertain` cases, only 6 were correctly
  identified, and 5 were mistaken for `clear` — the opposite end of the ordinal scale. This is the
  most clinically concerning error type this model makes (a 2-level miss, not a 1-level one) and is
  worth extra scrutiny if this model is ever used to triage genuinely uncertain cases.
- `borderline` performs reasonably (f1 0.550); most of its errors are adjacent-level confusions
  (uncertain↔borderline, borderline↔clear) rather than 2-level jumps — consistent with the low
  aggregate ordinal distance (0.440) despite middling per-class F1.
- No class collapses to zero recall — a genuine 3-way discriminator, not a majority-class
  impersonator wearing a probabilistic model's clothing.

Full machine-readable output (including the exact `sklearn.metrics.classification_report`
`output_dict=True` payload):
`experiments/exp_12/results/loo_confusion_matrix_confidence_direct_scalar_23col/metrics.json`.

## 4. Important caveats

- **This is a small-data model (N=91).** All published performance numbers already account for
  that (cross-validation, leave-one-out, and repeated held-out splits, not a single train/test
  score) — but predictions on any individual new case should still be read with real uncertainty,
  not treated as ground truth.
- **The in-sample check `export_model.py` prints (accuracy=1.000) is not a generalization
  estimate.** This exported model is refit on *all* 91 labeled cases, and because it's
  memory-based, every one of those 91 rows is literally one of the model's own prototypes — calling
  `predict()` on a row the model was trained on will essentially always return that row's own label.
  Only call this model on genuinely new cases, and trust §2's numbers (not a re-run on training
  data) for what accuracy to expect.
- **ARD was tried and found to actively hurt this specific target** (`experiments/exp_11`) — this
  model deliberately uses the plain scalar-bandwidth backbone, not ARD, even though ARD helps the
  separate decision model in this project. Don't "upgrade" this specific model to ARD without
  re-validating; the two targets disagree.
- **MRI is optional but the model expects the column to exist.** If a case has no MRI embedding,
  fill the 1024 `mri_emb_*` columns with `NaN` (or omit them from the DataFrame entirely) — the
  wrapper falls back to the fitted PCA space's origin plus a `mri_missing=1` flag, matching how
  missing-MRI training cases were handled. Passing zeros instead of `NaN` would be treated as a
  real (mistaken) MRI reading, not "missing."

## 5. Files in this directory

| File | What it is |
|---|---|
| `confidence_kdm_23col.pkl` | The deployable artifact — a pickled `ConfidenceKDMPredictor` bundling the fitted MRI-PCA, imputer, scaler, and trained KDM together. |
| `predictor.py` | The `ConfidenceKDMPredictor` class definition. **Required to unpickle the `.pkl` file** — Python's `pickle` module needs this exact class importable at load time. Keep it on `sys.path` (or just keep this directory intact) before calling `pickle.load`. |
| `export_model.py` | The training/export script that produced `confidence_kdm_23col.pkl`. Re-run it to regenerate the artifact from scratch (e.g., after new labeled cases are added to `data/ground_truth.csv`). |
| `loo_confusion_matrix.py` | Reproduces §3's confusion matrix and classification report from scratch (91-fold leave-one-out on the 23-column frame). |
| `MODEL_CARD.md` | This file. |

## 6. Requirements

Python 3.10+ (built/tested on 3.14), with:

| Package | Version used to build this artifact |
|---|---|
| `torch` | 2.13.0 (CPU) |
| `kdm-torch` | 2.0.0 |
| `scikit-learn` | 1.9.0 |
| `pandas` | 3.0.5 |
| `numpy` | 2.5.1 |

Pickle files are sensitive to library version drift (especially `torch`/`scikit-learn` major
versions). If loading fails on a different environment, re-run `export_model.py` there instead of
trying to force-load a `.pkl` built elsewhere.

## 7. Input schema

Call the model with a `pandas.DataFrame` shaped like `data/inputs.csv` — one row per case, at
minimum the columns below (any extra columns are ignored). No manual preprocessing needed: the
wrapper handles imputation, scaling, MRI-PCA projection, and the categorical→ordinal/flag encodings
internally, using the exact same code path as training.

| Column | Type | Meaning | Missing-value handling |
|---|---|---|---|
| `cli_psa` | float | Serum PSA (ng/mL) | Median-imputed |
| `cli_psap` | float | Prior/previous PSA value (ng/mL) | Median-imputed |
| `cli_psav` | float | PSA velocity (ng/mL/year) | Median-imputed |
| `cli_psad` | float | PSA density (PSA / prostate volume) | Median-imputed |
| `cli_vol` | float | Prostate volume (cm³) | Median-imputed |
| `cli_age` | float | Patient age (years) | Median-imputed |
| `cli_cspca` | float | Model/nomogram-estimated probability of clinically significant PCa (0–1) | Median-imputed |
| `cli_pirads` | float | PI-RADS MRI score (1–5) | Median-imputed |
| `cli_dre` | string | Digital rectal exam finding — one of `Normal`, `Abnormal`, `Nodus`, `Suspicious`, `Not done` | `Not done`/missing → its own flag, not treated as "Normal" |
| `cli_bx` | string | Prior biopsy result — `Positive`, `Negative`, or missing | Missing → its own flag, not treated as "Negative" |
| `txt_comorbidities` | string | Comma-joined free-text comorbidity list (e.g. `"Hypertension, Obesity"`) | Empty/missing → all 6 comorbidity flags 0 |
| `path_hist_bx_isup` | float | ISUP grade group from prior biopsy pathology (0–5), if available | Median-imputed |
| `vit_bmi` | float | Body-mass index | Median-imputed |
| `mri_emb_0` … `mri_emb_1023` | float | 1024-dim MRI lesion embedding | See MRI caveat above — omit entirely or set to `NaN` if unavailable |

The model internally derives 23 engineered columns from the above (comorbidity → 6 binary group
flags, `dre`/`bx` → ordinal/binary encodings, MRI embedding → 2-component PCA + missing flag,
everything median-imputed then standardized) — this is handled by `predictor.py`, not something the
caller needs to replicate.

## 8. How to use

```python
import pickle
import sys

# predictor.py must be importable -- either run from this directory, or add it to sys.path first:
sys.path.insert(0, "experiments/exp_12/model")

import pandas as pd

with open("experiments/exp_12/model/confidence_kdm_23col.pkl", "rb") as f:
    predictor = pickle.load(f)

# `new_cases` is a DataFrame shaped like data/inputs.csv (see the input schema above) -- can be a
# single row or many.
new_cases = pd.read_csv("path/to/new_cases.csv")

# Simplest entry point: predicted label + all 3 class probabilities, one row per case.
result = predictor.predict_full(new_cases)
print(result)
#   predicted_confidence  p_uncertain  p_borderline   p_clear
# 0                clear     0.006790      0.000890  0.992320

# Or, if you only need the labels or only the raw probabilities:
labels = predictor.predict(new_cases)              # list[str], one of "uncertain"/"borderline"/"clear"
probs = predictor.predict_proba(new_cases)          # np.ndarray, shape (n, 3), columns in that order
```

## 9. Output format

- **`predict_full(df)`** — a `pandas.DataFrame`, same index as the input, with columns
  `predicted_confidence` (string label), `p_uncertain`, `p_borderline`, `p_clear` (floats summing to
  1 per row). Recommended default entry point.
- **`predict(df)`** — a plain `list[str]`, one label per input row.
- **`predict_proba(df)`** — a `numpy.ndarray` of shape `(n_rows, 3)`, columns ordered
  `[uncertain, borderline, clear]`.

## 10. Retraining

Re-run `export_model.py` (from the project root, using the project's `.venv`) whenever
`data/ground_truth.csv`/`data/inputs.csv` change — it always refits on every currently-labeled case
from scratch, with the same fixed hyperparameters exp_12 validated (300 epochs, Adam lr=1e-2,
sigma-only trainable, no hyperparameter search). It overwrites `confidence_kdm_23col.pkl` in place.

## 11. Provenance

Full experimental detail, ablations (ARD vs. scalar, 19-col vs. 23-col frame), and the reasoning
behind every design choice above: `experiments/exp_12/DESIGN.md` and
`experiments/exp_12/reports/summary.md`. Project-wide comparison against every other approach tried:
`experiments/INDEX.md`.
