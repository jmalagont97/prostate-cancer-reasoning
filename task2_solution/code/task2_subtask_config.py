"""Per-sub-task {feature set, model, required tools} configuration for task2 -- mirrors
subtask_config.py (task1) exactly, including its central finding: reveal_sequence is one
case-level list shared by all three sub-tasks' scoring, so the real reveal policy at inference
time is the union across whichever configs are wired in (compute_called_tools), and every swap
must be measured against the real ranking_score, not a sub-task's isolated component score.

All three sub-tasks call zero tools by default -- MEASURED, not assumed. Ground-truth
reveal_sequence is always [] for every one of the 72 labeled cases (matches task1/task3's finding),
so cost_aware_tool_score gives a guaranteed tool_score=1.0 for calling nothing versus a guaranteed
0.0 for calling anything (score = |actual ∩ expected|/|actual|, expected is always empty). Swept via
task2_variant_tester.py: grounding variable_weights with {get_mri_report, get_pathology_report}
(the naive "ground everything you can" choice) raised var_weight_score (0.673->0.803) and
factor_f1 (0.431->0.633) but LOST on ranking_score (0.7166->0.6848) because it zeroes tool_score
outright -- paired bootstrap on the per-case gap: mean +0.0636, 95% CI [+0.0525, +0.0748], 100% of
resamples favor zero tools. Every tool-grounding variant tried (mri-only, pathology-only,
mri+path+fh) also lost to zero tools. See experiments/exp_11/.

`decision`/`confidence` use `knn` (task1's proven winner, re-confirmed for task2 in exp_12's
8-model sweep -- no challenger beat it on either sub-task; `random_forest`'s apparent +0.0149 on
`decision` did not survive bootstrap, 95% CI [-0.0725,+0.1091]). `variable_weights` uses
`mord.LogisticIT` (a genuine ordinal classifier -- proportional-odds/threshold-based, fits one
linear score + learned ordered cutpoints, NOT the same as exp_13's regress-and-round proxy, which
assumed evenly-spaced integer distances and regressed toward the majority class). Promoted in
exp_14: ranking_score 0.7166->0.7186, borderline on bootstrap (95% CI [-0.0005,+0.0047], 93.7% of
resamples favor it) but decisively corroborated by three independent per-class diagnostics vs.
nominal knn -- accuracy 0.53->0.56, macro-F1 0.47->0.51, and critically `decisive`-class recall
0.18->0.25 (the exact weak spot this line of investigation started from; exp_13's proxy made the
same recall WORSE, 0.18->0.10). Same promotion logic as exp_6 (task3): borderline on the primary
composite metric, promoted anyway because a second, independently-verified diagnostic moved
decisively in the right direction, not just the headline number. `mord`'s ordinal classifiers were
also tried for `confidence` and lost outright (0.914->0.897, uniformly worse, no ambiguity) --
`confidence` stays `knn`. See experiments/exp_13/, experiments/exp_14/.
"""

import sys
from pathlib import Path

import mord  # noqa: E402

EXP2_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "experiments" / "exp_2" / "scripts"
sys.path.insert(0, str(EXP2_SCRIPTS))
from models import knn  # noqa: E402


def mord_logistic_it():
    return mord.LogisticIT()

# All 10 numeric features that are either always-available or already-extracted engineered
# columns (ct_ordinal, comorbidity_count from scripts/extract_features_task2.py) -- no gated text
# extraction needed for the base variable set, unlike task3's pathology-text parse (task2's
# structured.csv already carries the graded numbers directly, confirmed in notebooks/task2_eda.ipynb).
TASK2_BASE_NUMERIC = [
    "age", "psa", "psad", "pirads", "bx_isup", "bx_gl_prim", "bx_gl_sec", "cspca",
    "ct_ordinal", "comorbidity_count",
]
TASK2_BASE_CATEGORICAL: list[str] = []

# exp_16: PSA-kinetics features (psa_trend velocity/ratio, NOT redundant with `psa` -- confirmed
# `psa` == psa_trend's last reading == the lab panel's own PSA entry, same value everywhere; only
# the trajectory is new signal). Added to `decision` only -- +0.0243 ranking_score, bootstrap
# borderline (95% CI [+0.0000,+0.0622], 85.8% favoring, just short of the 90% bar used elsewhere in
# this project) but promoted anyway: the classification report shows a coherent, no-regression gain
# across every real class (active_surveillance F1 0.76->0.80, active_treatment F1 0.81->0.85,
# accuracy 0.81->0.83), the same "corroborating second diagnostic" logic that promoted exp_14's
# borderline variable_weights result, plus real a priori clinical grounding (PSA velocity is a
# guideline-recognized treatment-escalation factor, not a speculative feature). Labs (free PSA,
# %free PSA, etc.) were also tried and LOST badly (0.7186->0.6548) -- likely dimensionality noise
# from 7 new features on 72 cases; not added. See experiments/exp_16/.
TASK2_DECISION_NUMERIC = TASK2_BASE_NUMERIC + ["psa_velocity_per_year", "psa_ratio_last_first"]


def default_configs(df=None) -> dict:
    """Fresh copy each call -- callers (a future variant_tester) merge overrides onto this, never
    mutate a shared dict across tests. `df` accepted for signature parity with task1's
    default_configs (unused here -- task2's feature list isn't resolved from engineered_features.csv
    at call time the way task1's naive_73feat is)."""
    return {
        "decision": {
            "numeric": list(TASK2_DECISION_NUMERIC), "categorical": list(TASK2_BASE_CATEGORICAL),
            "model_factory": knn, "required_tools": set(),
        },
        "confidence": {
            "numeric": list(TASK2_BASE_NUMERIC), "categorical": list(TASK2_BASE_CATEGORICAL),
            "model_factory": knn, "required_tools": set(),
        },
        "variable_weights": {
            "numeric": list(TASK2_BASE_NUMERIC), "categorical": list(TASK2_BASE_CATEGORICAL),
            "model_factory": mord_logistic_it, "required_tools": set(),
        },
    }


def compute_called_tools(configs: dict) -> set[str]:
    return set().union(*(cfg["required_tools"] for cfg in configs.values()))


def merge_overrides(df, overrides: dict) -> dict:
    configs = default_configs(df)
    for subtask, patch in (overrides or {}).items():
        configs[subtask] = {**configs[subtask], **patch}
    return configs
