"""Confidence + variable-weights target construction for task2, ported from
experiments/exp_2/scripts/targets.py (task1) -- same CONF_MAP/WEIGHT_MAP, same evaluator formulas,
extended for task2's 11-variable vocabulary. decision itself is left as raw strings (4-class:
sklearn classifiers accept string labels natively -- no need for task1's bool-encoding trick).
"""

import json

import numpy as np

CONF_MAP = {"uncertain": 0, "borderline": 1, "clear": 2}
WEIGHT_MAP = {"not_used": 0, "noted": 1, "important": 2, "decisive": 3}
IMPORTANT_OR_DECISIVE = {2, 3}

VALID_TREATMENT_DECISIONS = (
    "watchful_waiting", "active_surveillance", "continued_surveillance", "active_treatment",
)

# Exact TASK2_VARIABLES order, confirmed from task2_schema.py / section_variable_mapping.json.
TASK2_VARIABLES = [
    "age", "psa", "psad", "pirads", "cspca", "ct", "bx_isup", "bx_gl_prim", "bx_gl_sec",
    "comorbidity", "fh",
]


def parse_decision(labels_df):
    return labels_df["decision"].to_numpy()


def parse_confidence(labels_df):
    return labels_df["confidence"].map(CONF_MAP).to_numpy()


def parse_variable_weights(labels_df):
    """Returns an (n_cases, 11) int array, columns in TASK2_VARIABLES order."""
    rows = []
    for s in labels_df["variable_weights"]:
        vw = json.loads(s) if isinstance(s, str) else s
        rows.append([WEIGHT_MAP[vw.get(var, "not_used")] for var in TASK2_VARIABLES])
    return np.array(rows)


def confidence_score(y_true, y_pred):
    """Evaluator's exact formula: 1 - ordinal_distance/max_dist, max_dist=2."""
    return 1.0 - np.abs(np.asarray(y_true) - np.asarray(y_pred)) / 2.0


def variable_weight_score_per_case(true_row, pred_row):
    """1 - mean(|true-pred|)/3 across the 11 variables, for one case."""
    return 1.0 - np.mean(np.abs(np.asarray(true_row) - np.asarray(pred_row))) / 3.0


def _set_f1(gt_set, pred_set):
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def important_decisive_factor_score_per_case(true_row, pred_row):
    gt_set = {i for i, v in enumerate(true_row) if v in IMPORTANT_OR_DECISIVE}
    pred_set = {i for i, v in enumerate(pred_row) if v in IMPORTANT_OR_DECISIVE}
    return _set_f1(gt_set, pred_set)
