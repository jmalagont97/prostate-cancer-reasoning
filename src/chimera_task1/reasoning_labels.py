"""Ordinal label encodings + set/order utilities for the step-4 reasoning models.

Confidence and per-factor variable weights are ordinal scales in the
schema (``Weight``/``Confidence`` enums in the baseline's
``output/schema.py``). We keep the *model-facing* representation as plain
multiclass targets (simplest to fit at N=91) but score everything through
these ordinal encodings, since the official rubric scores "ordinal
distance" / "mean ordinal error", not classification accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

WEIGHT_LEVELS = ["not_used", "noted", "important", "decisive"]
WEIGHT_RANK = {level: i for i, level in enumerate(WEIGHT_LEVELS)}

CONFIDENCE_LEVELS = ["uncertain", "borderline", "clear"]
CONFIDENCE_RANK = {level: i for i, level in enumerate(CONFIDENCE_LEVELS)}

# The 10 schema variable_weights keys, in the order they appear in ground_truth.csv's
# target_weight_* columns (see features.TASK1_VARIABLE_TO_FEATURE for the feature mapping).
TASK1_FACTORS = ["age", "fh", "cspca", "pirads", "vol", "psa", "comorbidity", "psad", "dre", "bx"]

# Ground-truth reveal_sequence uses these section names; TOOL_REVEAL_INFO in the baseline's
# schema.py maps each to an MCP tool. pathology_report/family_history are rare/absent in the
# 91-case sample but valid per the schema so kept here for completeness.
REVEAL_SECTIONS = [
    "previous_notes",
    "laboratory_results",
    "psa_trend",
    "radiology_report",
    "family_history",
    "pathology_report",
]

# Canonical order observed empirically across the 91 annotated cases (see exploration):
# the two dominant non-empty sequences are subsets/permutations of this ordering.
CANONICAL_REVEAL_ORDER = ["previous_notes", "laboratory_results", "psa_trend", "radiology_report"]


def weight_col(factor: str) -> str:
    return f"target_weight_{factor}"


def ordinal_distance(y_true_labels: list[str], y_pred_labels: list[str], rank_map: dict[str, int]) -> float:
    """Mean absolute rank difference between two label sequences (the official rubric's metric)."""
    true_ranks = np.array([rank_map[label] for label in y_true_labels])
    pred_ranks = np.array([rank_map[label] for label in y_pred_labels])
    return float(np.mean(np.abs(true_ranks - pred_ranks)))


def decisive_set_f1(y_true_labels: list[str], y_pred_labels: list[str]) -> float:
    """Set F1 over which cases are marked important/decisive (matches the rubric's factor-set component)."""
    true_flags = np.array([label in ("important", "decisive") for label in y_true_labels])
    pred_flags = np.array([label in ("important", "decisive") for label in y_pred_labels])
    tp = int(np.sum(true_flags & pred_flags))
    fp = int(np.sum(~true_flags & pred_flags))
    fn = int(np.sum(true_flags & ~pred_flags))
    if tp == 0:
        return 1.0 if fp == 0 and fn == 0 else 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def parse_reveal_sequences(series: pd.Series) -> list[list[str]]:
    return [json.loads(s) if isinstance(s, str) else [] for s in series]


def reveal_set_precision(true_seq: list[str], pred_seq: list[str]) -> float:
    """Precision of predicted revealed sections against the target's (the rubric's "tool efficiency")."""
    if not pred_seq:
        return 1.0 if not true_seq else 0.0
    true_set, pred_set = set(true_seq), set(pred_seq)
    return len(true_set & pred_set) / len(pred_set)


def order_by_canonical(section_set: set[str]) -> list[str]:
    """Order a predicted section subset by the empirically canonical sequence, extras appended."""
    ordered = [s for s in CANONICAL_REVEAL_ORDER if s in section_set]
    ordered += [s for s in section_set if s not in CANONICAL_REVEAL_ORDER]
    return ordered
