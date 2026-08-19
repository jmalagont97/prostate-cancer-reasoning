#!/usr/bin/env python3
"""
agent/tools.py

Tool Abstraction Layer for the Pathology Reasoning Agent.
Provides a unified, tool-agnostic API wrapper around deploy/subtask_1.1,
deploy/subtask_1.2, and deploy/subtask_1.3 inference engines.
"""

import sys
import importlib.util
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
DEPLOY_DIR = PROJECT_ROOT / "deploy"

# Dynamically load inference functions from deploy subtasks
def _load_tool_module(subtask_name):
    subtask_dir = DEPLOY_DIR / subtask_name
    infer_file = subtask_dir / "predict_inference.py"
    if not infer_file.exists():
        raise FileNotFoundError(f"Tool inference file not found at: {infer_file}")
    
    spec = importlib.util.spec_from_file_location(f"{subtask_name}_module", infer_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_SUBTASK_1_1 = _load_tool_module("subtask_1.1")
_SUBTASK_1_2 = _load_tool_module("subtask_1.2")
_SUBTASK_1_3 = _load_tool_module("subtask_1.3")


def tool_predict_biopsy_decision(raw_case):
    """
    Tool 1: Predicts biopsy decision (Subtask 1.1).
    Returns standardized dict with 'biopsy_decision_binary', 'decision_label' ('yes'/'no'), 'fused_probability'.
    """
    res = _SUBTASK_1_1.predict_subtask_1_1(raw_case)
    decision_label = "yes" if res["biopsy_decision_binary"] == 1 else "no"
    return {
        "biopsy_decision_binary": res["biopsy_decision_binary"],
        "biopsy_decision_label": decision_label,
        "fused_probability": res["fused_probability"],
        "modality_probabilities": res.get("modality_probabilities", {})
    }


def tool_predict_clinical_confidence(raw_case):
    """
    Tool 2: Predicts clinical confidence (Subtask 1.2).
    Returns standardized dict with 'confidence_ordinal', 'confidence_label' ('uncertain'/'borderline'/'clear').
    """
    res = _SUBTASK_1_2.predict_subtask_1_2(raw_case)
    return {
        "confidence_ordinal": res["confidence_ordinal"],
        "confidence_label": res["confidence_label"],
        "decision_risk_score": res["decision_risk_score"]
    }


def tool_predict_relevance_and_reveal_sequence(raw_case):
    """
    Tool 3: Predicts 10 relevance weights & section reveal sequence (Subtask 1.3).
    Returns standardized dict with 'relevance_weights', 'relevance_labels', 'reveal_sequence'.
    """
    res = _SUBTASK_1_3.predict_subtask_1_3(raw_case)
    return {
        "relevance_weights": res["relevance_weights"],
        "relevance_labels": res["relevance_labels"],
        "reveal_sequence": res["reveal_sequence"]
    }
