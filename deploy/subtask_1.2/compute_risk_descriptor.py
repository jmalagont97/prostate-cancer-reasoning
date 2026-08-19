#!/usr/bin/env python3
"""
deploy/subtask_1.2/compute_risk_descriptor.py

Calculates the continuous Decision Risk Score Omega(c_fn, lambda) from Subtask 1.1 predictions.
Formula (exp_18 winning condition: c_fn=0.65, lambda=0.00):
  Omega = 2 * | 0.65 * p_bar - 0.5 |
"""

import sys
import importlib.util
from pathlib import Path
import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parent
SUBTASK_1_1_DIR = DEPLOY_DIR.parent / "subtask_1.1"

# Dynamically import predict_subtask_1_1 from subtask_1.1 without namespace collision
spec_1_1 = importlib.util.spec_from_file_location("subtask_1_1_infer", SUBTASK_1_1_DIR / "predict_inference.py")
subtask_1_1_module = importlib.util.module_from_spec(spec_1_1)
spec_1_1.loader.exec_module(subtask_1_1_module)
predict_subtask_1_1 = subtask_1_1_module.predict_subtask_1_1


def compute_decision_risk(raw_case, c_fn=0.65, lambda_param=0.00):
    """
    Computes the continuous Decision Risk metric Omega from raw patient inputs.

    Parameters:
      raw_case (dict): Raw patient case containing tabular, text, and mri_emb keys.
      c_fn (float): False negative penalty weight (0.65)
      lambda_param (float): Uncertainty penalty multiplier (0.00)

    Returns:
      dict:
        {
          "omega_risk_score": float,
          "p_bar": float,
          "subtask_1_1_res": dict
        }
    """
    res_1_1 = predict_subtask_1_1(raw_case)
    p_fused = res_1_1["fused_probability"]

    # Decision Risk Metric Omega
    omega = 2.0 * abs(c_fn * p_fused - 0.5)

    return {
        "omega_risk_score": float(np.round(omega, 4)),
        "p_bar": float(np.round(p_fused, 4)),
        "subtask_1_1_res": res_1_1
    }
