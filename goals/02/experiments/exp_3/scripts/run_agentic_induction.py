#!/usr/bin/env python3
"""run_agentic_induction.py

Closed-loop literature-guided multi-agent decision tree induction for Goal 02 Task 1.1.
Implements Two-Stage Induction Decoupling:
- Phase A: Feature Prioritization (Oracle selects the single most important clinical variable based on extensive clinical context and EAU/ESMO guidelines)
- Phase B: Boundary & Operator Calibration (Oracle formulates the splitting threshold grounded on feature-specific empirical bounds and guideline evidence)
- Evaluation: Quantitative evaluation strictly on Train partition (N=45) via compute_gini_split

Outputs:
- results/artifacts/decision_tree.json
- results/artifacts/model.py
- results/blackboard.md
- results/version.txt
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add scripts directory to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from agent_runner import SingleAgentRunner, load_agent_declaration, resolve_paths
from blackboard import Blackboard
from tools.compute_gini_split import compute_gini_split


RCT_HUMAN_DIRECTIVE = """# [R] ROLE
You are a senior consulting specialist in Urological Oncology and Evidence-Based Medicine, with comprehensive mastery of international prostate cancer guidelines (European Association of Urology - EAU, and ESMO).

# [C] CONTEXT
You are assessing an outpatient clinical cohort evaluated for suspected or monitored prostate disease. You have full access to international guidelines via ChromaDB (`query_clinical_literature`) and each patient is characterized strictly by the 10 official Ground Truth clinical variables evaluated for diagnostic relevance in Task 1:

### Extensive Clinical Profiles of the 10 Ground Truth Variables:
1. `radiology_report__pirads` (mpMRI PI-RADS v2.1 Score, Scale 1 to 5):
   - Biological meaning: Standardized assessment of the probability of clinically significant prostate cancer (csPCa, Gleason score >= 7 / ISUP grade group >= 2) in suspicious intraprostatic lesions on multiparametric MRI.
   - Use in clinical practice: Primary anatomical triaging tool. PI-RADS 1-2 indicates very low likelihood of aggressive malignancy; PI-RADS 3 represents indeterminate / equivocal findings; PI-RADS 4-5 indicates high to very high probability of invasive malignancy. Used to determine whether invasive biopsy is indicated or can be safely omitted/deferred.

2. `radiology_report__psa_density` (PSA Density - PSAD, ng/ml/cc):
   - Biological meaning: Ratio of serum total PSA to prostate gland volume measured by imaging (Total PSA / Prostate Volume).
   - Use in clinical practice: Corrects serum PSA for benign prostatic hyperplasia (BPH) volume enlargement. Malignant epithelial tissue secretes substantially more PSA per unit of volume than benign hyperplastic tissue. Key clinical discriminator in intermediate or indeterminate cases (especially PI-RADS 3) and active surveillance, distinguishing benign enlargement from aggressive disease.

3. `structured_prompt__bx` (Histopathological Finding of Prior Prostate Biopsies):
   - Biological meaning: Documented histological tissue diagnosis resulting from prior prostate core needle biopsies ('Positive' for confirmed adenocarcinoma, 'Negative' for benign/negative cores, or missing/naive).
   - Use in clinical practice: Crucial for determining whether a new biopsy is diagnostically necessary to establish a primary cancer diagnosis, whether it is redundant because malignancy is already histologically proven (guiding directly to definitive curative therapy), or whether re-biopsy is needed due to rising suspicion despite prior negative cores.

4. `laboratory_results__psa` (Total Serum PSA, ng/ml):
   - Biological meaning: Glycoprotein kallikrein enzyme synthesized by prostate luminal epithelial cells; bloodstream concentrations rise following basement membrane disruption.
   - Use in clinical practice: Primary biochemical screening and disease monitoring biomarker. Levels > 4.0 ng/ml indicate elevated risk of pathology (BPH, prostatitis, or malignancy); markedly high levels (> 10.0 or > 20.0 ng/ml) significantly increase the probability of high-grade, aggressive disease.

5. `structured_prompt__age` (Chronological Age, Years):
   - Biological meaning: Patient chronological age, acting as a direct surrogate for life expectancy and cumulative medical comorbidity.
   - Use in clinical practice: Urological guidelines apply the 10-year life expectancy benchmark. In elderly patients with limited life expectancy, diagnostic biopsy is frequently omitted to prevent overdiagnosis and overtreatment of indolent, non-lethal tumors that would never affect survival.

6. `structured_prompt__dre` (Digital Rectal Examination - DRE):
   - Biological meaning: Physical digital palpation of the posterior and lateral peripheral zones of the prostate gland ('Normal', 'Nodus', 'Suspicious', 'Not done').
   - Use in clinical practice: Detection of a palpable firm induration or nodule (clinical stage >= cT2) is an independent clinical predictor of aggressive cancer, representing a classical biopsy indication even when PSA is borderline or normal.

7. `clinical_data__family_history` (Family History of Prostate Cancer):
   - Biological meaning: Documented confirmed history of prostate cancer in first-degree relatives (father, brothers) or hereditary syndromes (BRCA1/2).
   - Use in clinical practice: Confers a 2- to 4-fold higher relative risk of developing prostate cancer, frequently associated with earlier onset or more aggressive phenotypes, lowering the clinical threshold for recommending imaging and diagnostic biopsy.

8. `clinical_data__cspca` (Algorithmic Risk Estimate of csPCa, 0.0 to 1.0):
   - Biological meaning: Continuous probabilistic score generated by multivariable risk calculators or machine learning models predicting clinically significant prostate cancer.
   - Use in clinical practice: Quantitative decision-support adjunct used to communicate individualized risk and facilitate shared clinical decision-making.

9. `radiology_report__prostate_volume` (Prostate Gland Volume by Imaging, cc):
   - Biological meaning: Anatomical volume of the prostate gland estimated via ultrasound or mpMRI.
   - Use in clinical practice: Quantifies the degree of benign enlargement (BPH). Markedly enlarged prostates (> 60-80 cc) frequently produce elevated total serum PSA purely from benign transitional zone tissue, explaining discrepancies between high PSA and low cancer risk.

10. `previous_notes__prior_cancer_confirmed` (Documented Previous Cancer Diagnosis, 1=Yes, 0=No):
    - Biological meaning: Formal binary medical record entry confirming a previous clinical diagnosis of prostate neoplasia.
    - Use in clinical practice: Identifies patients who are not in the primary diagnostic pathway but rather in active surveillance, watchful waiting, or post-treatment follow-up, where repeating an invasive biopsy requires explicit justification of clinical or radiological disease progression.

# [T] TASK
Determine with urological rigor which patients require a prostate biopsy ('yes') and which do not ('no').
For every decision boundary or patient subgroup, you must weigh:
1. Clinical Severity: Assess the likelihood of harboring undetected clinically significant prostate cancer (csPCa).
2. Prognostic Necessity: Determine if histological tissue confirmation will alter clinical management, or if biopsy is redundant due to already established histopathological malignancy.
3. Risk-Benefit Balance: Weigh procedure risks (sepsis, hematuria, urinary retention, pain, overdiagnosis of indolent disease) against omission risks, ensuring invasive biopsy is recommended only when clinical benefit clearly outweighs procedural harm.
"""

GROUND_TRUTH_VARS = [
    "radiology_report__pirads",
    "radiology_report__psa_density",
    "structured_prompt__bx",
    "laboratory_results__psa",
    "structured_prompt__age",
    "structured_prompt__dre",
    "clinical_data__family_history",
    "clinical_data__cspca",
    "radiology_report__prostate_volume",
    "previous_notes__prior_cancer_confirmed",
]


def _calculate_gini(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    probabilities = series.value_counts(normalize=True).values
    return float(round(1.0 - np.sum(probabilities ** 2), 4))


def _resolve_feature_name(raw_feature: str) -> Optional[str]:
    """Resolve raw feature input to an exact column name in GROUND_TRUTH_VARS."""
    cleaned = raw_feature.strip()
    if cleaned in GROUND_TRUTH_VARS:
        return cleaned

    alias_map = {
        "pirads": "radiology_report__pirads",
        "pirads_score": "radiology_report__pirads",
        "pi_rads": "radiology_report__pirads",
        "psa_density": "radiology_report__psa_density",
        "psad": "radiology_report__psa_density",
        "bx": "structured_prompt__bx",
        "biopsy": "structured_prompt__bx",
        "prior_bx": "structured_prompt__bx",
        "psa": "laboratory_results__psa",
        "total_psa": "laboratory_results__psa",
        "serum_psa": "laboratory_results__psa",
        "age": "structured_prompt__age",
        "dre": "structured_prompt__dre",
        "family_history": "clinical_data__family_history",
        "fh": "clinical_data__family_history",
        "cspca": "clinical_data__cspca",
        "prostate_volume": "radiology_report__prostate_volume",
        "volume": "radiology_report__prostate_volume",
        "vol": "radiology_report__prostate_volume",
        "prior_cancer_confirmed": "previous_notes__prior_cancer_confirmed",
        "prior_cancer": "previous_notes__prior_cancer_confirmed",
        "cancer_confirmed": "previous_notes__prior_cancer_confirmed",
        "comorbidity": "previous_notes__prior_cancer_confirmed",
    }
    norm = cleaned.lower().replace(" ", "_").replace("-", "_")
    if norm in alias_map:
        return alias_map[norm]

    stem = re.sub(r"_(level|value|score|result|status|finding)$", "", norm)
    if stem in alias_map:
        return alias_map[stem]

    matches = [c for c in GROUND_TRUTH_VARS if norm in c.lower() or stem in c.lower()]
    if len(matches) == 1:
        return matches[0]

    return None


def compute_general_overview(train_df: pd.DataFrame, sample_cases: List[str]) -> str:
    """Compute high-level summary of the 10 Ground Truth variables in active partition for Phase A."""
    sub_df = train_df[train_df["case_id"].isin(sample_cases)].copy()
    n_samples = len(sub_df)
    dist = sub_df["biopsy_decision"].value_counts().to_dict()
    gini = _calculate_gini(sub_df["biopsy_decision"])

    lines = [
        f"### ACTIVE COHORT SUMMARY (N={n_samples}, Gini={gini:.4f}, Distribution={dist})",
        "Candidate Clinical Decision Axes (10 Ground Truth Variables):",
    ]

    for col in GROUND_TRUTH_VARS:
        if col not in sub_df.columns:
            continue
        s = sub_df[col]
        if pd.api.types.is_numeric_dtype(s):
            s_clean = s.dropna()
            if not s_clean.empty:
                val_min, val_max, val_med = round(float(s_clean.min()), 4), round(float(s_clean.max()), 4), round(float(s_clean.median()), 4)
                if s_clean.nunique() <= 6:
                    counts = {int(k) if isinstance(k, (int, float)) and k == int(k) else k: int(v) for k, v in s_clean.value_counts().sort_index().items()}
                    lines.append(f"- `{col}`: observed_values={counts}, range=[{val_min}, {val_max}], median={val_med}")
                else:
                    lines.append(f"- `{col}`: range=[{val_min}, {val_max}], median={val_med}")
        else:
            s_clean = s.fillna("Missing")
            ct = pd.crosstab(s_clean, sub_df["biopsy_decision"]).to_dict()
            bd = {str(k): f"{ct.get('yes', {}).get(k, 0)} yes / {ct.get('no', {}).get(k, 0)} no" for k in s_clean.unique()}
            lines.append(f"- `{col}`: {bd}")

    return "\n".join(lines)


def compute_feature_specific_bounds(train_df: pd.DataFrame, sample_cases: List[str], feature_col: str) -> str:
    """Compute detailed empirical distribution and viable cut points for the selected feature in Phase B."""
    sub_df = train_df[train_df["case_id"].isin(sample_cases)].copy()
    n_samples = len(sub_df)
    s = sub_df[feature_col]

    lines = [
        f"### EMPIRICAL PROFILE OF SELECTED VARIABLE: `{feature_col}` (Active Node N={n_samples})",
    ]

    is_num = pd.api.types.is_numeric_dtype(s)
    if is_num:
        s_clean = s.dropna()
        if s_clean.empty:
            return f"Variable `{feature_col}` has all missing values in this partition."

        val_min, val_max, val_med = round(float(s_clean.min()), 4), round(float(s_clean.max()), 4), round(float(s_clean.median()), 4)
        lines.append(f"- Observed Range: [{val_min}, {val_max}] (Median: {val_med})")

        if s_clean.nunique() <= 8:
            counts = {int(k) if isinstance(k, (int, float)) and k == int(k) else k: int(v) for k, v in s_clean.value_counts().sort_index().items()}
            lines.append(f"- Value Distribution: {counts}")

            viable_cuts = []
            for val in sorted(s_clean.unique()):
                n_le = int((s_clean <= val).sum())
                n_gt = int((s_clean > val).sum())
                if n_le >= 3 and n_gt >= 3:
                    viable_cuts.append(f"'<= {val}' (splits {n_le} vs {n_gt})")
            if viable_cuts:
                lines.append(f"- Viable Balanced Cuts (both branches >= 3 samples): {', '.join(viable_cuts)}")
            else:
                lines.append("- Note: Thresholds with '<=' or '>=' must be chosen carefully to ensure both branches have >= 3 samples.")

        elif feature_col == "radiology_report__psa_density":
            # For continuous PSA density, report sample counts around standard guideline cutoffs
            n_15 = int((s_clean >= 0.15).sum())
            n_lt15 = int((s_clean < 0.15).sum())
            n_20 = int((s_clean >= 0.20).sum())
            n_lt20 = int((s_clean < 0.20).sum())
            n_25 = int((s_clean >= 0.25).sum())
            n_lt25 = int((s_clean < 0.25).sum())
            lines.append(f"- Standard Guideline Thresholds in this node:")
            lines.append(f"  * '>= 0.15' splits {n_15} vs {n_lt15}")
            lines.append(f"  * '>= 0.20' splits {n_20} vs {n_lt20}")
            lines.append(f"  * '>= 0.25' splits {n_25} vs {n_lt25}")

    else:
        s_clean = s.fillna("Missing")
        ct = pd.crosstab(s_clean, sub_df["biopsy_decision"]).to_dict()
        breakdown = {str(k): f"{ct.get('yes', {}).get(k, 0)} yes / {ct.get('no', {}).get(k, 0)} no" for k in s_clean.unique()}
        lines.append(f"- Category Breakdown by Class: {breakdown}")

        # Highlight viable categorical splits
        for cat_val in s_clean.unique():
            n_eq = int((s_clean == cat_val).sum())
            n_neq = n_samples - n_eq
            if n_eq >= 3 and n_neq >= 3:
                lines.append(f"- Viable Cut: `== '{cat_val}'` splits {n_eq} vs {n_neq}")

    return "\n".join(lines)


def generate_python_model(tree: Dict[str, Any]) -> str:
    """Generate standalone deterministic Python code from serialized decision tree."""
    code_lines = [
        "#!/usr/bin/env python3",
        "# Auto-generated decision tree model derived from clinical guidelines (EAU/ESMO)",
        "# Generated by goals/02/experiments/exp_3/scripts/run_agentic_induction.py",
        "",
        "import math",
        "",
        "def _safe_num(val, default=0.0) -> float:",
        "    if val is None:",
        "        return default",
        "    try:",
        "        f = float(val)",
        "        return default if math.isnan(f) else f",
        "    except (ValueError, TypeError):",
        "        return default",
        "",
        "def predict_biopsy(row: dict) -> tuple[str, float]:",
        '    """Predict prostate biopsy decision (\'yes\' or \'no\') and positive probability.',
        "    Args:",
        "        row: Dict of clinical patient features.",
        "    Returns:",
        "        Tuple of (decision: 'yes'|'no', probability_yes: float)",
        '    """',
    ]

    def _recurse(node_id: str, indent: str) -> List[str]:
        node = tree[node_id]
        lines = []
        if node.get("is_leaf"):
            leaf_cls = node.get("leaf_class", "no")
            prob_yes = float(node.get("prob_yes", 1.0 if leaf_cls == "yes" else 0.0))
            lines.append(f"{indent}return '{leaf_cls}', {prob_yes:.4f}")
            return lines

        rule = node.get("rule") or {}
        feat = rule.get("feature", "radiology_report__pirads")
        op = rule.get("operator", "<=")
        thresh = rule.get("threshold", 3.0)

        is_numeric = isinstance(thresh, (int, float))
        if is_numeric:
            cond_str = f"_safe_num(row.get('{feat}')) {op} {thresh}"
        else:
            cond_str = f"str(row.get('{feat}', '')).strip().lower() {op} '{str(thresh).strip().lower()}'"

        lines.append(f"{indent}if {cond_str}:")
        left_child_id = node.get("left_child")
        lines.extend(_recurse(left_child_id, indent + "    "))

        lines.append(f"{indent}else:")
        right_child_id = node.get("right_child")
        lines.extend(_recurse(right_child_id, indent + "    "))

        return lines

    code_lines.extend(_recurse("root", "    "))
    return "\n".join(code_lines) + "\n"


def format_ascii_tree(tree: Dict[str, Any], node_id: str = "root", prefix: str = "", is_left: bool = True) -> str:
    """Render ASCII tree view."""
    node = tree.get(node_id)
    if not node:
        return ""

    if node.get("is_leaf"):
        leaf_cls = (node.get("leaf_class") or "unresolved").upper()
        return f"{prefix}└── [LEAF: {leaf_cls}] (n={node.get('n_samples')}, Gini={node.get('gini'):.4f}, dist={node.get('class_counts')})\n"

    rule = node.get("rule")
    if not rule:
        return f"{prefix}├── [NODE: {node_id} (Evaluating)] (n={node.get('n_samples')}, Gini={node.get('gini'):.4f}, dist={node.get('class_counts')})\n"

    feat = rule.get("feature", "feat")
    op = rule.get("operator", "<=")
    thresh = rule.get("threshold", 0)
    rule_str = f"{feat} {op} {thresh}"

    line = f"{prefix}├── [NODE: {node_id}] {rule_str} (n={node.get('n_samples')}, Gini={node.get('gini'):.4f})\n"
    left_str = format_ascii_tree(tree, node.get("left_child"), prefix + "│   ", is_left=True)
    right_str = format_ascii_tree(tree, node.get("right_child"), prefix + "    ", is_left=False)
    return line + left_str + right_str


def run_induction_pipeline(
    max_retries_per_node: int = 5,
    min_delta_gini: float = 0.0,
    min_samples_leaf: int = 3,
    max_global_turns: int = 30,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete closed-loop induction pipeline with Two-Stage Induction Decoupling."""
    scripts_dir, agents_dir, tools_decl_dir, code_tools_dir = resolve_paths()
    exp_dir = scripts_dir.parent
    goal_dir = exp_dir.parent.parent

    data_dir = goal_dir / "data"
    results_dir = exp_dir / "results"
    artifacts_dir = results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    blackboard_path = results_dir / "blackboard.md"
    blackboard = Blackboard(blackboard_path)

    # Ingest human directive into blackboard in English
    blackboard.write_section("human_guidance", RCT_HUMAN_DIRECTIVE)

    # Load Train partition
    inputs_df = pd.read_csv(data_dir / "inputs.csv")
    outputs_df = pd.read_csv(data_dir / "outputs.csv")[["case_id", "biopsy_decision"]]
    design_df = pd.read_csv(data_dir / "experimental-design.csv")[["case_id", "split"]]

    merged = inputs_df.merge(outputs_df, on="case_id").merge(design_df, on="case_id")
    train_df = merged[merged["split"] == "train"].copy()

    total_train = len(train_df)
    class_dist = train_df["biopsy_decision"].value_counts().to_dict()
    initial_gini = _calculate_gini(train_df["biopsy_decision"])

    problem_spec_text = (
        f"### 2. Clinical Problem & Dataset Specification\n"
        f"- **Task**: Binary prostate biopsy recommendation (`biopsy_decision`: 'yes' vs 'no')\n"
        f"- **Partition**: Strictly `split == 'train'` (N = {total_train} samples)\n"
        f"- **Distribution**: {class_dist} (Initial Root Gini = {initial_gini:.4f})\n"
        f"- **Clinical Scope**: Official 10 Ground Truth Clinical Variables (EAU / ESMO Guidelines via ChromaDB)\n"
        f"- **Induction Architecture**: Two-Stage Decoupling (Phase A: Variable Prioritization -> Phase B: Boundary Calibration)\n"
        f"- **Constraints**: Impurity reduction Delta Gini > {min_delta_gini}, Min samples/leaf >= {min_samples_leaf}, Patience = {max_retries_per_node} retries/node, Natural stopping."
    )
    blackboard.write_section("problem_spec", problem_spec_text)

    # Instantiate agents
    if verbose:
        print(f"[Induction] Initializing agent runners for Goal 02...")
    oracle_decl = load_agent_declaration("oracle_agent", agents_dir)
    evaluator_decl = load_agent_declaration("evaluator_agent", agents_dir)
    consolidator_decl = load_agent_declaration("consolidator_agent", agents_dir)

    oracle = SingleAgentRunner(oracle_decl, verbose=verbose)
    evaluator = SingleAgentRunner(evaluator_decl, verbose=verbose)
    consolidator = SingleAgentRunner(consolidator_decl, verbose=verbose)

    # Initialize root tree state
    tree: Dict[str, Any] = {
        "root": {
            "id": "root",
            "parent": None,
            "depth": 0,
            "active_filters": {},
            "sample_cases": train_df["case_id"].tolist(),
            "n_samples": total_train,
            "class_counts": class_dist,
            "gini": initial_gini,
            "is_leaf": False,
            "leaf_class": None,
            "prob_yes": class_dist.get("yes", 0) / total_train,
            "retries": 0,
            "rejected_rules": [],
            "rule": None,
            "left_child": None,
            "right_child": None,
        }
    }

    turn = 0
    start_time = time.time()

    while turn < max_global_turns:
        turn += 1

        # Find open candidate nodes:
        # Must have: not is_leaf, rule is None, gini > 0, n_samples >= 2 * min_samples_leaf
        open_nodes = [
            n for n in tree.values()
            if not n["is_leaf"] and n["rule"] is None and n["gini"] > 0.0 and n["n_samples"] >= (2 * min_samples_leaf)
        ]

        if not open_nodes:
            if verbose:
                print(f"[Induction] No open candidate nodes remaining. Tree fully consolidated.")
            break

        # Select node with largest sample size, tie-break by highest Gini
        open_nodes.sort(key=lambda x: (x["n_samples"], x["gini"]), reverse=True)
        active_node = open_nodes[0]
        node_id = active_node["id"]

        if verbose:
            print(f"\n{'='*70}\n>>> Turn #{turn} | Active Node: {node_id} (n={active_node['n_samples']}, Gini={active_node['gini']:.4f}, Depth={active_node['depth']})")

        # -------------------------------------------------------------
        # PHASE A: CLINICAL VARIABLE PRIORITIZATION (FEATURE SELECTION)
        # -------------------------------------------------------------
        general_overview_str = compute_general_overview(train_df, active_node["sample_cases"])

        rejected_features_clause = ""
        if active_node["rejected_rules"]:
            past_failed = [r.get("rule", "") for r in active_node["rejected_rules"]]
            rejected_features_clause = (
                f"\n🚨 PREVIOUS FAILED CANDIDATE ATTEMPTS FOR THIS NODE:\n"
                + "\n".join(f"- {f}" for f in past_failed)
                + f"\nYou must prioritize a clinical axis that has NOT failed or test a different viable cutoff.\n"
            )

        phase_a_prompt = (
            f"### PHASE A: CLINICAL VARIABLE PRIORITIZATION FOR NODE '{node_id}' (Depth: {active_node['depth']})\n"
            f"Active patient cohort distribution: {active_node['class_counts']} (N={active_node['n_samples']}, Gini={active_node['gini']:.4f}).\n"
            f"Prior branch filters leading to this node: {active_node['active_filters']}.\n"
            f"{rejected_features_clause}\n"
            f"Evaluate the 10 Ground Truth clinical variables and their clinical roles in the Context.\n"
            f"Which single variable represents the primary diagnostic priority to evaluate at this decision point according to EAU/ESMO guidelines?\n"
            f"Query ChromaDB via 'query_clinical_literature' if needed to substantiate your clinical choice.\n\n"
            f"Respond strictly in JSON with format:\n"
            f'{{\n  "status": "completed",\n  "selected_feature": "exact_variable_name_from_10_variables",\n  "clinical_rationale": "Medical explanation of why this variable is the primary decision priority here",\n  "guideline_citation": "EAU/ESMO Guidelines section"\n}}'
        )

        simulated_inputs_a = (
            f"# CLINICAL MISSION & 10 GROUND TRUTH VARIABLES\n"
            f"{RCT_HUMAN_DIRECTIVE}\n\n"
            f"# ACTIVE BLACKBOARD STATE\n"
            f"{blackboard.load_content()}\n\n"
            f"{general_overview_str}\n"
        )

        oracle_out_a = oracle.run(user_prompt=phase_a_prompt, simulated_inputs=simulated_inputs_a)
        parsed_a = oracle_out_a.get("parsed") or {}
        raw_feat = parsed_a.get("selected_feature") or parsed_a.get("feature_expression") or "radiology_report__pirads"
        resolved_feat = _resolve_feature_name(str(raw_feat)) or "radiology_report__pirads"
        rationale_a = parsed_a.get("clinical_rationale", "Primary clinical diagnostic discriminator.")
        citation_a = parsed_a.get("guideline_citation") or parsed_a.get("literature_citation") or "EAU Guidelines 2024"

        if verbose:
            print(f"  [Oracle - Phase A] Selected Feature: `{resolved_feat}` | Rationale: {rationale_a[:80]}...")

        # -------------------------------------------------------------
        # PHASE B: BOUNDARY & OPERATOR CALIBRATION
        # -------------------------------------------------------------
        feature_bounds_str = compute_feature_specific_bounds(train_df, active_node["sample_cases"], resolved_feat)

        retry_feedback_clause = ""
        matching_rejections = [r for r in active_node["rejected_rules"] if resolved_feat in r.get("rule", "")]
        if matching_rejections:
            retry_feedback_clause = (
                f"\n⚠️ PREVIOUS ATTEMPT FEEDBACK ON '{resolved_feat}':\n"
                + "\n".join(f"- Rule `{r['rule']}` failed: {r.get('feedback', '')}" for r in matching_rejections)
                + f"\nYou MUST choose a different operator/threshold from the viable cuts listed below.\n"
            )

        phase_b_prompt = (
            f"### PHASE B: BOUNDARY & OPERATOR CALIBRATION FOR '{resolved_feat}' AT NODE '{node_id}'\n"
            f"Active patient cohort distribution: {active_node['class_counts']} (N={active_node['n_samples']}, Gini={active_node['gini']:.4f}).\n"
            f"Selected clinical axis: `{resolved_feat}` (Rationale: {rationale_a})\n\n"
            f"{feature_bounds_str}\n\n"
            f"{retry_feedback_clause}\n"
            f"Propose the optimal urological decision splitting rule on `{resolved_feat}`.\n"
            f"Select an operator (<=, <, >=, >, ==, !=, in) and threshold that aligns with guideline evidence and partitions the active samples into valid branches (>= {min_samples_leaf} samples per branch).\n"
            f"Query ChromaDB via 'query_clinical_literature' if needed for specific cutoff thresholds.\n\n"
            f"Respond strictly in JSON with format:\n"
            f'{{\n  "status": "completed",\n  "result": {{\n    "target_leaf_id": "{node_id}",\n    "feature_expression": "{resolved_feat}",\n    "operator": "<=",\n    "threshold": 3.0,\n    "unit": "scale",\n    "target_class_if_true": "no",\n    "literature_citation": "{citation_a}",\n    "clinical_rationale": "Medical justification for this threshold..."\n  }}\n}}'
        )

        simulated_inputs_b = (
            f"# CLINICAL MISSION & 10 GROUND TRUTH VARIABLES\n"
            f"{RCT_HUMAN_DIRECTIVE}\n\n"
            f"# ACTIVE BLACKBOARD STATE\n"
            f"{blackboard.load_content()}\n\n"
            f"{feature_bounds_str}\n"
        )

        oracle_out_b = oracle.run(user_prompt=phase_b_prompt, simulated_inputs=simulated_inputs_b)
        oracle_res_b = (oracle_out_b.get("parsed") or {}).get("result", {})
        if not oracle_res_b and isinstance(oracle_out_b.get("parsed"), dict):
            oracle_res_b = oracle_out_b.get("parsed", {})

        feat_expr = _resolve_feature_name(str(oracle_res_b.get("feature_expression") or resolved_feat)) or resolved_feat
        op = oracle_res_b.get("operator", "<=")
        thresh = oracle_res_b.get("threshold", 3.0)
        rationale_b = oracle_res_b.get("clinical_rationale") or rationale_a
        citation_b = oracle_res_b.get("literature_citation") or citation_a

        if verbose:
            print(f"  [Oracle - Phase B] Proposed Split: `{feat_expr} {op} {thresh}` | Citation: {citation_b}")

        # -------------------------------------------------------------
        # EVALUATOR TURN
        # -------------------------------------------------------------
        eval_tool_res = compute_gini_split(
            feature=feat_expr,
            operator=op,
            threshold=thresh,
            target_leaf_id=node_id,
            sample_indices=active_node["sample_cases"],
            active_filter=active_node["active_filters"],
            split="train",
            min_delta_gini=min_delta_gini,
            min_samples_leaf=min_samples_leaf,
        )

        is_accepted = bool(eval_tool_res.get("ok") and eval_tool_res.get("decision") == "ACCEPTED")
        delta_gini = eval_tool_res.get("delta_gini", 0.0)

        if verbose:
            print(f"  [Evaluator] Verdict: {eval_tool_res.get('decision')} (Delta Gini: {delta_gini:.4f})")
            if not is_accepted:
                print(f"    Reason: {eval_tool_res.get('feedback')}")

        # -------------------------------------------------------------
        # CONSOLIDATOR TURN
        # -------------------------------------------------------------
        if is_accepted:
            resolved_col = eval_tool_res["resolved_feature"]
            eval_op = eval_tool_res["operator"]
            eval_thresh = eval_tool_res["threshold"]

            active_node["rule"] = {
                "feature": resolved_col,
                "operator": eval_op,
                "threshold": eval_thresh,
                "delta_gini": delta_gini,
                "clinical_rationale": rationale_b,
                "literature_citation": citation_b,
            }

            left_id = f"{node_id}_L"
            right_id = f"{node_id}_R"
            active_node["left_child"] = left_id
            active_node["right_child"] = right_id

            # Create Left Child (Condition is True)
            left_cases = eval_tool_res["left_case_ids"]
            left_dist = eval_tool_res["left_distribution"]
            left_samples = eval_tool_res["samples_left"]
            left_gini = eval_tool_res["left_gini"]
            left_prob_yes = left_dist.get("yes", 0) / max(left_samples, 1)

            left_filter = dict(active_node["active_filters"])
            left_filter[resolved_col] = {"operator": eval_op, "threshold": eval_thresh}

            # Terminal condition: pure or too few samples to split further (< 2 * min_samples_leaf)
            left_is_pure = bool(left_gini == 0.0 or left_samples < (2 * min_samples_leaf))
            left_maj_class = max(left_dist, key=left_dist.get) if left_dist else "no"

            tree[left_id] = {
                "id": left_id,
                "parent": node_id,
                "depth": active_node["depth"] + 1,
                "active_filters": left_filter,
                "sample_cases": left_cases,
                "n_samples": left_samples,
                "class_counts": left_dist,
                "gini": left_gini,
                "prob_yes": left_prob_yes,
                "is_leaf": left_is_pure,
                "leaf_class": left_maj_class if left_is_pure else None,
                "retries": 0,
                "rejected_rules": [],
                "rule": None,
                "left_child": None,
                "right_child": None,
            }

            # Create Right Child (Condition is False)
            right_cases = eval_tool_res["right_case_ids"]
            right_dist = eval_tool_res["right_distribution"]
            right_samples = eval_tool_res["samples_right"]
            right_gini = eval_tool_res["right_gini"]
            right_prob_yes = right_dist.get("yes", 0) / max(right_samples, 1)

            right_filter = dict(active_node["active_filters"])
            inv_map = {
                "<=": ">",
                "<": ">=",
                ">=": "<",
                ">": "<=",
                "==": "!=",
                "!=": "==",
            }
            inv_op = inv_map.get(eval_op, "!=")
            right_filter[resolved_col] = {"operator": inv_op, "threshold": eval_thresh}

            # Terminal condition: pure or too few samples to split further (< 2 * min_samples_leaf)
            right_is_pure = bool(right_gini == 0.0 or right_samples < (2 * min_samples_leaf))
            right_maj_class = max(right_dist, key=right_dist.get) if right_dist else "yes"

            tree[right_id] = {
                "id": right_id,
                "parent": node_id,
                "depth": active_node["depth"] + 1,
                "active_filters": right_filter,
                "sample_cases": right_cases,
                "n_samples": right_samples,
                "class_counts": right_dist,
                "gini": right_gini,
                "prob_yes": right_prob_yes,
                "is_leaf": right_is_pure,
                "leaf_class": right_maj_class if right_is_pure else None,
                "retries": 0,
                "rejected_rules": [],
                "rule": None,
                "left_child": None,
                "right_child": None,
            }

            # Log accepted hypothesis to blackboard
            blackboard.append_section(
                "hypothesis_log",
                f"#### Turn {turn}: Split ACCEPTED for Node '{node_id}'\n"
                f"- **Rule**: `{resolved_col} {eval_op} {eval_thresh}`\n"
                f"- **Literature Citation**: {citation_b}\n"
                f"- **Clinical Rationale**: {rationale_b}\n"
                f"- **Metrics**: Parent Gini={active_node['gini']:.4f} -> Weighted Child={eval_tool_res.get('weighted_child_gini'):.4f} (Delta Gini = {delta_gini:.4f})\n"
                f"- **Partitions**: Left (n={left_samples}, {left_dist}) | Right (n={right_samples}, {right_dist})"
            )

        else:
            # Rejection handling
            active_node["retries"] += 1
            rule_summary = f"{feat_expr} {op} {thresh}"
            rejection_entry = {
                "rule": rule_summary,
                "delta_gini": delta_gini,
                "decision": eval_tool_res.get("decision", "REJECTED"),
                "feedback": eval_tool_res.get("feedback", "No impurity reduction or degenerate partition."),
                "error": eval_tool_res.get("error"),
                "samples_left": eval_tool_res.get("samples_left", 0),
                "samples_right": eval_tool_res.get("samples_right", 0),
            }
            active_node["rejected_rules"].append(rejection_entry)

            blackboard.append_section(
                "rejected_hypotheses",
                f"- **Node '{node_id}' (Attempt {active_node['retries']}/{max_retries_per_node})**: Rule `{rule_summary}` REJECTED. {eval_tool_res.get('feedback')}"
            )

            if active_node["retries"] >= max_retries_per_node:
                # Patience budget exhausted: seal as terminal leaf
                active_node["is_leaf"] = True
                active_node["leaf_class"] = max(active_node["class_counts"], key=active_node["class_counts"].get)
                if verbose:
                    print(f"  [Consolidator] Node '{node_id}' reached patience budget ({max_retries_per_node}). Sealed as terminal leaf: {active_node['leaf_class']}.")

        # Update live tree state on blackboard
        ascii_tree = format_ascii_tree(tree, "root")
        blackboard.write_section("active_tree_state", f"```text\n{ascii_tree}\n```")

    # Seal any remaining open nodes as leaves
    for n in tree.values():
        if not n["is_leaf"] and n["rule"] is None:
            n["is_leaf"] = True
            n["leaf_class"] = max(n["class_counts"], key=n["class_counts"].get) if n["class_counts"] else "no"

    duration = time.time() - start_time
    ascii_tree = format_ascii_tree(tree, "root")
    print(f"\n{'='*70}\n[Consolidation Complete] Execution duration: {duration:.2f}s | Global Turns: {turn}")
    print(f"Final Decision Tree Topology:\n{ascii_tree}")

    # Generate Python model
    python_code = generate_python_model(tree)

    # Validate syntax with ast.parse
    try:
        ast.parse(python_code)
        if verbose:
            print("[Validation] Python model syntax verified with ast.parse() (Valid).")
    except SyntaxError as e:
        print(f"[Error] Generated model.py has syntax error: {e}")
        raise

    # Persist artifacts
    model_py_path = artifacts_dir / "model.py"
    model_py_path.write_text(python_code, encoding="utf-8")

    # Clean serializable tree
    clean_tree = {}
    for nid, ndata in tree.items():
        clean_tree[nid] = {
            "id": nid,
            "depth": ndata["depth"],
            "is_leaf": ndata["is_leaf"],
            "leaf_class": ndata.get("leaf_class"),
            "prob_yes": round(float(ndata.get("prob_yes", 0.0)), 4),
            "n_samples": ndata["n_samples"],
            "class_counts": ndata["class_counts"],
            "gini": ndata["gini"],
            "rule": ndata.get("rule"),
            "left_child": ndata.get("left_child"),
            "right_child": ndata.get("right_child"),
        }

    tree_json_path = artifacts_dir / "decision_tree.json"
    with open(tree_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_tree, f, indent=2)

    # Version tracking
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_hash = "git_hash_unavailable"

    version_path = results_dir / "version.txt"
    version_text = f"Git SHA: {git_hash}\nTimestamp: {datetime.now().isoformat()}\nConda Env: ml-harness\nNodes: {len(tree)}\n"
    version_path.write_text(version_text, encoding="utf-8")

    # Evaluate Train accuracy
    exec_scope: Dict[str, Any] = {}
    exec(python_code, exec_scope)
    predict_fn = exec_scope["predict_biopsy"]

    correct = 0
    for _, row in train_df.iterrows():
        pred_cls, _ = predict_fn(row.to_dict())
        if pred_cls == row["biopsy_decision"]:
            correct += 1
    train_acc = round(correct / total_train, 4)

    synthesis_text = (
        f"### 6. Final Model Synthesis & Verification\n"
        f"- **Status**: Induction Complete (Duration: {duration:.2f}s, Turns: {turn})\n"
        f"- **Total Nodes**: {len(tree)}\n"
        f"- **Train Accuracy**: {train_acc * 100:.2f}% ({correct}/{total_train})\n"
        f"- **Exported Artifacts**:\n"
        f"  * Topology: `{tree_json_path.relative_to(goal_dir)}`\n"
        f"  * Executable Routine: `{model_py_path.relative_to(goal_dir)}`\n\n"
        f"```text\n{ascii_tree}\n```"
    )
    blackboard.write_section("final_model_synthesis", synthesis_text)

    if verbose:
        print(f"[Telemetry] Train Accuracy: {train_acc * 100:.2f}% ({correct}/{total_train})")
        print(f"[Success] Artifacts persisted to {artifacts_dir.relative_to(goal_dir)}")

    return {
        "status": "success",
        "train_accuracy": train_acc,
        "nodes": len(tree),
        "duration": duration,
        "tree_json": str(tree_json_path),
        "model_py": str(model_py_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run literature-guided decision tree induction with Two-Stage Decoupling.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per node (default: 5)")
    parser.add_argument("--min-delta-gini", type=float, default=0.0, help="Min delta Gini (default: 0.0, any positive gain)")
    parser.add_argument("--min-samples-leaf", type=int, default=3, help="Min samples per leaf (default: 3)")
    parser.add_argument("--max-turns", type=int, default=30, help="Max global turns (default: 30)")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")

    args = parser.parse_args()
    run_induction_pipeline(
        max_retries_per_node=args.max_retries,
        min_delta_gini=args.min_delta_gini,
        min_samples_leaf=args.min_samples_leaf,
        max_global_turns=args.max_turns,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
