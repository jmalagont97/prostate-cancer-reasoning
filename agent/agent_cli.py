#!/usr/bin/env python3
"""
agent/agent_cli.py

Tool-Agnostic Pathology Reasoning Agent Executable.
Orchestrates sequential Tool calls (Subtasks 1.1 -> 1.2 -> 1.3) from deploy/,
queries local Ollama Gemma LLM for clinical rationale as a Urology Specialist,
and enforces Pydantic validation matching 'prostate-biopsy-decision-reasoning.json'.

Usage:
  python agent_cli.py --input_file path/to/structured-prompt.json --model gemma4:e2b
  python agent_cli.py --input_file path/to/raw_case_folder/
  python agent_cli.py --case_id PT-pseudo_0020cfca66c8
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "chimera26" / "preprocessed" / "task1"

sys.path.insert(0, str(AGENT_DIR))

from schemas import PathologyReasoningOutput, VariableWeights
from tools import (
    tool_predict_biopsy_decision,
    tool_predict_clinical_confidence,
    tool_predict_relevance_and_reveal_sequence
)
from ollama_client import query_ollama_gemma, check_ollama_status, DEFAULT_MODEL


SYSTEM_PROMPT = """You are a Senior Specialist in Urology (Especialista en Urología).
Your role is to analyze a prostate patient's case and synthesize a concise, expert urological rationale (free_text) explaining the exact medical reasons behind the diagnostic decision.

IMPORTANT RULE (IMMUTABLE TOOLS):
The decisions returned by the diagnostic tools (biopsy_decision, confidence, variable_weights, reveal_sequence) are STAGE-1 IMMUTABLE LAWS and CANNOT be altered or overridden. Your task is to provide the expert urological explanation ('free_text') justifying these IMMUTABLE decisions.

You MUST produce a valid JSON object strictly matching this schema:
{
  "biopsy_decision": "yes" or "no",
  "confidence": "uncertain", "borderline", or "clear",
  "variable_weights": {
    "age": "not_used" | "noted" | "important" | "decisive",
    "fh": "not_used" | "noted" | "important" | "decisive",
    "cspca": "not_used" | "noted" | "important" | "decisive",
    "pirads": "not_used" | "noted" | "important" | "decisive",
    "vol": "not_used" | "noted" | "important" | "decisive",
    "psa": "not_used" | "noted" | "important" | "decisive",
    "comorbidity": "not_used" | "noted" | "important" | "decisive",
    "psad": "not_used" | "noted" | "important" | "decisive",
    "dre": "not_used" | "noted" | "important" | "decisive",
    "bx": "not_used" | "noted" | "important" | "decisive"
  },
  "reveal_sequence": ["radiology_report", ...],
  "free_text": "Direct, concise urological specialist rationale (e.g., 'PIRADS 5 lesion in 68 year old man with elevated PSA')."
}
"""


def load_patient_case_from_dataset(case_id):
    inputs_df = pd.read_csv(DATA_DIR / "inputs.csv").set_index("case_id")
    text_df = pd.read_csv(DATA_DIR / "full_prompt_narrative.csv").set_index("case_id")
    images_df = pd.read_csv(DATA_DIR / "images.csv").set_index("case_id")

    if case_id not in inputs_df.index:
        raise ValueError(f"case_id '{case_id}' not found in preprocessed dataset.")

    tab_data = inputs_df.loc[case_id].to_dict()
    text_data = text_df.loc[case_id, "txt_full_prompt_narrative"]
    mri_cols = [c for c in images_df.columns if c.startswith("mri_emb_")]
    mri_data = images_df.loc[case_id, mri_cols].values

    return {"tabular": tab_data, "text": text_data, "mri_emb": mri_data, "case_id": case_id}


def load_patient_case_from_file(file_path):
    """
    Loads patient data from a JSON file or directory path, handling raw structured-prompt.json format.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Specified input path does not exist: {path}")

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "tabular" in data or "text" in data or "mri_emb" in data:
            tab_data = data.get("tabular", {})
            text_data = data.get("text", "")
            mri_data = np.array(data.get("mri_emb", np.zeros(1024)))
        else:
            # Raw structured-prompt.json format
            tab_data = data

            # Extract narrative text
            if "full_prompt_narrative" in data and isinstance(data["full_prompt_narrative"], str):
                text_data = data["full_prompt_narrative"]
            elif "note_sections" in data and isinstance(data["note_sections"], list):
                sec_texts = [f"{sec.get('s', '')}: {sec.get('t', '')}" for sec in data["note_sections"]]
                text_data = " | ".join(sec_texts)
                if "notes" in data and isinstance(data["notes"], str):
                    text_data += " | " + data["notes"]
            else:
                text_data = str(data.get("notes", ""))

            mri_data = np.array(data.get("mri_embedding", data.get("mri_emb", np.zeros(1024))))

        return {
            "tabular": tab_data,
            "text": text_data,
            "mri_emb": mri_data,
            "case_id": data.get("case_id", path.parent.name if path.parent.name != "task1" else path.stem)
        }

    elif path.is_dir():
        struct_file = path / "structured-prompt.json"
        cli_file = path / "prostate-biopsy-decision-clinical-data.json"
        mri_file = path / "prostate-modality-level-neural-representations.json"

        tab_data = {}
        text_data = ""
        mri_data = np.zeros(1024)

        if struct_file.exists():
            with open(struct_file, "r") as f:
                s_data = json.load(f)
                if "full_prompt_narrative" in s_data:
                    text_data = s_data["full_prompt_narrative"]
                elif "note_sections" in s_data:
                    sec_texts = [f"{sec.get('s', '')}: {sec.get('t', '')}" for sec in s_data["note_sections"]]
                    text_data = " | ".join(sec_texts)
                tab_data.update(s_data)

        if cli_file.exists():
            with open(cli_file, "r") as f:
                c_data = json.load(f)
                tab_data.update(c_data)

        if mri_file.exists():
            with open(mri_file, "r") as f:
                m_data = json.load(f)
                if isinstance(m_data, list):
                    mri_data = np.array(m_data)

        return {
            "tabular": tab_data,
            "text": text_data,
            "mri_emb": mri_data,
            "case_id": path.name
        }


def run_pathology_reasoning_agent(raw_case, model_name=DEFAULT_MODEL, mock_llm=False):
    """
    Tool-Agnostic Agent Execution Flow:
      1. Calls Tool 1.1 -> Biopsy Decision (yes/no) & probabilities (IMMUTABLE)
      2. Calls Tool 1.2 -> Clinical Confidence (uncertain/borderline/clear) (IMMUTABLE)
      3. Calls Tool 1.3 -> 10 Variable Weights & Section Reveal Sequence (IMMUTABLE)
      4. Queries Ollama Gemma as a Urology Specialist to write concise urological rationale ('free_text')
      5. Enforces Pydantic schema validation
    """
    # 1. Execute Subtask 1.1 Tool (IMMUTABLE)
    res_1_1 = tool_predict_biopsy_decision(raw_case)
    
    # 2. Execute Subtask 1.2 Tool (IMMUTABLE)
    res_1_2 = tool_predict_clinical_confidence(raw_case)
    
    # 3. Execute Subtask 1.3 Tool (IMMUTABLE)
    res_1_3 = tool_predict_relevance_and_reveal_sequence(raw_case)

    # Consolidate IMMUTABLE tool predictions
    biopsy_decision = res_1_1["biopsy_decision_label"]
    confidence = res_1_2["confidence_label"]
    variable_weights_dict = res_1_3["relevance_labels"]
    reveal_sequence = res_1_3["reveal_sequence"]

    # Extract clinical values for urological reasoning (handling both cli_ prefixed and raw key names)
    tab = raw_case.get("tabular", {})
    pirads_val = tab.get("cli_pirads", tab.get("pirads", "N/A"))
    psa_val = tab.get("cli_psa", tab.get("psa", "N/A"))
    age_val = tab.get("cli_age", tab.get("age", "N/A"))

    # 4. Generate Urology Specialist Free-Text Rationale
    if mock_llm:
        if biopsy_decision == "yes":
            free_text = f"PIRADS {pirads_val} lesion in {int(age_val) if isinstance(age_val, (int, float)) else age_val} year old man with elevated PSA of {psa_val} ng/mL."
        else:
            free_text = f"PIRADS {pirads_val} with PSA level of {psa_val} ng/mL and no decisive high-risk findings."
    else:
        prompt_content = f"""PATIENT CLINICAL DATA (UROLOGY CONSULTATION):
- Age: {age_val} years
- PSA: {psa_val} ng/mL
- PI-RADS Score: {pirads_val}
- PSA Density (PSAD): {tab.get('cli_psad', tab.get('psad', 'N/A'))}
- DRE: {tab.get('cli_dre', tab.get('dre', 'N/A'))}
- Comorbidities Count: {tab.get('cli_comorbidity_count', tab.get('comorbidity', 'N/A'))}

IMMUTABLE DIAGNOSTIC TOOL DECISIONS:
- Biopsy Decision: {biopsy_decision.upper()} (Probability: {res_1_1['fused_probability']:.4f}) [IMMUTABLE]
- Clinical Confidence Grade: {confidence.upper()} [IMMUTABLE]
- Assigned Relevance Weights: {json.dumps(variable_weights_dict)} [IMMUTABLE]
- Recommended Reveal Sequence: {json.dumps(reveal_sequence)} [IMMUTABLE]

INSTRUCTION FOR UROLOGY SPECIALIST:
As a Senior Specialist in Urology, write a direct, 1-sentence expert clinical explanation ('free_text') justifying why a biopsy decision of '{biopsy_decision.upper()}' with '{confidence}' confidence is indicated.
Incorporate key clinical markers like PI-RADS score ({pirads_val}), PSA level ({psa_val} ng/mL), and age ({age_val}).
Do NOT change any of the IMMUTABLE tool outputs. Return the valid JSON object."""

        try:
            llm_response = query_ollama_gemma(prompt_content, system_prompt=SYSTEM_PROMPT, model=model_name)
            free_text = llm_response.get("free_text", f"PIRADS {pirads_val} lesion with PSA of {psa_val} ng/mL supporting biopsy decision '{biopsy_decision}'.")
        except Exception as e:
            print(f"[Warning] Ollama LLM query failed ({e}). Falling back to baseline urologist rationale.")
            if biopsy_decision == "yes":
                free_text = f"PIRADS {pirads_val} lesion in {int(age_val) if isinstance(age_val, (int, float)) else age_val} year old man with elevated PSA of {psa_val} ng/mL."
            else:
                free_text = f"PIRADS {pirads_val} with PSA level of {psa_val} ng/mL and no decisive high-risk findings."

    # 5. Enforce Pydantic Schema Validation with IMMUTABLE tool values
    var_weights_pydantic = VariableWeights(**variable_weights_dict)
    
    final_output = PathologyReasoningOutput(
        biopsy_decision=biopsy_decision,
        confidence=confidence,
        variable_weights=var_weights_pydantic,
        reveal_sequence=reveal_sequence,
        free_text=free_text
    )

    return final_output.model_dump()


def main():
    parser = argparse.ArgumentParser(description="Pathology Reasoning Agent CLI")
    parser.add_argument("--input_file", "-i", type=str, help="Path to patient raw JSON file or directory")
    parser.add_argument("--case_id", type=str, help="Evaluate patient case_id from dataset")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama LLM model name (default: gemma:2b)")
    parser.add_argument("--mock_llm", action="store_true", help="Run without calling Ollama server")
    parser.add_argument("--output_file", "-o", type=str, help="Path to save output JSON file")
    args = parser.parse_args()

    if args.input_file:
        case_source = f"File/Dir: '{args.input_file}'"
        raw_case = load_patient_case_from_file(args.input_file)
    elif args.case_id:
        case_source = f"Case ID: '{args.case_id}'"
        raw_case = load_patient_case_from_dataset(args.case_id)
    else:
        default_case_id = "PT-pseudo_0020cfca66c8"
        case_source = f"Default Case ID: '{default_case_id}'"
        raw_case = load_patient_case_from_dataset(default_case_id)

    print("\n" + "=" * 80)
    print(f"EXECUTING UROLOGY SPECIALIST AGENT ON {case_source.upper()}")
    print("=" * 80)

    result_json = run_pathology_reasoning_agent(raw_case, model_name=args.model, mock_llm=args.mock_llm)

    print("\nFINAL PYDANTIC-VALIDATED PATHOLOGY REASONING OUTPUT:")
    print("=" * 80)
    print(json.dumps(result_json, indent=2))
    print("=" * 80)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"✓ Saved agent output JSON to: {out_path}")


if __name__ == "__main__":
    main()
