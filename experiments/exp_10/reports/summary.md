# Experiment Report: Exploratory Full-Cohort Evaluation of `deploy/agent` on Task 1

**Experiment**: `experiments/exp_10/`  
**Project**: `prostate-cancer-reasoning`  
**Report date**: 2026-08-25  
**Plan date**: 2026-08-25  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete (exploratory)

---

## 1. Summary

`deploy/agent` generated valid Task-1 artifacts for all 91 annotated cases from the `exp_1` cohort. Using the same `chimera_metrics`-based deterministic aggregation used by `exp_1`, the agent obtained `R_1 = 0.8678`, compared with `0.6103` for the previous baseline (`Δ = +0.2576`). The improvement is driven by substantially better decision specificity and near-perfect deterministic reasoning fields. This is an optimistic in-sample result because the evaluated cases were part of the development/training cohort.

---

## 2. Hypothesis & Verdict

**Hypothesis:** `deploy/agent` produces valid official Task-1 artifacts and an optimistic in-sample ranking score comparable to `exp_1`.

**Verdict:** ✅ Supported for the exploratory objective.

**Evidence:** All 91 cases produced both official output files and passed the package-compatible evaluation path. The ranking score improved from `0.6103` to `0.8678`.

---

## 3. Experimental Setup (as run)

- **Dataset:** `data/raw/train_release/task1/`; 91 cases with both annotated decision and reasoning files, matching the `exp_1` cohort.
- **Agent:** `deploy/agent`, Task 1 baseline, Gemma 4 E2B through local Ollama.
- **Models:** `deploy/tasks/task1/training/models/` (`subtask1_ensemble.pkl`, `subtask2_kdm.pkl`, `subtask3_extratrees.pkl`).
- **Execution:** `tmux:0`; one smoke case followed by the complete 91-case run.
- **Runtime:** 690.9 s for the full generation run (approximately 7.6 s/case).
- **Evaluation:** `experiments/exp_10/scripts/evaluate_deploy_task1.py`, using the same deterministic `chimera_metrics` API and aggregation convention as `exp_1`.
- **Deviation:** The official authors' container was not invoked; this is an exp_1-compatible local evaluation, not an official leaderboard submission.

---

## 4. Artifact Verification

| Artifact | Observed |
|---|---:|
| Annotated cases targeted | 91 |
| Decision files generated | 91 |
| Reasoning files generated | 91 |
| Internal prediction files generated | 91 |
| Missing paired outputs | 0 |

The pilot case was also checked manually: the generated decision was `"no"`, matching its annotation, and both official filenames were present.

---

## 5. Results

### 5.1 Primary and decision metrics

| Metric | `exp_1` baseline | `exp_10` deploy/agent | Δ |
|---|---:|---:|---:|
| **Official-style `R_1`** | **0.6103** | **0.8678** | **+0.2576** |
| Mean case score | 0.4348 | 0.8126 | +0.3778 |
| Decision gate accuracy | 0.6703 (61/91) | 0.9121 (83/91) | +0.2418 |
| Biopsy F1 (yes) | 0.7857 | 0.9231 | +0.1374 |
| Precision (yes) | 0.6548 | 1.0000 | +0.3452 |
| Recall (yes) | 0.9821 | 0.8571 | −0.1250 |

### 5.2 Confusion matrix

| | Predicted biopsy | Predicted no biopsy |
|---|---:|---:|
| **GT biopsy** | TP = 48 | FN = 8 |
| **GT no biopsy** | FP = 0 | TN = 35 |

The main change relative to `exp_1` is the elimination of false positives, at the cost of 8 false negatives.

### 5.3 Reasoning and agentic metrics

| Metric | `exp_1` baseline | `exp_10` deploy/agent | Δ |
|---|---:|---:|---:|
| Confidence score | 0.7473 | 0.9945 | +0.2473 |
| Variable-weights score | 0.6463 | 1.0000 | +0.3537 |
| Important/decisive set-F1 | 0.5672 | 1.0000 | +0.4328 |
| Tool efficiency | 1.0000 | 0.4800 | −0.5200 |
| Section grounding | 0.2431 | 0.9863 | +0.7432 |

The LLM rationale judge was not executed; its evaluation remains external to the authors' evaluator environment.

---

## 6. Interpretation

The result strongly supports operational compatibility and shows a large optimistic improvement over the `exp_1` baseline under the same local deterministic scoring convention. The zero false-positive count is clinically notable, but the recall reduction from `0.9821` to `0.8571` means the model became more conservative toward recommending biopsy.

The perfect variable-weight and decisive-factor scores should not be interpreted as independent evidence of generalization: the evaluated cohort was used during development, and the deployment output derives structured fields from trained submodels. The near-perfect grounding score is likewise an in-sample diagnostic result.

---

## 7. Missing Data & Caveats

- No official authors' evaluator container run was performed.
- No external/held-out cohort was evaluated.
- No MCCV-30 result was produced.
- No LLM-as-a-judge score was available locally.
- The result is therefore a **tentative optimistic implementation result**, not a challenge-performance estimate.

---

## 8. Conclusion and Next Steps

- `deploy/agent` successfully generated the expected Task-1 decision and reasoning files for all 91 annotated cases.
- Under the exp_1-compatible deterministic evaluator, `R_1` increased from `0.6103` to `0.8678`.
- The next scientifically meaningful step is an evaluation on cases excluded from model development, followed by the official evaluator container when its complete execution environment is available.
