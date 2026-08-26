# Experiment Design: Exploratory Full-Cohort Evaluation of `deploy/agent` on Task 1

**Experiment**: `experiments/exp_10/`  
**Project**: `prostate-cancer-reasoning`  
**Date**: 2026-08-25  
**Author**: Juan Sebastian Malagón Torres  
**Status**: Complete (exploratory)

---

## 1. Hypothesis

`deploy/agent` produces valid CHIMERA Task-1 artifacts and, when evaluated against all paired annotated cases, obtains a deterministic ranking score comparable to or better than the `exp_1` ReAct baseline; because the models were trained/developed with these data, this result is an optimistic in-sample upper-bound diagnostic, not an estimate of generalization.

## 2. Experimental Setup

- **Task**: CHIMERA-Agent Task 1 — pre-biopsy decision.
- **Cohort**: every case under `data/raw/train_release/task1/` containing both official ground-truth files; `exp_1` reports 91 paired cases, which is the expected primary cohort size.
- **Prediction system**: `deploy/agent`, including its deterministic submodels and LLM reasoning output.
- **Ground truth**: the same paired annotations used by `exp_1`, without changing labels or mappings.
- **Reference evaluator**: `docs/chimera-agent-evalaution/evaluation/evaluate.py`; `chimera_metrics` is used only after smoke-level parity checks.
- **Comparison**: `exp_1/results/baseline/aggregate_metrics.json` and `exp_1/reports/summary.md`.
- **Interpretation**: exploratory resubstitution/in-sample evaluation; no claim of external validity.

## 3. Conditions

| Condition | Description | Purpose |
|---|---|---|
| `cond_agent_full_cohort` | `deploy/agent` predictions on all paired annotated Task-1 cases | Main exploratory result |
| `cond_exp1_baseline` | Existing official ReAct baseline from `exp_1` | Direct historical comparison |
| `cond_oracle` | Ground truth compared with itself, if supported by the evaluator | Verify metric upper bounds and parser correctness |

No hyperparameter sweep, retraining, ablation, or test-set claim is included in this experiment.

## 4. Evaluation Metrics

The primary metric must be reported exactly as defined by the authors:

```text
R_1 = 0.5 * mean_case_score + 0.5 * F1_yes
```

Report the following metrics separately, not only the aggregate score:

| Priority | Metric | Required interpretation |
|---|---|---|
| Primary | Official Ranking Score `R_1` | Main comparison with `exp_1` value `0.6103` |
| Primary | `mean_case_score` | Reasoning/composite contribution; compare with `0.4348` |
| Decision | Gate accuracy / pass rate | Compare with `0.6703` (`61/91`) |
| Decision | Positive-class biopsy `F1_yes` | Compare with `0.7857`; report precision and recall |
| Decision | Confusion matrix | TP, FP, FN, TN; emphasize over-biopsy bias and specificity |
| Reasoning | Confidence score | Compare with `0.7473` |
| Reasoning | Variable-weights score | Compare with `0.6463` over the 10 clinical variables |
| Reasoning | Important/decisive set-F1 | Compare with `0.5672` |
| Agentic | Cost-aware tool efficiency | Compare with `1.0000`, while noting the deployment trace limitation |
| Agentic | Section grounding | Compare with `0.2431`; report grounded, ungrounded and ungradable counts |

For every metric report: absolute value, difference versus `exp_1`, relative direction (`improved`/`degraded`), and denominator (`N`).

## 5. Decision Rules

- **Artifact validity**: pass only if every annotated case produces the two official files with valid JSON and the expected schema.
- **Evaluator compatibility**: pass only if the official evaluator can parse every generated case without manual repair.
- **Exploratory improvement**: report improvement only when `R_1` exceeds `0.6103`; also inspect whether the change is caused by decision F1 or reasoning metrics.
- **Clinical decision warning**: any increase in `F1_yes` must be accompanied by the confusion matrix, precision, recall, false-positive rate and specificity. A higher score driven by more positive predictions is not automatically clinically preferable.
- **Reasoning interpretation**: do not claim that a high reasoning score proves grounded clinical reasoning; report the external judge component separately if available from the authors' evaluator.
- **No generalization claim**: results must be labelled optimistic/in-sample because model development used the same cohort.

## 6. Biases, Risks, and Controls

- **Training-set leakage / resubstitution bias**: disclose prominently and do not compare directly with an out-of-sample benchmark as if equivalent.
- **Metric implementation drift**: use the authors' `evaluate.py` as the normative oracle; verify `chimera_metrics` only with small differential smoke cases.
- **Artifact mismatch**: evaluate the two official output files (`prostate-biopsy-decision.json` and `prostate-biopsy-decision-reasoning.json`), not the internal boolean `prediction.json`.
- **Partial cohort execution**: require an exact case-count and case-ID reconciliation against the paired ground-truth cohort; missing cases invalidate the aggregate result.
- **Jueces LLM externos**: do not replace or reinterpret the organizers' reasoning judge; if unavailable, mark that metric unavailable rather than imputing it.

## 7. Reproducibility Requirements

- Record the exact code version and dirty working-tree state.
- Preserve the complete per-case predictions and evaluator output.
- Preserve the case-ID manifest used for pairing.
- Record model/checkpoint identifiers and inference configuration.
- Record the exact evaluator source/version and whether the official container or local deterministic implementation was used.
- Store results under `experiments/exp_10/results/` and the comparison report under `experiments/exp_10/reports/`.

## 8. Next Step

After approval of this research design, create a separate implementation plan describing the minimal scripts/configuration needed to generate all Task-1 predictions and run the official evaluator. Do not launch the full-cohort run until that implementation plan is approved.
