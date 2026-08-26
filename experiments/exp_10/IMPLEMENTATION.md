# Implementation Plan: exp_10 full-cohort Task 1 evaluation

**Design:** `experiments/exp_10/DESIGN.md`  
**Status:** Approved for execution

## Scope

- Use the 91 cases under `data/raw/train_release/task1` that contain both official Task-1 annotation files.
- Do not modify the source dataset or `deploy/agent`.
- Stage only symlinks under `experiments/exp_10/runtime/task1/agent_input/`.
- Reuse the existing Task-1 runner and the local `gemma4:e2b` Ollama model.
- Save raw official-style artifacts, normalized records, metrics, logs, and comparison with `exp_1`.

## Execution stages

1. Discover annotated cases and write a manifest.
2. Run one-case smoke execution in `tmux:0`.
3. Validate the two required artifacts and their JSON contracts.
4. Run the full annotated cohort in the same tmux pane.
5. Evaluate with the authors' deterministic evaluator logic (without the unavailable/external LLM judge).
6. Produce a concise report comparing the result with `experiments/exp_1/reports/summary.md`.

## Runtime reporting

The runner must emit one compact line per case:
`[exp10] i/N case_id status elapsed_s`

A final line must report completion, successful/failed counts, and elapsed time. Detailed model output goes to `run.log`, not to the live progress stream.

## Acceptance criteria

- All annotated cases are attempted and counted explicitly.
- No source data are overwritten.
- Every successful case has both official output files.
- Malformed/incomplete cases are recorded as failures, never silently skipped.
- Metrics are labelled as optimistic in-sample and compared to exp_1's 91-case baseline.
- No full MCCV or large external evaluation is run in this experiment.
