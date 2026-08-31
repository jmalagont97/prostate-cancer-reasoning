# Implementation Plan: exp_14

## Scope

Run the already implemented Task 3 agent over every case directory under `data/raw/train_release/task3/` without changing raw data. Stage a disposable copy under `/tmp`, mount it as the agent input, and preserve the Docker/GPU deployment used by the smoke test.

## Execution

1. Discover case directories containing `structured-prompt.json`; record the expected manifest.
2. Copy only those case directories to `/tmp/chimera-task3-exp14-input/` in the flat layout expected by `deploy/agent/inference.py`.
3. Run `chimera-agent:task3-cox` with `--gpus all`, `--network none`, input mounted read-only, and output mounted at `experiments/exp_14/results/agent_all_cases/outputs/`.
4. Capture Docker stdout/stderr and image/model/GPU metadata.

## Route audit correction

The first attempted batch exposed that the base entrypoint defaults to Task 1. The generic `/input/<case_id>/` path must therefore infer the task from every `structured-prompt.json` before selecting the staging namespace. Mixed-task batches are invalid and must fail closed. The regression suite covers Task 3 detection and mixed-task rejection.

The validated Docker invocation is:

```bash
docker run --rm --gpus all --network none \
  -v /tmp/chimera-task3-exp14-input:/input:ro \
  -v experiments/exp_14/results/agent_all_cases/outputs:/output \
  chimera-agent:task3-cox
```

Expected output root: `/output/task3/<case_id>/`. A successful Task 3 run must not create `/output/task1/`.
5. Validate both required JSON files for every discovered case and compare outcome fields with the deterministic Cox tool outputs available in the execution logs/agent implementation.
6. Join valid outcomes with the ground-truth outcome JSON files only after inference completes.
7. Compute Task 3 metrics exclusively through `utils/pkg/chimera_metrics`; record unavailable metrics explicitly as `null` rather than substituting local implementations.
8. Write a manifest, contract/coverage report, metrics JSON/CSV, and `version.txt` containing the exact image digest and repository state. Keep failures visible and do not silently drop cases.

## Conditions and artifacts

- `agent_all_cases`: all discovered cases; contract, consistency, latency, logs, and outputs.
- `annotated_metrics`: annotated subset of `agent_all_cases`; in-sample exploratory metrics only.

No source-code changes are planned for this execution. If the current runner cannot process the corpus, stop and report the blocking interface mismatch rather than silently altering the pipeline.
