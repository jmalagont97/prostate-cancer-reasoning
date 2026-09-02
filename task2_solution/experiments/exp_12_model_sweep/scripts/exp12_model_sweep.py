"""exp_12: model-family sweep for task2, one sub-task axis at a time (mirrors task3's exp_4 /
task1's exp_2 isolation discipline -- vary one sub-task's model_factory, hold the other two at
their current promoted default (knn, zero tools), measure the FULL ranking_score each time, not an
isolated component). 8-model roster from experiments/exp_2/scripts/models.py, unchanged.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402

EXP2_SCRIPTS = Path("experiments/exp_2/scripts")
sys.path.insert(0, str(EXP2_SCRIPTS))
from models import MODEL_REGISTRY  # noqa: E402


def main():
    df = vt.load_labeled_df()
    results = {"decision": {}, "confidence": {}, "variable_weights": {}}

    for subtask in ["decision", "confidence", "variable_weights"]:
        print(f"\n=== Sweeping {subtask} ===")
        for model_name, model_factory in MODEL_REGISTRY.items():
            r = vt.test_variant(
                f"{subtask}={model_name}",
                {subtask: {"model_factory": model_factory}},
                df=df, verbose=False,
            )
            results[subtask][model_name] = r
            print(f"  {model_name:14s} ranking_score={r['ranking_score']:.4f}  "
                  f"case_score={r['mean_case_score']:.4f}  decision_f1={r['decision_weighted_f1']:.4f}")

    print("\n=== Best per sub-task ===")
    baseline = vt.test_variant("baseline_all_knn", {}, df=df, verbose=False)
    print(f"baseline (knn everywhere): ranking_score={baseline['ranking_score']:.4f}")
    for subtask in results:
        best_name = max(results[subtask], key=lambda k: results[subtask][k]["ranking_score"])
        best = results[subtask][best_name]
        print(f"{subtask}: best={best_name} ranking_score={best['ranking_score']:.4f} "
              f"(vs baseline {baseline['ranking_score']:.4f}, diff={best['ranking_score']-baseline['ranking_score']:+.4f})")

    out = {"baseline": baseline, "sweeps": results}
    Path("exp12_sweep_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nWrote exp12_sweep_results.json")


if __name__ == "__main__":
    main()
