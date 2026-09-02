"""exp_16: does adding psa-kinetics (psa_trend velocity) and/or lab-panel features to `decision`'s
feature set improve ranking_score? Isolates the feature-set axis: model stays at the current
promoted default (knn) for this first pass, mirroring task3's exp_10 isolation discipline.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402
from task2_subtask_config import TASK2_BASE_NUMERIC  # noqa: E402

PSA_KINETICS = ["psa_velocity_per_year", "psa_ratio_last_first"]
LABS = ["free_psa", "pct_free_psa", "testosterone", "creatinine", "alk_phos", "ldh", "egfr_reduced"]

VARIANTS = {
    "baseline_10feat": TASK2_BASE_NUMERIC,
    "+psa_kinetics": TASK2_BASE_NUMERIC + PSA_KINETICS,
    "+labs": TASK2_BASE_NUMERIC + LABS,
    "+psa_kinetics+labs": TASK2_BASE_NUMERIC + PSA_KINETICS + LABS,
}


def main():
    df = vt.load_labeled_df()
    print(f"{'variant':>22s} {'n_feat':>7s} {'ranking_score':>14s} {'decision_f1':>12s}")
    results = {}
    for name, feats in VARIANTS.items():
        r = vt.test_variant(name, {"decision": {"numeric": feats}}, df=df, verbose=False)
        results[name] = r
        print(f"{name:>22s} {len(feats):7d} {r['ranking_score']:14.4f} {r['decision_weighted_f1']:12.4f}")

    best = max(results, key=lambda k: results[k]["ranking_score"])
    print(f"\nBest: {best} (ranking_score={results[best]['ranking_score']:.4f}) "
          f"vs baseline {results['baseline_10feat']['ranking_score']:.4f}")


if __name__ == "__main__":
    main()
