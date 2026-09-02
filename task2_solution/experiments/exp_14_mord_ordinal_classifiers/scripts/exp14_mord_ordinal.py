"""exp_14: does a GENUINE ordinal classifier (mord's proportional-odds / cumulative-link models)
beat exp_11's nominal knn or exp_13's regress-and-round proxy for confidence/variable_weights?
Structurally different mechanism from exp_13: mord's LogisticIT/LogisticAT/LogisticSE fit ordinal
threshold boundaries directly (discriminative), not a regression pulled toward the mean.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402

import mord  # noqa: E402

MORD_MODELS = {
    "mord_LogisticIT": lambda: mord.LogisticIT(),
    "mord_LogisticAT": lambda: mord.LogisticAT(),
    "mord_LogisticSE": lambda: mord.LogisticSE(),
}


def main():
    df = vt.load_labeled_df()
    baseline = vt.test_variant("baseline (knn nominal)", {}, df=df, verbose=False)
    print(f"baseline (knn, nominal classification): ranking_score={baseline['ranking_score']:.4f}\n")

    print("=== confidence: mord ordinal classifiers ===")
    for name, factory in MORD_MODELS.items():
        r = vt.test_variant(f"confidence={name}", {"confidence": {"model_factory": factory}}, df=df, verbose=False)
        print(f"  {name:18s} ranking_score={r['ranking_score']:.4f}  confidence_score={r['mean_confidence_score']:.4f}")

    print("\n=== variable_weights: mord ordinal classifiers ===")
    for name, factory in MORD_MODELS.items():
        r = vt.test_variant(f"variable_weights={name}", {"variable_weights": {"model_factory": factory}}, df=df, verbose=False)
        print(f"  {name:18s} ranking_score={r['ranking_score']:.4f}  var_weight_score={r['mean_variable_weight_score']:.4f}  "
              f"factor_f1={r['mean_important_decisive_factor_score']:.4f}")


if __name__ == "__main__":
    main()
