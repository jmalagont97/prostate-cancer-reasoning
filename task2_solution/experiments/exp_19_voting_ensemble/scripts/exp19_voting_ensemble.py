"""exp_19: a 3-model majority-vote ensemble (knn + logistic + random_forest) for `decision`, tested
against the current promoted single-model default (knn alone), on decision's already-promoted
12-feature set (10 base + psa_kinetics, exp_16). confidence/variable_weights held at their own
promoted defaults throughout.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402

sys.path.insert(0, "experiments/exp_2/scripts")
from models import knn, logistic, random_forest  # noqa: E402

from sklearn.ensemble import VotingClassifier  # noqa: E402


def voting_hard():
    return VotingClassifier(
        estimators=[("knn", knn()), ("logistic", logistic()), ("random_forest", random_forest())],
        voting="hard",
    )


def voting_soft():
    return VotingClassifier(
        estimators=[("knn", knn()), ("logistic", logistic()), ("random_forest", random_forest())],
        voting="soft",
    )


def main():
    df = vt.load_labeled_df()
    baseline = vt.test_variant("decision=knn (current promoted)", {}, df=df, verbose=False)
    print(f"baseline (decision=knn): ranking_score={baseline['ranking_score']:.4f}\n")

    for name, factory in [("voting_hard (knn+logistic+rf, majority vote)", voting_hard),
                            ("voting_soft (knn+logistic+rf, averaged proba)", voting_soft)]:
        r = vt.test_variant(name, {"decision": {"model_factory": factory}}, df=df, verbose=False)
        print(f"{name}: ranking_score={r['ranking_score']:.4f}  "
              f"(vs baseline {r['ranking_score']-baseline['ranking_score']:+.4f})  "
              f"decision_f1={r['decision_weighted_f1']:.4f}  case_score={r['mean_case_score']:.4f}")


if __name__ == "__main__":
    main()
