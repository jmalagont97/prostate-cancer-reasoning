"""exp_20: KDMClassifier (fagonzalezo/kdm's KDMClassModel, wrapped) for `decision`, tested against
the current promoted knn default, on decision's already-promoted 12-feature set (exp_16).
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402
from kdm_models import KDMClassifier  # noqa: E402


def kdm_default():
    return KDMClassifier(encoded_size=16, sigma=0.5, epochs=300, lr=1e-2)


def main():
    df = vt.load_labeled_df()
    baseline = vt.test_variant("decision=knn (current promoted)", {}, df=df, verbose=False)
    print(f"baseline (decision=knn): ranking_score={baseline['ranking_score']:.4f}")

    t0 = time.time()
    r = vt.test_variant("decision=KDMClassifier", {"decision": {"model_factory": kdm_default}}, df=df, verbose=False)
    dt = time.time() - t0
    print(f"decision=KDMClassifier: ranking_score={r['ranking_score']:.4f}  "
          f"(vs baseline {r['ranking_score']-baseline['ranking_score']:+.4f})  "
          f"decision_f1={r['decision_weighted_f1']:.4f}  case_score={r['mean_case_score']:.4f}  "
          f"[{dt:.0f}s]")


if __name__ == "__main__":
    main()
