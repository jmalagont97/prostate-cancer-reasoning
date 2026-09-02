"""exp_22: does KDMClassModel's documented generative co-training (freeze encoder + add
-generative*log_marginal to the loss for the back half of training) fix exp_20's KDMClassifier
underperformance on `decision`? Swept over generative weight; exp_20's baseline (generative=0.0,
ranking_score=0.6944) is the comparison point, not re-run.
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402
from kdm_models import KDMClassifier  # noqa: E402


def main():
    df = vt.load_labeled_df()
    print("exp_20 baseline for reference: decision=knn ranking_score=0.7429, "
          "decision=KDMClassifier(generative=0.0) ranking_score=0.6944\n")

    for gen_weight in [0.5, 1.0, 2.0]:
        def factory(g=gen_weight):
            return KDMClassifier(encoded_size=16, sigma=0.5, epochs=300, lr=1e-2,
                                  generative=g, freeze_epoch_frac=0.5)
        t0 = time.time()
        r = vt.test_variant(f"decision=KDMClassifier(generative={gen_weight})",
                             {"decision": {"model_factory": factory}}, df=df, verbose=False)
        dt = time.time() - t0
        print(f"generative={gen_weight}: ranking_score={r['ranking_score']:.4f}  "
              f"(vs knn baseline {r['ranking_score']-0.7429:+.4f})  "
              f"decision_f1={r['decision_weighted_f1']:.4f}  case_score={r['mean_case_score']:.4f}  [{dt:.0f}s]")


if __name__ == "__main__":
    main()
