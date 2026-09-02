"""exp_21: KDMRegressor for `decision`, encoding the 4 treatment categories as an integer scale
(0=watchful_waiting ... 3=active_treatment, a clinically-motivated least->most-interventional
ordering, not derived from the evaluator -- decision_score/decision_weighted_f1 are both confirmed
NOMINAL, no partial credit for a close wrong category). Predict a continuous value, round+clip to
[0,3], map back to the label. Tested against the current promoted knn default.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402
from kdm_models import KDMRegressor  # noqa: E402

DECISION_ORDER = ["watchful_waiting", "active_surveillance", "continued_surveillance", "active_treatment"]
CODE_TO_LABEL = {i: lab for i, lab in enumerate(DECISION_ORDER)}
LABEL_TO_CODE = {lab: i for i, lab in enumerate(DECISION_ORDER)}


class KDMRegressorForDecision:
    """Wraps KDMRegressor behind the same .fit(X, y_labels)/.predict(X)->labels interface
    task2_variant_tester expects -- y is fit/predicted as the ordinal code internally."""

    def __init__(self, **kdm_kwargs):
        self.kdm_kwargs = kdm_kwargs

    def fit(self, X, y):
        y_codes = np.array([LABEL_TO_CODE[v] for v in y], dtype=np.float32)
        self.reg = KDMRegressor(**self.kdm_kwargs)
        self.reg.fit(X, y_codes)
        return self

    def predict(self, X):
        raw = self.reg.predict(X)
        codes = np.clip(np.rint(raw), 0, 3).astype(int)
        return np.array([CODE_TO_LABEL[c] for c in codes])


def kdm_regressor_default():
    return KDMRegressorForDecision(encoded_size=16, sigma_x=0.5, sigma_y=0.5, epochs=300, lr=1e-2)


def main():
    df = vt.load_labeled_df()
    baseline = vt.test_variant("decision=knn (current promoted)", {}, df=df, verbose=False)
    print(f"baseline (decision=knn): ranking_score={baseline['ranking_score']:.4f}")

    r = vt.test_variant("decision=KDMRegressor (ordinal-coded, rounded)",
                         {"decision": {"model_factory": kdm_regressor_default}}, df=df, verbose=False)
    print(f"decision=KDMRegressor: ranking_score={r['ranking_score']:.4f}  "
          f"(vs baseline {r['ranking_score']-baseline['ranking_score']:+.4f})  "
          f"decision_f1={r['decision_weighted_f1']:.4f}  case_score={r['mean_case_score']:.4f}")


if __name__ == "__main__":
    main()
