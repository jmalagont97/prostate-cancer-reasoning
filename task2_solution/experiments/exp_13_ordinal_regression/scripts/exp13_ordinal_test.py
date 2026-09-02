"""exp_13: does treating confidence/variable_weights as ORDINAL (regression on the integer code,
rounded+clipped to the valid range) beat treating them as nominal classification (all 8 models in
exp_12, none ordinal-aware)? decision is excluded -- it's genuinely nominal (4 unordered treatment
categories), not ordinal, so no ordinal treatment applies there.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402


class RoundedRegressor:
    """Wraps a regressor; predict() rounds to the nearest integer and clips to [0, max_code] --
    a drop-in .fit/.predict model for the existing model_factory slot, no harness changes needed."""

    def __init__(self, base_factory, max_code):
        self.base = base_factory()
        self.max_code = max_code

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

    def predict(self, X):
        raw = self.base.predict(X)
        rounded = np.clip(np.rint(raw), 0, self.max_code).astype(int)
        return rounded


def ordinal_factory(base_factory, max_code):
    return lambda: RoundedRegressor(base_factory, max_code)


CONF_MAX, WEIGHT_MAX = 2, 3

ORDINAL_MODELS = {
    "knn_ordinal": lambda: KNeighborsRegressor(n_neighbors=7, weights="distance"),
    "ridge_ordinal": lambda: Ridge(alpha=1.0),
    "rf_ordinal": lambda: RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=10, random_state=0),
}


def main():
    df = vt.load_labeled_df()
    baseline = vt.test_variant("baseline (knn nominal everywhere)", {}, df=df, verbose=False)
    print(f"baseline (knn, nominal classification): ranking_score={baseline['ranking_score']:.4f}\n")

    print("=== confidence: ordinal regression vs nominal classification ===")
    for name, base in ORDINAL_MODELS.items():
        factory = ordinal_factory(base, CONF_MAX)
        r = vt.test_variant(f"confidence={name}", {"confidence": {"model_factory": factory}}, df=df, verbose=False)
        print(f"  {name:14s} ranking_score={r['ranking_score']:.4f}  confidence_score={r['mean_confidence_score']:.4f}")

    print("\n=== variable_weights: ordinal regression vs nominal classification ===")
    for name, base in ORDINAL_MODELS.items():
        factory = ordinal_factory(base, WEIGHT_MAX)
        r = vt.test_variant(f"variable_weights={name}", {"variable_weights": {"model_factory": factory}}, df=df, verbose=False)
        print(f"  {name:14s} ranking_score={r['ranking_score']:.4f}  var_weight_score={r['mean_variable_weight_score']:.4f}  "
              f"factor_f1={r['mean_important_decisive_factor_score']:.4f}")


if __name__ == "__main__":
    main()
