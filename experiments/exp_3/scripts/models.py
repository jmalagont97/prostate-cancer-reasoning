"""8-model registry for exp_3's decision/confidence comparison.

KDM is handled separately in each runner via
chimera_task1.train_confidence_kdm.fit_predict_kdm (not a plain sklearn Estimator, so it doesn't
fit this registry's shape). Hyperparameters here are hand-chosen for N=91-195 (shallow/
regularized), not tuned via a validation search -- same limitation as exp_1/exp_2's model
choices, noted in the eventual report rather than hidden.
"""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

RANDOM_STATE = 0

# Models with no class-imbalance handling at all (neither class_weight nor sample_weight
# supported by sklearn for these) -- recorded explicitly per-condition so this isn't silently
# inconsistent with the other six models, which do get some form of balancing.
NO_IMBALANCE_HANDLING = {"mlp", "knn"}


def build_sklearn_models() -> dict[str, tuple[object, bool]]:
    """Returns {name: (estimator, use_sample_weight)}.

    use_sample_weight=True means the caller should fit with
    sample_weight=sklearn.utils.class_weight.compute_sample_weight("balanced", y_train) --
    for models that support sample_weight but not a native class_weight="balanced" the way
    sklearn's own linear/tree/SVM estimators do. XGBoost auto-infers binary vs. multiclass
    objective from y at fit time, so no explicit `objective=` is set here.
    """
    return {
        "svm": (
            SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=RANDOM_STATE),
            False,
        ),
        "rf": (
            RandomForestClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=10,
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
            False,
        ),
        "xgb": (
            XGBClassifier(
                n_estimators=100, max_depth=3, min_child_weight=5,
                learning_rate=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
            ),
            True,
        ),
        "extratrees": (
            ExtraTreesClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=10,
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
            False,
        ),
        "mlp": (
            MLPClassifier(
                hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000,
                early_stopping=True, random_state=RANDOM_STATE,
            ),
            False,
        ),
        "nb": (GaussianNB(), True),
        "knn": (KNeighborsClassifier(n_neighbors=7, weights="distance"), False),
    }
