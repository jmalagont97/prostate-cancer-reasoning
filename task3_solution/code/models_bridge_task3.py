"""Final production fit for task3: trains task3_config.DEFAULT_MODEL_KEY's model on all 75 labeled
cases and saves it. Mirrors agent/src/models_bridge.py (task1) -- swapping the deployed model is a
one-line change to task3_config.DEFAULT_MODEL_KEY, then rerun this script.
"""

import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from task3_config import DEFAULT_MODEL_KEY, MODEL_FACTORIES, MODELS_DIR, NUMERIC_FEATURES  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task3"


def load_task3():
    structured = pd.read_csv(TASK_DIR / "structured.csv", keep_default_na=False, na_values=[""])
    engineered = pd.read_csv(TASK_DIR / "engineered_features.csv", keep_default_na=False, na_values=[""])
    labels = pd.read_csv(TASK_DIR / "labels.csv")
    df = structured.merge(engineered, on="case_id", how="inner").merge(labels, on="case_id", how="inner")
    for col in NUMERIC_FEATURES + ["age", "psa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def fit_and_save():
    df = load_task3()
    model = MODEL_FACTORIES[DEFAULT_MODEL_KEY]().fit(df)
    joblib.dump({"model": model, "model_key": DEFAULT_MODEL_KEY}, MODELS_DIR / "task3_agent_model.joblib")
    print(f"Saved task3 model ({DEFAULT_MODEL_KEY}, {len(df)} cases) to "
          f"{MODELS_DIR / 'task3_agent_model.joblib'}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fit_and_save()
