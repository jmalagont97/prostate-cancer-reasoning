"""Task3 swap-test harness: test_variant(model_key) -> LOOCV ranking_score (= Harrell's C-index,
via evaluate.py's OWN concordance_index -- same convention it uses: preds are literal predicted
months, positively correlated with survival time, NOT partial hazard). Mirrors task1's
variant_tester.py: real evaluator functions, in-memory LOOCV, no disk writes needed per test.
"""

import json
import sys
import time
import warnings
from pathlib import Path
from statistics import mean

warnings.filterwarnings("ignore")

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from task3_config import MODEL_FACTORIES, NUMERIC_FEATURES  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "CHIMERA-agent" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))
from evaluate import concordance_index, recurrence_time_score  # noqa: E402

TASK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task3"
RESULTS_LOG = Path(__file__).resolve().parent.parent / "results" / "task3_variant_log.json"

import pandas as pd  # noqa: E402


def _load_df():
    structured = pd.read_csv(TASK_DIR / "structured.csv", keep_default_na=False, na_values=[""])
    engineered = pd.read_csv(TASK_DIR / "engineered_features.csv", keep_default_na=False, na_values=[""])
    labels = pd.read_csv(TASK_DIR / "labels.csv")
    df = structured.merge(engineered, on="case_id", how="inner").merge(labels, on="case_id", how="inner")
    for col in NUMERIC_FEATURES + ["age", "psa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def test_variant(model_key: str, log: bool = True) -> dict:
    df = _load_df()
    n = len(df)
    preds, n_failed = [None] * n, 0

    for i in range(n):
        tr, va = df.drop(index=i), df.iloc[[i]]
        try:
            model = MODEL_FACTORIES[model_key]().fit(tr)
            preds[i] = float(model.predict_months(va)[0])
        except Exception:
            n_failed += 1

    ok_idx = [i for i in range(n) if preds[i] is not None]
    times = df["months_to_recurrence"].to_numpy()[ok_idx].tolist()
    events = df["event"].to_numpy()[ok_idx].tolist()
    preds_ok = [preds[i] for i in ok_idx]

    c_index = concordance_index(times, preds_ok, events)

    # Diagnostic only (not part of ranking_score, per evaluate.py's own comment) -- reported for
    # visibility into calibration quality, not used to pick a winner.
    time_scores = [
        recurrence_time_score({"event": e, "months_to_recurrence": t}, {"months_to_recurrence": p})
        for t, p, e in zip(times, preds_ok, events)
    ]

    result = {
        "model_key": model_key, "ranking_score": c_index, "n_used": len(ok_idx), "n_total": n,
        "n_failed": n_failed, "mean_time_score_diagnostic": mean(time_scores) if time_scores else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    c_str = f"{c_index:.4f}" if c_index is not None else "None (no admissible pairs)"
    ts_str = f"{result['mean_time_score_diagnostic']:.3f}" if result["mean_time_score_diagnostic"] is not None else "n/a"
    print(f"{model_key:18s}  ranking_score(C-index)={c_str}  mean_time_score={ts_str}  "
          f"(n_used={len(ok_idx)}/{n}, {n_failed} failed)")

    if log:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        history = json.loads(RESULTS_LOG.read_text()) if RESULTS_LOG.exists() else []
        history.append(result)
        RESULTS_LOG.write_text(json.dumps(history, indent=2, default=str))

    return result


if __name__ == "__main__":
    print("=== Task3 model sweep (LOOCV, real evaluate.py concordance_index) ===")
    results = {key: test_variant(key) for key in MODEL_FACTORIES}
    best = max(results.values(), key=lambda r: r["ranking_score"])
    print(f"\nBest: {best['model_key']}  ranking_score={best['ranking_score']:.4f}")
