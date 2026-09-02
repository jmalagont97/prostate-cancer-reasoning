"""Bootstrap the decision=random_forest vs decision=knn ranking_score gain, paired per-case
(both case_score and the decision-correctness indicator, since ranking_score combines
mean_case_score with decision_weighted_f1 -- F1 isn't a simple per-case mean, so the bootstrap
resamples case indices and recomputes both components fresh each resample, exactly mirroring
exp_8/9/10's bootstrap pattern for task3.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402

sys.path.insert(0, "experiments/exp_2/scripts")
from models import knn, random_forest  # noqa: E402


def get_rows(model_factory):
    df = vt.load_labeled_df()
    configs = vt.merge_overrides(df, {"decision": {"model_factory": model_factory}})
    called_tools = vt.compute_called_tools(configs)
    reveal_sequence = [vt.TOOL_TO_SECTION[t] for t in called_tools]
    y_decision = vt.parse_decision(df)
    y_confidence = vt.parse_confidence(df)
    y_weights = vt.parse_variable_weights(df)
    n = len(df)
    rows = []
    for i in range(n):
        tr_idx = [j for j in range(n) if j != i]
        tr, va = df.iloc[tr_idx], df.iloc[[i]]
        dc = configs["decision"]
        decision_pred = str(vt._fit_predict_one(tr, va, dc["numeric"], dc["categorical"], dc["model_factory"], y_decision[tr_idx]))
        cc = configs["confidence"]
        conf_code = int(vt._fit_predict_one(tr, va, cc["numeric"], cc["categorical"], cc["model_factory"], y_confidence[tr_idx]))
        confidence_label = vt.CONF_CODE_TO_LABEL[conf_code]
        wc = configs["variable_weights"]
        raw_weights = {}
        for v_idx, var in enumerate(vt.TASK2_VARIABLES):
            code = int(vt._fit_predict_one(tr, va, wc["numeric"], wc["categorical"], wc["model_factory"], y_weights[tr_idx, v_idx]))
            raw_weights[var] = vt.WEIGHT_CODE_TO_LABEL[code]
        grounded_weights = vt.enforce_eligibility(raw_weights, called_tools)
        full_weights = vt.normalise_to_full_shape(grounded_weights)
        gt_row = df.iloc[i]
        gt = {"treatment_recommendation": {"primary": gt_row["decision"]}, "confidence": gt_row["confidence"],
              "variable_weights": {var: vt.WEIGHT_CODE_TO_LABEL[y_weights[i, v_idx]] for v_idx, var in enumerate(vt.TASK2_VARIABLES)},
              "reveal_sequence": []}
        pred = {"treatment_recommendation": {"primary": decision_pred}, "confidence": confidence_label,
                "variable_weights": full_weights, "reveal_sequence": reveal_sequence}
        row = vt._score_case(gt, pred)
        row["gt_decision"] = gt_row["decision"]
        row["pred_decision"] = decision_pred
        rows.append(row)
    return rows


def ranking_score_for_indices(rows, idx):
    from statistics import mean
    from sklearn.metrics import f1_score
    sub = [rows[i] for i in idx]
    mean_case_score = mean(r["case_score"] for r in sub)
    y_true = [r["gt_decision"] for r in sub]
    y_pred = [r["pred_decision"] for r in sub]
    f1 = f1_score(y_true, y_pred, labels=sorted(vt.VALID_TREATMENT_DECISIONS), average="weighted", zero_division=0)
    return (mean_case_score + f1) / 2.0


def main():
    print("Computing rows for knn and random_forest (decision sub-task)...")
    rows_knn = get_rows(knn)
    rows_rf = get_rows(random_forest)

    n = len(rows_knn)
    rs_knn = ranking_score_for_indices(rows_knn, list(range(n)))
    rs_rf = ranking_score_for_indices(rows_rf, list(range(n)))
    print(f"Point estimates: knn={rs_knn:.4f}  random_forest={rs_rf:.4f}  diff={rs_rf-rs_knn:+.4f}")

    rng = np.random.default_rng(42)
    n_boot = 2000
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n).tolist()
        rs_k = ranking_score_for_indices(rows_knn, idx)
        rs_r = ranking_score_for_indices(rows_rf, idx)
        diffs.append(rs_r - rs_k)
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    frac = (diffs > 0).mean()
    sig = "significant" if ci_lo > 0 else ("borderline" if frac > 0.9 else "not significant")
    print(f"\nBootstrap ({n_boot} resamples): mean diff={diffs.mean():+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], "
          f"frac favoring random_forest={frac:.1%}  -> {sig}")


if __name__ == "__main__":
    main()
