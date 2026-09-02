"""Bootstrap decision +psa_kinetics vs decision baseline_10feat ranking_score gain, holding
confidence=knn and variable_weights=mord.LogisticIT at their promoted defaults.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402
from task2_subtask_config import TASK2_BASE_NUMERIC  # noqa: E402

PSA_KINETICS = ["psa_velocity_per_year", "psa_ratio_last_first"]


def get_rows(decision_numeric):
    df = vt.load_labeled_df()
    configs = vt.merge_overrides(df, {"decision": {"numeric": decision_numeric}})
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
        dp = str(vt._fit_predict_one(tr, va, dc["numeric"], dc["categorical"], dc["model_factory"], y_decision[tr_idx]))
        cc = configs["confidence"]
        cc_code = int(vt._fit_predict_one(tr, va, cc["numeric"], cc["categorical"], cc["model_factory"], y_confidence[tr_idx]))
        conf_label = vt.CONF_CODE_TO_LABEL[cc_code]
        wc = configs["variable_weights"]
        raw_weights = {}
        for v_idx, var in enumerate(vt.TASK2_VARIABLES):
            code = int(vt._fit_predict_one(tr, va, wc["numeric"], wc["categorical"], wc["model_factory"], y_weights[tr_idx, v_idx]))
            raw_weights[var] = vt.WEIGHT_CODE_TO_LABEL[code]
        gw = vt.enforce_eligibility(raw_weights, called_tools)
        fw = vt.normalise_to_full_shape(gw)
        gt_row = df.iloc[i]
        gt = {"treatment_recommendation": {"primary": gt_row["decision"]}, "confidence": gt_row["confidence"],
              "variable_weights": {var: vt.WEIGHT_CODE_TO_LABEL[y_weights[i, v_idx]] for v_idx, var in enumerate(vt.TASK2_VARIABLES)},
              "reveal_sequence": []}
        pred = {"treatment_recommendation": {"primary": dp}, "confidence": conf_label, "variable_weights": fw,
                "reveal_sequence": reveal_sequence}
        row = vt._score_case(gt, pred)
        row["gt_decision"] = gt_row["decision"]
        row["pred_decision"] = dp
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
    print("Computing rows for baseline_10feat and +psa_kinetics (decision sub-task)...")
    rows_base = get_rows(TASK2_BASE_NUMERIC)
    rows_psa = get_rows(TASK2_BASE_NUMERIC + PSA_KINETICS)

    n = len(rows_base)
    rs_base = ranking_score_for_indices(rows_base, list(range(n)))
    rs_psa = ranking_score_for_indices(rows_psa, list(range(n)))
    print(f"Point estimates: baseline={rs_base:.4f}  +psa_kinetics={rs_psa:.4f}  diff={rs_psa-rs_base:+.4f}")

    rng = np.random.default_rng(42)
    n_boot = 2000
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n).tolist()
        diffs.append(ranking_score_for_indices(rows_psa, idx) - ranking_score_for_indices(rows_base, idx))
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    frac = (diffs > 0).mean()
    sig = "significant" if ci_lo > 0 else ("borderline" if frac > 0.9 else "not significant")
    print(f"\nBootstrap ({n_boot} resamples): mean diff={diffs.mean():+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], "
          f"frac favoring +psa_kinetics={frac:.1%}  -> {sig}")


if __name__ == "__main__":
    main()
