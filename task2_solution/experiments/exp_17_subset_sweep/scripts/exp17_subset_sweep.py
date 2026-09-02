"""exp_17: feature-subset sweep across all 3 sub-tasks, floor = the evaluator's 11-variable scored
vocabulary (task2_schema.TASK2_VARIABLES / section_variable_mapping.json), always included per the
user's explicit choice. Optional additions (psa_kinetics, labs) swept on top, independently per
sub-task, each held at its own currently-promoted model (decision=knn, confidence=knn,
variable_weights=mord.LogisticIT).
"""
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402

FLOOR_11 = [
    "age", "psa", "psad", "pirads", "bx_isup", "bx_gl_prim", "bx_gl_sec", "cspca",
    "ct_ordinal", "comorbidity_count", "family_history_present",
]
PSA_KINETICS = ["psa_velocity_per_year", "psa_ratio_last_first"]
LABS = ["free_psa", "pct_free_psa", "testosterone", "creatinine", "alk_phos", "ldh", "egfr_reduced"]

VARIANTS = {
    "floor_11_only": FLOOR_11,
    "floor+psa_kinetics": FLOOR_11 + PSA_KINETICS,
    "floor+labs": FLOOR_11 + LABS,
    "floor+psa_kinetics+labs": FLOOR_11 + PSA_KINETICS + LABS,
}


def main():
    df = vt.load_labeled_df()
    for subtask in ["decision", "confidence", "variable_weights"]:
        print(f"\n=== {subtask} (floor=11 evaluator-scored variables) ===")
        for name, feats in VARIANTS.items():
            r = vt.test_variant(name, {subtask: {"numeric": feats}}, df=df, verbose=False)
            print(f"  {name:>24s}  n_feat={len(feats):2d}  ranking_score={r['ranking_score']:.4f}  "
                  f"case_score={r['mean_case_score']:.4f}  decision_f1={r['decision_weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
