"""exp_18: radiology-group (EPE, SVI) and pathology-group (AI-ISUP discordance, cribriform, PNI)
features, tested in isolation per sub-task, against each sub-task's currently-promoted feature set
(not just the 10-feature base) -- same isolation discipline as exp_16/17: one group at a time, not
combined, so a win or loss is attributable to a specific source.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "agent/src")
import task2_variant_tester as vt  # noqa: E402
from task2_subtask_config import TASK2_BASE_NUMERIC, TASK2_DECISION_NUMERIC  # noqa: E402

RADIOLOGY_GROUP = ["epe_present", "svi_present"]
PATHOLOGY_GROUP = ["ai_isup_discordance", "cribriform_present", "perineural_invasion_present"]

# decision's baseline already includes psa_kinetics (exp_16); confidence/variable_weights use the
# plain 10-feature base (exp_11/14) -- test additions on top of EACH sub-task's own real baseline.
SUBTASK_BASE = {
    "decision": TASK2_DECISION_NUMERIC,
    "confidence": TASK2_BASE_NUMERIC,
    "variable_weights": TASK2_BASE_NUMERIC,
}


def main():
    df = vt.load_labeled_df()
    baseline = vt.test_variant("baseline (all promoted defaults)", {}, df=df, verbose=False)
    print(f"baseline (current full promoted pipeline): ranking_score={baseline['ranking_score']:.4f}\n")

    for subtask, base_feats in SUBTASK_BASE.items():
        print(f"=== {subtask} (base={len(base_feats)} feat) ===")
        variants = {
            "base_only": base_feats,
            "+radiology (EPE,SVI)": base_feats + RADIOLOGY_GROUP,
            "+pathology (AI-ISUP,cribriform,PNI)": base_feats + PATHOLOGY_GROUP,
            "+radiology+pathology": base_feats + RADIOLOGY_GROUP + PATHOLOGY_GROUP,
        }
        for name, feats in variants.items():
            r = vt.test_variant(name, {subtask: {"numeric": feats}}, df=df, verbose=False)
            print(f"  {name:>36s}  n_feat={len(feats):2d}  ranking_score={r['ranking_score']:.4f}  "
                  f"(vs baseline {r['ranking_score']-baseline['ranking_score']:+.4f})")
        print()


if __name__ == "__main__":
    main()
