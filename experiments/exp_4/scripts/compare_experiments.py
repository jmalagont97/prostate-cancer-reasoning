# Paired comparison exp_3 (39 features) vs exp_4 (37 features, >50% missingness filter).
# Same frozen MCCV splits and identical pipeline except the feature subset.
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP3_DIR = PROJECT_ROOT / "experiments" / "exp_3"
EXP4_DIR = PROJECT_ROOT / "experiments" / "exp_4"

METRIC = "Macro_F1"
EXP3_WINNER = "manhattan_fuzzy_confidence_uniform_k11"


def load_fold_metrics(exp_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(exp_dir / "results" / "mccv_fold_metrics.csv")
    assert len(df) == 128 * 50, f"{exp_dir.name}: {len(df)} rows"
    return df


def main() -> None:
    f39 = load_fold_metrics(EXP3_DIR)
    f37 = load_fold_metrics(EXP4_DIR)

    exp4_hp = json.loads((EXP4_DIR / "results" / "selected_config" / "hyperparameters.json").read_text())
    exp4_winner = exp4_hp["config_id"]

    ids = sorted(f39["config_id"].unique())
    assert ids == sorted(f37["config_id"].unique())
    assert len(ids) == 128

    rows = []
    for cid in ids:
        a = f39.loc[f39["config_id"].eq(cid)].sort_values("split_id")[METRIC].to_numpy()
        b = f37.loc[f37["config_id"].eq(cid)].sort_values("split_id")[METRIC].to_numpy()
        assert len(a) == 50 and len(b) == 50
        d = b - a
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        delta_mean = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        ci95 = float(se * stats.t.ppf(0.975, len(d) - 1))
        w = stats.wilcoxon(d)
        rows.append({
            "config_id": cid,
            f"Macro_F1_mean_39": mean_a,
            f"Macro_F1_mean_37": mean_b,
            "delta_mean_37_minus_39": delta_mean,
            "delta_std": float(d.std(ddof=1)),
            "delta_se": se,
            "delta_ci95_half": ci95,
            "delta_ci95_low": delta_mean - ci95,
            "delta_ci95_high": delta_mean + ci95,
            "wilcoxon_p": float(w.pvalue),
            "n_folds_positive_delta": int((d > 0).sum()),
            "n_folds_negative_delta": int((d < 0).sum()),
            "is_exp3_winner": cid == EXP3_WINNER,
            "is_exp4_winner": cid == exp4_winner,
        })

    cmp = pd.DataFrame(rows).sort_values("delta_mean_37_minus_39", ascending=False)
    cmp.to_csv(EXP4_DIR / "results" / "comparison_39_vs_37.csv", index=False)
    print(f"[compare] exp4 winner={exp4_winner}; wrote comparison_39_vs_37.csv")

    # Scatter of the 128 configs (mean Macro-F1 39 vs 37) + identity line + winners.
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(cmp["Macro_F1_mean_39"], cmp["Macro_F1_mean_37"],
               s=18, alpha=0.6, color="#4C72B0", label="128 configs")
    lo = min(cmp["Macro_F1_mean_39"].min(), cmp["Macro_F1_mean_37"].min()) - 0.01
    hi = max(cmp["Macro_F1_mean_39"].max(), cmp["Macro_F1_mean_37"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=1, label="identity (no change)")
    for cid, color, label in [(EXP3_WINNER, "#C44E52", "exp_3 winner"),
                              (exp4_winner, "#55A868", "exp_4 winner")]:
        r = cmp.loc[cmp["config_id"].eq(cid)].iloc[0]
        ax.scatter([r["Macro_F1_mean_39"]], [r["Macro_F1_mean_37"]],
                   s=70, color=color, zorder=5, edgecolor="black", label=label)
        ax.annotate(cid.replace("_", "\n"),
                    (r["Macro_F1_mean_39"], r["Macro_F1_mean_37"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_xlabel(f"MCCV mean {METRIC} — exp_3 (39 features)")
    ax.set_ylabel(f"MCCV mean {METRIC} — exp_4 (37 features)")
    ax.set_title("Feature-subset ablation: 39 vs 37 features\n"
                 "(drop >50% missingness, essentials always kept)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = EXP4_DIR / "reports" / "figures" / "comparison_macro_f1_39_vs_37.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
