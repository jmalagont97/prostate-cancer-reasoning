import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
goal_data_dir = Path("../../data")
results_dir = Path("results")
reports_dir = Path("reports")
figures_dir = reports_dir / "figures"
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# 1. Version recording
try:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    git_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip()
    version_str = f"{git_sha}  {git_msg}"
except Exception:
    version_str = "unversioned_run"

with open(results_dir / "version.txt", "w") as f:
    f.write(version_str + "\n")

# 2. Load data
inputs_df = pd.read_csv(goal_data_dir / "inputs.csv")
outputs_df = pd.read_csv(goal_data_dir / "outputs.csv")

# Merge labeled set
df = pd.merge(outputs_df, inputs_df, on="case_id", how="inner")

# 3. Basic metrics
total_labeled = len(df)
decision_counts = df["biopsy_decision"].value_counts().to_dict()
confidence_counts = df["confidence"].value_counts().to_dict()

# Contingency: PIRADS vs Decision
pirads_crosstab = pd.crosstab(df["radiology_report__pirads"], df["biopsy_decision"], margins=True).to_dict()

# Discordant cases: PIRADS >= 4 with biopsy_decision == 'no'
disc_high_pirads_no = df[(df["radiology_report__pirads"] >= 4) & (df["biopsy_decision"] == "no")][
    ["case_id", "radiology_report__pirads", "radiology_report__psa_density", "structured_prompt__bx", "previous_notes__prior_cancer_confirmed", "free_text"]
].to_dict(orient="records")

# Discordant cases: PIRADS <= 2 with biopsy_decision == 'yes'
disc_low_pirads_yes = df[(df["radiology_report__pirads"] <= 2) & (df["biopsy_decision"] == "yes")][
    ["case_id", "radiology_report__pirads", "radiology_report__psa_density", "psa_trend__latest_psa", "clinical_data__cspca", "free_text"]
].to_dict(orient="records")

# Numerical summaries by class
num_cols = [
    "radiology_report__pirads",
    "radiology_report__psa_density",
    "radiology_report__prostate_volume",
    "psa_trend__latest_psa",
    "clinical_data__cspca",
    "structured_prompt__age"
]

stats_by_class = {}
for col in num_cols:
    grouped = df.groupby("biopsy_decision")[col].describe().to_dict(orient="index")
    stats_by_class[col] = grouped

# Tool sequence frequencies
tool_counts = {}
for seq_str in df["reveal_sequence"]:
    seq = json.loads(seq_str)
    for tool in seq:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

# Variable weights distribution across 10 target variables
weight_cols = [c for c in df.columns if c.startswith("weight__")]
weights_distribution = {}
for w_col in weight_cols:
    var_name = w_col.replace("weight__", "")
    weights_distribution[var_name] = df[w_col].value_counts().to_dict()

summary = {
    "total_labeled": total_labeled,
    "decision_counts": decision_counts,
    "confidence_counts": confidence_counts,
    "pirads_contingency": pirads_crosstab,
    "discordant_pirads_ge4_decision_no": disc_high_pirads_no,
    "discordant_pirads_le2_decision_yes": disc_low_pirads_yes,
    "numerical_stats_by_class": stats_by_class,
    "tool_frequencies": tool_counts,
    "weights_distribution": weights_distribution
}

with open(results_dir / "statistical_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# 4. Generate Figures
sns.set_theme(style="whitegrid")

# Figure 1: Contingency PIRADS vs Decision
plt.figure(figsize=(7, 5))
ct = pd.crosstab(df["radiology_report__pirads"], df["biopsy_decision"])
ax = ct.plot(kind="bar", stacked=True, color=["#e74c3c", "#2ecc71"], figsize=(8, 5))
plt.title("Biopsy Decision by PI-RADS Score (N=91)", fontsize=14, fontweight="bold")
plt.xlabel("PI-RADS Score", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Decision", labels=["No", "Yes"])
plt.tight_layout()
plt.savefig(figures_dir / "pirads_vs_decision.png", dpi=150)
plt.close()

# Figure 2: Distributions by class (Boxplots of Key Biomarkers)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
features_to_plot = [
    ("radiology_report__psa_density", "PSA Density (ng/mL/mL)"),
    ("clinical_data__cspca", "csPCa DL Probability"),
    ("psa_trend__latest_psa", "Latest PSA (ng/mL)")
]

for idx, (col, title) in enumerate(features_to_plot):
    sns.boxplot(x="biopsy_decision", y=col, data=df, ax=axes[idx], palette=["#e74c3c", "#2ecc71"], showfliers=False)
    sns.stripplot(x="biopsy_decision", y=col, data=df, ax=axes[idx], color="black", alpha=0.4, jitter=0.2)
    axes[idx].set_title(title, fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Biopsy Decision", fontsize=11)
    axes[idx].set_ylabel("")

plt.tight_layout()
plt.savefig(figures_dir / "distributions_by_class.png", dpi=150)
plt.close()

# Figure 3: Tool Usage and Variable Weights Heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Tools
tools_s = pd.Series(tool_counts).sort_values(ascending=True)
tools_s.plot(kind="barh", ax=ax1, color="#3498db")
ax1.set_title("EHR Section Consultation Frequency (reveal_sequence)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Frequency (N=91)", fontsize=11)

# Subplot 2: Variable weights heatmap
w_df = pd.DataFrame(weights_distribution).fillna(0).T
order_cols = ["not_used", "noted", "important", "decisive"]
w_df = w_df.reindex(columns=[c for c in order_cols if c in w_df.columns])
sns.heatmap(w_df, annot=True, fmt=".0f", cmap="Blues", ax=ax2, cbar=True)
ax2.set_title("Variable Weight Level Assignment Counts", fontsize=12, fontweight="bold")
ax2.set_xlabel("Weight Level", fontsize=11)
ax2.set_ylabel("Clinical Variable", fontsize=11)

plt.tight_layout()
plt.savefig(figures_dir / "tool_usage_and_weights.png", dpi=150)
plt.close()

print(f"EDA complete: summary saved to {results_dir / 'statistical_summary.json'} and figures in {figures_dir}")
