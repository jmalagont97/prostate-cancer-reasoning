import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

script_path = Path(__file__).resolve()
project_root = script_path.parents[3]
exp_dir = project_root / "experiments" / "exp_21"
results_dir = exp_dir / "results"
figures_dir = exp_dir / "reports" / "figures"
brain_figures_dir = Path("/home/jmalagont/.gemini/antigravity-cli/brain/7884c29e-c602-4c6a-bff9-83df54c2ad16/figures")

brain_figures_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "feature_attribution_metrics.json", "r") as f:
    metrics = json.load(f)

features = list(metrics.keys())
macro_f1s = [metrics[feat]["macro_f1"] for feat in features]
abs_rhos = [abs(metrics[feat]["spearman_rho"]) for feat in features]
raw_rhos = [metrics[feat]["spearman_rho"] for feat in features]

# 1. Macro-F1 Bar Chart / Histogram
plt.figure(figsize=(9, 5.5))
colors_f1 = plt.cm.plasma(np.linspace(0.2, 0.85, len(features)))
bars1 = plt.bar(features, macro_f1s, color=colors_f1, edgecolor="black", linewidth=1.2, width=0.55)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.title("4-Class Macro-F1 Performance per Clinical Feature (exp_21 SHAP)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Clinical Tabular Features", fontsize=11, fontweight="bold")
plt.ylabel("4-Class Macro-F1 Score", fontsize=11, fontweight="bold")
plt.ylim(0, max(macro_f1s) + 0.04)
plt.grid(True, linestyle="--", alpha=0.4, axis="y")
plt.tight_layout()
fig1_path = figures_dir / "histogram_macro_f1.png"
plt.savefig(fig1_path, dpi=300)
plt.savefig(brain_figures_dir / "histogram_macro_f1.png", dpi=300)
plt.close()

# 2. Absolute Spearman Rho (|Rho|) Bar Chart / Histogram
plt.figure(figsize=(9, 5.5))
colors_rho = plt.cm.viridis(np.linspace(0.25, 0.9, len(features)))
bars2 = plt.bar(features, abs_rhos, color=colors_rho, edgecolor="black", linewidth=1.2, width=0.55)

for i, bar in enumerate(bars2):
    yval = bar.get_height()
    raw_val = raw_rhos[i]
    sign_str = "+" if raw_val >= 0 else "-"
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"|{yval:.4f}| ({sign_str})", ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.title("Absolute Spearman Rank Correlation (|Rho|) per Clinical Feature (exp_21 SHAP)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Clinical Tabular Features", fontsize=11, fontweight="bold")
plt.ylabel("Absolute Spearman Correlation |Rho|", fontsize=11, fontweight="bold")
plt.ylim(0, max(abs_rhos) + 0.06)
plt.grid(True, linestyle="--", alpha=0.4, axis="y")
plt.tight_layout()
fig2_path = figures_dir / "histogram_abs_spearman_rho.png"
plt.savefig(fig2_path, dpi=300)
plt.savefig(brain_figures_dir / "histogram_abs_spearman_rho.png", dpi=300)
plt.close()

# 3. Combined Side-by-Side Figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Macro-F1
bars_a = axes[0].bar(features, macro_f1s, color=colors_f1, edgecolor="black", linewidth=1.2, width=0.55)
for bar in bars_a:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
axes[0].set_title("A) 4-Class Macro-F1 Score per Feature", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Clinical Feature", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
axes[0].set_ylim(0, max(macro_f1s) + 0.04)
axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

# Panel B: Absolute Rho
bars_b = axes[1].bar(features, abs_rhos, color=colors_rho, edgecolor="black", linewidth=1.2, width=0.55)
for i, bar in enumerate(bars_b):
    yval = bar.get_height()
    raw_val = raw_rhos[i]
    sign_str = "+" if raw_val >= 0 else "-"
    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f"|{yval:.4f}| ({sign_str})", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
axes[1].set_title("B) Absolute Spearman Correlation |Rho| per Feature", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Clinical Feature", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Absolute Spearman Correlation |Rho|", fontsize=10, fontweight="bold")
axes[1].set_ylim(0, max(abs_rhos) + 0.06)
axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

plt.suptitle("SHAP Clinical Feature Relevance Attribution Summary (exp_21)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig_combined_path = figures_dir / "histograms_combined_f1_rho.png"
plt.savefig(fig_combined_path, dpi=300)
plt.savefig(brain_figures_dir / "histograms_combined_f1_rho.png", dpi=300)
plt.close()

print(f"Generated figures successfully: \n  - {fig1_path}\n  - {fig2_path}\n  - {fig_combined_path}")
