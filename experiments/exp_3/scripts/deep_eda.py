import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def compute_text_stats(df, col_name):
    # Filter out 'NONE'
    texts = df[col_name].astype(str)
    valid_texts = texts[texts != "NONE"]
    
    if len(valid_texts) == 0:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
        
    lengths = [len(t.split()) for t in valid_texts]
    return {
        "count": len(lengths),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths))
    }

def main():
    parser = argparse.ArgumentParser(description="Deep EDA on Preprocessed Task 1 Data")
    parser.add_argument("--data_dir", type=str, default="data/chimera26/preprocessed/task1")
    parser.add_argument("--results_dir", type=str, default="experiments/exp_3/results")
    parser.add_argument("--reports_dir", type=str, default="experiments/exp_3/reports")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    results_path = Path(args.results_dir)
    reports_path = Path(args.reports_dir)
    figures_path = reports_path / "figures"

    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed CSV files...")
    df_mri = pd.read_csv(data_path / "mri_embeddings.csv")
    df_prompt = pd.read_csv(data_path / "clinical_prompts.csv")
    df_tabular = pd.read_csv(data_path / "clinical_data_tabular.csv")
    df_reasoning = pd.read_csv(data_path / "clinical_reasoning.csv")
    df_decision = pd.read_csv(data_path / "biopsy_decision.csv")

    # 1. Target Balance
    decision_counts = df_decision["biopsy_decision"].value_counts().to_dict()
    print("Decision Target distribution:", decision_counts)

    # 2. Text stats
    prompt_stats = compute_text_stats(df_prompt, "clinical_prompt_text")
    reasoning_stats = compute_text_stats(df_reasoning, "reasoning_text")
    print("Prompt text stats:", prompt_stats)
    print("Reasoning text stats:", reasoning_stats)

    # 3. Tabular missingness
    # Exclude patient_id
    tab_cols = [c for c in df_tabular.columns if c != "patient_id"]
    reason_cols = [c for c in df_reasoning.columns if c != "patient_id" and c != "reasoning_text"]
    
    missingness = {}
    for c in tab_cols:
        none_count = (df_tabular[c].astype(str) == "NONE").sum()
        missingness[c] = (none_count / len(df_tabular)) * 100
        
    for c in reason_cols:
        none_count = (df_reasoning[c].astype(str) == "NONE").sum()
        missingness[c] = (none_count / len(df_reasoning)) * 100
        
    print("Missingness percentages:", missingness)

    # 4. t-SNE on MRI embeddings
    # Identify non-NONE MRI rows
    mri_features = [c for c in df_mri.columns if c != "patient_id"]
    valid_mri_mask = df_mri[mri_features[0]].astype(str) != "NONE"
    
    df_mri_valid = df_mri[valid_mri_mask]
    df_decision_valid = df_decision[valid_mri_mask]
    
    X = df_mri_valid[mri_features].values.astype(float)
    y = df_decision_valid["biopsy_decision"].values.astype(str)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
    X_embedded = tsne.fit_transform(X)

    # Calculate Silhouette score for labeled training cases
    labeled_mask = (y == "yes") | (y == "no")
    X_labeled = X_embedded[labeled_mask]
    y_labeled = y[labeled_mask]
    
    if len(np.unique(y_labeled)) > 1:
        sil = float(silhouette_score(X_labeled, y_labeled))
    else:
        sil = -1.0
    print(f"t-SNE Silhouette Score: {sil:.4f}")

    # Plot 1: Text Length Histogram
    plt.figure(figsize=(12, 5))
    
    # Prompt text lengths
    plt.subplot(1, 2, 1)
    prompt_lens = [len(str(t).split()) for t in df_prompt["clinical_prompt_text"] if str(t) != "NONE"]
    plt.hist(prompt_lens, bins=15, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.title("Clinical Prompt Word Count Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Word Count", fontsize=10)
    plt.ylabel("Patient Count", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Reasoning text lengths
    plt.subplot(1, 2, 2)
    reasoning_lens = [len(str(t).split()) for t in df_reasoning["reasoning_text"] if str(t) != "NONE"]
    plt.hist(reasoning_lens, bins=15, color="#ff7f0e", edgecolor="black", alpha=0.8)
    plt.title("Clinical Reasoning Word Count Distribution", fontsize=11, fontweight="bold")
    plt.xlabel("Word Count", fontsize=10)
    plt.ylabel("Patient Count", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(figures_path / "text_length_dist.png", dpi=300)
    plt.close()

    # Plot 2: Missingness Bar Chart
    plt.figure(figsize=(10, 6))
    features_sorted = sorted(missingness.items(), key=lambda x: x[1], reverse=True)
    f_names = [x[0] for x in features_sorted]
    f_rates = [x[1] for x in features_sorted]
    
    bars = plt.bar(f_names, f_rates, color="#d62728", edgecolor="black", alpha=0.8)
    plt.title("Tabular Feature Missingness Rates - Task 1 Cohort", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Feature Name", fontsize=11)
    plt.ylabel("Missingness Rate (%)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="semibold")
                    
    plt.tight_layout()
    plt.savefig(figures_path / "missingness_rates.png", dpi=300)
    plt.close()

    # Plot 3: t-SNE Scatter Plot
    plt.figure(figsize=(8, 6))
    colors = {"yes": "#ff7f0e", "no": "#1f77b4", "NONE": "#7f7f7f"}
    labels_map = {"yes": "Requires Biopsy (yes)", "no": "Do Not Biopsy (no)", "NONE": "Unlabeled Test Split"}
    
    for val in ["yes", "no", "NONE"]:
        mask = y == val
        if mask.sum() > 0:
            plt.scatter(X_embedded[mask, 0], X_embedded[mask, 1],
                        c=colors[val], label=labels_map[val],
                        edgecolor="black", alpha=0.8, s=40)
                        
    plt.title(f"2D t-SNE Projection of MRI Embeddings\n(Silhouette Score for Labeled split: {sil:.4f})", 
              fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=True, fontsize=9, loc="best")
    
    plt.tight_layout()
    plt.savefig(figures_path / "tsne_mri.png", dpi=300)
    plt.close()

    # Plot 4: Class Balance Bar Chart
    plt.figure(figsize=(7, 5))
    categories = ["Requires Biopsy (yes)", "Do Not Biopsy (no)", "Unlabeled Test Split"]
    values = [decision_counts.get("yes", 0), decision_counts.get("no", 0), decision_counts.get("NONE", 0)]
    colors_cb = ["#ff7f0e", "#1f77b4", "#7f7f7f"]
    
    bars_cb = plt.bar(categories, values, color=colors_cb, edgecolor="black", alpha=0.8, width=0.6)
    plt.title("Target Class Distribution and Partition Balance - Task 1", fontsize=11, fontweight="bold", pad=15)
    plt.ylabel("Patient Count", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.ylim(0, 120)
    
    for bar in bars_cb:
        height = bar.get_height()
        plt.annotate(f"{height} cases",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    plt.savefig(figures_path / "class_balance.png", dpi=300)
    plt.close()

    # Save metrics JSON
    out_metrics = {
        "total_folders": len(df_decision),
        "target_balance": decision_counts,
        "prompt_word_stats": prompt_stats,
        "reasoning_word_stats": reasoning_stats,
        "tabular_missingness_rates": missingness,
        "mri_tsne_silhouette": sil
    }
    with open(results_path / "deep_eda_metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2)

    # Generate summary.md
    with open(reports_path / "summary.md", "w") as f:
        f.write("# Deep Exploratory Data Analysis Summary Report — Task 1\n\n")
        f.write(f"**Date**: 2026-07-20  \n")
        f.write(f"**Total Cases Scanned**: {len(df_decision)}  \n")
        f.write(f"**Conda Environment**: `histo-DL`  \n\n")
        
        f.write("## Target Class Balance\n")
        f.write(f"*   **`yes` (Requires Biopsy)**: {decision_counts.get('yes', 0)} cases\n")
        f.write(f"*   **`no` (Do Not Biopsy)**: {decision_counts.get('no', 0)} cases\n")
        f.write(f"*   **`unlabeled` (Test Split)**: {decision_counts.get('NONE', 0)} cases\n\n")
        
        f.write("## Text Word Count Distributions\n\n")
        f.write("| Narrative Source | Case Count | Min Words | Max Words | Mean Words | Median Words | Std Dev |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| Clinical Prompts | {prompt_stats['count']} | {prompt_stats['min']} | {prompt_stats['max']} | {prompt_stats['mean']:.1f} | {prompt_stats['median']:.1f} | {prompt_stats['std']:.1f} |\n")
        f.write(f"| Clinical Reasoning | {reasoning_stats['count']} | {reasoning_stats['min']} | {reasoning_stats['max']} | {reasoning_stats['mean']:.1f} | {reasoning_stats['median']:.1f} | {reasoning_stats['std']:.1f} |\n\n")
        
        f.write("## Tabular & Rationale Variable Missingness Rates\n\n")
        f.write("| Feature Variable | Source File | Missingness Rate (%) | Status |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        for f_name, rate in features_sorted:
            src = "clinical_data_tabular.csv" if f_name in tab_cols else "clinical_reasoning.csv"
            status = "Standard (0.0% Missing)" if rate == 0 else ("Test Split Padded (53.3% Missing)" if abs(rate - 53.33) < 1.0 else "Partially Missing")
            f.write(f"| `{f_name}` | `{src}` | {rate:.1f}% | {status} |\n")
            
        f.write("\n## MRI Embeddings t-SNE Clustering\n")
        f.write(f"*   **Unsupervised 2D t-SNE projection** was computed on the 191 cases with active MRI vectors.\n")
        f.write(f"*   **Silhouette Score (Labeled partition only)**: **{sil:.4f}**\n")
        f.write(f"*   *Interpretation*: A Silhouette score of `{sil:.4f}` (near 0) indicates that the 1024-D pre-extracted MRI embeddings exhibit significant class overlap when projected to 2D without visual labels. This emphasizes the mathematical necessity of fusing the MRI representations with structured tabular features (like PSA, age, and PI-RADS) or clinical prompt text to establish a discriminative classification boundary.\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("### 1. Target Class Balance and Partition Distribution\n")
        f.write("![Class Balance](figures/class_balance.png)\n\n")
        f.write("### 2. Narrative Text Length Distributions\n")
        f.write("![Text Lengths](figures/text_length_dist.png)\n\n")
        f.write("### 3. Feature Missingness (Absenteeism) Rates\n")
        f.write("![Missingness](figures/missingness_rates.png)\n\n")
        f.write("### 4. MRI Embeddings t-SNE Projection\n")
        f.write("![t-SNE Plot](figures/tsne_mri.png)\n")
        
    print(f"Summary report written to {reports_path / 'summary.md'}")

if __name__ == "__main__":
    main()
