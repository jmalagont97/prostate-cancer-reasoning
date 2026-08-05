import os
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

def scan_cohort(data_dir: Path) -> dict:
    total_folders = 0
    # Files presence counters
    presence = {
        "prostate-biopsy-decision.json": 0,
        "prostate-biopsy-decision-reasoning.json": 0,
        "prostate-biopsy-decision-clinical-data.json": 0,
        "prostate-modality-level-neural-representations.json": 0,
        "structured-prompt.json": 0
    }
    
    # Active sources counters
    active_sources = {
        "MRI Embedding": 0,
        "Structured Clinical Notes": 0,
        "Clinical Lab Data": 0,
        "Biopsy Label": 0
    }
    
    label_distribution = {
        "yes": 0,
        "no": 0,
        "unlabeled": 0
    }

    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("PT-pseudo_"):
            continue
        total_folders += 1
        
        # Check files existence
        files_found = {}
        for fname in presence.keys():
            fpath = case_dir / fname
            exists = fpath.exists()
            files_found[fname] = exists
            if exists:
                presence[fname] += 1
                
        # Parse targets
        decision_file = case_dir / "prostate-biopsy-decision.json"
        if decision_file.exists():
            try:
                with open(decision_file) as f:
                    label = json.load(f)
                    if label in ["yes", "no"]:
                        label_distribution[label] += 1
                        active_sources["Biopsy Label"] += 1
                    else:
                        label_distribution["unlabeled"] += 1
            except Exception:
                label_distribution["unlabeled"] += 1
        else:
            label_distribution["unlabeled"] += 1
            
        # Parse representations
        rep_file = case_dir / "prostate-modality-level-neural-representations.json"
        if rep_file.exists():
            try:
                with open(rep_file) as f:
                    rep = json.load(f)
                    if rep.get("MRI image") is not None and len(rep["MRI image"]) > 0:
                        active_sources["MRI Embedding"] += 1
            except Exception:
                pass
                
        # Parse prompt notes
        prompt_file = case_dir / "structured-prompt.json"
        if prompt_file.exists():
            try:
                with open(prompt_file) as f:
                    prompt = json.load(f)
                    if prompt.get("note_sections") and len(prompt["note_sections"]) > 0:
                        active_sources["Structured Clinical Notes"] += 1
            except Exception:
                pass
                
        # Parse clinical labs
        clinical_file = case_dir / "prostate-biopsy-decision-clinical-data.json"
        if clinical_file.exists():
            try:
                with open(clinical_file) as f:
                    clinical = json.load(f)
                    if clinical.get("laboratory_results") and len(clinical["laboratory_results"]) > 0:
                        active_sources["Clinical Lab Data"] += 1
            except Exception:
                pass

    return {
        "total_cases": total_folders,
        "file_presence": presence,
        "active_sources": active_sources,
        "label_distribution": label_distribution
    }

def main():
    parser = argparse.ArgumentParser(description="Task 1 Cohort Composition Analysis")
    parser.add_argument("--data_dir", type=str, default="data/chimera26/raw/task1")
    parser.add_argument("--results_dir", type=str, default="experiments/exp_1/results")
    parser.add_argument("--reports_dir", type=str, default="experiments/exp_1/reports")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    results_path = Path(args.results_dir)
    reports_path = Path(args.reports_dir)
    figures_path = reports_path / "figures"

    results_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning cohort at {data_path}...")
    stats = scan_cohort(data_path)
    print(f"Total cases scanned: {stats['total_cases']}")

    # Save metrics JSON
    metrics_file = results_path / "eda_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    # Plot 1: File Presence
    plt.figure(figsize=(10, 6))
    files = list(stats["file_presence"].keys())
    counts = list(stats["file_presence"].values())
    short_names = [f.replace("prostate-biopsy-decision-", "").replace("prostate-", "") for f in files]
    
    bars = plt.bar(short_names, counts, color="#2b5c8f", edgecolor="black", alpha=0.9)
    plt.title("File Ingestion Completeness - Task 1 Cohort (n = 195)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Patient Count", fontsize=12)
    plt.xlabel("File Name Suffix", fontsize=12)
    plt.ylim(0, stats["total_cases"] + 20)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        pct = (height / stats["total_cases"]) * 100
        plt.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="semibold")
    
    plt.tight_layout()
    plot1_path = figures_path / "file_presence.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to {plot1_path}")

    # Plot 2: Active Sources
    plt.figure(figsize=(9, 6))
    sources = list(stats["active_sources"].keys())
    s_counts = list(stats["active_sources"].values())
    
    bars2 = plt.bar(sources, s_counts, color="#2ca02c", edgecolor="black", alpha=0.9)
    plt.title("Available Information Sources - Task 1 Cohort (n = 195)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Patient Count", fontsize=12)
    plt.xlabel("Information Source / Modality", fontsize=12)
    plt.ylim(0, stats["total_cases"] + 20)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Annotate bars
    for bar in bars2:
        height = bar.get_height()
        pct = (height / stats["total_cases"]) * 100
        plt.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="semibold")
                    
    plt.tight_layout()
    plot2_path = figures_path / "active_sources.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to {plot2_path}")

    # Write summary.md
    summary_file = reports_path / "summary.md"
    total = stats["total_cases"]
    labeled = stats["label_distribution"]["yes"] + stats["label_distribution"]["no"]
    unlabeled = stats["label_distribution"]["unlabeled"]
    
    with open(summary_file, "w") as f:
        f.write("# Cohort Composition Summary Report — Task 1 (Biopsy Decision)\n\n")
        f.write(f"**Date**: 2026-07-20  \n")
        f.write(f"**Total Patient Folders Scanned**: {total}  \n")
        f.write(f"**Labeled Cases (Train Partition)**: {labeled} ({labeled/total*100:.1f}%)  \n")
        f.write(f"**Unlabeled Cases (Test Partition)**: {unlabeled} ({unlabeled/total*100:.1f}%)  \n\n")
        
        f.write("## Target Variable Distribution (Labeled Split)\n")
        f.write(f"*   **`yes` (Requires Biopsy)**: {stats['label_distribution']['yes']} cases ({stats['label_distribution']['yes']/labeled*100:.1f}%)\n")
        f.write(f"*   **`no` (Do Not Biopsy)**: {stats['label_distribution']['no']} cases ({stats['label_distribution']['no']/labeled*100:.1f}%)\n\n")
        
        f.write("## File Completeness Audit\n\n")
        f.write("| File Name | Present Cases | Presence Rate (% of total) |\n")
        f.write("| :--- | :---: | :---: |\n")
        for fname, count in stats["file_presence"].items():
            f.write(f"| `{fname}` | {count} | {count/total*100:.1f}% |\n")
        f.write("\n")
        
        f.write("## Modality Availability Audit\n\n")
        f.write("| Information Source | Present Cases | Presence Rate (% of total) | Notes |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        for src, count in stats["active_sources"].items():
            notes = ""
            if src == "MRI Embedding":
                notes = "4 cases missing (PT-pseudo_4bfd4ec864d8, PT-pseudo_4d54f04e26ae, PT-pseudo_7dbdcd6f9064, PT-pseudo_8636aa471ef7)"
            elif src == "Biopsy Label":
                notes = "Only available for labeled training split (91 cases)"
            f.write(f"| {src} | {count} | {count/total*100:.1f}% | {notes} |\n")
        f.write("\n")
        
        f.write("## Visualizations\n\n")
        f.write("### File Presence Completeness\n")
        f.write("![File Presence](figures/file_presence.png)\n\n")
        f.write("### Available Modalities\n")
        f.write("![Active Sources](figures/active_sources.png)\n")
        
    print(f"Summary report written to {summary_file}")

if __name__ == "__main__":
    main()
