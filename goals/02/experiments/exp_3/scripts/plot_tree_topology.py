#!/usr/bin/env python3
"""plot_tree_topology.py
Renders a visual diagram of the induced neurosymbolic decision tree topology
and saves it to reports/figures/decision_tree_topology.png.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_tree():
    exp_dir = Path(__file__).resolve().parent.parent
    tree_path = exp_dir / "results" / "artifacts" / "decision_tree.json"
    output_path = exp_dir / "reports" / "figures" / "decision_tree_topology.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 105)
    ax.set_ylim(-2, 102)
    ax.axis("off")

    coords = {
        "root": (50, 95),
        "root_L": (20, 80),
        "root_R": (72, 80),
        "root_L_L": (12, 65),
        "root_L_R": (28, 65),
        "root_R_L": (56, 65),
        "root_R_R": (88, 65),
        "root_R_L_L": (45, 50),
        "root_R_L_R": (65, 50),
        "root_R_R_L": (80, 50),
        "root_R_R_R": (95, 50),
        "root_R_L_R_L": (56, 35),
        "root_R_L_R_R": (73, 35),
        "root_R_R_R_L": (89, 35),
        "root_R_R_R_R": (98, 35),
        "root_R_L_R_R_L": (65, 20),
        "root_R_L_R_R_R": (80, 20),
        "root_R_L_R_R_R_L": (74, 5),
        "root_R_L_R_R_R_R": (86, 5),
    }

    # Draw edges
    for node_id, data in tree.items():
        if not data.get("is_leaf", False):
            left = data.get("left_child")
            right = data.get("right_child")
            x, y = coords[node_id]
            
            if left and left in coords:
                lx, ly = coords[left]
                ax.annotate(
                    "", xy=(lx, ly + 2.8), xytext=(x, y - 2.8),
                    arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.5)
                )
                ax.text(
                    (x + lx) / 2 - 1.5, (y + ly) / 2, "True",
                    fontsize=7.5, fontweight="bold", color="#27ae60",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
                )

            if right and right in coords:
                rx, ry = coords[right]
                ax.annotate(
                    "", xy=(rx, ry + 2.8), xytext=(x, y - 2.8),
                    arrowprops=dict(arrowstyle="->", color="#34495e", lw=1.5)
                )
                ax.text(
                    (x + rx) / 2 + 1.5, (y + ly) / 2, "False",
                    fontsize=7.5, fontweight="bold", color="#c0392b",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
                )

    # Draw nodes
    for node_id, data in tree.items():
        x, y = coords[node_id]
        is_leaf = data.get("is_leaf", False)
        n = data.get("n_samples", 0)
        p_yes = data.get("prob_yes", 0.0)
        counts = data.get("class_counts", {})
        n_yes = counts.get("yes", 0)
        n_no = counts.get("no", 0)

        if is_leaf:
            leaf_cls = data.get("leaf_class", "no")
            fc = "#d4edda" if leaf_cls == "yes" else "#f8d7da"
            ec = "#28a745" if leaf_cls == "yes" else "#dc3545"
            pred_text = f"{leaf_cls.upper()}\nP={p_yes:.2f}\n(n={n})"
            bbox_props = dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.8)
            ax.text(x, y, pred_text, ha="center", va="center", fontsize=7.5, fontweight="bold", bbox=bbox_props)
        else:
            rule = data.get("rule", {})
            feat = rule.get("feature", "").replace("radiology_report__", "").replace("structured_prompt__", "").replace("laboratory_results__", "")
            op = rule.get("operator", "")
            thr = rule.get("threshold", "")
            rule_text = f"{feat} {op} {thr}\n(n={n})"
            bbox_props = dict(boxstyle="square,pad=0.4", fc="#e8f4f8", ec="#2980b9", lw=1.8)
            ax.text(x, y, rule_text, ha="center", va="center", fontsize=7.5, fontweight="bold", bbox=bbox_props)

    plt.title("Neuro-Symbolic Decision Tree Topology (Goal 02 — exp_3)\nInduced via Two-Stage Decoupling on EAU/ESMO Guidelines",
              fontsize=13, fontweight="bold", pad=15)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Topology saved successfully to: {output_path}")

if __name__ == "__main__":
    plot_tree()
