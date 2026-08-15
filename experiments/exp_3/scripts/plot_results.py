# Confusion-matrix figures for exp_3 (selected config only).
# MCCV = aggregation of 900 validation events across the 50 frozen splits (NOT per-patient).
# LOO  = one prediction per patient over the 88 usable_labeled cases.
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = PROJECT_ROOT / "experiments" / "exp_3"
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "reports" / "figures"

LABELS = ["no", "yes"]


def load_selected_id() -> str:
    hp = json.loads((RESULTS_DIR / "selected_config" / "hyperparameters.json").read_text())
    return hp["config_id"]


def build_matrix(cdf: pd.DataFrame, selected_id: str) -> np.ndarray:
    sub = cdf.loc[cdf["config_id"].eq(selected_id)]
    agg = sub.groupby(["true_label", "pred_label"], as_index=False)["count"].sum()
    mat = np.zeros((2, 2), dtype=int)
    for _, r in agg.iterrows():
        mat[int(r["true_label"]), int(r["pred_label"])] = int(r["count"])
    return mat


def draw(ax, mat: np.ndarray, title: str, normalize: bool, annot_size: int = 14) -> None:
    disp = mat.astype(float)
    if normalize:
        row_sums = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, row_sums, out=np.zeros_like(disp), where=row_sums > 0)
    im = ax.imshow(disp, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else disp.max())
    for i in range(2):
        for j in range(2):
            if normalize:
                text = f"{disp[i, j]:.1%}"
            else:
                text = f"{int(mat[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=annot_size,
                    color="white" if disp[i, j] > (0.6 if normalize else disp.max() * 0.6) else "black")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    return im


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    selected_id = load_selected_id()

    cdf = pd.read_csv(RESULTS_DIR / "confusion_matrices_mccv.csv")
    loo_cdf = pd.read_csv(RESULTS_DIR / "confusion_matrix_loo.csv")

    mccv_mat = build_matrix(cdf, selected_id)
    loo_mat = build_matrix(loo_cdf, selected_id)
    mccv_total = int(mccv_mat.sum())
    loo_total = int(loo_mat.sum())

    # 1-4: individual figures
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw(ax, mccv_mat,
         f"Selected: {selected_id}\nMCCV aggregated ({mccv_total} validation events, 50 splits)",
         normalize=False)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_mccv_counts.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw(ax, mccv_mat,
         f"Selected: {selected_id}\nMCCV aggregated, row-normalized ({mccv_total} validation events)",
         normalize=True)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_mccv_normalized.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw(ax, loo_mat,
         f"Selected: {selected_id}\nLOO pooled ({loo_total} predictions, one per patient)",
         normalize=False)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_loo_counts.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    draw(ax, loo_mat,
         f"Selected: {selected_id}\nLOO pooled, row-normalized ({loo_total} predictions)",
         normalize=True)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_loo_normalized.png", dpi=200)
    plt.close(fig)

    # 5: side-by-side MCCV vs LOO
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    draw(axes[0], mccv_mat,
         f"MCCV aggregated\n({mccv_total} validation events)",
         normalize=False)
    draw(axes[1], loo_mat,
         f"LOO pooled\n({loo_total} predictions, one per patient)",
         normalize=False)
    fig.suptitle(f"Confusion matrices — selected config: {selected_id}", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_mccv_vs_loo.png", dpi=200)
    plt.close(fig)

    print("[figures] wrote 5 figures:")
    for p in sorted(FIGURES_DIR.glob("confusion_matrix*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
