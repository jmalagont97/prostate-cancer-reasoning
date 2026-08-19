"""Figure/artifact writers shared across experiments. No experiment targets
this module yet (it's added ahead of need per the plan — low cost, and every
exp_9-24 script re-implements a subset of it). Must never import `kdm`.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np


class _StrictEncoder(json.JSONEncoder):
    """Rejects NaN/Infinity instead of silently emitting the bare (invalid
    JSON) literals `json.dump` writes by default — exp_24/results/
    confidence_metrics.json contains bare `NaN` from exactly this, which
    strict JSON parsers (not `json.load`) reject."""

    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _check_finite(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_finite(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite value at {path!r}: {obj!r} — NaN/Infinity are not valid JSON")


def write_json(obj, path: Path, allow_nan: bool = False) -> None:
    if not allow_nan:
        _check_finite(obj)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, cls=_StrictEncoder, allow_nan=allow_nan)


def record_git_commit(results_dir: Path) -> None:
    """Writes `results/git_commit.txt` — a checklist item every DESIGN.md in
    exp_13-17 lists but none of their `results/` actually contain."""
    out = subprocess.run(["git", "log", "-1", "--format=%H %s"], capture_output=True, text=True, check=True)
    (Path(results_dir) / "git_commit.txt").write_text(out.stdout.strip() + "\n")


def plot_confusion_matrix(y_true, y_pred, labels, title: str, out_path: Path, cmap="Blues"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=getattr(plt.cm, cmap))
    plt.title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_roc_curves(curves: dict, y_true, title: str, out_path: Path):
    """`curves`: {label: p_soft_array}."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    plt.figure(figsize=(7, 6))
    for label, p_soft in curves.items():
        fpr, tpr, _ = roc_curve(y_true, p_soft)
        auc_val = roc_auc_score(y_true, p_soft)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_grid_search_curves(df_grid, x_col: str, y_col: str, group_cols: list, title: str, out_path: Path,
                             xscale: str = "linear"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    for keys, grp in df_grid.groupby(group_cols):
        grp_sorted = grp.sort_values(x_col)
        label = ",".join(f"{c}={v}" for c, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        plt.plot(grp_sorted[x_col], grp_sorted[y_col], marker="o", alpha=0.8, label=label)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if xscale != "linear":
        plt.xscale(xscale)
    plt.title(title, fontweight="bold")
    plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_signal_scatter(x, y, labels, colors: dict, xlabel: str, ylabel: str, title: str, out_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for label, color in colors.items():
        mask = np.asarray(labels) == label
        plt.scatter(np.asarray(x)[mask], np.asarray(y)[mask], c=color, label=label,
                    alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
