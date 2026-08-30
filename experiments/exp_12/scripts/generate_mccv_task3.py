#!/usr/bin/env python3
"""Generate and validate deterministic 30-fold MCCV splits for Task 3 survival data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

N_SPLITS = 30
N_CASES = 75
N_VALIDATION = 15
N_VALIDATION_EVENTS = 4
N_CANDIDATES = 2000
BASE_SEED = 42000
SPLIT_COLUMNS = tuple(f"split_{i}" for i in range(N_SPLITS))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_labels(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"case_id", "months_to_recurrence", "event"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Labels must contain {sorted(required)}")
    records = []
    for row in rows:
        try:
            time = float(row["months_to_recurrence"])
            event = int(row["event"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid survival label for {row.get('case_id')}") from exc
        if not row["case_id"] or time < 0 or event not in (0, 1):
            raise ValueError(f"Invalid survival label for {row.get('case_id')}")
        records.append({"case_id": row["case_id"], "time": time, "event": event})
    records.sort(key=lambda row: row["case_id"])
    ids = [row["case_id"] for row in records]
    if len(records) != N_CASES or len(set(ids)) != N_CASES:
        raise ValueError(f"Expected {N_CASES} unique cases, found {len(set(ids))}")
    if sum(row["event"] for row in records) != 19:
        raise ValueError("Expected 19 observed recurrence events in Task 3")
    return records


def time_bins(times: np.ndarray) -> np.ndarray:
    """Return four rank-based temporal bins, robust to tied follow-up times."""
    order = np.argsort(times, kind="stable")
    ranks = np.empty(len(times), dtype=int)
    ranks[order] = np.arange(len(times))
    return np.minimum((ranks * 4) // len(times), 3)


def balance_loss(
    validation: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    bins: np.ndarray,
) -> float:
    train = np.ones(len(events), dtype=bool)
    train[validation] = False
    val_events = events[validation]
    train_events = events[train]
    loss = abs(float(val_events.mean()) - float(train_events.mean())) * 10.0
    val_hist = np.bincount(bins[validation], minlength=4) / len(validation)
    train_hist = np.bincount(bins[train], minlength=4) / train.sum()
    loss += float(np.abs(val_hist - train_hist).sum())
    loss += abs(float(times[validation].mean()) - float(times[train].mean())) / max(float(times.mean()), 1.0)
    for quantile in (0.25, 0.5, 0.75):
        loss += abs(float(np.quantile(times[validation], quantile)) - float(np.quantile(times[train], quantile))) / max(float(times.mean()), 1.0)
    return loss


def generate(records: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    events = np.asarray([row["event"] for row in records], dtype=int)
    times = np.asarray([row["time"] for row in records], dtype=float)
    event_indices = np.flatnonzero(events == 1)
    censor_indices = np.flatnonzero(events == 0)
    bins = time_bins(times)
    output = [{"case_id": row["case_id"]} for row in records]
    diagnostics: dict[str, Any] = {"n_cases": len(records), "n_splits": N_SPLITS, "base_seed": BASE_SEED, "validation_size": N_VALIDATION, "target_validation_events": N_VALIDATION_EVENTS, "folds": []}

    for split_index in range(N_SPLITS):
        seed = BASE_SEED + 1000 * split_index
        rng = np.random.default_rng(seed)
        best: np.ndarray | None = None
        best_loss = float("inf")
        for _ in range(N_CANDIDATES):
            candidate = np.concatenate((rng.choice(event_indices, N_VALIDATION_EVENTS, replace=False), rng.choice(censor_indices, N_VALIDATION - N_VALIDATION_EVENTS, replace=False)))
            candidate.sort()
            loss = balance_loss(candidate, events, times, bins)
            if loss < best_loss:
                best, best_loss = candidate, loss
        if best is None:
            raise RuntimeError(f"Could not generate split {split_index}")
        validation_set = set(int(index) for index in best)
        for index, row in enumerate(output):
            row[f"split_{split_index}"] = "val" if index in validation_set else "train"
        train_mask = np.ones(len(events), dtype=bool)
        train_mask[best] = False
        diagnostics["folds"].append({
            "split": split_index,
            "seed": seed,
            "validation_case_ids": [records[index]["case_id"] for index in best],
            "train_cases": int(train_mask.sum()),
            "validation_cases": int(best.size),
            "train_events": int(events[train_mask].sum()),
            "validation_events": int(events[best].sum()),
            "train_mean_months": float(times[train_mask].mean()),
            "validation_mean_months": float(times[best].mean()),
            "balance_loss": best_loss,
        })
    return output, diagnostics


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", *SPLIT_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def validate(path: Path, labels: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = [row["case_id"] for row in labels]
    if len(rows) != N_CASES or [row["case_id"] for row in rows] != expected_ids:
        raise ValueError("Split rows do not match lexicographically ordered labels")
    if list(rows[0]) != ["case_id", *SPLIT_COLUMNS]:
        raise ValueError("Split columns are not exactly case_id plus split_0..split_29")
    for column in SPLIT_COLUMNS:
        values = [row[column] for row in rows]
        if set(values) != {"train", "val"} or values.count("train") != 60 or values.count("val") != 15:
            raise ValueError(f"Invalid train/validation counts in {column}")
        event_by_id = {row["case_id"]: row["event"] for row in labels}
        validation_events = sum(event_by_id[row["case_id"]] for row in rows if row[column] == "val")
        if validation_events != N_VALIDATION_EVENTS:
            raise ValueError(f"Expected {N_VALIDATION_EVENTS} validation events in {column}, found {validation_events}")
    return {"rows": len(rows), "splits": len(SPLIT_COLUMNS), "validated": True}


def git_version() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=repo_root(), text=True).strip() or "clean"
        return f"commit={commit}\nstatus={status}\n"
    except (OSError, subprocess.CalledProcessError):
        return "git_version=unavailable\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "validate", "all"), nargs="?", default="all")
    parser.add_argument("--labels", type=Path, default=Path("data/processed/task3/survival_labels.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/task3/mccv_splits_30.csv"))
    parser.add_argument("--diagnostics", type=Path, default=Path("experiments/exp_12/results/mccv_diagnostics.json"))
    args = parser.parse_args()
    root = repo_root()
    labels_path = args.labels if args.labels.is_absolute() else root / args.labels
    output_path = args.output if args.output.is_absolute() else root / args.output
    diagnostics_path = args.diagnostics if args.diagnostics.is_absolute() else root / args.diagnostics
    records = load_labels(labels_path)
    if args.mode in ("generate", "all"):
        rows, diagnostics = generate(records)
        write_csv(output_path, rows)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        (diagnostics_path.parent / "version.txt").write_text(git_version(), encoding="utf-8")
        print(f"GENERATED: {len(rows)} cases, {N_SPLITS} splits -> {output_path}")
    if args.mode in ("validate", "all"):
        result = validate(output_path, records)
        print(f"VALID: {result['rows']} cases, {result['splits']} splits, 60 train/15 val and 4 events per validation fold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
