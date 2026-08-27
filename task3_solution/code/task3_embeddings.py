"""Embedding loading + masked-mean pooling for task3, promoted from experiments/exp_5's exploration
code into production once exp_5 confirmed a real, bootstrap-significant win (prostatectomy-slide
embeddings, PCA-reduced, added to the tabular+text feature set: LOOCV C-index 0.7655 -> 0.8575,
95% CI [+0.028, +0.176], see experiments/exp_5/). Same EmbeddingSource design as
experiments/exp_5/scripts/common.py -- aligns by case_id explicitly, never by row order.

Two data paths, not one:
- The pre-compiled `data/processed/task3/embeddings/*.npz` (built once by scripts/build_dataset.py
  from the 75 cases in our downloaded training release) -- fast, used for fitting/LOOCV over the
  known dataset.
- A raw-file fallback, reading a case's `prostate-modality-level-neural-representations.json`
  directly (same file scripts/build_dataset.py itself reads, same key convention: "MRI image" list
  of 1, "Biopsy slide"/"Prostatectomy slide" variable-length lists, no padding at this stage --
  padding to a fixed bag size is an artifact build_dataset.py introduces for the compiled .npz, not
  present in the raw file). This is the path a genuinely new case at real submission time would
  need: it isn't in the pre-compiled npz at all (that file only ever covers the 75 training cases),
  so without this fallback the agent would silently treat every new case as having zero embedding
  data, quietly defeating the entire point of the exp_5 promotion. Caught during a gap-analysis
  pass, not by a failing test -- our current downloaded release has 0 unlabeled cases locally to
  exercise this path against directly, so it's validated instead by checking it reproduces the
  npz-derived pooled vector exactly for a case already in both (see task3_embeddings_test.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EMB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "task3" / "embeddings"
RAW_TASK3_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "train_release_v2"
                  / "train_release" / "task3")
REP_FILENAME = "prostate-modality-level-neural-representations.json"
REP_KEY = {"mri": "MRI image", "biopsy_slides": "Biopsy slide", "prostatectomy_slides": "Prostatectomy slide"}


class EmbeddingSource:
    """One embedding source, indexed by case_id. `bag` sources (biopsy/prostatectomy slides)
    expose per-case masked-mean pooling; `single` sources (MRI) expose the vector directly."""

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind  # "single" or "bag"
        d = np.load(EMB_DIR / f"{name}.npz", allow_pickle=True)
        case_ids = d["case_ids"]
        self.emb = {cid: d["emb"][i] for i, cid in enumerate(case_ids)}
        self.mask = {cid: d["mask"][i] for i, cid in enumerate(case_ids)}

    def raw_dim(self) -> int:
        return next(iter(self.emb.values())).shape[-1]

    def _pooled_from_compiled(self, case_id: str) -> np.ndarray | None:
        emb, mask = self.emb.get(case_id), self.mask.get(case_id)
        if emb is None:
            return None
        if self.kind == "single":
            return emb.astype(float) if mask else None
        valid = emb[mask]
        return valid.mean(axis=0).astype(float) if len(valid) > 0 else None

    def _pooled_from_raw_case(self, case_id: str) -> np.ndarray | None:
        """Fallback for a case not in the pre-compiled npz: reads its raw per-case
        neural-representations.json directly, matching scripts/build_dataset.py's exact parsing
        convention (same dict key, same "first element" handling for MRI) rather than re-deriving
        it and risking a subtle divergence."""
        rep_path = RAW_TASK3_DIR / case_id / REP_FILENAME
        if not rep_path.exists():
            return None
        rep = json.loads(rep_path.read_text())
        vectors = rep.get(REP_KEY[self.name], [])
        if not vectors:
            return None
        if self.kind == "single":
            return np.asarray(vectors[0], dtype=float)
        return np.mean([np.asarray(v, dtype=float) for v in vectors], axis=0)

    def pooled_vector(self, case_id: str) -> np.ndarray | None:
        """Masked mean over valid slots (bag) or the vector itself (single). None if no valid data
        for this case at all (either a real gap for a known case, e.g. biopsy_slides' 24 zero-slide
        training cases, or a genuinely new case with no recorded scan of that modality)."""
        if case_id in self.emb:
            return self._pooled_from_compiled(case_id)
        return self._pooled_from_raw_case(case_id)


PROSTATECTOMY = "prostatectomy_slides"
BIOPSY = "biopsy_slides"
MRI = "mri"
