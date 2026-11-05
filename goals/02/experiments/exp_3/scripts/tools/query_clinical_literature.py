#!/usr/bin/env python3
"""query_clinical_literature.py

Wrapper tool for exp_3 to query the ChromaDB vector store of clinical prostate cancer guidelines
(EAU & ESMO) in goals/02/data/chroma_db/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import importlib.util

# Load underlying package module directly by path to avoid circular self-import
_pkg_query = None
current_file = Path(__file__).resolve()
for p in current_file.parents:
    for candidate in [p / "pkg", p / "utils" / "pkg"]:
        target_script = candidate / "query_clinical_literature.py"
        if target_script.exists():
            spec = importlib.util.spec_from_file_location("pkg_query_literature", target_script)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _pkg_query = getattr(mod, "query_clinical_literature", None)
            break
    if _pkg_query:
        break

if not _pkg_query:
    raise ImportError("Could not find underlying query_clinical_literature.py in pkg or utils/pkg")


def query_clinical_literature(
    query: str,
    n_results: int = 3,
    doc_name: Optional[str] = None,
    topic: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute semantic query against clinical guidelines (EAU/ESMO)."""
    return _pkg_query(query=query, n_results=n_results, doc_name=doc_name, topic=topic)


def main():
    parser = argparse.ArgumentParser(description="Query prostate cancer guidelines.")
    parser.add_argument("query", help="Clinical question or query string")
    parser.add_argument("-n", "--num-results", type=int, default=3, help="Chunks to retrieve")
    parser.add_argument("--doc", choices=["EAU", "ESMO"], help="Filter by guideline")
    parser.add_argument("--topic", help="Filter by topic")

    args = parser.parse_args()
    res = query_clinical_literature(
        query=args.query,
        n_results=args.num_results,
        doc_name=args.doc,
        topic=args.topic,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
