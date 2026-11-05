#!/usr/bin/env python3
"""query_clinical_literature.py

ChromaDB Vector Store and Semantic Search for Clinical Prostate Cancer Guidelines (EAU & ESMO).
Designed for the Clinical Oracle Agent in Goal 02 (Neuro-Symbolic Decision Tree Induction).

Capabilities:
- Hierarchical sliding-window chunking of clinical guidelines (ESMO.md & EANM.md).
- Embedded contextual section headers for high-fidelity semantic embeddings.
- Persistent local ChromaDB storage in goals/02/data/chroma_db/.
- Self-healing collection initialization and automatic ingestion.
- Standardized tool interface compatible with agent declarations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection


def _resolve_paths() -> Tuple[Path, Path, Path]:
    """Resolve repository root, guidelines directory, and ChromaDB persistence directory."""
    current_file = Path(__file__).resolve()
    # Path hierarchy: goals/02/pkg/query_clinical_literature.py -> goals/02/pkg -> goals/02
    goal_dir = current_file.parent.parent
    context_dir = goal_dir / "data" / "context"
    chroma_dir = goal_dir / "data" / "chroma_db"
    return goal_dir, context_dir, chroma_dir


def _detect_topic(text: str) -> str:
    """Assign a primary clinical topic tag based on keywords in the chunk."""
    lower = text.lower()
    if any(k in lower for k in ["biopsy", "biopsies", "cores", "transperineal", "transrectal"]):
        return "biopsy_strategy"
    if any(k in lower for k in ["pi-rads", "pirads", "mpmri", "mri"]):
        return "imaging_pirads"
    if any(k in lower for k in ["psa density", "psad", "psa level", "kinetics"]):
        return "psa_biomarkers"
    if any(k in lower for k in ["gleason", "isup", "grade group", "grading"]):
        return "histology_grading"
    if any(k in lower for k in ["active surveillance", "watchful waiting"]):
        return "active_surveillance"
    if any(k in lower for k in ["staging", "t-stage", "n-stage", "m-stage", "risk group", "d'amico"]):
        return "risk_stratification"
    return "general_clinical"


def parse_and_chunk_guideline(
    doc_path: Path, 
    doc_name: str, 
    chunk_size: int = 35, 
    overlap: int = 10
) -> List[Dict[str, Any]]:
    """Parse a markdown guideline line-by-line, tracking clinical sections and building chunks with overlap."""
    if not doc_path.exists():
        raise FileNotFoundError(f"Guideline file not found at: {doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]

    current_section = f"{doc_name} Overview"
    line_metadata: List[str] = []

    # First pass: map each line to its enclosing markdown section
    for line in lines:
        stripped = line.strip()
        # Markdown headings (# , ## , ### )
        if stripped.startswith("#"):
            heading_text = re.sub(r"^#+\s*", "", stripped)
            current_section = heading_text
        # Numbered chapters in EAU (e.g. '5. DIAGNOSTIC EVALUATION', '5.6 Biopsy strategy')
        elif re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped):
            current_section = stripped

        line_metadata.append(current_section)

    # Second pass: sliding window chunking with embedded context headers
    chunks: List[Dict[str, Any]] = []
    step = max(1, chunk_size - overlap)

    for idx, i in enumerate(range(0, len(lines), step)):
        window = lines[i : i + chunk_size]
        window_text = "\n".join(window).strip()
        if not window_text:
            continue

        w_metas = line_metadata[i : i + chunk_size]
        section_name = w_metas[len(w_metas) // 2]  # representative section for the window
        topic = _detect_topic(window_text)

        header = f"### Source: {doc_name} Clinical Guideline [{section_name}] (Topic: {topic})\n"
        full_text = header + "\n".join(window)

        chunks.append({
            "id": f"{doc_name.lower()}_chunk_{idx:04d}",
            "text": full_text,
            "doc_name": doc_name,
            "section": section_name,
            "topic": topic,
            "start_line": i + 1,
            "end_line": min(i + chunk_size, len(lines)),
        })

    return chunks


def build_or_get_clinical_collection(
    collection_name: str = "prostate_clinical_guidelines",
    chroma_path: Optional[Path] = None,
    context_dir: Optional[Path] = None,
    force_reindex: bool = False,
) -> Collection:
    """Connect to persistent ChromaDB collection in goals/02/data/chroma_db/. Ingests guidelines if empty."""
    _, default_context, default_chroma = _resolve_paths()
    chroma_path = chroma_path or default_chroma
    context_dir = context_dir or default_context

    chroma_path.mkdir(parents=True, exist_ok=True)
    client: ClientAPI = chromadb.PersistentClient(path=str(chroma_path))

    if force_reindex:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "EAU & ESMO Clinical Practice Guidelines for Prostate Cancer"}
    )

    # Ingest documents if collection is new or empty
    if collection.count() == 0:
        all_chunks: List[Dict[str, Any]] = []
        guidelines = [
            ("ESMO", context_dir / "ESMO.md"),
            ("EAU", context_dir / "EANM.md"),
        ]

        for doc_name, path in guidelines:
            if path.exists():
                doc_chunks = parse_and_chunk_guideline(path, doc_name=doc_name)
                all_chunks.extend(doc_chunks)

        if not all_chunks:
            raise RuntimeError(f"No clinical chunks could be parsed from {context_dir}")

        # Batch ingestion to ChromaDB (batches of 100)
        batch_size = 100
        for b_idx in range(0, len(all_chunks), batch_size):
            batch = all_chunks[b_idx : b_idx + batch_size]
            collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "doc_name": c["doc_name"],
                        "section": c["section"],
                        "topic": c["topic"],
                        "start_line": c["start_line"],
                        "end_line": c["end_line"],
                    }
                    for c in batch
                ],
            )

    return collection


def query_clinical_literature(
    query: Optional[str] = None,
    n_results: int = 3,
    topic: Optional[str] = None,
    doc_name: Optional[str] = None,
    collection_name: str = "prostate_clinical_guidelines",
    chroma_path: Optional[Path] = None,
    context_dir: Optional[Path] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute semantic query against the clinical guidelines collection.

    Returns a standardized tool contract payload for the Oracle Agent.
    """
    tool_id = "query_clinical_literature"
    version = "1.0.0"

    clean_query = (query or "").strip()
    if not clean_query and kwargs:
        clean_query = " ".join(f"{k} {v}" for k, v in kwargs.items() if v).strip()

    if not clean_query:
        return {
            "ok": False,
            "tool_id": tool_id,
            "version": version,
            "error": {"type": "input_error", "message": "Query string cannot be empty."},
        }

    try:
        collection = build_or_get_clinical_collection(
            collection_name=collection_name,
            chroma_path=chroma_path,
            context_dir=context_dir,
        )

        n_samples = max(1, min(n_results, 10))
        
        # Build optional metadata filter
        where_filter: Optional[Dict[str, Any]] = None
        filters = []
        if doc_name:
            filters.append({"doc_name": {"$eq": doc_name}})
        if topic:
            filters.append({"topic": {"$eq": topic}})

        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        results = collection.query(
            query_texts=[clean_query],
            n_results=n_samples,
            where=where_filter,
        )

        retrieved_chunks = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, cid, dist in zip(docs, metas, ids, distances):
                retrieved_chunks.append({
                    "chunk_id": cid,
                    "doc_name": meta.get("doc_name", "Unknown"),
                    "section": meta.get("section", "Unknown"),
                    "topic": meta.get("topic", "general"),
                    "text": doc,
                    "relevance_distance": round(float(dist), 4),
                })

        return {
            "ok": True,
            "tool_id": tool_id,
            "version": version,
            "result": {
                "total_retrieved": len(retrieved_chunks),
                "chunks": retrieved_chunks,
            },
        }

    except Exception as exc:
        return {
            "ok": False,
            "tool_id": tool_id,
            "version": version,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _run_cli() -> None:
    """Interactive CLI test runner for query_clinical_literature."""
    parser = argparse.ArgumentParser(description="Query prostate cancer guidelines via ChromaDB.")
    parser.add_argument("query", nargs="?", help="Clinical question or search query")
    parser.add_argument("-n", "--num-results", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--doc", choices=["EAU", "ESMO"], help="Filter by specific guideline")
    parser.add_argument("--topic", help="Filter by clinical topic")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of guidelines")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")

    args = parser.parse_args()

    if args.reindex:
        print("Re-indexing clinical guidelines into ChromaDB (goals/02/data/chroma_db/)...")
        col = build_or_get_clinical_collection(force_reindex=True)
        print(f"Re-indexing complete. Total chunks indexed: {col.count()}")
        if not args.query:
            return

    test_queries = [args.query] if args.query else [
        "PIRADS 4 or 5 biopsy recommendation",
        "PIRADS 3 and PSA density threshold for biopsy",
        "Prior positive biopsy confirmed cancer repeat biopsy indication",
        "PIRADS 1 or 2 negative MRI when is biopsy avoided",
    ]

    for q in test_queries:
        res = query_clinical_literature(
            query=q,
            n_results=args.num_results,
            doc_name=args.doc,
            topic=args.topic,
        )

        if args.json:
            print(json.dumps(res, indent=2))
            continue

        print(f"\n{'='*70}\nQUERY: '{q}'\n{'='*70}")
        if not res["ok"]:
            print(f"Error: {res['error']['message']}")
            continue

        for idx, c in enumerate(res["result"]["chunks"], start=1):
            print(f"--- Chunk #{idx} [{c['doc_name']} - {c['section']}] (Topic: {c['topic']}, dist: {c['relevance_distance']}) ---")
            print(c["text"][:350] + "\n...")


if __name__ == "__main__":
    _run_cli()
