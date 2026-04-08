"""
Retrieval agent — FAISS bi-encoder retrieval + cross-encoder reranking.

Two-stage pipeline:
  1. Embed the patient query with neuml/pubmedbert-base-embeddings, pull top-50
     from the 64K-trial PubMedBERT FAISS index (cosine via IndexFlatIP).
  2. Score each (query, chunk_text) pair with ClinicalReranker (ms-marco
     MiniLM L-6-v2 cross-encoder), re-sort. Reranking is toggled by
     use_reranker flag — set False to return raw FAISS order.

Embedding model upgrade (2026-04-08): switched from all-MiniLM-L6-v2 (10K index)
to neuml/pubmedbert-base-embeddings (64K index). Manual relevance audit confirmed
top-10 retrievals are clinically coherent for 4 of 5 audited queries.

Known limitation: ms-marco cross-encoder was trained on web passage retrieval,
not clinical eligibility text. It is the documented baseline to beat — a
biomedical cross-encoder fine-tuned on clinical trial text is the planned upgrade.

The FAISS index and embedding model are loaded at module import time.
SQLite is opened per-call — cheap enough, avoids holding a connection.
"""

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.reranker import ClinicalReranker

# ---------------------------------------------------------------------------
# Paths — DB_PATH overridable via environment so tests can point at a fixture DB
# ---------------------------------------------------------------------------

_INDEX_PATH  = Path("data/trials_pubmedbert.index")
_NCTIDS_PATH = Path("data/nct_ids_pubmedbert.npy")
DB_PATH = Path(os.environ.get("DB_PATH", "data/trialcompass.db"))

_BI_MODEL_NAME = "neuml/pubmedbert-base-embeddings"

# ---------------------------------------------------------------------------
# Module-level model / index load — pay the cost once, not per call
# ---------------------------------------------------------------------------

if not _INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found at {_INDEX_PATH}. "
        "Run: python src/embeddings/embed_biobert.py to build the PubMedBERT index."
    )

_index: faiss.IndexFlatIP = faiss.read_index(str(_INDEX_PATH))
_nct_ids: np.ndarray = np.load(str(_NCTIDS_PATH), allow_pickle=True)
_bi_encoder: SentenceTransformer = SentenceTransformer(_BI_MODEL_NAME)
_reranker: ClinicalReranker = ClinicalReranker()


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_query_string(profile: dict[str, Any]) -> str:
    """
    Turn a parsed patient profile dict into a natural language retrieval query.

    Field precedence roughly mirrors what the bi-encoder index is sensitive to:
    cancer type first (drives ~70% of cosine similarity in practice), then
    treatment context, then biomarkers last because the 256-token truncation
    means biomarker terms often don't make it into the embedding.

    Missing fields are silently skipped — a sparse profile still gets a query.
    """
    parts: list[str] = []

    cancer_type: Optional[str] = profile.get("cancer_type")
    if cancer_type:
        metastatic: Optional[bool] = profile.get("metastatic")
        prefix = "metastatic " if metastatic else ""
        parts.append(f"{prefix}{cancer_type}")

    # Can't use `or` here — 0 prior lines is meaningful and 0 is falsy
    prior: Optional[int] = profile.get("prior_treatment_lines")
    if prior is None:
        prior = profile.get("prior_treatments")
    if prior is not None:
        if prior == 0:
            parts.append("treatment naive")
        elif prior == 1:
            parts.append("one prior line of therapy")
        else:
            parts.append(f"{prior} prior lines of therapy heavily pretreated")

    # Biomarkers — dict or nested dict both handled
    biomarkers = profile.get("biomarkers") or {}
    # parser_agent returns a nested dict; direct callers may pass flat key=value pairs
    if hasattr(biomarkers, "__iter__") and not isinstance(biomarkers, str):
        if isinstance(biomarkers, dict):
            active = [
                f"{marker} {status}"
                for marker, status in biomarkers.items()
                if status and status not in ("None", "null")
            ]
            if active:
                parts.append(" ".join(active))

    ecog: Optional[int] = profile.get("ecog")
    if ecog is not None:
        parts.append(f"ECOG performance status {ecog}")

    age: Optional[int] = profile.get("age")
    if age is not None:
        parts.append(f"age {age}")

    # Should never happen given parser_agent always fills cancer_type, but handle it
    if not parts:
        return "oncology clinical trial cancer treatment"

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve_and_rerank(
    patient_profile: dict[str, Any],
    top_k: int = 10,
    faiss_candidates: int = 50,
    use_reranker: bool = True,
) -> list[dict[str, Any]]:
    """
    Full two-stage retrieval for a patient profile.

    Parameters
    ----------
    patient_profile : dict
        Structured output from parser_agent.parse_patient_profile(), or any dict
        with the same keys: cancer_type, biomarkers, ecog, prior_treatment_lines,
        age, metastatic.
    top_k : int
        How many results to return after reranking. Default 10.
    faiss_candidates : int
        How many candidates to pull from FAISS before cross-encoder reranking.
        50 is the recommended value for 64K-trial corpus.
    use_reranker : bool
        If True (default), run ClinicalReranker cross-encoder on top-50 candidates
        and sort by ce_score. If False, return the raw FAISS bi-encoder order
        (useful for ablation comparisons and latency-sensitive contexts).

    Returns
    -------
    list[dict]
        Ranked list of trials, best match first. Each dict has:
        nct_id, brief_title, conditions, eligibility_text, chunk_text,
        bi_encoder_score, ce_score (if use_reranker), rank (1-indexed).
    """
    query = build_query_string(patient_profile)
    print(f"[retrieval] query: {query!r}")

    # --- Stage 1: bi-encoder FAISS retrieval ---
    t0 = time.time()
    q_emb = _bi_encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine via inner product on unit vectors
    ).astype("float32")
    bi_scores, bi_indices = _index.search(q_emb, faiss_candidates)
    bi_elapsed = time.time() - t0
    print(f"[retrieval] FAISS top-{faiss_candidates} retrieved in {bi_elapsed:.3f}s")

    # Map index positions → (nct_id, bi_score)
    candidates = [
        (_nct_ids[i], float(bi_scores[0][rank]))
        for rank, i in enumerate(bi_indices[0])
    ]

    # --- Fetch chunk_text from SQLite ---
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    nct_list = [nct for nct, _ in candidates]
    placeholders = ",".join("?" * len(nct_list))
    cur.execute(
        f"SELECT nct_id, brief_title, conditions, eligibility_text, chunk_text "
        f"FROM trials WHERE nct_id IN ({placeholders})",
        nct_list,
    )
    rows = {r["nct_id"]: dict(r) for r in cur.fetchall()}
    con.close()

    # Preserve FAISS order; attach bi_encoder_score
    ordered_rows = []
    bi_score_map = {nct: score for nct, score in candidates}
    for nct, _ in candidates:
        if nct in rows:
            row = rows[nct]
            row["bi_encoder_score"] = bi_score_map[nct]
            ordered_rows.append(row)

    # --- Stage 2: cross-encoder reranking (optional) ---
    if use_reranker:
        t1 = time.time()
        reranked = _reranker.rerank(query, ordered_rows)
        ce_elapsed = time.time() - t1
        print(
            f"[retrieval] cross-encoder reranked {len(ordered_rows)} candidates "
            f"in {ce_elapsed:.3f}s"
        )
        print(
            f"[retrieval] top-3 ce_scores: "
            + ", ".join(f"{r['ce_score']:.3f}" for r in reranked[:3])
        )
    else:
        reranked = ordered_rows
        print("[retrieval] reranker disabled — returning raw FAISS order")

    # Add 1-indexed rank and trim to top_k
    for i, row in enumerate(reranked[:top_k]):
        row["rank"] = i + 1

    return reranked[:top_k]
