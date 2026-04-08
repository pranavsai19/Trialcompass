"""
Retrieval agent — hybrid structured pre-filter + FAISS bi-encoder + cross-encoder rerank.

Three-stage pipeline:
  1. Structured pre-filter: apply hard constraints (age, status, phase, condition
     keywords) via StructuredFilter to narrow 64K corpus to a manageable candidate
     set. Resolves structured clinical fields that semantic search cannot distinguish
     (e.g. Child-Pugh score, ECOG, age limits). Falls back to full corpus if filter
     returns < MIN_FILTER_SIZE candidates.
  2. Bi-encoder FAISS retrieval: embed query with neuml/pubmedbert-base-embeddings,
     search within the filtered sub-index (top-50).
  3. Cross-encoder reranking: ClinicalReranker (ms-marco MiniLM L-6-v2) scores
     each (query, chunk_text) pair and re-sorts. Toggled by use_reranker flag.

Embedding model: neuml/pubmedbert-base-embeddings (64K index).
Manual relevance audit: top-10 retrievals clinically coherent for 4 of 5 queries.
Known limitation: ms-marco cross-encoder trained on web passages, not clinical text.
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
from src.retrieval.structured_filter import StructuredFilter

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
_struct_filter: StructuredFilter = StructuredFilter(DB_PATH)

# Minimum filtered corpus size — fall back to full index below this threshold
MIN_FILTER_SIZE = 100


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
    use_structured_filter: bool = True,
) -> list[dict[str, Any]]:
    """
    Full three-stage retrieval for a patient profile.

    Parameters
    ----------
    patient_profile : dict
        Structured output from parser_agent.parse_patient_profile(), or any dict
        with keys: cancer_type, biomarkers, ecog, prior_treatment_lines, age,
        metastatic, phase (list), status (list).
    top_k : int
        How many results to return after reranking. Default 10.
    faiss_candidates : int
        How many candidates to pull from FAISS before cross-encoder reranking.
        50 is the recommended value for the 64K-trial corpus.
    use_reranker : bool
        If True (default), run ClinicalReranker cross-encoder on top-50 candidates
        and sort by ce_score. If False, return the raw FAISS bi-encoder order.
    use_structured_filter : bool
        If True (default), apply StructuredFilter pre-filter to narrow the corpus
        before FAISS search. Falls back to full corpus if filter is too restrictive
        (< MIN_FILTER_SIZE results). Set False to bypass for ablation testing.

    Returns
    -------
    list[dict]
        Ranked list of trials, best match first. Each dict has:
        nct_id, brief_title, conditions, eligibility_text, chunk_text,
        bi_encoder_score, ce_score (if use_reranker), rank (1-indexed).
    """
    query = build_query_string(patient_profile)
    print(f"[retrieval] query: {query!r}")

    # --- Stage 1: Structured pre-filter ---
    search_index = _index        # default: full 64K index
    search_nct_ids = _nct_ids

    if use_structured_filter:
        t_sf = time.time()
        age    = patient_profile.get("age")
        phase  = patient_profile.get("phase")       # list[str] or None
        status = patient_profile.get("status")      # list[str] or None
        # Derive condition keywords from cancer_type if not explicitly provided
        cond_kw = patient_profile.get("conditions_keywords")
        if cond_kw is None:
            cancer_type = patient_profile.get("cancer_type", "")
            if cancer_type:
                cond_kw = [cancer_type] + ["cancer", "carcinoma", "tumor", "neoplasm"]

        filtered_ncts = _struct_filter.filter(
            age=age,
            phase=phase,
            status=status,
            conditions_keywords=cond_kw,
        )
        sf_elapsed = time.time() - t_sf

        if len(filtered_ncts) >= MIN_FILTER_SIZE:
            positions = _struct_filter.filter_to_index_positions(filtered_ncts, _nct_ids)
            # Build a sub-index from the filtered positions' vectors
            sub_vecs = np.vstack([_index.reconstruct(int(p)) for p in positions]).astype("float32")
            sub_index = faiss.IndexFlatIP(sub_vecs.shape[1])
            sub_index.add(sub_vecs)
            sub_nct_ids = np.array([_nct_ids[p] for p in positions])
            search_index   = sub_index
            search_nct_ids = sub_nct_ids
            print(
                f"[retrieval] structured filter: {len(_nct_ids):,} → {len(filtered_ncts):,} "
                f"trials in {sf_elapsed:.3f}s"
            )
        else:
            print(
                f"[retrieval] structured filter returned {len(filtered_ncts)} candidates "
                f"(< {MIN_FILTER_SIZE} threshold) — falling back to full corpus"
            )

    # --- Stage 2: bi-encoder FAISS retrieval ---
    actual_candidates = min(faiss_candidates, len(search_nct_ids))
    t0 = time.time()
    q_emb = _bi_encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    bi_scores, bi_indices = search_index.search(q_emb, actual_candidates)
    bi_elapsed = time.time() - t0
    print(
        f"[retrieval] FAISS top-{actual_candidates} retrieved in {bi_elapsed:.3f}s "
        f"(searched {len(search_nct_ids):,} vectors)"
    )

    # Map sub-index positions → (nct_id, bi_score)
    candidates = [
        (str(search_nct_ids[i]), float(bi_scores[0][rank]))
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
