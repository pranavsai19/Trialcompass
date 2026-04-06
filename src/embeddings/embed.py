"""
Embed trial chunk_text fields from SQLite using sentence-transformers.

Truncation note (documented in notebooks/02_embedding_experiments.ipynb):
all-MiniLM-L6-v2 max seq length = 256 tokens. 89% of trial chunks exceed
~1000 chars. The bi-encoder sees title + phase + conditions + first ~100 words
of eligibility. This is the primary motivation for cross-encoder reranking.
"""

import logging
import time
from pathlib import Path

import numpy as np
import sqlite_utils
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data/trialcompass.db")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_chunks(db_path: Path = DB_PATH) -> tuple[list[str], list[str]]:
    """Return (nct_ids, chunk_texts) for all trials in the DB."""
    db = sqlite_utils.Database(db_path)
    rows = list(db["trials"].rows)
    nct_ids = [r["nct_id"] for r in rows]
    chunks = [r["chunk_text"] for r in rows]
    return nct_ids, chunks


def embed_chunks(
    chunks: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Embed a list of text chunks. Returns float32 numpy array of shape (N, dim).
    """
    model = SentenceTransformer(model_name)
    log.info(f"Model: {model_name} | max_seq_length={model.max_seq_length} | dim={model.get_sentence_embedding_dimension()}")

    t0 = time.time()
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity via dot product
    )
    elapsed = time.time() - t0
    log.info(f"Embedded {len(chunks):,} chunks in {elapsed:.1f}s ({len(chunks)/elapsed:.0f} records/sec)")
    return embeddings.astype("float32")


if __name__ == "__main__":
    nct_ids, chunks = load_chunks()
    log.info(f"Loaded {len(chunks):,} chunks from DB")
    embeddings = embed_chunks(chunks)
    log.info(f"Embeddings shape: {embeddings.shape}")
