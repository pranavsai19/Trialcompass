"""
Build, save, and load a FAISS flat L2 index over trial embeddings.

Index choice: IndexFlatIP (inner product) over L2-normalized vectors
= cosine similarity, no approximation, exact search. Appropriate for
10K–60K vectors — no need for IVF/HNSW at this scale.
"""

import logging
from pathlib import Path

import faiss
import numpy as np

from src.embeddings.embed import load_chunks, embed_chunks

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss_index.bin"
NCTIDS_PATH = DATA_DIR / "nct_ids.npy"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS flat inner product index from L2-normalized embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"Index built: {index.ntotal:,} vectors, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatIP, nct_ids: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(NCTIDS_PATH), np.array(nct_ids))
    log.info(f"Saved index to {INDEX_PATH}, NCT IDs to {NCTIDS_PATH}")


def load_index() -> tuple[faiss.IndexFlatIP, np.ndarray]:
    index = faiss.read_index(str(INDEX_PATH))
    nct_ids = np.load(str(NCTIDS_PATH), allow_pickle=True)
    log.info(f"Loaded index: {index.ntotal:,} vectors")
    return index, nct_ids


def search(
    query: str,
    index: faiss.IndexFlatIP,
    nct_ids: np.ndarray,
    model_name: str = "BAAI/bge-large-en-v1.5",
    top_k: int = 50,
) -> list[tuple[str, float]]:
    """
    Embed a query string and return top-k (nct_id, score) pairs.
    Scores are cosine similarities (0–1, higher = more similar).
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)
    return [(nct_ids[i], float(scores[0][rank])) for rank, i in enumerate(indices[0])]


if __name__ == "__main__":
    nct_ids, chunks = load_chunks()
    log.info(f"Loaded {len(chunks):,} chunks")

    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)
    save_index(index, nct_ids)

    # Sanity check
    results = search(
        "metastatic breast cancer BRCA2 mutation PARP inhibitor",
        index, np.array(nct_ids)
    )
    log.info("Top-5 results for BRCA2 breast cancer query:")
    for nct_id, score in results[:5]:
        log.info(f"  {nct_id}  score={score:.4f}")
