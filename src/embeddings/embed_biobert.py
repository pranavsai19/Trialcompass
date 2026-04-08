"""
Embed trial chunk_text fields using dmis-lab/biobert-v1.1 with explicit mean pooling
of the last hidden state. Builds a new FAISS IndexFlatIP on L2-normalized vectors.

Why mean pooling instead of CLS:
BioBERT was pretrained with MLM — the CLS token is not trained for sentence-level
similarity. Mean pooling over the last hidden state (masked to exclude padding)
produces more stable sentence embeddings for biomedical retrieval.

Outputs:
  data/embeddings_biobert.npy    — float32 array (N, 768)
  data/nct_ids_biobert.npy       — NCT ID array aligned to rows above
  data/trials_biobert.index      — FAISS IndexFlatIP
"""

import logging
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME   = "dmis-lab/biobert-v1.1"
DB_PATH      = Path("data/trialcompass.db")
EMB_PATH     = Path("data/embeddings_biobert.npy")
NCTIDS_PATH  = Path("data/nct_ids_biobert.npy")
INDEX_PATH   = Path("data/trials_biobert.index")
MAX_LENGTH   = 512
BATCH_SIZE   = 32    # safe for M3 Pro MPS at 512 tokens, BioBERT-base (768-dim)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool the last hidden state, excluding padding tokens.

    last_hidden_state: (batch, seq_len, hidden_dim)
    attention_mask:    (batch, seq_len)  — 1 for real tokens, 0 for padding
    Returns:           (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).float()          # (batch, seq_len, 1)
    summed = (last_hidden_state * mask).sum(dim=1)       # (batch, hidden_dim)
    counts = mask.sum(dim=1).clamp(min=1e-9)             # (batch, 1)
    return summed / counts


def load_chunks(db_path: Path = DB_PATH) -> tuple[list[str], list[str]]:
    """Return (nct_ids, chunk_texts) for all rows in SQLite."""
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("SELECT nct_id, chunk_text FROM trials ORDER BY nct_id")
    rows = cur.fetchall()
    con.close()
    nct_ids = [r[0] for r in rows]
    chunks  = [r[1] or "" for r in rows]
    log.info(f"Loaded {len(chunks):,} chunks from {db_path}")
    return nct_ids, chunks


def embed_chunks(
    chunks: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    device = _get_device()
    log.info(f"Device: {device}")
    log.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    log.info(f"Model loaded. max_length={max_length} | hidden_dim=768")

    all_embeddings: list[np.ndarray] = []
    n_batches = (len(chunks) + batch_size - 1) // batch_size

    t0 = time.time()
    for i in range(n_batches):
        batch = chunks[i * batch_size : (i + 1) * batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        pooled = _mean_pool(outputs.last_hidden_state, attention_mask)

        # L2 normalize for cosine similarity via inner product
        norms = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        pooled = pooled / norms

        all_embeddings.append(pooled.cpu().float().numpy())

        if (i + 1) % 50 == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            done = (i + 1) * batch_size
            log.info(
                f"Batch {i+1}/{n_batches} | "
                f"{min(done, len(chunks)):,}/{len(chunks):,} chunks | "
                f"{elapsed:.0f}s elapsed"
            )

    elapsed = time.time() - t0
    embeddings = np.vstack(all_embeddings).astype("float32")
    log.info(f"Embedding complete: {embeddings.shape} in {elapsed:.1f}s "
             f"({len(chunks)/elapsed:.0f} records/sec)")
    return embeddings


def build_and_save_index(embeddings: np.ndarray, nct_ids: list[str]) -> None:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"FAISS index built: {index.ntotal:,} vectors, dim={dim}")

    faiss.write_index(index, str(INDEX_PATH))
    np.save(str(EMB_PATH), embeddings)
    np.save(str(NCTIDS_PATH), np.array(nct_ids))

    log.info(f"Saved index  → {INDEX_PATH}")
    log.info(f"Saved embeds → {EMB_PATH}")
    log.info(f"Saved IDs    → {NCTIDS_PATH}")


if __name__ == "__main__":
    nct_ids, chunks = load_chunks()
    embeddings = embed_chunks(chunks)
    build_and_save_index(embeddings, nct_ids)
