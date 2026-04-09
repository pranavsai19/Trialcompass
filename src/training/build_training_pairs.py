"""
Build training pairs for LoRA cross-encoder fine-tuning.

Strategy: hard-negative mining from FAISS.
  - POSITIVE pairs: (patient_query, chunk_text) for each labeled NCT ID
  - NEGATIVE pairs: top FAISS retrievals that are NOT labeled positives
    These are the trickiest wrong answers — hard negatives make the
    fine-tuning clinically meaningful, not just random noise.

Output: data/training_pairs.json
  [{"query": str, "passage": str, "label": 0|1, "nct_id": str}, ...]

Run:
  PYTHONPATH=. python src/training/build_training_pairs.py
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Eval set — hardcoded (identical to eval_three_configs.py EVAL_SET)
# 10 queries, 14 labeled NCT IDs
# ---------------------------------------------------------------------------

EVAL_SET = [
    {
        "query": "EGFR mutant NSCLC, erlotinib failed, ECOG 1",
        "relevant": ["NCT05037331", "NCT05020275"],
    },
    {
        "query": "HER2 positive breast cancer, trastuzumab and pertuzumab prior, ECOG 0",
        "relevant": ["NCT05319873", "NCT05113251"],
    },
    {
        "query": "Metastatic colorectal cancer, KRAS wild-type, bevacizumab naive",
        "relevant": ["NCT03874026"],
    },
    {
        "query": "Relapsed diffuse large B-cell lymphoma, 2 prior lines, CAR-T eligible",
        "relevant": ["NCT07473167"],
    },
    {
        "query": "Metastatic melanoma, PD-1 refractory, BRAF V600E positive",
        "relevant": ["NCT04439292"],
    },
    {
        "query": "Pancreatic adenocarcinoma, locally advanced unresectable, gemcitabine naive",
        "relevant": ["NCT06998940", "NCT06333314"],
    },
    {
        "query": "Prostate mCRPC, enzalutamide resistant, no prior taxane",
        "relevant": ["NCT04471974"],
    },
    {
        "query": "AML newly diagnosed, FLT3-ITD positive, age 60+, not fit for intensive chemo",
        "relevant": ["NCT03013998", "NCT05520567"],
    },
    {
        "query": "Ovarian cancer platinum resistant, BRCA1 mutant, 3 prior lines",
        "relevant": ["NCT06856499"],
    },
    {
        "query": "Hepatocellular carcinoma, Child-Pugh A, sorafenib failed",
        "relevant": ["NCT02115373"],
    },
]

INDEX_PATH  = Path("data/trials_pubmedbert.index")
NCTIDS_PATH = Path("data/nct_ids_pubmedbert.npy")
DB_PATH     = Path("data/trialcompass.db")
OUTPUT_PATH = Path("data/training_pairs.json")

BI_MODEL         = "neuml/pubmedbert-base-embeddings"
HARD_NEG_POOL    = 20   # retrieve top-20 from FAISS per query
HARD_NEG_PER_Q   = 5    # keep top-5 as hard negatives after removing positives


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_chunk_texts(nct_ids: list[str]) -> dict[str, str]:
    con = sqlite3.connect(str(DB_PATH))
    ph  = ",".join(["?"] * len(nct_ids))
    rows = con.execute(
        f"SELECT nct_id, chunk_text, eligibility_text FROM trials "
        f"WHERE nct_id IN ({ph})",
        nct_ids,
    ).fetchall()
    con.close()
    # Prefer chunk_text; fall back to eligibility_text if chunk_text empty
    result = {}
    for nct_id, chunk_text, elig_text in rows:
        result[nct_id] = chunk_text or elig_text or ""
    return result


def get_hard_negatives(
    query: str,
    positive_ncts: set[str],
    index: faiss.IndexFlatIP,
    nct_list: list[str],
    bi: SentenceTransformer,
    k: int = HARD_NEG_POOL,
    n_neg: int = HARD_NEG_PER_Q,
) -> list[str]:
    """Return top-k FAISS results that are not labeled positives."""
    q_emb = bi.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, idxs = index.search(q_emb, k)
    candidates = [nct_list[i] for i in idxs[0]]
    negatives = [nct for nct in candidates if nct not in positive_ncts]
    return negatives[:n_neg]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading FAISS index and bi-encoder...", flush=True)
    index    = faiss.read_index(str(INDEX_PATH))
    nct_ids  = np.load(str(NCTIDS_PATH), allow_pickle=True)
    nct_list = nct_ids.tolist()
    bi       = SentenceTransformer(BI_MODEL)
    print(f"  Index: {index.ntotal:,} vectors\n", flush=True)

    pairs: list[dict] = []
    n_pos = 0
    n_neg = 0

    for item in EVAL_SET:
        query        = item["query"]
        positive_set = set(item["relevant"])

        print(f"Query: {query[:60]}", flush=True)

        # ── Positive pairs ────────────────────────────────────────────────
        pos_texts = fetch_chunk_texts(list(positive_set))
        for nct_id in item["relevant"]:
            text = pos_texts.get(nct_id, "")
            if not text:
                print(f"  WARN: no chunk_text for positive {nct_id} — skipping")
                continue
            pairs.append({
                "query":   query,
                "passage": text,
                "label":   1,
                "nct_id":  nct_id,
            })
            n_pos += 1
            print(f"  + POSITIVE: {nct_id} ({len(text)} chars)")

        # ── Hard negative pairs ───────────────────────────────────────────
        neg_ncts  = get_hard_negatives(query, positive_set, index, nct_list, bi)
        neg_texts = fetch_chunk_texts(neg_ncts)
        for nct_id in neg_ncts:
            text = neg_texts.get(nct_id, "")
            if not text:
                print(f"  WARN: no chunk_text for negative {nct_id} — skipping")
                continue
            pairs.append({
                "query":   query,
                "passage": text,
                "label":   0,
                "nct_id":  nct_id,
            })
            n_neg += 1
            print(f"  - NEGATIVE: {nct_id} ({len(text)} chars)")

        print()

    # ── Save ─────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    print("=" * 60)
    print(f"Training pairs saved to {OUTPUT_PATH}")
    print(f"  Total pairs : {len(pairs)}")
    print(f"  Positives   : {n_pos}")
    print(f"  Negatives   : {n_neg}")
    print(f"  Pos/Neg ratio: 1:{n_neg/n_pos:.1f}" if n_pos else "  (no positives)")


if __name__ == "__main__":
    main()
