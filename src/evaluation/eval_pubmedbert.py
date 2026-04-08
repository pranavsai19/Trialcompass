"""
Canonical eval script for the PubMedBERT full-corpus index.

Index  : data/trials_pubmedbert.index  (FAISS IndexFlatIP, 64K+ trials)
NCT IDs: data/nct_ids_pubmedbert.npy
Model  : neuml/pubmedbert-base-embeddings (via sentence-transformers)

Metrics: P@5 and MRR over a 10-query oncology eval set with manually
         verified ground-truth labels.

Usage:
    python src/evaluation/eval_pubmedbert.py
"""

import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Eval set — manually verified against ClinicalTrials.gov eligibility text
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
        "query": "AML newly diagnosed, FLT3-ITD positive, age 60 or older, not fit for intensive chemo",
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

# ---------------------------------------------------------------------------
# Manual relevance audit — 5 of 10 queries inspected 2026-04-08
# Top-10 retrieved trials were read against ClinicalTrials.gov records.
# ---------------------------------------------------------------------------

MANUAL_RELEVANCE = {
    "EGFR mutant NSCLC, erlotinib failed, ECOG 1": (
        "Top-10 clinically coherent: all EGFR NSCLC TKI-resistance trials. "
        "Bi-encoder correctly scopes search space."
    ),
    "HER2 positive breast cancer, trastuzumab and pertuzumab prior, ECOG 0": (
        "Top-10 clinically coherent: rank 5 is pertuzumab retreatment in "
        "previously pertuzumab-treated HER2+ breast — near-exact query match."
    ),
    "Metastatic colorectal cancer, KRAS wild-type, bevacizumab naive": (
        "Top-10 clinically coherent: every trial is KRAS-wt mCRC anti-EGFR. "
        "Rank 2 contains bevacizumab + KRAS status terms explicitly."
    ),
    "Prostate mCRPC, enzalutamide resistant, no prior taxane": (
        "Top-10 clinically coherent: labeled trial NCT04471974 found at "
        "rank 28 among 28 clinically similar trials. Model understands query."
    ),
    "Hepatocellular carcinoma, Child-Pugh A, sorafenib failed": (
        "Mixed: HCC + sorafenib found but Child-Pugh A/B not "
        "distinguishable from text. Weakest query — structured field gap."
    ),
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INDEX_PATH  = Path("data/trials_pubmedbert.index")
NCTIDS_PATH = Path("data/nct_ids_pubmedbert.npy")
DB_PATH     = Path("data/trialcompass.db")
MODEL_NAME  = "neuml/pubmedbert-base-embeddings"
TOP_K       = 50
DISPLAY_K   = 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved, relevant_set, k=5):
    return sum(1 for r in retrieved[:k] if r in relevant_set) / k


def mrr(retrieved, relevant_set):
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load index
    print(f"Loading FAISS index from {INDEX_PATH} ...")
    index = faiss.read_index(str(INDEX_PATH))
    nct_ids = np.load(str(NCTIDS_PATH), allow_pickle=True)
    print(f"Index size: {index.ntotal:,} vectors | dim: {index.d}")

    # Load title lookup from DB
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT nct_id, brief_title FROM trials")
    title_map = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()

    # Load model
    print(f"Loading model: {MODEL_NAME}\n")
    model = SentenceTransformer(MODEL_NAME)

    # Encode all queries
    queries = [ex["query"] for ex in EVAL_SET]
    q_embs = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Run eval
    SEP = "=" * 80
    p5_list, mrr_list = [], []

    for i, (ex, q_emb) in enumerate(zip(EVAL_SET, q_embs)):
        rel_set = set(ex["relevant"])
        _, I = index.search(q_emb.reshape(1, -1), TOP_K)
        retrieved = [str(nct_ids[j]) for j in I[0]]

        p5  = precision_at_k(retrieved, rel_set)
        mrr_ = mrr(retrieved, rel_set)
        p5_list.append(p5)
        mrr_list.append(mrr_)

        hit_rank = next(
            (rank for rank, r in enumerate(retrieved, 1) if r in rel_set), None
        )

        print(SEP)
        print(f"Q{i+1:02d}: {ex['query']}")
        print(f"     Relevant : {', '.join(ex['relevant'])}")
        print(f"     P@5={p5:.3f}  MRR={mrr_:.3f}  "
              f"First hit rank: {hit_rank if hit_rank else 'none (>50)'}")
        print(f"     Top-{DISPLAY_K} retrieved:")
        for rank, nct in enumerate(retrieved[:DISPLAY_K], 1):
            marker = "<<< HIT" if nct in rel_set else ""
            title  = title_map.get(nct, "(title not in DB)")[:72]
            print(f"       [{rank}] {nct}  {title}  {marker}")

    # Summary table
    print("\n" + SEP)
    print(f"{'SUMMARY':^80}")
    print(SEP)
    hdr = f"{'Query':<62} {'P@5':>6} {'MRR':>8}"
    print(hdr)
    print("-" * 80)
    for ex, p5, mrr_ in zip(EVAL_SET, p5_list, mrr_list):
        sq = ex["query"][:60]
        print(f"{sq:<62} {p5:>6.3f} {mrr_:>8.3f}")
    print("-" * 80)
    print(f"{'MEAN':<62} {np.mean(p5_list):>6.3f} {np.mean(mrr_list):>8.3f}")
    print(SEP)


def print_manual_audit():
    SEP = "=" * 80
    print(f"\n{SEP}")
    print(f"{'MANUAL RELEVANCE AUDIT (5 of 10 queries)':^80}")
    print(SEP)
    for query, verdict in MANUAL_RELEVANCE.items():
        print(f"\nQuery  : {query}")
        print(f"Verdict: {verdict}")
    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
    print_manual_audit()
