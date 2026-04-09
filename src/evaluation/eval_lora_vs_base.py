"""
LoRA cross-encoder vs. base cross-encoder comparison — Config C benchmark.

Runs the canonical 10-query hybrid retrieval benchmark (Config C: structured
pre-filter + PubMedBERT bi-encoder k=500 + CE rerank) twice:
  1. Base model : cross-encoder/ms-marco-MiniLM-L-6-v2
  2. LoRA model : models/crossencoder_lora_v1  (fine-tuned by finetune_crossencoder.py)

Injection approach: ClinicalReranker stores its CrossEncoder in self._model.
We instantiate a second reranker and replace ._model with the LoRA CrossEncoder
loaded from the saved path — no changes to the reranker class needed.

Prerequisites:
  - Run build_training_pairs.py first:
      PYTHONPATH=. python src/training/build_training_pairs.py
  - Run finetune_crossencoder.py first:
      PYTHONPATH=. python src/training/finetune_crossencoder.py
  - models/crossencoder_lora_v1/ must exist

Output:
  - results/eval_lora_comparison.csv
  - results/eval_lora_comparison.json  (per-query detail)

Run:
  PYTHONPATH=. python src/evaluation/eval_lora_vs_base.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.retrieval.reranker import ClinicalReranker
from src.retrieval.structured_filter import StructuredFilter

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

INDEX_PATH      = Path("data/trials_pubmedbert.index")
NCTIDS_PATH     = Path("data/nct_ids_pubmedbert.npy")
DB_PATH         = Path("data/trialcompass.db")
LORA_MODEL_DIR  = Path("models/crossencoder_lora_v1")
RESULTS_CSV     = Path("results/eval_lora_comparison.csv")
RESULTS_JSON    = Path("results/eval_lora_comparison.json")

BI_MODEL = "neuml/pubmedbert-base-embeddings"
BASE_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

K_C         = 500   # bi-encoder candidates on filtered sub-corpus
TOP_P       = 5     # P@5
TOP_MRR     = 10    # MRR@10
MIN_FILTER  = 100   # fallback to full corpus if filter too restrictive

# ---------------------------------------------------------------------------
# Canonical eval set — identical to eval_three_configs.py
# ---------------------------------------------------------------------------

EVAL_SET = [
    {
        "query": "EGFR mutant NSCLC, erlotinib failed, ECOG 1",
        "relevant": ["NCT05037331", "NCT05020275"],
        "filter": {"age": 55, "conditions_keywords": ["lung", "cancer"]},
    },
    {
        "query": "HER2 positive breast cancer, trastuzumab and pertuzumab prior, ECOG 0",
        "relevant": ["NCT05319873", "NCT05113251"],
        "filter": {"age": 50, "conditions_keywords": ["breast", "cancer"]},
    },
    {
        "query": "Metastatic colorectal cancer, KRAS wild-type, bevacizumab naive",
        "relevant": ["NCT03874026"],
        "filter": {"age": 58, "conditions_keywords": ["colorectal", "colon", "cancer"]},
    },
    {
        "query": "Relapsed diffuse large B-cell lymphoma, 2 prior lines, CAR-T eligible",
        "relevant": ["NCT07473167"],
        "filter": {"age": 52, "conditions_keywords": ["lymphoma", "cancer"]},
    },
    {
        "query": "Metastatic melanoma, PD-1 refractory, BRAF V600E positive",
        "relevant": ["NCT04439292"],
        "filter": {"age": 48, "conditions_keywords": ["melanoma", "cancer"]},
    },
    {
        "query": "Pancreatic adenocarcinoma, locally advanced unresectable, gemcitabine naive",
        "relevant": ["NCT06998940", "NCT06333314"],
        "filter": {"age": 62, "conditions_keywords": ["pancreatic", "cancer"]},
    },
    {
        "query": "Prostate mCRPC, enzalutamide resistant, no prior taxane",
        "relevant": ["NCT04471974"],
        "filter": {"age": 65, "conditions_keywords": ["prostate", "cancer"]},
    },
    {
        "query": "AML newly diagnosed, FLT3-ITD positive, age 60+, not fit for intensive chemo",
        "relevant": ["NCT03013998", "NCT05520567"],
        "filter": {"age": 67, "conditions_keywords": ["leukemia", "AML", "cancer"]},
    },
    {
        "query": "Ovarian cancer platinum resistant, BRCA1 mutant, 3 prior lines",
        "relevant": ["NCT06856499"],
        "filter": {"age": 55, "conditions_keywords": ["ovarian", "cancer"]},
    },
    {
        "query": "Hepatocellular carcinoma, Child-Pugh A, sorafenib failed",
        "relevant": ["NCT02115373"],
        "filter": {"age": 60, "conditions_keywords": ["hepatocellular", "liver", "cancer"]},
        # NCT02115373 is COMPLETED — correctly excluded by structured filter.
        # Label kept to surface the coverage gap (same as eval_three_configs.py).
    },
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for n in ranked[:k] if n in relevant) / k


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, n in enumerate(ranked[:k], 1):
        if n in relevant:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _embed(bi: SentenceTransformer, query: str) -> np.ndarray:
    return bi.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")


def run_config_c_with_reranker(
    query: str,
    filter_params: dict,
    index: faiss.IndexFlatIP,
    nct_ids: np.ndarray,
    bi: SentenceTransformer,
    reranker: ClinicalReranker,
    sf: StructuredFilter,
) -> tuple[list[str], int]:
    """
    Config C: structured filter → bi-encoder k=500 → CE rerank.

    The reranker argument is either a base or LoRA-injected ClinicalReranker.
    Returns (ranked_ncts, n_filtered).
    """
    filtered = sf.filter(
        age=filter_params.get("age"),
        status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
        conditions_keywords=filter_params.get("conditions_keywords"),
    )
    n_filtered = len(filtered)

    if n_filtered >= MIN_FILTER:
        positions = sf.filter_to_index_positions(filtered, nct_ids)
        sub_vecs  = np.vstack(
            [index.reconstruct(int(p)) for p in positions]
        ).astype("float32")
        sub_index = faiss.IndexFlatIP(sub_vecs.shape[1])
        sub_index.add(sub_vecs)
        sub_ncts  = np.array([nct_ids[p] for p in positions])
        actual_k  = min(K_C, sub_index.ntotal)
        q_emb     = _embed(bi, query)
        _, idxs   = sub_index.search(q_emb, actual_k)
        candidates = [str(sub_ncts[i]) for i in idxs[0]]
    else:
        q_emb = _embed(bi, query)
        _, idxs = index.search(q_emb, K_C)
        candidates = [nct_ids[i] for i in idxs[0]]

    # reranker.rerank returns list[dict] sorted by ce_score; extract nct_ids
    # ClinicalReranker.rerank needs (query, rows) where rows have nct_id key.
    # Build minimal row dicts — reranker only needs nct_id and chunk_text.
    import sqlite3
    con = sqlite3.connect(str(DB_PATH))
    ph  = ",".join(["?"] * len(candidates))
    rows_db = con.execute(
        f"SELECT nct_id, chunk_text, eligibility_text FROM trials WHERE nct_id IN ({ph})",
        candidates,
    ).fetchall()
    con.close()

    text_map = {r[0]: r[1] or r[2] or "" for r in rows_db}
    rows = [
        {"nct_id": nct, "chunk_text": text_map.get(nct, "")}
        for nct in candidates
    ]

    reranked = reranker.rerank(query, rows)
    return [r["nct_id"] for r in reranked], n_filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Check LoRA model exists ───────────────────────────────────────────
    if not LORA_MODEL_DIR.exists():
        print(f"ERROR: {LORA_MODEL_DIR} not found.")
        print("Run: PYTHONPATH=. python src/training/finetune_crossencoder.py")
        sys.exit(1)

    # ── Load shared resources ─────────────────────────────────────────────
    print("Loading FAISS index and models...", flush=True)
    index   = faiss.read_index(str(INDEX_PATH))
    nct_ids = np.load(str(NCTIDS_PATH), allow_pickle=True)
    bi      = SentenceTransformer(BI_MODEL)
    sf      = StructuredFilter(DB_PATH)
    print(f"  Index: {index.ntotal:,} vectors\n", flush=True)

    # ── Build base reranker ───────────────────────────────────────────────
    print(f"Loading base CE: {BASE_CE_MODEL}", flush=True)
    base_reranker = ClinicalReranker(model_name=BASE_CE_MODEL)

    # ── Build LoRA reranker (inject saved model into reranker._model) ─────
    print(f"Loading LoRA CE from: {LORA_MODEL_DIR}", flush=True)
    lora_ce = CrossEncoder(str(LORA_MODEL_DIR), num_labels=1, max_length=512)
    lora_reranker = ClinicalReranker(model_name=BASE_CE_MODEL)  # init for other attrs
    lora_reranker._model = lora_ce  # swap in the fine-tuned model
    print("  LoRA model injected into reranker._model\n", flush=True)

    # ── Run eval ──────────────────────────────────────────────────────────
    results = []
    for i, item in enumerate(EVAL_SET, 1):
        q   = item["query"]
        rel = set(item["relevant"])
        fp  = item["filter"]
        print(f"[{i:02d}/10] {q[:65]}", flush=True)

        base_ranked, n_filt = run_config_c_with_reranker(
            q, fp, index, nct_ids, bi, base_reranker, sf
        )
        lora_ranked, _      = run_config_c_with_reranker(
            q, fp, index, nct_ids, bi, lora_reranker, sf
        )

        base_p5  = precision_at_k(base_ranked, rel, TOP_P)
        base_mrr = mrr_at_k(base_ranked, rel, TOP_MRR)
        lora_p5  = precision_at_k(lora_ranked, rel, TOP_P)
        lora_mrr = mrr_at_k(lora_ranked, rel, TOP_MRR)

        results.append({
            "query":     q,
            "relevant":  item["relevant"],
            "n_filtered": n_filt,
            "base_p5":   base_p5,
            "base_mrr":  base_mrr,
            "lora_p5":   lora_p5,
            "lora_mrr":  lora_mrr,
            "delta_p5":  round(lora_p5 - base_p5, 4),
            "delta_mrr": round(lora_mrr - base_mrr, 4),
            "base_top5": base_ranked[:5],
            "lora_top5": lora_ranked[:5],
        })

        print(f"         Base P@5={base_p5:.3f} MRR={base_mrr:.3f} | "
              f"LoRA P@5={lora_p5:.3f} MRR={lora_mrr:.3f} | "
              f"ΔP@5={lora_p5-base_p5:+.3f} ΔMRR={lora_mrr-base_mrr:+.3f}",
              flush=True)

    # ── Summary table ─────────────────────────────────────────────────────
    mean_base_p5  = sum(r["base_p5"]  for r in results) / len(results)
    mean_base_mrr = sum(r["base_mrr"] for r in results) / len(results)
    mean_lora_p5  = sum(r["lora_p5"]  for r in results) / len(results)
    mean_lora_mrr = sum(r["lora_mrr"] for r in results) / len(results)

    HDR = "{:<37} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7}".format(
        "Query (first 35 chars)",
        "B-P@5", "B-MRR",
        "L-P@5", "L-MRR",
        "ΔP@5", "ΔMRR",
    )
    SEP = "-" * len(HDR)

    print("\n" + SEP)
    print(HDR)
    print(SEP)
    for r in results:
        print("{:<37} | {:>7.3f} {:>7.3f} | {:>7.3f} {:>7.3f} | {:>+7.3f} {:>+7.3f}".format(
            r["query"][:35],
            r["base_p5"], r["base_mrr"],
            r["lora_p5"], r["lora_mrr"],
            r["delta_p5"], r["delta_mrr"],
        ))
    print(SEP)
    print("{:<37} | {:>7.3f} {:>7.3f} | {:>7.3f} {:>7.3f} | {:>+7.3f} {:>+7.3f}".format(
        "MEAN",
        mean_base_p5, mean_base_mrr,
        mean_lora_p5, mean_lora_mrr,
        mean_lora_p5 - mean_base_p5,
        mean_lora_mrr - mean_base_mrr,
    ))
    print(SEP)

    print(f"\n  LoRA delta: P@5 {mean_lora_p5 - mean_base_p5:+.3f}   "
          f"MRR {mean_lora_mrr - mean_base_mrr:+.3f}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "n_filtered",
            "base_p5", "base_mrr", "lora_p5", "lora_mrr",
            "delta_p5", "delta_mrr",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

        # Append MEAN row
        writer.writerow({
            "query":      "MEAN",
            "n_filtered": "",
            "base_p5":    round(mean_base_p5,  4),
            "base_mrr":   round(mean_base_mrr, 4),
            "lora_p5":    round(mean_lora_p5,  4),
            "lora_mrr":   round(mean_lora_mrr, 4),
            "delta_p5":   round(mean_lora_p5  - mean_base_p5,  4),
            "delta_mrr":  round(mean_lora_mrr - mean_base_mrr, 4),
        })

    print(f"\nCSV saved to {RESULTS_CSV}")

    # ── Save JSON (per-query detail with top-5 lists) ─────────────────────
    summary = {
        "base_model":  BASE_CE_MODEL,
        "lora_model":  str(LORA_MODEL_DIR),
        "mean_base_p5":  round(mean_base_p5,  4),
        "mean_base_mrr": round(mean_base_mrr, 4),
        "mean_lora_p5":  round(mean_lora_p5,  4),
        "mean_lora_mrr": round(mean_lora_mrr, 4),
        "delta_p5":      round(mean_lora_p5  - mean_base_p5,  4),
        "delta_mrr":     round(mean_lora_mrr - mean_base_mrr, 4),
        "per_query":     results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON saved to {RESULTS_JSON}")

    print("\n" + "=" * 60)
    print("LORA vs BASE COMPARISON COMPLETE")
    print(f"  Base  : P@5={mean_base_p5:.3f}  MRR={mean_base_mrr:.3f}")
    print(f"  LoRA  : P@5={mean_lora_p5:.3f}  MRR={mean_lora_mrr:.3f}")
    print(f"  Delta : P@5={mean_lora_p5-mean_base_p5:+.3f}  MRR={mean_lora_mrr-mean_base_mrr:+.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
