"""
Model comparison eval: llama3 vs mistral on the 10-query oncology benchmark.

Runs the full three-stage pipeline (parse → retrieve → explain) for each model
and compares on: P@5, MRR, average per-trial LLM latency, ELIGIBLE rate.

Monkey-patching strategy
------------------------
Retrieval (PubMedBERT bi-encoder + MS-MARCO cross-encoder) is model-agnostic
and loads once at module import — swapping LLM does not affect it.
We patch three module-level variables at runtime before each model's run:
  - src.config.MODEL_NAME
  - src.agents.parser_agent.DEFAULT_MODEL
  - src.agents.explanation_agent.MODEL
Both agents read these at call time (not at import time), so the patch takes
effect for all subsequent Ollama calls without reloading the module.

Usage
-----
  PYTHONPATH=. python src/evaluation/eval_model_comparison.py

Output
------
  Prints comparison table to stdout.
  Saves results/model_comparison.csv.

Note: each full run is ~200-300s (10 queries × ~20s per trial × 10 trials).
Total wall time for both models: ~400-600s. Run with Ollama serving.
"""

import csv
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Eval set — same 10 queries and 14 labeled NCT IDs as eval_three_configs.py
# ---------------------------------------------------------------------------

EVAL_SET = [
    {
        "query": "EGFR mutant NSCLC, erlotinib failed, ECOG 1",
        "relevant": ["NCT05037331", "NCT05020275"],
        "patient_text": (
            "55-year-old patient with EGFR-mutant non-small cell lung cancer, "
            "erlotinib treatment failed, ECOG performance status 1"
        ),
    },
    {
        "query": "HER2 positive breast cancer, trastuzumab and pertuzumab prior, ECOG 0",
        "relevant": ["NCT05319873", "NCT05113251"],
        "patient_text": (
            "50-year-old female with HER2-positive metastatic breast cancer, "
            "prior trastuzumab and pertuzumab therapy, ECOG 0"
        ),
    },
    {
        "query": "Metastatic colorectal cancer, KRAS wild-type, bevacizumab naive",
        "relevant": ["NCT03874026"],
        "patient_text": (
            "58-year-old with metastatic colorectal cancer, KRAS wild-type, "
            "bevacizumab naive, 2 prior lines of chemotherapy"
        ),
    },
    {
        "query": "Relapsed diffuse large B-cell lymphoma, 2 prior lines, CAR-T eligible",
        "relevant": ["NCT07473167"],
        "patient_text": (
            "52-year-old with relapsed diffuse large B-cell lymphoma, "
            "2 prior lines of therapy, eligible for CAR-T cell therapy"
        ),
    },
    {
        "query": "Metastatic melanoma, PD-1 refractory, BRAF V600E positive",
        "relevant": ["NCT04439292"],
        "patient_text": (
            "48-year-old with metastatic melanoma, BRAF V600E mutation positive, "
            "refractory to PD-1 checkpoint inhibitor"
        ),
    },
    {
        "query": "Pancreatic adenocarcinoma, locally advanced unresectable, gemcitabine naive",
        "relevant": ["NCT06998940", "NCT06333314"],
        "patient_text": (
            "62-year-old with locally advanced unresectable pancreatic adenocarcinoma, "
            "gemcitabine naive, no prior systemic therapy"
        ),
    },
    {
        "query": "Prostate mCRPC, enzalutamide resistant, no prior taxane",
        "relevant": ["NCT04471974"],
        "patient_text": (
            "65-year-old male with metastatic castration-resistant prostate cancer, "
            "enzalutamide resistant, no prior taxane chemotherapy"
        ),
    },
    {
        "query": "AML newly diagnosed, FLT3-ITD positive, age 60+, not fit for intensive chemo",
        "relevant": ["NCT03013998", "NCT05520567"],
        "patient_text": (
            "67-year-old with newly diagnosed acute myeloid leukemia, FLT3-ITD mutation, "
            "not fit for intensive chemotherapy"
        ),
    },
    {
        "query": "Ovarian cancer platinum resistant, BRCA1 mutant, 3 prior lines",
        "relevant": ["NCT06856499"],
        "patient_text": (
            "55-year-old female with platinum-resistant ovarian cancer, BRCA1 mutation, "
            "3 prior lines of therapy including carboplatin"
        ),
    },
    {
        "query": "Hepatocellular carcinoma, Child-Pugh A, sorafenib failed",
        "relevant": ["NCT02115373"],
        "patient_text": (
            "60-year-old with hepatocellular carcinoma, Child-Pugh class A liver function, "
            "sorafenib treatment failed"
        ),
    },
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(returned_ncts: list[str], relevant: list[str], k: int = 5) -> float:
    rel_set = set(relevant)
    return sum(1 for n in returned_ncts[:k] if n in rel_set) / k


def mrr_at_k(returned_ncts: list[str], relevant: list[str], k: int = 10) -> float:
    rel_set = set(relevant)
    for i, n in enumerate(returned_ncts[:k], 1):
        if n in rel_set:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------

def _set_model(model_name: str) -> None:
    """
    Patch MODEL_NAME into config and both agent modules.
    Must be called before pipeline.invoke() — agents read these at call time.
    """
    import src.config as cfg
    import src.agents.parser_agent as pa
    import src.agents.explanation_agent as ea

    cfg.MODEL_NAME = model_name
    pa.DEFAULT_MODEL = model_name
    ea.MODEL = model_name


# ---------------------------------------------------------------------------
# Single-model run
# ---------------------------------------------------------------------------

def run_one_model(model_name: str) -> dict[str, Any]:
    """
    Run all 10 eval queries through the full pipeline with the given model.

    Returns a summary dict:
      model, p5, mrr, avg_response_time_s, eligible_rate,
      per_query (list of per-query result dicts)
    """
    # Import pipeline here so retrieval models load once regardless of model_name
    from src.orchestration.graph import pipeline

    _set_model(model_name)

    per_query = []
    total_llm_time = 0.0
    total_trial_count = 0

    for item in EVAL_SET:
        q   = item["query"]
        rel = item["relevant"]

        initial_state = {
            "raw_input": item["patient_text"],
            "patient_profile": {},
            "retrieved_trials": [],
            "explained_matches": [],
            "error": None,
        }

        t0 = time.time()
        try:
            result = pipeline.invoke(initial_state)
        except Exception as exc:
            per_query.append({
                "query": q,
                "relevant": rel,
                "error": str(exc),
                "returned_ncts": [],
                "p5": 0.0,
                "mrr": 0.0,
                "eligible_count": 0,
                "trial_count": 0,
                "elapsed_s": time.time() - t0,
            })
            continue

        elapsed = time.time() - t0

        if result.get("error"):
            per_query.append({
                "query": q,
                "relevant": rel,
                "error": result["error"],
                "returned_ncts": [],
                "p5": 0.0,
                "mrr": 0.0,
                "eligible_count": 0,
                "trial_count": 0,
                "elapsed_s": elapsed,
            })
            continue

        matches = result.get("explained_matches") or []
        returned_ncts = [m.get("nct_id", "") for m in matches]
        eligible_count = sum(1 for m in matches if m.get("verdict") == "ELIGIBLE")

        total_llm_time += elapsed
        total_trial_count += len(matches)

        per_query.append({
            "query": q,
            "relevant": rel,
            "error": None,
            "returned_ncts": returned_ncts,
            "p5": precision_at_k(returned_ncts, rel, k=5),
            "mrr": mrr_at_k(returned_ncts, rel, k=10),
            "eligible_count": eligible_count,
            "trial_count": len(matches),
            "elapsed_s": elapsed,
        })

    # Aggregate
    valid = [r for r in per_query if r["error"] is None]
    mean_p5  = sum(r["p5"]  for r in valid) / len(valid) if valid else 0.0
    mean_mrr = sum(r["mrr"] for r in valid) / len(valid) if valid else 0.0

    total_eligible = sum(r["eligible_count"] for r in valid)
    total_trials   = sum(r["trial_count"]    for r in valid)
    eligible_rate  = total_eligible / total_trials if total_trials else 0.0

    # Avg time per trial (LLM portion is ~total_llm_time / total_trial_count)
    avg_per_trial = total_llm_time / total_trial_count if total_trial_count else 0.0

    return {
        "model": model_name,
        "p5": mean_p5,
        "mrr": mean_mrr,
        "avg_response_time_s": avg_per_trial,
        "eligible_rate": eligible_rate,
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    models = ["llama3", "mistral"]
    summaries = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Running eval: model={model}")
        print(f"{'='*60}")
        summary = run_one_model(model)
        summaries.append(summary)
        print(f"  Done — P@5={summary['p5']:.3f}  MRR={summary['mrr']:.3f}  "
              f"eligible_rate={summary['eligible_rate']:.2f}  "
              f"avg_s/trial={summary['avg_response_time_s']:.1f}s")

    # ── comparison table ──────────────────────────────────────────────────

    HDR   = f"{'Model':<10} | {'P@5':>6} | {'MRR':>6} | {'Avg s/trial':>12} | {'ELIGIBLE rate':>14}"
    SEP   = "-" * len(HDR)

    print(f"\n{SEP}")
    print(HDR)
    print(SEP)
    for s in summaries:
        print(f"{s['model']:<10} | {s['p5']:>6.3f} | {s['mrr']:>6.3f} | "
              f"{s['avg_response_time_s']:>12.1f} | {s['eligible_rate']:>14.2f}")
    print(SEP)

    # ── per-query breakdown ───────────────────────────────────────────────

    print("\nPer-query P@5 breakdown:")
    q_hdr = f"{'Query (35 chars)':<37}"
    for s in summaries:
        q_hdr += f"  {s['model']:>8} P@5"
    print("-" * len(q_hdr))
    print(q_hdr)
    print("-" * len(q_hdr))

    for i in range(len(EVAL_SET)):
        row = f"{EVAL_SET[i]['query'][:35]:<37}"
        for s in summaries:
            pq = s["per_query"][i]
            err = "ERROR" if pq["error"] else f"{pq['p5']:.3f}"
            row += f"  {err:>12}"
        print(row)

    # ── save CSV ──────────────────────────────────────────────────────────

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "model_comparison.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "p5", "mrr", "avg_response_time_s", "eligible_rate"],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "model": s["model"],
                "p5": round(s["p5"], 4),
                "mrr": round(s["mrr"], 4),
                "avg_response_time_s": round(s["avg_response_time_s"], 2),
                "eligible_rate": round(s["eligible_rate"], 4),
            })

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
