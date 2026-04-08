# TrialCompass

**Agentic clinical trial matching for oncology patients**

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![pytest](https://img.shields.io/badge/pytest-passing-brightgreen)

---

## Overview

TrialCompass takes a free-text patient description — cancer type, biomarkers, prior treatments, performance status — and returns a ranked, explained list of matching clinical trials from a 10,000-trial ClinicalTrials.gov corpus. Keyword search fails for this problem because eligibility criteria are written in clinical shorthand: "EGFR exon 19 deletion, treatment-naive, ECOG ≤1" does not surface from a query like "lung cancer trial." The agentic approach solves this by decomposing the task — a parser agent structures the free text into a validated schema, a retrieval agent does semantic search and cross-encoder reranking over FAISS-indexed trial documents, and an explanation agent runs chain-of-thought eligibility reasoning per trial with SELF-RAG confidence flags and provenance citations. The result is a pipeline that produces ranked verdicts with reasoning, not a bag of keyword hits.

---

## Architecture

```
Patient Profile (free text)
        │
        ▼
┌─────────────────┐
│   Parse Node    │  llama3 via Ollama — extracts cancer type, biomarkers,
│                 │  ECOG, prior treatments, age, metastatic status
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Retrieve Node  │  all-MiniLM-L6-v2 → FAISS top-50 → ms-marco cross-encoder rerank
│                 │  10,000 oncology trials from ClinicalTrials.gov
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Explain Node   │  Chain-of-thought reasoning per trial
│                 │  SELF-RAG confidence flags, provenance tagging
└────────┬────────┘
         │
         ▼
Ranked matches with verdicts, reasoning, and human-review flags
```

The four nodes are wired as a linear LangGraph state machine. Errors propagate through a state field — no conditional edges, no silent swallowing. Each node is a thin wrapper around its agent module so the graph stays decoupled from inference logic.

---

## Evaluation Results

Measured on a 10-query oncology eval set with manually labeled relevant trials.

| Metric | Bi-encoder only | + Cross-encoder rerank |
|---|---|---|
| Mean P@5 | 0.080 | 0.060 |
| Mean MRR | 0.320 | 0.305 |
| Hallucination rate | — | 0.40 |

The cross-encoder regression is a domain mismatch: ms-marco-MiniLM-L-12-v2 was trained on web passage retrieval, not clinical eligibility text, and it actively hurts ranking quality on biomarker-specific queries. The 40% hallucination rate traces to two root causes: eligibility text is truncated at 2,000 characters before being passed to the LLM, so disqualifying criteria buried deeper in the document are never seen; and llama3 regularly misses implicit disease-subtype exclusions that a clinician would infer from context. Both are documented findings with clear remediation paths, not surprises.

---

## Quick Start

```bash
git clone git@github.com:pranavsai19/Trialcompass.git
cd Trialcompass

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull the local LLM (requires Ollama installed and running)
ollama pull llama3

# Run the CLI pipeline
PYTHONPATH=. python src/orchestration/run_pipeline.py \
  --patient "58-year-old female, stage IV NSCLC, EGFR exon 19 deletion, ECOG 1, failed carboplatin+pemetrexed, metastatic"

# Launch the Streamlit UI
PYTHONPATH=. streamlit run app/streamlit_app.py
```

Ollama must be running (`ollama serve`) before any inference call. The FAISS index and SQLite database are in `data/` — no rebuild step needed for the 10K-trial corpus.

---

## Project Structure

```
trialcompass/
├── app/
│   └── streamlit_app.py          # Streamlit UI — full pipeline demo
├── data/
│   ├── faiss_index.bin           # FAISS flat IP index, 10K trials
│   ├── nct_ids.npy               # NCT ID array aligned to FAISS index
│   └── trialcompass.db           # SQLite — trials table with chunk_text, eligibility_text
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_experiments.ipynb
│   ├── 03_parser_agent_dev.ipynb
│   ├── 04_retrieval_eval.ipynb
│   └── 05_explanation_agent_dev.ipynb
├── src/
│   ├── agents/
│   │   ├── parser_agent.py       # Ollama/llama3 + Pydantic schema extraction
│   │   ├── retrieval_agent.py    # FAISS bi-encoder + cross-encoder reranker
│   │   └── explanation_agent.py  # CoT eligibility reasoning, SELF-RAG flags, provenance
│   ├── embeddings/
│   │   ├── embed.py              # Batch embedding with all-MiniLM-L6-v2
│   │   └── faiss_index.py        # Index build and persistence
│   ├── ingestion/
│   │   ├── fetch_trials.py       # ClinicalTrials.gov API v2 pull
│   │   └── preprocess.py         # JSON → flat text chunks → SQLite
│   ├── orchestration/
│   │   ├── graph.py              # LangGraph state machine (4 nodes, linear)
│   │   └── run_pipeline.py       # CLI wrapper with tabulate output
│   └── evaluation/
│       └── metrics.py            # Precision@K, MRR, hallucination rate
└── tests/                        # pytest — parser, retrieval, explanation agents
```

---

## Known Limitations

- **Eligibility text truncation at 2,000 characters.** The LLM only sees the first ~500 tokens of each trial's eligibility criteria. Disqualifying criteria buried in the second half of the document are invisible to the explanation agent. This is the primary driver of the 40% hallucination rate.
- **ms-marco cross-encoder domain mismatch.** The cross-encoder was trained on web retrieval pairs, not clinical text. It degrades P@5 from 0.080 to 0.060 on this eval set. A biomedical cross-encoder fine-tuned on clinical trial text is the planned replacement.
- **llama3 confidence is not calibrated.** The model reports HIGH confidence on approximately every verdict, including wrong ones. The SELF-RAG human-review flag is driven by uncertainty phrases in the reasoning text and eligibility text length — the confidence field itself should be ignored.

---

## Citation

If you use this work, please cite as:

> Vishnuvajjhula, P. (2026). TrialCompass: Agentic RAG for Oncology Clinical Trial Matching.
