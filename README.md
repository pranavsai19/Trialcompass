# TrialCompass

I kept wondering why oncologists still manually scan through thousands of trials to find one match for a patient.

The problem is not access — ClinicalTrials.gov has over 400,000 registered studies, all publicly available. The problem is signal. A patient with triple-negative breast cancer, BRCA2 mutation, two prior chemo lines, and an ECOG score of 1 needs a very different trial than a patient with the same diagnosis and no prior treatment. Keyword search does not capture that. It returns noise. Clinicians either over-rely on what they already know, or spend hours they do not have reading eligibility criteria one by one.

TrialCompass is my attempt to build something better — an agentic AI system that takes a structured patient profile and returns a ranked, explainable list of matching oncology trials from ClinicalTrials.gov.

---

## What the System Does

TrialCompass is not a search engine. It is a multi-agent pipeline built on LangGraph where each agent owns a distinct piece of the reasoning problem:

1. **Patient Profile Parser** — Takes free-text or structured patient input and extracts structured fields: primary cancer type (ICD-10 code), biomarker status (BRCA1/2, PD-L1, KRAS, HER2), ECOG performance status, prior treatment lines, age, metastasis status. Runs locally via Ollama using Llama 3 8B or BioMistral-7B. Output is validated with Pydantic so nothing downstream breaks on a missing field.

2. **Retrieval Agent** — Queries a FAISS index of 60,000+ oncology trial documents embedded with a biomedical sentence encoder. Pulls the top-50 candidates by semantic similarity. This is fast and broad — precision is not the goal here, recall is.

3. **Re-Ranking Agent** — Runs a cross-encoder (ms-marco-MiniLM-L12-v2) over the top-50 to re-score based on actual query-document relevance. This is where the quality jump happens. The delta between bi-encoder retrieval and retrieve-then-rerank is one of the things I am measuring.

4. **Explanation Agent** — Takes the top-10 reranked trials and reasons through eligibility criteria for this specific patient. Uses chain-of-thought prompting to flag inclusion matches, exclusion violations, and uncertain criteria. Applies SELF-RAG-style confidence scoring — if the LLM is not sure, it says so rather than fabricating a match. Every claim is tagged with the source chunk that drove it.

5. **Orchestration** — LangGraph manages the state machine: parse → retrieve → rerank → explain → return. Each agent can be swapped or extended without touching the others.

---

## The Five Components

| Component | What It Does | Key Tech |
|-----------|-------------|----------|
| Data Ingestion | Pull + preprocess 60K+ oncology trials from ClinicalTrials.gov API v2 | `requests`, `pandas`, `sqlite-utils` |
| Vector Store | Embed trials, build FAISS index, benchmark retrieval | `sentence-transformers`, `faiss-cpu` |
| Parser Agent | Extract structured patient profile from input | Ollama, Pydantic, LangChain |
| Retrieval + Reranking | FAISS top-50 → cross-encoder rerank → top-10 | `sentence-transformers` cross-encoders |
| Explanation Agent | CoT reasoning over eligibility criteria, confidence scoring, provenance | LangGraph, BioMistral-7B |

---

## Evaluation

The metrics that matter here:

- **Precision@5** — Of the top 5 returned trials, how many are actually eligible for this patient?
- **MRR (Mean Reciprocal Rank)** — How far down the list do you have to go to find the first correct match?
- **Hallucination Rate** — When the explanation agent makes a claim about eligibility, what fraction of those claims are unsupported by the source documents?

Baseline is bi-encoder retrieval only. The goal is to show that LangGraph orchestration with reranking and structured reasoning produces a measurable lift on all three.

---

## Tech Stack

- **Orchestration**: LangGraph, LangChain
- **LLMs**: BioMistral-7B or Llama 3 8B via Ollama (local, Mac MPS backend)
- **Embeddings**: `all-MiniLM-L6-v2` → BioBERT / BioLinkBERT
- **Vector Store**: FAISS (flat L2 index)
- **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L12-v2`
- **Data**: ClinicalTrials.gov API v2 (public, no key required)
- **Storage**: SQLite
- **Validation**: Pydantic
- **UI**: Streamlit
- **Evaluation**: scikit-learn, numpy
- **Language**: Python 3.11

---

## Project Structure

```
trialcompass/
├── CLAUDE.md                        # Engineering context and project spec
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                         # Raw JSON from ClinicalTrials.gov API
│   ├── processed/                   # Cleaned, flat text chunks
│   └── trialcompass.db              # SQLite store
├── notebooks/
│   ├── 01_data_exploration.ipynb    # API structure, field coverage analysis
│   ├── 02_embedding_experiments.ipynb  # MiniLM vs BioBERT comparison
│   ├── 03_parser_agent_dev.ipynb    # Prompt engineering for patient parser
│   ├── 04_retrieval_eval.ipynb      # Precision@5, MRR benchmarks
│   └── 05_explanation_agent_dev.ipynb  # CoT + SELF-RAG experiments
├── src/
│   ├── ingestion/
│   │   ├── fetch_trials.py          # ClinicalTrials.gov API client
│   │   └── preprocess.py            # JSON → flat text chunks
│   ├── embeddings/
│   │   ├── embed.py                 # Embedding pipeline
│   │   └── faiss_index.py           # Build, save, load FAISS index
│   ├── agents/
│   │   ├── parser_agent.py          # Patient profile extraction
│   │   ├── retrieval_agent.py       # FAISS + cross-encoder reranking
│   │   └── explanation_agent.py     # CoT eligibility reasoning
│   ├── orchestration/
│   │   └── graph.py                 # LangGraph state machine
│   └── evaluation/
│       └── metrics.py               # Precision@K, MRR, hallucination rate
├── tests/
└── app/
    └── streamlit_app.py             # Demo UI
```

---

## How to Run Locally

**Prerequisites**: Python 3.11, Ollama installed and running (`ollama serve`), Llama 3 8B pulled (`ollama pull llama3`).

```bash
# Clone and set up
git clone https://github.com/pranavsai19/trialcompass.git
cd trialcompass
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy env template
cp .env.example .env

# Pull trial data (starts with 10K, scales to 60K+)
python src/ingestion/fetch_trials.py

# Build FAISS index
python src/embeddings/faiss_index.py

# Run evaluation
python src/evaluation/metrics.py

# Launch Streamlit UI
streamlit run app/streamlit_app.py
```

---

## What Comes Next

A few things I want to tackle that are not in the current scope:

**Upgrade the embedding model.** `all-MiniLM-L6-v2` is fast but not trained on biomedical text. BioBERT or BioLinkBERT should give a meaningful retrieval quality improvement on biomarker-specific queries. I want to measure that delta explicitly before claiming it is worth the latency tradeoff.

**Scale to the full ClinicalTrials.gov corpus.** 60K oncology trials is the target. Getting there requires batched API calls, incremental SQLite inserts, and a more careful chunking strategy — eligibility criteria sections alone can be 2,000+ tokens.

**HPC scaling path.** The embedding pipeline and cross-encoder reranking are both parallelizable. On a single Mac the index build takes hours. On an HPC cluster with GPU nodes it should take minutes. This is the piece I want to explore at DL4Sci 2026 — how the architecture scales when you are not resource-constrained.

**Better hallucination detection.** The current SELF-RAG approach is a first pass. A more rigorous approach would use NLI (natural language inference) to verify each LLM claim against the source document it cited. That is a cleaner signal than asking the model to rate its own confidence.

---

*Built by Pranav Vishnuvajjhula — Data Scientist, UTD MSBA. Applying to DL4Sci 2026 at Lawrence Berkeley National Laboratory.*
