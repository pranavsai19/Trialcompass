"""
Central configuration for TrialCompass.
Change MODEL_NAME here to switch the LLM across the entire pipeline.
"""

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Active model — swap between "llama3" and "mistral" here
MODEL_NAME = "llama3"

# Retrieval settings
TOP_K_RERANK = 10
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_MODEL = "neuml/pubmedbert-base-embeddings"
