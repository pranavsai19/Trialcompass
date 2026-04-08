"""
Cross-encoder re-ranking for clinical trial retrieval.

The bi-encoder (PubMedBERT FAISS index) retrieves the top-50 candidates fast
using approximate semantic similarity. The cross-encoder reads the query and
each candidate document together in a single forward pass, producing a precise
pairwise relevance score at the cost of O(N) inference.

This is the standard retrieve-then-rerank pattern: bi-encoder for recall,
cross-encoder for precision. The cross-encoder score is the primary ranking
signal returned to the pipeline.

Known limitation: ms-marco-MiniLM-L-6-v2 was trained on web passage retrieval
(MS MARCO dataset), not clinical eligibility text. It understands query-document
relevance in general but does not know that "Child-Pugh A" and "Child-Pugh B"
are clinically distinct, or that "enzalutamide resistant" differs from "enzalutamide
naive". A biomedical cross-encoder fine-tuned on clinical text would close this gap.
That is the documented next upgrade (Component 5).
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder


class ClinicalReranker:
    """
    Wraps a cross-encoder model to rerank a list of candidate trials
    against a patient query.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for the cross-encoder. Default is the
        MS MARCO MiniLM L-6 model — fast (6 layers) and good enough
        for a baseline. Swap for L-12 for ~1.5× more accuracy at ~2×
        the latency, or for a biomedical model when available.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "chunk_text",
    ) -> list[dict]:
        """
        Score each (query, candidate_text) pair and return candidates
        sorted by cross-encoder score descending.

        Parameters
        ----------
        query : str
            The patient profile query string.
        candidates : list[dict]
            Each dict must contain at minimum: nct_id, and the field
            named by `text_key` (default "chunk_text"). Any additional
            keys (title, bi_encoder_score, etc.) are preserved.
        text_key : str
            Key in each candidate dict to use as the document text.
            Defaults to "chunk_text".

        Returns
        -------
        list[dict]
            Same dicts as input, each with a new "ce_score" key (float),
            sorted by ce_score descending (highest relevance first).
        """
        if not candidates:
            return []

        texts = [c.get(text_key) or "" for c in candidates]
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["ce_score"] = float(score)

        return sorted(candidates, key=lambda c: c["ce_score"], reverse=True)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "EGFR mutant NSCLC, erlotinib failed, ECOG 1"

    candidates = [
        {
            "nct_id": "NCT_RELEVANT_1",
            "title": "Osimertinib for Advanced EGFR+ NSCLC Patients",
            "chunk_text": (
                "Trial ID: NCT_RELEVANT_1 Title: Osimertinib for Advanced EGFR+ NSCLC "
                "Conditions: EGFR Positive Non-small Cell Lung Cancer Status: RECRUITING "
                "Eligibility: Inclusion Criteria: Histologically confirmed NSCLC with "
                "EGFR mutation. Prior treatment with first- or second-generation EGFR TKI "
                "(erlotinib, gefitinib, or afatinib) with documented progression. ECOG "
                "performance status 0 or 1."
            ),
        },
        {
            "nct_id": "NCT_RELEVANT_2",
            "title": "Phase II Osimertinib + Chemo After TKI Failure",
            "chunk_text": (
                "Trial ID: NCT_RELEVANT_2 Title: Phase II Osimertinib + Chemo After TKI Failure "
                "Conditions: NSCLC EGFR-TKI Resistant Status: ACTIVE_NOT_RECRUITING "
                "Eligibility: Patients with EGFR-mutant NSCLC who have progressed on erlotinib, "
                "gefitinib, or osimertinib. ECOG PS 0-1 required."
            ),
        },
        {
            "nct_id": "NCT_IRRELEVANT_1",
            "title": "Pembrolizumab in Triple Negative Breast Cancer",
            "chunk_text": (
                "Trial ID: NCT_IRRELEVANT_1 Title: Pembrolizumab in Triple Negative Breast Cancer "
                "Conditions: Triple Negative Breast Cancer Status: RECRUITING "
                "Eligibility: Histologically confirmed triple-negative breast cancer. "
                "No prior chemotherapy in the metastatic setting."
            ),
        },
        {
            "nct_id": "NCT_IRRELEVANT_2",
            "title": "Sorafenib After Failure in Hepatocellular Carcinoma",
            "chunk_text": (
                "Trial ID: NCT_IRRELEVANT_2 Title: Sorafenib After Failure in HCC "
                "Conditions: Hepatocellular Carcinoma Status: COMPLETED "
                "Eligibility: Child-Pugh Class A liver function. Prior sorafenib treatment "
                "with radiographic progression."
            ),
        },
        {
            "nct_id": "NCT_PARTIAL_1",
            "title": "Gefitinib vs Erlotinib in EGFR+ Lung Cancer",
            "chunk_text": (
                "Trial ID: NCT_PARTIAL_1 Title: Gefitinib vs Erlotinib in EGFR+ Lung Cancer "
                "Conditions: Non-Small Cell Lung Cancer Status: COMPLETED "
                "Eligibility: Treatment-naive EGFR mutation positive NSCLC. No prior TKI therapy."
            ),
        },
    ]

    reranker = ClinicalReranker()
    reranked = reranker.rerank(query, candidates)

    print(f"Query: {query!r}\n")
    print(f"{'Rank':<6} {'NCT ID':<20} {'ce_score':>10}  Title")
    print("-" * 80)
    for rank, c in enumerate(reranked, 1):
        print(f"{rank:<6} {c['nct_id']:<20} {c['ce_score']:>10.4f}  {c['title']}")
