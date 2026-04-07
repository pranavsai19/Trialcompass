"""
Tests for retrieval_agent.build_query_string.

We only test the query builder here — retrieve_and_rerank requires a live FAISS
index and SQLite, which belong in an integration test, not unit tests that run in CI.
"""

import pytest
from src.agents.retrieval_agent import build_query_string


# Full profile — matches what parser_agent.parse_patient_profile() returns
FULL_PROFILE = {
    "cancer_type": "non-small cell lung cancer",
    "icd10_code": "C34.90",
    "biomarkers": {
        "EGFR": "mutant",
        "KRAS": "wildtype",
        "PD_L1": None,
        "BRCA1": None,
        "BRCA2": None,
        "HER2": None,
        "FLT3": None,
    },
    "ecog": 1,
    "prior_treatment_lines": 2,
    "age": 58,
    "metastatic": True,
    "extraction_notes": None,
}

# Sparse profile — only cancer_type, everything else missing
SPARSE_PROFILE = {
    "cancer_type": "triple negative breast cancer",
}


def test_build_query_full_profile():
    q = build_query_string(FULL_PROFILE)
    assert isinstance(q, str)
    assert len(q) > 0
    # cancer type and metastatic prefix should appear
    assert "non-small cell lung cancer" in q
    assert "metastatic" in q
    # active biomarkers should be included
    assert "EGFR" in q
    assert "mutant" in q
    # ECOG and age
    assert "ECOG" in q
    assert "58" in q


def test_build_query_sparse_profile():
    q = build_query_string(SPARSE_PROFILE)
    assert isinstance(q, str)
    assert len(q) > 0
    # should still include the cancer type
    assert "triple negative breast cancer" in q


def test_build_query_empty_profile_returns_fallback():
    # completely empty dict — should not raise, should return fallback string
    q = build_query_string({})
    assert isinstance(q, str)
    assert len(q) > 0


def test_build_query_zero_prior_lines():
    q = build_query_string({"cancer_type": "AML", "prior_treatment_lines": 0})
    assert "treatment naive" in q


def test_build_query_metastatic_false_no_prefix():
    q = build_query_string({"cancer_type": "prostate cancer", "metastatic": False})
    assert "metastatic" not in q
    assert "prostate cancer" in q
