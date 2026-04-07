"""
Tests for explanation_agent — flagging logic and JSON parsing only.

No Ollama calls here. explain_matches() requires a live LLM and belongs in
an integration test. We test the two pure functions that drive the safety logic:
_should_flag_for_review() and _extract_json().
"""

import pytest
from src.agents.explanation_agent import _should_flag_for_review, _extract_json


# ---------------------------------------------------------------------------
# _should_flag_for_review tests
# ---------------------------------------------------------------------------

# A trial dict with substantial eligibility text (normal case).
# Must be clearly above _MIN_ELIGIBILITY_LENGTH (200 chars) to avoid the short-text flag.
TRIAL_FULL = {
    "nct_id": "NCT03260491",
    "brief_title": "U3-1402 in Metastatic NSCLC",
    "eligibility_text": (
        "Inclusion Criteria:\n"
        "1. Has locally advanced or metastatic NSCLC, not amenable to curative surgery or radiation.\n"
        "2. Has at least one measurable lesion per RECIST version 1.1 criteria.\n"
        "3. Has Eastern Cooperative Oncology Group (ECOG) performance status of 0 or 1 at Screening.\n"
        "4. Has histologically or cytologically documented adenocarcinoma NSCLC.\n"
        "5. Has acquired resistance to an EGFR TKI per Jackman criteria including sensitizing EGFR mutation.\n"
        "Exclusion Criteria:\n"
        "1. Has received any prior HER3-directed therapy.\n"
        "2. Has uncontrolled or significant cardiovascular disease.\n"
        "3. Has active CNS metastases (stable treated CNS metastases are allowed).\n"
        "4. ECOG performance status of 2 or greater.\n"
    ),
}

# A trial with almost no eligibility text — truncated at source
TRIAL_SHORT_ELIG = {
    "nct_id": "NCT99999999",
    "brief_title": "Some Trial",
    "eligibility_text": "Inclusion: cancer diagnosis.",  # 28 chars — below threshold
}


def test_flag_on_uncertain_verdict():
    parsed = {"verdict": "UNCERTAIN", "confidence": "HIGH", "reasoning": "Looks fine overall."}
    flag, reason = _should_flag_for_review(parsed, TRIAL_FULL)
    assert flag is True
    assert "UNCERTAIN" in reason


def test_flag_on_low_confidence():
    parsed = {"verdict": "ELIGIBLE", "confidence": "LOW", "reasoning": "Seems like a match."}
    flag, reason = _should_flag_for_review(parsed, TRIAL_FULL)
    assert flag is True
    assert "LOW" in reason


def test_flag_on_uncertainty_phrase_in_reasoning():
    parsed = {
        "verdict": "ELIGIBLE",
        "confidence": "HIGH",
        "reasoning": "The patient meets most criteria. However, ECOG status is not specified in the profile.",
    }
    flag, reason = _should_flag_for_review(parsed, TRIAL_FULL)
    assert flag is True
    assert "not specified" in reason


def test_flag_on_short_eligibility_text():
    # Even with a clean HIGH/ELIGIBLE result, short eligibility text = unreliable
    parsed = {"verdict": "ELIGIBLE", "confidence": "HIGH", "reasoning": "Patient clearly qualifies."}
    flag, reason = _should_flag_for_review(parsed, TRIAL_SHORT_ELIG)
    assert flag is True
    assert "chars" in reason  # reason mentions char count


def test_no_flag_on_clean_result():
    parsed = {
        "verdict": "ELIGIBLE",
        "confidence": "HIGH",
        "reasoning": "Patient meets all inclusion criteria including EGFR mutation and ECOG 1. No exclusion criteria apply.",
    }
    flag, reason = _should_flag_for_review(parsed, TRIAL_FULL)
    assert flag is False
    assert reason is None


def test_no_flag_on_ineligible_with_full_text():
    # INELIGIBLE with HIGH confidence and full text = not flagged
    parsed = {
        "verdict": "INELIGIBLE",
        "confidence": "HIGH",
        "reasoning": "Patient has colorectal cancer, not NSCLC. Disease type does not match trial criteria.",
    }
    flag, reason = _should_flag_for_review(parsed, TRIAL_FULL)
    assert flag is False
    assert reason is None


def test_flag_on_cannot_determine_phrase():
    parsed = {
        "verdict": "ELIGIBLE",
        "confidence": "MEDIUM",
        "reasoning": "The erlotinib duration cannot determine from the available information.",
    }
    flag, reason = _should_flag_for_review(parsed, TRIAL_FULL)
    assert flag is True


# ---------------------------------------------------------------------------
# _extract_json tests
# ---------------------------------------------------------------------------

def test_extract_clean_json():
    raw = '{"verdict": "ELIGIBLE", "confidence": "HIGH", "reasoning": "Good match."}'
    result = _extract_json(raw)
    assert result is not None
    assert result["verdict"] == "ELIGIBLE"


def test_extract_json_with_markdown_fences():
    raw = '```json\n{"verdict": "INELIGIBLE", "confidence": "LOW"}\n```'
    result = _extract_json(raw)
    assert result is not None
    assert result["verdict"] == "INELIGIBLE"


def test_extract_json_strips_comments():
    # llama3 inserts // comments — must be stripped before json.loads
    raw = '{"verdict": "ELIGIBLE", "exclusion_flags": ["Prior therapy"] // no info provided\n}'
    result = _extract_json(raw)
    assert result is not None
    assert result["verdict"] == "ELIGIBLE"


def test_extract_json_returns_none_on_no_json():
    raw = "I cannot determine eligibility from the provided information."
    result = _extract_json(raw)
    assert result is None


def test_extract_json_truncation_recovery():
    # Simulate a truncated response — missing closing brace
    raw = '{"verdict": "ELIGIBLE", "confidence": "HIGH", "reasoning": "Good match.'
    result = _extract_json(raw)
    # Should either recover or return None — must not raise
    assert result is None or isinstance(result, dict)
