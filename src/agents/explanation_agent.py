"""
Explanation agent — chain-of-thought eligibility reasoning over retrieved trials.

Takes the output of retrieval_agent.retrieve_and_rerank() and runs each trial through
llama3 via Ollama to produce: a structured eligibility verdict, confidence, inclusion/
exclusion analysis, a reasoning summary, and provenance quotes from the eligibility text.

Prompt design documented in notebooks/05_explanation_agent_dev.ipynb.

Key findings from the notebook:
- Variant C (structured JSON) is best: ~8s latency, parseable, includes CoT in reasoning field.
- llama3 inserts // comments inside JSON and sometimes uses $ref pointer syntax in lists.
  Both handled in _extract_json() before passing to json.loads().
- Confidence field is unreliable — model reports HIGH on everything including wrong verdicts.
  Do NOT use confidence==HIGH as a signal that a result is trustworthy.
- Hallucination rate on 10 pairs: 40%. Primary driver: 2000-char eligibility truncation.
  The model never sees disqualifying criteria buried deeper in the text.
- SELF-RAG flag must be truncation-aware, not just confidence-aware.

No local model weights loaded here — all inference goes through Ollama HTTP.
"""

import json
import re
import time
from typing import Any, Optional

import requests

from src.config import MODEL_NAME, OLLAMA_URL

MODEL = MODEL_NAME

# Phrases in the model's reasoning that indicate it is working from incomplete information.
# The model won't say "I cannot determine" in its confidence field — it will say HIGH —
# but these phrases leak through into the reasoning text.
_UNCERTAINTY_PHRASES = (
    "not specified",
    "unclear",
    "cannot determine",
    "not mentioned",
    "truncated",
    "no information",
    "not provided",
    "not available",
)

# Eligibility text below this length almost certainly means the DB stored only the
# condition name with no actual criteria — anything the model says is extrapolation.
_MIN_ELIGIBILITY_LENGTH = 200


# ---------------------------------------------------------------------------
# Ollama HTTP wrapper
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, timeout: int = 45) -> Optional[str]:
    """
    POST to Ollama and return the response string, or None on any failure.

    Callers handle None — this function never raises. Ollama can be slow
    on first inference (model warmup) and occasionally drops connections,
    so we want graceful degradation rather than a crash mid-pipeline.
    """
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 750},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.RequestException:
        return None


# ---------------------------------------------------------------------------
# JSON parsing with comment stripping and truncation recovery
# ---------------------------------------------------------------------------

def _strip_json_comments(text: str) -> str:
    # llama3 sometimes inserts // comments inside JSON — invalid, breaks json.loads
    return re.sub(r"//[^\n]*", "", text)


def _extract_json(raw: str) -> Optional[dict]:
    """
    Parse a JSON dict from raw LLM output.

    Handles three failure modes seen in notebook experiments:
    1. Markdown code fences wrapping the JSON.
    2. JS-style // comments inside the JSON object.
    3. Output truncated mid-object (model hit num_predict limit).
       Recovery: try progressively shorter suffixes until json.loads succeeds.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    cleaned = _strip_json_comments(cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    fragment = m.group()
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        for end in range(len(fragment) - 1, 0, -1):
            try:
                return json.loads(fragment[:end] + "}")
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# Flagging logic — extracted into a standalone function so tests don't need Ollama
# ---------------------------------------------------------------------------

def _should_flag_for_review(
    parsed: dict[str, Any],
    trial: dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """
    Return (flag: bool, reason: str | None).

    Does NOT rely on confidence == "HIGH" as a positive signal.
    From the notebook: llama3 reports HIGH on everything, including wrong verdicts.

    Flags if ANY condition is true:
    1. Model reported confidence LOW.
    2. Model returned verdict UNCERTAIN.
    3. Reasoning text contains a phrase that signals missing information.
    4. eligibility_text in the trial dict is very short (truncation likely at source).
    """
    confidence = parsed.get("confidence", "")
    verdict = parsed.get("verdict", "")
    reasoning = parsed.get("reasoning", "").lower()
    elig_len = len(trial.get("eligibility_text") or "")

    if confidence == "LOW":
        return True, "model confidence LOW"
    if verdict == "UNCERTAIN":
        return True, "model returned UNCERTAIN verdict"
    for phrase in _UNCERTAINTY_PHRASES:
        if phrase in reasoning:
            return True, f"uncertainty phrase in reasoning: '{phrase}'"
    if elig_len < _MIN_ELIGIBILITY_LENGTH:
        return True, f"eligibility text too short ({elig_len} chars) — likely truncated in DB"
    # Flag on thin reasoning regardless of LLM confidence
    if len(parsed.get("reasoning", "").strip()) < 100:
        return True, "reasoning text too short — low quality output"
    return False, None


# ---------------------------------------------------------------------------
# Computed confidence and provenance helpers
# ---------------------------------------------------------------------------

def _extract_provenance(reasoning: str, trial: dict) -> list[str]:
    """
    Extract which specific eligibility criteria phrases from the trial
    appear to be referenced in the model's reasoning.
    Returns a list of up to 3 short matched phrases (max 80 chars each).
    This is a lightweight keyword overlap — not semantic matching.
    """
    import re as _re
    eligibility = trial.get("eligibility_text", "")
    if not eligibility or not reasoning:
        return []

    clauses = _re.split(r'[;\n•\-]+', eligibility)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 20]

    reasoning_lower = reasoning.lower()
    matched = []
    for clause in clauses:
        words = clause.lower().split()
        if len(words) < 3:
            continue
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3])
            if trigram in reasoning_lower:
                matched.append(clause[:80])
                break
        if len(matched) >= 3:
            break

    return matched


def _compute_confidence(
    parsed: dict,
    reasoning: str,
    eligibility_text: str,
    ce_score: float,
    patient_profile_text: str,
) -> str:
    """
    Compute a calibrated confidence score from observable signals.
    Ignores the LLM's self-reported confidence field entirely.

    Returns: "HIGH", "MEDIUM", or "LOW"

    Scoring (starts at 100, deductions applied):
    - Each uncertainty phrase found in reasoning: -15 points
    - eligibility_text shorter than 200 chars: -30 points
    - eligibility_text shorter than 500 chars: -15 points
    - verdict is UNCERTAIN: -40 points
    - verdict is ELIGIBLE but ce_score < 0.0: -20 points
      (cross-encoder thinks it's a weak match but LLM says eligible — suspicious)
    - verdict is INELIGIBLE but ce_score > 5.0: -10 points
      (cross-encoder thinks it's a strong match but LLM says ineligible — worth reviewing)
    - reasoning text shorter than 100 chars: -25 points
      (model gave almost no reasoning — low quality output)

    Thresholds:
    - score >= 70: HIGH
    - score >= 45: MEDIUM
    - score < 45: LOW
    """
    score = 100
    reasoning_lower = reasoning.lower()
    verdict = parsed.get("verdict", "UNCERTAIN")

    # Uncertainty phrase penalty
    phrase_hits = sum(1 for p in _UNCERTAINTY_PHRASES if p in reasoning_lower)
    score -= phrase_hits * 15

    # Eligibility text quality penalty
    if len(eligibility_text) < 200:
        score -= 30
    elif len(eligibility_text) < 500:
        score -= 15

    # Uncertain verdict penalty
    if verdict == "UNCERTAIN":
        score -= 40

    # CE score vs verdict disagreement penalties
    if verdict == "ELIGIBLE" and ce_score < 0.0:
        score -= 20
    if verdict == "INELIGIBLE" and ce_score > 5.0:
        score -= 10

    # Thin reasoning penalty
    if len(reasoning.strip()) < 100:
        score -= 25

    # Clamp to 0-100
    score = max(0, min(100, score))

    if score >= 70:
        return "HIGH"
    elif score >= 45:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Variant C from the notebook — structured JSON with CoT in reasoning field.
# Critical guards: "plain text strings only — not $ref objects" prevents the model from
# producing {"$ref": "#/paths/..."} in list fields; "no // comments" prevents JS comments.
_ELIGIBILITY_PROMPT = """\
You are a clinical trial eligibility screener. Analyze the patient-trial match carefully.

PATIENT:
{patient_summary}

TRIAL: {trial_title} ({nct_id})
ELIGIBILITY CRITERIA (may be truncated):
{eligibility_text}

Return ONLY a valid JSON object. Lists must contain plain text strings only — not $ref \
objects, not JSON schema references, no // comments inside the JSON.
{{
  "verdict": "ELIGIBLE" | "INELIGIBLE" | "UNCERTAIN",
  "inclusion_met": ["criterion text in plain English"],
  "inclusion_failed": ["criterion text in plain English"],
  "exclusion_flags": ["criterion text in plain English"],
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "reasoning": "2-3 sentence clinical explanation. If eligibility text appears truncated \
or key criteria are missing, note this and use confidence LOW or return UNCERTAIN."
}}

JSON only. No markdown fences. No text outside the object."""

_PROVENANCE_PROMPT = """\
A patient was matched to clinical trial {nct_id}.

These inclusion criteria were determined to be met:
{inclusion_met}

Eligibility text:
{eligibility_text}

For each inclusion criterion above, quote the EXACT sentence or clause from the eligibility \
text that supports it. If no matching sentence exists, write "NOT FOUND IN TEXT".

Return ONLY a JSON object: each key is a criterion string, each value is the supporting quote.
No // comments. No $ref. No text outside the JSON object."""


# ---------------------------------------------------------------------------
# Patient summary builder
# ---------------------------------------------------------------------------

def _build_patient_summary(profile: "dict[str, Any] | str") -> str:
    """Prose summary of the patient profile for the LLM prompt.
    Accepts either a structured dict (normal pipeline path) or a plain
    string (direct testing / smoke tests)."""
    if isinstance(profile, str):
        return profile
    parts: list[str] = []

    cancer = profile.get("cancer_type")
    if cancer:
        meta = profile.get("metastatic")
        parts.append(f"Cancer: {'metastatic ' if meta else ''}{cancer}")

    age = profile.get("age")
    if age is not None:
        parts.append(f"Age: {age}")

    ecog = profile.get("ecog")
    if ecog is not None:
        parts.append(f"ECOG performance status: {ecog}")

    # Can't use `or` — 0 prior lines is meaningful and 0 is falsy
    prior = profile.get("prior_treatment_lines")
    if prior is None:
        prior = profile.get("prior_treatments")
    if prior is not None:
        label = "treatment naive (0 prior lines)" if prior == 0 else f"{prior} prior treatment line(s)"
        parts.append(f"Prior treatments: {label}")

    biomarkers = profile.get("biomarkers") or {}
    if isinstance(biomarkers, dict):
        active = [
            f"{m} {s}"
            for m, s in biomarkers.items()
            if s and s not in ("None", "null")
        ]
        if active:
            parts.append(f"Biomarkers: {', '.join(active)}")

    notes = profile.get("extraction_notes")
    if notes:
        parts.append(f"Note: {notes}")

    return "\n".join(parts) if parts else "Oncology patient (no structured profile available)"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def explain_matches(
    patient_profile: dict[str, Any],
    trials: list[dict[str, Any]],
    max_trials: int = 10,
) -> list[dict[str, Any]]:
    """
    Run chain-of-thought eligibility reasoning over retrieved and reranked trials.

    Parameters
    ----------
    patient_profile : dict
        Structured output from parser_agent — fields: cancer_type, biomarkers, ecog,
        prior_treatment_lines, age, metastatic. Missing fields handled gracefully.
    trials : list[dict]
        Output of retrieval_agent.retrieve_and_rerank(). Each dict must have at minimum:
        nct_id, brief_title, eligibility_text. Fields like bi_encoder_score,
        cross_encoder_score are passed through to the output unchanged.
    max_trials : int
        Cap on how many trials to explain. LLM calls are ~8-12s each plus ~10s provenance
        — budget ~20s per trial, so 10 trials ≈ 200s total.

    Returns
    -------
    list[dict]
        One dict per trial, preserving all input fields and adding:
        verdict, confidence, inclusion_met, inclusion_failed, exclusion_flags,
        reasoning, human_review_flag, flag_reason, provenance.
        rank is re-assigned 1-indexed based on position in output list.
    """
    patient_summary = _build_patient_summary(patient_profile)
    results: list[dict[str, Any]] = []

    for i, trial in enumerate(trials[:max_trials]):
        nct_id = trial.get("nct_id", "UNKNOWN")
        title = trial.get("brief_title", "")
        elig = trial.get("eligibility_text") or ""

        print(f"[explain] {i+1}/{min(len(trials), max_trials)}  {nct_id}  ({len(elig)} chars eligibility)")
        t0 = time.time()

        # --- Stage 1: eligibility verdict ---
        prompt = _ELIGIBILITY_PROMPT.format(
            patient_summary=patient_summary,
            trial_title=title[:100],
            nct_id=nct_id,
            # 4000 chars is ~1000 tokens — fits in llama3's 8K context with room for prompt overhead.
            # Increased from 2000 to capture exclusion criteria in longer trials.
            eligibility_text=elig[:4000],
        )
        raw = _call_ollama(prompt)
        elapsed = time.time() - t0

        if raw is None:
            # Ollama unreachable or timed out — flag and continue
            results.append({
                **trial,
                "rank": i + 1,
                "verdict": "UNCERTAIN",
                "confidence": "LOW",
                "inclusion_met": [],
                "inclusion_failed": [],
                "exclusion_flags": [],
                "reasoning": "LLM call failed — Ollama did not respond.",
                "human_review_flag": True,
                "flag_reason": "Ollama call failed",
                "provenance": None,
            })
            continue

        parsed = _extract_json(raw) or {}
        print(f"           verdict={parsed.get('verdict', '?')}  conf={parsed.get('confidence', '?')}  ({elapsed:.1f}s)")

        flag, flag_reason = _should_flag_for_review(parsed, trial)

        # --- Stage 2: provenance tagging (best-effort, second LLM call) ---
        provenance: Optional[dict] = None
        inclusion_met: list[str] = parsed.get("inclusion_met") or []

        if inclusion_met:
            prov_prompt = _PROVENANCE_PROMPT.format(
                nct_id=nct_id,
                inclusion_met=json.dumps(inclusion_met, indent=2),
                eligibility_text=elig[:4000],
            )
            t1 = time.time()
            prov_raw = _call_ollama(prov_prompt, timeout=45)
            prov_elapsed = time.time() - t1
            if prov_raw:
                provenance = _extract_json(prov_raw)
                n_citations = len(provenance) if provenance else 0
                print(f"           provenance: {n_citations} citations ({prov_elapsed:.1f}s)")
            else:
                print(f"           provenance: call failed or timed out")

        _reasoning = parsed.get("reasoning", "")
        _elig      = trial.get("eligibility_text", "")
        _ce_score  = trial.get("ce_score", 0.0)

        results.append({
            **trial,
            "rank": i + 1,
            "verdict": parsed.get("verdict", "UNCERTAIN"),
            "confidence": _compute_confidence(
                parsed, _reasoning, _elig, _ce_score, patient_summary,
            ),
            "llm_raw_confidence": parsed.get("confidence", "UNKNOWN"),
            "inclusion_met": inclusion_met,
            "inclusion_failed": parsed.get("inclusion_failed") or [],
            "exclusion_flags": parsed.get("exclusion_flags") or [],
            "reasoning": _reasoning,
            "human_review_flag": flag,
            "flag_reason": flag_reason,
            "provenance": _extract_provenance(_reasoning, trial),
        })

    return results
