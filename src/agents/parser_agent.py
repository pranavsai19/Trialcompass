"""
Patient profile parser agent.

Takes free-text patient description, extracts structured clinical fields using
llama3 via Ollama, validates output with Pydantic. Never raises on missing fields
— unknown fields return None.

Prompt design documented in notebooks/03_parser_agent_dev.ipynb.

Key finding: few-shot examples are mandatory. Zero-shot returns prose + invalid JSON
and 25s latency. Two examples drop latency to ~4s and produce clean parseable output.

Known limitations (documented):
- ICD-10 codes approximate (AML C93.90 vs correct C92.00)
- Blood cancer metastatic field unreliable — treat as null for hematologic malignancies
- 'some prior treatment' inferred as 1 line — flagged in extraction_notes
"""

import json
import logging
import re
from typing import Literal, Optional

import requests
from pydantic import BaseModel, ValidationError, field_validator

from src.config import MODEL_NAME, OLLAMA_URL as _OLLAMA_URL

OLLAMA_URL = _OLLAMA_URL
DEFAULT_MODEL = MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

BiomarkerValue = Optional[Literal["positive", "negative", "mutant", "wildtype"]]


class BiomarkerStatus(BaseModel):
    BRCA1: BiomarkerValue = None
    BRCA2: BiomarkerValue = None
    PD_L1: BiomarkerValue = None
    KRAS: BiomarkerValue = None
    HER2: BiomarkerValue = None
    EGFR: BiomarkerValue = None
    FLT3: BiomarkerValue = None


class PatientProfile(BaseModel):
    cancer_type: Optional[str] = None
    icd10_code: Optional[str] = None
    biomarkers: BiomarkerStatus = BiomarkerStatus()
    ecog: Optional[int] = None
    prior_treatment_lines: Optional[int] = None
    age: Optional[int] = None
    metastatic: Optional[bool] = None
    extraction_notes: Optional[str] = None  # model flags uncertainty here

    @field_validator("ecog")
    @classmethod
    def ecog_range(cls, v):
        if v is not None and not (0 <= v <= 4):
            raise ValueError(f"ECOG must be 0-4, got {v}")
        return v

    @field_validator("prior_treatment_lines")
    @classmethod
    def lines_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"prior_treatment_lines cannot be negative, got {v}")
        return v


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a clinical NLP system. Extract structured patient data from free text.
Return ONLY a JSON object — no explanation, no markdown, no comments.
Use null for any field not mentioned. Use ICD-10 codes for cancer type.
If you had to infer a value (not explicitly stated), add a brief note in extraction_notes.

BIOMARKER VALUES must be one of: "positive", "negative", "mutant", "wildtype", or null.

EXAMPLE 1:
Input: 45-year-old male with stage IV non-small cell lung cancer, EGFR exon 19 deletion, never smoked, no prior systemic therapy, ECOG 0.
Output: {{"cancer_type": "Non-small cell lung cancer", "icd10_code": "C34.90", "biomarkers": {{"BRCA1": null, "BRCA2": null, "PD_L1": null, "KRAS": null, "HER2": null, "EGFR": "mutant", "FLT3": null}}, "ecog": 0, "prior_treatment_lines": 0, "age": 45, "metastatic": true, "extraction_notes": null}}

EXAMPLE 2:
Input: 62-year-old female with HR-positive HER2-negative metastatic breast cancer, CDK4/6 inhibitor refractory, PIK3CA mutation, ECOG 1, 2 prior lines.
Output: {{"cancer_type": "Breast cancer, HR-positive HER2-negative", "icd10_code": "C50.919", "biomarkers": {{"BRCA1": null, "BRCA2": null, "PD_L1": null, "KRAS": null, "HER2": "negative", "EGFR": null, "FLT3": null}}, "ecog": 1, "prior_treatment_lines": 2, "age": 62, "metastatic": true, "extraction_notes": null}}

EXAMPLE 3:
Input: Patient with lung cancer, KRAS G12C, some prior treatment.
Output: {{"cancer_type": "Lung cancer", "icd10_code": "C34.90", "biomarkers": {{"BRCA1": null, "BRCA2": null, "PD_L1": null, "KRAS": "mutant", "HER2": null, "EGFR": null, "FLT3": null}}, "ecog": null, "prior_treatment_lines": 1, "age": null, "metastatic": null, "extraction_notes": "prior_treatment_lines inferred as 1 from vague description"}}

Now extract from:
Input: {patient_text}
Output:"""


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """
    Pull the first JSON object from LLM output.
    Handles: JS // comments, truncated output (missing closing brace), whitespace.
    """
    text = text.strip()
    # Find opening brace
    start = text.find('{')
    if start == -1:
        return None
    raw = text[start:]
    raw = re.sub(r'//[^\n]*', '', raw)  # strip JS comments

    # Try parsing as-is first
    try:
        # Find the last closing brace to handle trailing text
        end = raw.rfind('}')
        if end != -1:
            return json.loads(raw[:end + 1])
    except json.JSONDecodeError:
        pass

    # Model truncated — attempt to close the object and parse what we have
    try:
        return json.loads(raw + '}')
    except json.JSONDecodeError:
        pass

    return None


def _normalize_biomarkers(bm_raw: dict) -> dict:
    """Handle key variants the model sometimes produces (PD-L1 vs PD_L1)."""
    normalized = dict(bm_raw)
    if "PD-L1" in normalized:
        normalized["PD_L1"] = normalized.pop("PD-L1")
    return {k: v for k, v in normalized.items() if k in BiomarkerStatus.model_fields}


def parse_patient(
    patient_text: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> PatientProfile:
    """
    Extract a structured PatientProfile from free-text patient description.

    Always returns a PatientProfile — missing fields are None, never raises.
    Validation errors (e.g. ECOG=9) are caught and logged; the field is set to None.
    """
    prompt = _PROMPT_TEMPLATE.format(patient_text=patient_text)

    try:
        resp = requests.post(
            ollama_url,
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=60,
        )
        resp.raise_for_status()
        raw_text = resp.json()["response"]
    except requests.RequestException as e:
        log.error(f"Ollama request failed: {e}")
        return PatientProfile()

    raw_dict = _extract_json(raw_text)
    if raw_dict is None:
        log.warning(f"JSON extraction failed. Raw response: {raw_text[:300]}")
        return PatientProfile()

    bm_raw = _normalize_biomarkers(raw_dict.get("biomarkers") or {})

    try:
        biomarkers = BiomarkerStatus(**bm_raw)
    except ValidationError:
        biomarkers = BiomarkerStatus()

    # Build profile field by field — coerce bad values to None rather than crash
    def safe_int(val):
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    try:
        profile = PatientProfile(
            cancer_type=raw_dict.get("cancer_type"),
            icd10_code=raw_dict.get("icd10_code"),
            biomarkers=biomarkers,
            ecog=safe_int(raw_dict.get("ecog")),
            prior_treatment_lines=safe_int(raw_dict.get("prior_treatment_lines")),
            age=safe_int(raw_dict.get("age")),
            metastatic=raw_dict.get("metastatic"),
            extraction_notes=raw_dict.get("extraction_notes"),
        )
    except ValidationError as e:
        log.warning(f"Pydantic validation error (partial recovery): {e}")
        # Re-build with only safe fields
        profile = PatientProfile(
            cancer_type=raw_dict.get("cancer_type"),
            biomarkers=biomarkers,
            age=safe_int(raw_dict.get("age")),
            extraction_notes=f"Validation error during extraction: {str(e)[:100]}",
        )

    log.info(f"Extracted: {profile.cancer_type} | ECOG={profile.ecog} | age={profile.age} | metastatic={profile.metastatic}")
    return profile


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_cases = [
        # Simple
        "58-year-old female with metastatic triple-negative breast cancer, BRCA2 mutation positive, 2 prior chemotherapy lines, ECOG 1.",
        # Complex — multiple biomarkers
        "67-year-old male with metastatic NSCLC, KRAS G12C mutation, PD-L1 TPS 60%, HER2 negative, 1 prior platinum-based chemo, ECOG 1, brain metastases.",
        # Missing fields
        "Patient with lung cancer, KRAS G12C, some prior treatment.",
    ]

    input_text = sys.argv[1] if len(sys.argv) > 1 else test_cases[0]

    if input_text == "--all":
        for text in test_cases:
            print(f"\nInput: {text}")
            profile = parse_patient(text)
            print(profile.model_dump_json(indent=2))
    else:
        profile = parse_patient(input_text)
        print(profile.model_dump_json(indent=2))
