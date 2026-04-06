"""
Preprocess raw NDJSON trial records into flat text chunks and store in SQLite.

Schema design and null rates documented in notebooks/01_data_exploration.ipynb.
"""

import json
import logging
import re
from pathlib import Path

import sqlite_utils

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/trialcompass.db")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# --- Field extractors (paths confirmed in 01_data_exploration.ipynb) ----------

def _get(d: dict, *keys, default=None):
    """Safe nested get."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


def extract_fields(study: dict) -> dict:
    proto = study.get("protocolSection", {})

    nct_id = _get(proto, "identificationModule", "nctId") or ""
    brief_title = _get(proto, "identificationModule", "briefTitle") or ""
    official_title = _get(proto, "identificationModule", "officialTitle") or ""

    phases = _get(proto, "designModule", "phases") or []
    phase_str = "|".join(phases) if phases else ""

    conditions = _get(proto, "conditionsModule", "conditions") or []
    conditions_str = "|".join(conditions) if conditions else ""

    overall_status = _get(proto, "statusModule", "overallStatus") or ""

    eligibility_text = _get(proto, "eligibilityModule", "eligibilityCriteria") or ""
    min_age = _get(proto, "eligibilityModule", "minimumAge") or ""
    max_age = _get(proto, "eligibilityModule", "maximumAge") or ""

    sponsor = _get(proto, "sponsorCollaboratorsModule", "leadSponsor", "name") or ""
    brief_summary = _get(proto, "descriptionModule", "briefSummary") or ""

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "phase": phase_str,
        "conditions": conditions_str,
        "overall_status": overall_status,
        "eligibility_text": eligibility_text,
        "min_age": min_age,
        "max_age": max_age,
        "sponsor": sponsor,
        "brief_summary": brief_summary,
        "has_eligibility": bool(eligibility_text.strip()),
    }


def build_chunk(row: dict) -> str:
    """
    Flatten a trial record into a single text string for embedding.
    Designed to maximize signal for biomedical semantic search.
    """
    parts = []
    if row["nct_id"]:        parts.append(f"Trial ID: {row['nct_id']}")
    if row["brief_title"]:   parts.append(f"Title: {row['brief_title']}")
    if row["phase"]:         parts.append(f"Phase: {row['phase'].replace('|', ', ')}")
    if row["conditions"]:    parts.append(f"Conditions: {row['conditions'].replace('|', ', ')}")
    if row["overall_status"]: parts.append(f"Status: {row['overall_status']}")
    if row["sponsor"]:       parts.append(f"Sponsor: {row['sponsor']}")
    if row["brief_summary"]: parts.append(f"Summary: {row['brief_summary'][:500]}")
    if row["eligibility_text"]: parts.append(f"Eligibility:\n{row['eligibility_text']}")
    return "\n".join(parts)


def preprocess(
    raw_path: Path,
    db_path: Path = DB_PATH,
) -> dict:
    """
    Read NDJSON, extract fields, build text chunks, upsert into SQLite.

    Returns a summary dict with counts and null rates.
    """
    db = sqlite_utils.Database(db_path)

    total = 0
    null_eligibility = 0
    rows = []

    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            study = json.loads(line)
            row = extract_fields(study)
            row["chunk_text"] = build_chunk(row)
            rows.append(row)
            total += 1
            if not row["has_eligibility"]:
                null_eligibility += 1

            # Batch upsert every 500 records
            if len(rows) >= 500:
                db["trials"].upsert_all(rows, pk="nct_id")
                rows = []

    if rows:
        db["trials"].upsert_all(rows, pk="nct_id")

    null_rate = null_eligibility / total * 100 if total else 0
    summary = {
        "total_trials": total,
        "null_eligibility": null_eligibility,
        "null_eligibility_pct": round(null_rate, 1),
        "db_path": str(db_path),
    }
    log.info(f"Preprocessed {total:,} trials — eligibility null rate: {null_rate:.1f}%")
    return summary


if __name__ == "__main__":
    import sys
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR / "trials_raw_10000.ndjson"
    summary = preprocess(raw_path)
    print(json.dumps(summary, indent=2))
