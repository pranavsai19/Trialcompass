"""
Fetch up to 60,000 oncology trials from ClinicalTrials.gov API v2 and insert
into the existing SQLite database using INSERT OR IGNORE on nct_id.

Does NOT drop or recreate the trials table. Safe to run against a DB that
already has rows — duplicates are silently skipped.

Differences from fetch_trials.py:
- Processes and upserts inline (no intermediate NDJSON file at 60K scale)
- Exponential backoff on HTTP 429 / transient errors
- Progress printed every 1,000 trials fetched from API
- Stops at max_trials from the API or when pages are exhausted
"""

import json
import logging
import sqlite3
import time
from pathlib import Path

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
DB_PATH = Path("data/trialcompass.db")
PAGE_SIZE = 1000
REQUEST_DELAY = 0.5     # polite baseline delay between pages
MAX_BACKOFF = 64        # seconds — cap on exponential backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field extraction (mirrors preprocess.py exactly)
# ---------------------------------------------------------------------------

def _get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is default:
            return default
    return d


def _extract_fields(study: dict) -> dict:
    proto = study.get("protocolSection", {})

    nct_id        = _get(proto, "identificationModule", "nctId") or ""
    brief_title   = _get(proto, "identificationModule", "briefTitle") or ""
    official_title= _get(proto, "identificationModule", "officialTitle") or ""
    phases        = _get(proto, "designModule", "phases") or []
    conditions    = _get(proto, "conditionsModule", "conditions") or []
    overall_status= _get(proto, "statusModule", "overallStatus") or ""
    eligibility_text = _get(proto, "eligibilityModule", "eligibilityCriteria") or ""
    min_age       = _get(proto, "eligibilityModule", "minimumAge") or ""
    max_age       = _get(proto, "eligibilityModule", "maximumAge") or ""
    sponsor       = _get(proto, "sponsorCollaboratorsModule", "leadSponsor", "name") or ""
    brief_summary = _get(proto, "descriptionModule", "briefSummary") or ""

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "phase": "|".join(phases),
        "conditions": "|".join(conditions),
        "overall_status": overall_status,
        "eligibility_text": eligibility_text,
        "min_age": min_age,
        "max_age": max_age,
        "sponsor": sponsor,
        "brief_summary": brief_summary,
        "has_eligibility": int(bool(eligibility_text.strip())),
    }


def _build_chunk(row: dict) -> str:
    parts = []
    if row["nct_id"]:          parts.append(f"Trial ID: {row['nct_id']}")
    if row["brief_title"]:     parts.append(f"Title: {row['brief_title']}")
    if row["phase"]:           parts.append(f"Phase: {row['phase'].replace('|', ', ')}")
    if row["conditions"]:      parts.append(f"Conditions: {row['conditions'].replace('|', ', ')}")
    if row["overall_status"]:  parts.append(f"Status: {row['overall_status']}")
    if row["sponsor"]:         parts.append(f"Sponsor: {row['sponsor']}")
    if row["brief_summary"]:   parts.append(f"Summary: {row['brief_summary'][:500]}")
    if row["eligibility_text"]:parts.append(f"Eligibility:\n{row['eligibility_text']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQLite insert — INSERT OR IGNORE (never touches existing rows)
# ---------------------------------------------------------------------------

def _insert_batch(con: sqlite3.Connection, rows: list[dict]) -> int:
    """Insert a batch of rows. Returns count actually inserted (not ignored)."""
    cur = con.cursor()
    inserted = 0
    for row in rows:
        try:
            cur.execute(
                """INSERT OR IGNORE INTO trials
                   (nct_id, brief_title, official_title, phase, conditions,
                    overall_status, eligibility_text, min_age, max_age,
                    sponsor, brief_summary, has_eligibility, chunk_text)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["nct_id"], row["brief_title"], row["official_title"],
                    row["phase"], row["conditions"], row["overall_status"],
                    row["eligibility_text"], row["min_age"], row["max_age"],
                    row["sponsor"], row["brief_summary"], row["has_eligibility"],
                    row["chunk_text"],
                ),
            )
            inserted += cur.rowcount
        except sqlite3.Error as e:
            log.warning(f"Insert error for {row.get('nct_id', '?')}: {e}")
    con.commit()
    return inserted


# ---------------------------------------------------------------------------
# API fetch with exponential backoff
# ---------------------------------------------------------------------------

def _fetch_page(query: str, page_token: str | None, page_size: int) -> dict:
    params = {
        "query.cond": query,
        "pageSize": page_size,
        "format": "json",
        "countTotal": "true",
    }
    if page_token:
        params["pageToken"] = page_token

    backoff = 1
    while True:
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                log.warning(f"Rate limited (429). Backing off {backoff}s.")
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning(f"Request error: {e}. Backing off {backoff}s.")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_and_store(
    query: str = "cancer",
    max_trials: int = 60_000,
    db_path: Path = DB_PATH,
) -> None:
    con = sqlite3.connect(str(db_path))

    # Confirm table exists
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM trials")
    existing = cur.fetchone()[0]
    log.info(f"Existing rows in SQLite: {existing:,}")

    total_fetched_api = 0   # raw count from API pages
    total_inserted    = 0   # net new rows inserted (OR IGNORE skips)
    page_token        = None
    page_num          = 0
    last_progress_log = 0

    while total_fetched_api < max_trials:
        page_num += 1
        batch_size = min(PAGE_SIZE, max_trials - total_fetched_api)

        data = _fetch_page(query, page_token, batch_size)
        studies = data.get("studies", [])

        if page_num == 1:
            log.info(f"API total available: {data.get('totalCount', 'unknown'):,}")

        if not studies:
            log.info("Empty page — API exhausted.")
            break

        # Extract, chunk, and insert
        rows = []
        for study in studies:
            row = _extract_fields(study)
            if not row["nct_id"]:
                continue
            row["chunk_text"] = _build_chunk(row)
            rows.append(row)

        n_inserted = _insert_batch(con, rows)
        total_fetched_api += len(studies)
        total_inserted    += n_inserted

        # Progress every 1,000 fetched from API
        milestone = (total_fetched_api // 1000) * 1000
        if milestone > last_progress_log:
            log.info(
                f"[page {page_num}] fetched from API: {total_fetched_api:,} | "
                f"net new inserted: {total_inserted:,} | "
                f"this page: {len(studies)} fetched, {n_inserted} new"
            )
            last_progress_log = milestone

        page_token = data.get("nextPageToken")
        if not page_token:
            log.info("No nextPageToken — API exhausted.")
            break

        time.sleep(REQUEST_DELAY)

    cur.execute("SELECT COUNT(*) FROM trials")
    final_count = cur.fetchone()[0]
    con.close()

    log.info(f"Fetch complete.")
    log.info(f"  Fetched from API:  {total_fetched_api:,}")
    log.info(f"  Net new inserted:  {total_inserted:,}")
    log.info(f"  Total in SQLite:   {final_count:,}")


if __name__ == "__main__":
    fetch_and_store(max_trials=60_000)
