"""
Fetch oncology trials from ClinicalTrials.gov API v2 and store raw NDJSON.

Field paths confirmed in notebooks/01_data_exploration.ipynb.
Pagination: nextPageToken chaining. Max pageSize=1000.
"""

import json
import time
import logging
from pathlib import Path

import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
RAW_DIR = Path("data/raw")
DEFAULT_QUERY = "cancer"
PAGE_SIZE = 1000
REQUEST_DELAY = 0.5  # seconds between pages — be polite to the API

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_page(query: str, page_token: str | None, page_size: int) -> dict:
    params = {
        "query.cond": query,
        "pageSize": page_size,
        "format": "json",
        "countTotal": "true",
    }
    if page_token:
        params["pageToken"] = page_token

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_trials(
    query: str = DEFAULT_QUERY,
    max_trials: int = 10_000,
    output_path: Path | None = None,
) -> Path:
    """
    Fetch up to `max_trials` oncology trials and write them as NDJSON.

    Returns the path to the output file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = RAW_DIR / f"trials_raw_{max_trials}.ndjson"

    total_fetched = 0
    page_token = None
    page_num = 0

    with open(output_path, "w") as f:
        while total_fetched < max_trials:
            page_num += 1
            batch_size = min(PAGE_SIZE, max_trials - total_fetched)

            data = fetch_page(query, page_token, batch_size)
            studies = data.get("studies", [])

            if page_num == 1:
                log.info(f"Total available: {data.get('totalCount', 'unknown'):,}")

            for study in studies:
                f.write(json.dumps(study) + "\n")

            total_fetched += len(studies)
            page_token = data.get("nextPageToken")

            log.info(f"Page {page_num}: fetched {len(studies)} studies (total so far: {total_fetched:,})")

            if not page_token or not studies:
                log.info("No more pages.")
                break

            time.sleep(REQUEST_DELAY)

    log.info(f"Done. {total_fetched:,} trials written to {output_path}")
    return output_path


if __name__ == "__main__":
    fetch_trials(max_trials=10_000)
