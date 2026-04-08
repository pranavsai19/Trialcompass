"""
Structured pre-filter for clinical trial retrieval.

The semantic retrieval pipeline (PubMedBERT bi-encoder + cross-encoder reranker)
cannot reliably resolve structured clinical constraints from free text:
  - Age limits: "65 Years" stored as a string field, not part of chunk_text
  - Trial status: RECRUITING vs COMPLETED is a status flag
  - Phase: PHASE2, PHASE3 are categorical
  - Condition: pipe-separated condition strings

This module narrows the 64,920-trial corpus to a structured-constraint-passing
subset BEFORE semantic search runs. Semantic search then runs only on that
subset, yielding better precision without sacrificing the recall of the full
FAISS index on the unconstrained candidates.

Design principle: be permissive on structured filters. A filter that drops the
right trial is worse than no filter at all. When in doubt, keep the trial in:
  - NULL min_age → assume 0 (no lower limit documented)
  - NULL/empty max_age → assume no upper limit
  - Unknown phase → include it (don't filter on missing data)
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np


def _parse_age_years(age_str: str | None) -> int | None:
    """
    Parse ClinicalTrials.gov age strings like '18 Years', '65 Years', '6 Months'.
    Returns the value in years (rounded down). Returns None if unparseable.
    '6 Months' → 0 (treat sub-year ages as 0 for integer year comparisons).
    """
    if not age_str:
        return None
    age_str = age_str.strip()
    m = re.match(r"(\d+)\s*(year|yr)", age_str, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"(\d+)\s*month", age_str, re.IGNORECASE)
    if m:
        return 0
    # bare integer
    m = re.match(r"^(\d+)$", age_str)
    if m:
        return int(m.group(1))
    return None


class StructuredFilter:
    """
    Apply hard structured constraints to narrow the trial corpus before
    semantic search.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database (data/trialcompass.db).
    """

    def __init__(self, db_path: str | Path = "data/trialcompass.db") -> None:
        self.db_path = str(db_path)

    def filter(
        self,
        age: int | None = None,
        phase: list[str] | None = None,
        status: list[str] | None = None,
        conditions_keywords: list[str] | None = None,
    ) -> list[str]:
        """
        Return NCT IDs of trials satisfying all specified constraints.

        Parameters
        ----------
        age : int, optional
            Patient age in years. Keeps trials where:
              min_age (parsed) <= age AND (max_age (parsed) >= age OR max_age missing)
            Trials with unparseable or missing min_age are kept (permissive).
        phase : list[str], optional
            List of acceptable phase strings, e.g. ['PHASE2', 'PHASE3'].
            Trials with empty/NULL phase are kept (permissive).
            A trial passes if ANY of its phases (pipe-separated) matches ANY
            requested phase.
        status : list[str], optional
            List of acceptable overall_status values.
            Defaults to ['RECRUITING', 'ACTIVE_NOT_RECRUITING'] if not provided.
        conditions_keywords : list[str], optional
            Keep trials where the conditions column contains ANY of these
            keywords (case-insensitive substring match). If None, no condition
            filter is applied.

        Returns
        -------
        list[str]
            NCT IDs passing all filters, unordered.
        """
        if status is None:
            status = ["RECRUITING", "ACTIVE_NOT_RECRUITING"]

        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()

        # Build WHERE clause incrementally
        clauses: list[str] = []
        params:  list      = []

        # Status filter (always applied — 100% field coverage)
        placeholders = ",".join("?" * len(status))
        clauses.append(f"overall_status IN ({placeholders})")
        params.extend(status)

        # Conditions keyword filter
        if conditions_keywords:
            kw_clauses = " OR ".join(
                "LOWER(conditions) LIKE ?" for _ in conditions_keywords
            )
            clauses.append(f"({kw_clauses})")
            params.extend(f"%{kw.lower()}%" for kw in conditions_keywords)

        # Phase filter — permissive: keep trials with empty/null phase
        if phase:
            ph_clauses = " OR ".join("phase LIKE ?" for _ in phase)
            clauses.append(f"(phase IS NULL OR phase = '' OR ({ph_clauses}))")
            params.extend(f"%{p}%" for p in phase)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT nct_id, min_age, max_age FROM trials WHERE {where}"
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()

        # Age filter in Python — avoids complex SQL string parsing
        if age is None:
            return [r[0] for r in rows]

        result: list[str] = []
        for nct_id, min_age_str, max_age_str in rows:
            min_age_val = _parse_age_years(min_age_str)
            max_age_val = _parse_age_years(max_age_str)

            # Permissive: if min_age is missing, assume patient is old enough
            if min_age_val is not None and age < min_age_val:
                continue
            # Permissive: if max_age is missing, assume no upper limit
            if max_age_val is not None and age > max_age_val:
                continue
            result.append(nct_id)

        return result

    def filter_to_index_positions(
        self,
        filtered_nct_ids: list[str],
        all_nct_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Map a list of NCT IDs to their integer positions in the FAISS index.

        Parameters
        ----------
        filtered_nct_ids : list[str]
            NCT IDs that passed the structured filter.
        all_nct_ids : np.ndarray
            The full nct_ids_pubmedbert.npy array — position i corresponds to
            FAISS vector i.

        Returns
        -------
        np.ndarray of int64
            Index positions in the FAISS index for the filtered trials.
            Trials in filtered_nct_ids that are not present in the FAISS index
            are silently skipped.
        """
        id_to_pos = {nct: i for i, nct in enumerate(all_nct_ids)}
        positions = [id_to_pos[nct] for nct in filtered_nct_ids if nct in id_to_pos]
        return np.array(positions, dtype=np.int64)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    sf = StructuredFilter("data/trialcompass.db")

    t0 = time.time()
    passed = sf.filter(
        age=65,
        status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
        conditions_keywords=[
            "cancer", "carcinoma", "lymphoma", "leukemia",
            "melanoma", "sarcoma", "myeloma", "glioma",
        ],
    )
    elapsed = time.time() - t0

    print(f"Structured filter results ({elapsed:.3f}s):")
    print(f"  Total corpus : 64,920")
    print(f"  After filter : {len(passed):,}")
    print(f"  Reduction    : {(1 - len(passed)/64920)*100:.1f}%")
    print(f"  First 5 NCT IDs: {passed[:5]}")
