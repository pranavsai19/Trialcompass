"""
CLI runner for the TrialCompass LangGraph pipeline.

Usage:
    python src/orchestration/run_pipeline.py --patient "58yo female, stage IV NSCLC, ..."

Prints a ranked table of matched trials: rank, NCT ID, title (truncated),
verdict, confidence, and human_review_flag.
"""

import argparse
import sys

from tabulate import tabulate

from src.orchestration.graph import pipeline, TrialMatchState


def run(patient_text: str) -> None:
    initial_state: TrialMatchState = {
        "raw_input": patient_text,
        "patient_profile": {},
        "retrieved_trials": [],
        "explained_matches": [],
        "error": None,
    }

    final_state = pipeline.invoke(initial_state)

    if final_state.get("error"):
        print(f"\n[ERROR] Pipeline aborted: {final_state['error']}", file=sys.stderr)
        sys.exit(1)

    profile = final_state["patient_profile"]
    print(f"\nPatient: {profile.get('cancer_type', 'unknown')} | "
          f"age={profile.get('age')} | ECOG={profile.get('ecog')} | "
          f"metastatic={profile.get('metastatic')}")
    print()

    matches = final_state["explained_matches"]
    if not matches:
        print("No matches returned.")
        return

    rows = []
    for m in matches:
        title = (m.get("brief_title") or "")[:50]
        rows.append([
            m.get("rank", ""),
            m.get("nct_id", ""),
            title,
            m.get("verdict", ""),
            m.get("confidence", ""),
            "YES" if m.get("human_review_flag") else "no",
        ])

    headers = ["Rank", "NCT ID", "Title", "Verdict", "Confidence", "Review?"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="TrialCompass — clinical trial matching pipeline")
    parser.add_argument("--patient", required=True, help="Free-text patient description")
    args = parser.parse_args()
    run(args.patient)


if __name__ == "__main__":
    main()
