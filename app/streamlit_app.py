import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import streamlit as st

from src.orchestration.graph import pipeline, TrialMatchState

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TrialCompass",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("System Info")
    st.markdown(
        """
**Reasoning LLM**
`llama3` via Ollama (local)

**Retrieval encoder**
`all-MiniLM-L6-v2`

**Re-ranker**
`ms-marco-MiniLM-L-12-v2`

**Corpus**
10,000 oncology trials
(ClinicalTrials.gov)
        """
    )
    st.divider()
    st.markdown(
        "**Source code**  \n"
        "[github.com/pranavsai19/Trialcompass](https://github.com/pranavsai19/Trialcompass)"
    )
    st.divider()
    st.caption(
        "Pipeline: parse → FAISS retrieve → cross-encoder rerank → LLM explain. "
        "⚠️ flags are set when the model expresses uncertainty or eligibility text is too short."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("TrialCompass")
st.markdown("**Agentic clinical trial matching for oncology patients**")
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

patient_text = st.text_area(
    "Patient Profile",
    height=130,
    placeholder=(
        "Example: 58-year-old female, stage IV NSCLC, EGFR exon 19 deletion, "
        "ECOG performance status 1, failed carboplatin + pemetrexed, metastatic disease"
    ),
)

run_button = st.button("Find Matching Trials", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Verdict badge helper
# ---------------------------------------------------------------------------

_VERDICT_STYLES = {
    "ELIGIBLE": "background:#1a7a4a;color:white;padding:2px 10px;border-radius:4px;font-weight:bold;",
    "INELIGIBLE": "background:#b52b2b;color:white;padding:2px 10px;border-radius:4px;font-weight:bold;",
    "UNCERTAIN": "background:#b58a00;color:white;padding:2px 10px;border-radius:4px;font-weight:bold;",
}


def verdict_badge(verdict: str) -> str:
    style = _VERDICT_STYLES.get(verdict, "background:#555;color:white;padding:2px 10px;border-radius:4px;")
    return f'<span style="{style}">{verdict}</span>'


# ---------------------------------------------------------------------------
# Pipeline execution and results
# ---------------------------------------------------------------------------

if run_button:
    if not patient_text.strip():
        st.warning("Enter a patient profile before running.")
        st.stop()

    with st.spinner("Running pipeline — parsing profile, retrieving trials, generating explanations…"):
        initial: TrialMatchState = {
            "raw_input": patient_text.strip(),
            "patient_profile": {},
            "retrieved_trials": [],
            "explained_matches": [],
            "error": None,
        }
        try:
            result = pipeline.invoke(initial)
        except Exception as exc:
            st.error(f"Pipeline crashed unexpectedly: {exc}")
            st.stop()

    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
        st.stop()

    profile = result["patient_profile"]
    matches = result["explained_matches"]

    # --- Patient summary strip ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cancer Type", profile.get("cancer_type") or "—")
    col2.metric("Age", profile.get("age") or "—")
    col3.metric("ECOG", profile.get("ecog") if profile.get("ecog") is not None else "—")
    col4.metric("Metastatic", "Yes" if profile.get("metastatic") else ("No" if profile.get("metastatic") is False else "—"))

    if profile.get("extraction_notes"):
        st.caption(f"Parser note: {profile['extraction_notes']}")

    st.divider()

    if not matches:
        st.info("No matches returned from the pipeline.")
        st.stop()

    # --- Summary table ---
    st.subheader(f"Top {len(matches)} Matching Trials")

    header_cols = st.columns([1, 2, 5, 2, 2, 2])
    for col, label in zip(header_cols, ["Rank", "NCT ID", "Title", "Verdict", "Confidence", "Review"]):
        col.markdown(f"**{label}**")

    st.markdown("---")

    for m in matches:
        rank = m.get("rank", "")
        nct_id = m.get("nct_id", "")
        title = m.get("brief_title") or ""
        verdict = m.get("verdict", "UNCERTAIN")
        confidence = m.get("confidence", "—")
        review_flag = m.get("human_review_flag", False)

        display_title = (title[:65] + "…") if len(title) > 65 else title
        review_icon = "⚠️" if review_flag else ""

        row_cols = st.columns([1, 2, 5, 2, 2, 2])
        row_cols[0].write(rank)
        row_cols[1].write(nct_id)
        row_cols[2].markdown(f"{review_icon} {display_title}")
        row_cols[3].markdown(verdict_badge(verdict), unsafe_allow_html=True)
        row_cols[4].write(confidence)
        row_cols[5].write("⚠️ Yes" if review_flag else "—")

        # Expandable detail panel
        with st.expander(f"Details — {nct_id}"):
            reasoning = m.get("reasoning", "")
            inclusion_met = m.get("inclusion_met") or []
            inclusion_failed = m.get("inclusion_failed") or []
            exclusion_flags = m.get("exclusion_flags") or []
            provenance = m.get("provenance") or {}
            flag_reason = m.get("flag_reason")

            if flag_reason:
                st.warning(f"Review flag reason: {flag_reason}")

            st.markdown(f"**Reasoning**  \n{reasoning or '*No reasoning returned.*'}")

            if inclusion_met:
                st.markdown("**Inclusion criteria met**")
                for c in inclusion_met:
                    prov_quote = provenance.get(c, "") if isinstance(provenance, dict) else ""
                    if prov_quote and prov_quote != "NOT FOUND IN TEXT":
                        st.markdown(f"- {c}  \n  > *{prov_quote}*")
                    else:
                        st.markdown(f"- {c}")

            if inclusion_failed:
                st.markdown("**Inclusion criteria not met**")
                for c in inclusion_failed:
                    st.markdown(f"- {c}")

            if exclusion_flags:
                st.markdown("**Exclusion flags**")
                for c in exclusion_flags:
                    st.markdown(f"- {c}")

            st.markdown(
                f"Bi-encoder score: `{m.get('bi_encoder_score', 0):.4f}` &nbsp;|&nbsp; "
                f"Cross-encoder score: `{m.get('cross_encoder_score', 0):.4f}`",
                unsafe_allow_html=True,
            )

        st.markdown("---")
