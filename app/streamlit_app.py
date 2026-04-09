import sys
import os
import time

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
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.verdict-eligible {
    background: #10B981; color: white; padding: 2px 12px;
    border-radius: 4px; font-weight: 700; font-size: 12px;
}
.verdict-ineligible {
    background: #EF4444; color: white; padding: 2px 12px;
    border-radius: 4px; font-weight: 700; font-size: 12px;
}
.verdict-uncertain {
    background: #F59E0B; color: white; padding: 2px 12px;
    border-radius: 4px; font-weight: 700; font-size: 12px;
}
.conf-high   { background: #D1FAE5; color: #065F46; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
.conf-medium { background: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
.conf-low    { background: #FEE2E2; color: #991B1B; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
.trial-card-eligible  { border-left: 5px solid #10B981; padding: 14px 18px; background: #F9FAFB; border-radius: 0 8px 8px 0; margin-bottom: 10px; }
.trial-card-ineligible{ border-left: 5px solid #EF4444; padding: 14px 18px; background: #F9FAFB; border-radius: 0 8px 8px 0; margin-bottom: 10px; }
.trial-card-uncertain { border-left: 5px solid #F59E0B; padding: 14px 18px; background: #F9FAFB; border-radius: 0 8px 8px 0; margin-bottom: 10px; }
.prov-tag { background: #F3F4F6; color: #6B7280; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-right: 4px; font-style: italic; }
.section-header { font-size: 13px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧬 Patient Profile")
    patient_text = st.text_area(
        label="Free-text patient description",
        height=200,
        placeholder=(
            "68 year old female with platinum-resistant BRCA1-mutant ovarian cancer, "
            "2 prior lines of therapy, ECOG 1, no prior PARP inhibitor"
        ),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("## Pipeline Settings")
    st.markdown(
        "**LLM:** `LLaMA 3` (local, Ollama)  \n"
        "**Embeddings:** `PubMedBERT`  \n"
        "**Re-ranker:** `MS-MARCO MiniLM-L-6-v2`"
    )
    n_results = st.slider("Max results", min_value=5, max_value=10, value=10)

    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        "Three-stage hybrid retrieval pipeline  \n"
        "SQL pre-filter → FAISS bi-encoder → Cross-encoder rerank  \n\n"
        "[![GitHub](https://img.shields.io/badge/GitHub-pranavsai19-black?logo=github)](https://github.com/pranavsai19/Trialcompass)"
    )

    st.markdown("---")
    run_button = st.button(
        "🔍 Find Matching Trials",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <h1 style="margin-bottom:2px;">🧬 TrialCompass</h1>
    <p style="color:#6B7280;margin-top:0;font-size:16px;">
    Agentic RAG for Oncology Clinical Trial Matching
    </p>
    <p style="color:#9CA3AF;font-size:13px;margin-top:-8px;">
    64,920 trials &nbsp;·&nbsp; PubMedBERT retrieval &nbsp;·&nbsp; LLaMA 3 reasoning &nbsp;·&nbsp; 3-stage hybrid pipeline
    </p>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Pre-search state — metrics + architecture diagram
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    col1, col2, col3 = st.columns(3)
    col1.metric("Trials in Corpus", "64,920")
    col2.metric("Retrieval Stages", "3")
    col3.metric("Benchmark P@5", "0.040")
    st.markdown("")
    arch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "architecture.png")
    if os.path.exists(arch_path):
        st.image(arch_path, caption="TrialCompass Pipeline Architecture", use_column_width=True)
    else:
        st.info("Architecture diagram not found at docs/architecture.png")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_button:
    if not patient_text.strip():
        st.warning("Enter a patient profile before running.")
        st.stop()

    t_total_start = time.time()

    # ── Step 1: Parse ──────────────────────────────────────────────────────
    with st.status("🔍 Parsing patient profile...", expanded=True) as status_parse:
        t0 = time.time()
        initial: TrialMatchState = {
            "raw_input": patient_text.strip(),
            "patient_profile": {},
            "retrieved_trials": [],
            "explained_matches": [],
            "error": None,
        }
        try:
            from src.agents.parser_agent import parse_patient
            profile_obj = parse_patient(patient_text.strip())
            profile_dict = profile_obj.model_dump()
            parse_time = time.time() - t0
            st.markdown(f"**Cancer type:** {profile_dict.get('cancer_type') or '—'}")
            st.markdown(f"**Age:** {profile_dict.get('age') or '—'} &nbsp;|&nbsp; **ECOG:** {profile_dict.get('ecog') if profile_dict.get('ecog') is not None else '—'} &nbsp;|&nbsp; **Metastatic:** {'Yes' if profile_dict.get('metastatic') else ('No' if profile_dict.get('metastatic') is False else '—')}", unsafe_allow_html=True)
            bm = profile_dict.get("biomarkers") or {}
            active_bm = [f"{k}: {v}" for k, v in bm.items() if v and v not in ("None", "null")]
            if active_bm:
                st.markdown(f"**Biomarkers:** {', '.join(active_bm)}")
            if profile_dict.get("extraction_notes"):
                st.caption(f"Parser note: {profile_dict['extraction_notes']}")
            st.caption(f"Parse time: {parse_time:.1f}s")
            status_parse.update(label=f"✅ Profile parsed ({parse_time:.1f}s)", state="complete")
        except Exception as exc:
            err_str = str(exc)
            if "Connection refused" in err_str or "connect" in err_str.lower():
                st.error("⚠️ Ollama is not running. Start it with: `ollama serve`")
            else:
                st.error(f"Parser failed: {exc}")
            st.stop()

    # ── Step 2: Full pipeline (retrieve + explain) ─────────────────────────
    with st.status("📊 Running hybrid retrieval + reasoning...", expanded=True) as status_main:
        st.markdown("**Stage 1 — SQL pre-filter:** narrowing 64,920 trials...")
        st.markdown("**Stage 2 — PubMedBERT FAISS bi-encoder:** top 500 candidates...")
        st.markdown("**Stage 3 — Cross-encoder rerank:** scoring top candidates...")
        st.markdown("**Explanation agent:** chain-of-thought reasoning over top 10 trials (~20s/trial)...")

        t1 = time.time()
        initial["patient_profile"] = profile_dict

        try:
            result = pipeline.invoke(initial)
        except Exception as exc:
            err_str = str(exc)
            if "Connection refused" in err_str or "connect" in err_str.lower():
                st.error("⚠️ Ollama is not running. Start it with: `ollama serve`")
            else:
                st.error(f"Pipeline crashed: {exc}")
            st.stop()

        pipeline_time = time.time() - t1

        if result.get("error"):
            st.error(f"Pipeline error: {result['error']}")
            st.stop()

        status_main.update(
            label=f"✅ Retrieval + reasoning complete ({pipeline_time:.0f}s)",
            state="complete",
        )

    t_total = time.time() - t_total_start

    # Store in session state so results persist on rerun
    st.session_state["results"] = result
    st.session_state["total_time"] = t_total

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if "results" in st.session_state:
    result = st.session_state["results"]
    t_total = st.session_state.get("total_time", 0)

    profile = result.get("patient_profile", {})
    matches = result.get("explained_matches", [])

    if not matches:
        st.info("No matches returned from the pipeline.")
        st.stop()

    # Summary metrics row
    n_eligible   = sum(1 for m in matches if m.get("verdict") == "ELIGIBLE")
    n_ineligible = sum(1 for m in matches if m.get("verdict") == "INELIGIBLE")
    n_flagged    = sum(1 for m in matches if m.get("human_review_flag"))

    st.divider()
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Total Eligible",    n_eligible)
    sm2.metric("Total Ineligible",  n_ineligible)
    sm3.metric("Flagged for Review", n_flagged)
    sm4.metric("Total Time",        f"{t_total:.0f}s")

    st.divider()
    st.subheader(f"Top {len(matches)} Trial Matches")

    # ---------------------------------------------------------------------------
    # Per-trial cards
    # ---------------------------------------------------------------------------
    for m in matches:
        rank        = m.get("rank", "")
        nct_id      = m.get("nct_id", "")
        title       = m.get("brief_title") or ""
        verdict     = m.get("verdict", "UNCERTAIN")
        confidence  = m.get("confidence", "—")
        review_flag = m.get("human_review_flag", False)
        flag_reason = m.get("flag_reason", "")
        reasoning   = m.get("reasoning", "")
        inclusion_met    = m.get("inclusion_met") or []
        inclusion_failed = m.get("inclusion_failed") or []
        exclusion_flags  = m.get("exclusion_flags") or []
        provenance       = m.get("provenance") or []
        phase            = m.get("phase") or ""
        conditions       = m.get("conditions") or ""
        bi_score         = m.get("bi_encoder_score", 0)
        ce_score         = m.get("ce_score", 0)

        # Verdict badge HTML
        verdict_css = {
            "ELIGIBLE":   "verdict-eligible",
            "INELIGIBLE": "verdict-ineligible",
            "UNCERTAIN":  "verdict-uncertain",
        }.get(verdict, "verdict-uncertain")

        conf_css = {
            "HIGH":   "conf-high",
            "MEDIUM": "conf-medium",
            "LOW":    "conf-low",
        }.get(confidence, "conf-medium")

        card_css = {
            "ELIGIBLE":   "trial-card-eligible",
            "INELIGIBLE": "trial-card-ineligible",
            "UNCERTAIN":  "trial-card-uncertain",
        }.get(verdict, "trial-card-uncertain")

        ct_link = f"https://clinicaltrials.gov/study/{nct_id}"

        flag_html = ""
        if review_flag:
            flag_html = '<span style="color:#F59E0B;font-size:12px;font-weight:600;">⚠️ Flagged for human review</span>'

        meta_parts = []
        if phase:
            meta_parts.append(f"Phase: {phase}")
        if conditions:
            meta_parts.append(f"Conditions: {conditions[:80]}")
        meta_str = "  ·  ".join(meta_parts) if meta_parts else ""

        st.markdown(
            f"""
            <div class="{card_css}">
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:4px;">
                <span style="font-size:13px;color:#6B7280;font-weight:600;">#{rank}</span>
                <span style="font-weight:700;font-size:14px;">{nct_id}</span>
                <span class="{verdict_css}">{verdict}</span>
                <span class="{conf_css}">{confidence}</span>
                {flag_html}
              </div>
              <div style="font-size:13px;color:#111827;margin-bottom:4px;">{title}</div>
              <div style="font-size:11px;color:#9CA3AF;margin-bottom:4px;">{meta_str}</div>
              <a href="{ct_link}" target="_blank" style="font-size:11px;color:#3B82F6;">
                🔗 {ct_link}
              </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"View reasoning — {nct_id}"):
            if flag_reason:
                st.warning(f"Review flag: {flag_reason}")

            st.markdown("**Reasoning**")
            st.markdown(reasoning or "*No reasoning returned.*")

            if inclusion_met:
                st.markdown("**Inclusion criteria met**")
                for c in inclusion_met:
                    st.markdown(f"- {c}")

            if inclusion_failed:
                st.markdown("**Inclusion criteria not met**")
                for c in inclusion_failed:
                    st.markdown(f"- {c}")

            if exclusion_flags:
                st.markdown("**Exclusion flags**")
                for c in exclusion_flags:
                    st.markdown(f"- {c}")

            if provenance:
                st.markdown("**Provenance citations**")
                prov_tags = " ".join(
                    f'<span class="prov-tag">"{p}"</span>'
                    for p in provenance
                )
                st.markdown(prov_tags, unsafe_allow_html=True)

            st.markdown(
                f"<span style='font-size:11px;color:#9CA3AF;'>"
                f"Bi-encoder: {bi_score:.4f} &nbsp;·&nbsp; Cross-encoder: {ce_score:.4f}"
                f"</span>",
                unsafe_allow_html=True,
            )
