"""
LangGraph state machine for TrialCompass.

Linear pipeline: START → parse → retrieve → explain → END

State carries raw_input through all nodes. Each node writes to the next field.
Errors are captured in state["error"] and cause downstream nodes to no-op —
no conditional edges, no explicit error routing. The error field is the signal.

Node functions are thin wrappers around the three agent modules so that the
graph stays decoupled from agent internals.
"""

from typing import Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.agents.explanation_agent import explain_matches
from src.agents.parser_agent import parse_patient
from src.agents.retrieval_agent import retrieve_and_rerank


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class TrialMatchState(TypedDict):
    raw_input: str
    patient_profile: dict
    retrieved_trials: list
    explained_matches: list
    error: Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_node(state: TrialMatchState) -> TrialMatchState:
    """Parse free-text patient description into a structured profile dict."""
    try:
        profile = parse_patient(state["raw_input"])
        return {**state, "patient_profile": profile.model_dump()}
    except Exception as exc:
        return {**state, "error": f"parse_node failed: {exc}"}


def retrieve_node(state: TrialMatchState) -> TrialMatchState:
    """Retrieve and rerank trials for the parsed patient profile."""
    if state.get("error"):
        return state
    try:
        trials = retrieve_and_rerank(state["patient_profile"])
        return {**state, "retrieved_trials": trials}
    except Exception as exc:
        return {**state, "error": f"retrieve_node failed: {exc}"}


def explain_node(state: TrialMatchState) -> TrialMatchState:
    """Run chain-of-thought eligibility reasoning over retrieved trials."""
    if state.get("error"):
        return state
    try:
        explained = explain_matches(state["patient_profile"], state["retrieved_trials"])
        return {**state, "explained_matches": explained}
    except Exception as exc:
        return {**state, "error": f"explain_node failed: {exc}"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(TrialMatchState)

    g.add_node("parse", parse_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("explain", explain_node)

    g.add_edge(START, "parse")
    g.add_edge("parse", "retrieve")
    g.add_edge("retrieve", "explain")
    g.add_edge("explain", END)

    return g.compile()


# Compiled graph — import this in run_pipeline.py
pipeline = build_graph()
