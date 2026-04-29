"""
Enterprise AI Financial Intelligence Platform
LangGraph State Machine — orchestrates Retriever, Analyst, and Synthesizer agents
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents.retriever import retriever_agent
from agents.analyst import analyst_agent
from agents.synthesizer import synthesizer_agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class FinancialQueryState(TypedDict):
    query: str                        # Original user question
    retrieved_chunks: List[dict]      # Pinecone results from Retriever
    verified_figures: dict            # MCP-verified financial data from Analyst
    final_answer: str                 # Synthesized answer
    sources: List[str]                # Source documents cited
    confidence: float                 # Answer confidence score


# ── Node functions ────────────────────────────────────────────────────────────

def run_retriever(state: FinancialQueryState) -> FinancialQueryState:
    logger.info(f"[Retriever] Query: {state['query'][:80]}...")
    chunks = retriever_agent(state["query"])
    logger.info(f"[Retriever] Retrieved {len(chunks)} chunks")
    return {**state, "retrieved_chunks": chunks}


def run_analyst(state: FinancialQueryState) -> FinancialQueryState:
    logger.info("[Analyst] Verifying financial figures via MCP server...")
    verified = analyst_agent(state["query"], state["retrieved_chunks"])
    logger.info(f"[Analyst] Verified {len(verified)} figures")
    return {**state, "verified_figures": verified}


def run_synthesizer(state: FinancialQueryState) -> FinancialQueryState:
    logger.info("[Synthesizer] Assembling final answer...")
    result = synthesizer_agent(
        state["query"],
        state["retrieved_chunks"],
        state["verified_figures"]
    )
    logger.info(f"[Synthesizer] Done. Confidence: {result['confidence']:.2f}")
    return {
        **state,
        "final_answer": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"]
    }


def should_verify(state: FinancialQueryState) -> str:
    """Route to Analyst only if query involves financial figures."""
    financial_keywords = [
        "revenue", "income", "profit", "margin", "eps", "earnings",
        "aum", "growth", "loss", "debt", "cash", "ratio", "return"
    ]
    needs_verification = any(kw in state["query"].lower() for kw in financial_keywords)
    if needs_verification:
        logger.info("[Router] Financial figures detected → Analyst")
        return "analyst"
    logger.info("[Router] No figures needed → Synthesizer")
    return "synthesizer"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(FinancialQueryState)

    graph.add_node("retriever", run_retriever)
    graph.add_node("analyst", run_analyst)
    graph.add_node("synthesizer", run_synthesizer)

    graph.set_entry_point("retriever")
    graph.add_conditional_edges("retriever", should_verify, {
        "analyst": "analyst",
        "synthesizer": "synthesizer"
    })
    graph.add_edge("analyst", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


app = build_graph()


def run_query(query: str) -> dict:
    """
    Main entry point.

    Example:
        result = run_query("What was Microsoft's operating income margin in FY2024?")
        print(result["final_answer"])
    """
    initial_state = FinancialQueryState(
        query=query,
        retrieved_chunks=[],
        verified_figures={},
        final_answer="",
        sources=[],
        confidence=0.0
    )
    return app.invoke(initial_state)


if __name__ == "__main__":
    query = "What was Apple's revenue growth year-over-year in FY2024?"
    result = run_query(query)
    print(f"\nAnswer: {result['final_answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Sources: {result['sources']}")
