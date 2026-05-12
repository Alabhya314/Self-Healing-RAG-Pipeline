"""
src/graph.py
============
Phase 3 — State Machine Wiring for the Self-Healing RAG Pipeline.

This module assembles the LangGraph StateGraph by:
  1. Registering all node functions under clean short names.
  2. Defining conditional-edge routers that implement the self-healing logic.
  3. Wiring static and conditional edges to form the complete state machine.
  4. Compiling the workflow into an invokable ``graph`` object.

Topology (full state machine)
------------------------------

    START
      │
      ▼
  [retrieve] ──────────────────────────────────────────┐
      │                                                  │ (loop-back)
      ▼                                                  │
  [grade_documents]                                      │
      │                                                  │
      ├─ (decide_to_generate) ──────────────────────────┤
      │      │                                           │
      │  docs exist?                              no docs left
      │      │                                           │
      │      ▼                                     [rewrite_query]
      │  [generate]                                      ▲
      │      │                                           │
      │      ▼                                           │
      │  [critic]                                        │
      │      │                                           │
      │      ├─ (grade_generation_v_documents_and_question)
      │             │
      │     ┌───────┼───────────┐
      │  hallucinated?    grounded &        grounded &
      │  (retries left)   irrelevant?       relevant?
      │       │              │                  │
      │  [rewrite_query]  [rewrite_query]     END
      │       │              │
      └───────┴──────────────┘

Usage
-----
>>> from src.graph import graph
>>> result = graph.invoke({
...     "question": "What is self-healing RAG?",
...     "search_query": "What is self-healing RAG?",
...     "retry_count": 0,
... })
>>> print(result["generation"])
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from src.nodes import (
    MAX_RETRIES,
    GraphState,
    critic_node,
    generate_node,
    grade_documents_node,
    retrieve_node,
    rewrite_query_node,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Routing / Conditional-Edge Functions
# ===========================================================================

def decide_to_generate(state: GraphState) -> str:
    """
    Conditional edge: ``grade_documents`` → ``generate`` OR ``rewrite_query``.

    Called after the document grader has filtered the retrieved context.
    Inspects the surviving document list and routes accordingly:

    Decision logic
    --------------
    - ``documents`` is non-empty  →  ``"generate"``
        At least one relevant document survived grading; proceed to answer
        generation.

    - ``documents`` is empty AND ``retry_count < MAX_RETRIES``  →  ``"rewrite_query"``
        No relevant context was found.  Trigger the self-healing loop by
        rewriting the query for a better retrieval attempt.

    - ``documents`` is empty AND ``retry_count >= MAX_RETRIES``  →  ``END``
        All retry budget is exhausted.  Terminate gracefully rather than
        looping infinitely.  The ``generation`` field will be empty; the
        caller is responsible for handling this edge case.

    Parameters
    ----------
    state : GraphState
        Current pipeline state.

    Returns
    -------
    str
        LangGraph node name to route to next.
    """
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if documents:
        logger.info(
            "🟢 [router:decide_to_generate] %d relevant doc(s) found — routing to GENERATE",
            len(documents),
        )
        return "generate"

    if retry_count < MAX_RETRIES:
        logger.warning(
            "🔴 [router:decide_to_generate] No relevant docs — "
            "routing to REWRITE_QUERY (attempt %d/%d)",
            retry_count + 1,
            MAX_RETRIES,
        )
        return "rewrite_query"

    logger.error(
        "⛔ [router:decide_to_generate] No relevant docs AND max retries (%d) reached — "
        "terminating pipeline",
        MAX_RETRIES,
    )
    return END


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    """
    Conditional edge: ``critic`` → ``rewrite_query`` OR ``END``.

    Called after the critic node has evaluated the generation on two axes:
      1. **Groundedness** (``is_hallucination``) — Is every claim in the
         answer supported by the retrieved documents?
      2. **Answer Relevance** (``is_relevant``) — Does the answer actually
         address what the user asked?

    Decision matrix
    ---------------
    ┌─────────────────────┬──────────────┬──────────────────────────────────┐
    │ is_hallucination    │ is_relevant  │ Route                            │
    ├─────────────────────┼──────────────┼──────────────────────────────────┤
    │ True                │ any          │ rewrite_query  (if retries left) │
    │ True                │ any          │ END            (max retries hit) │
    │ False               │ False        │ rewrite_query  (if retries left) │
    │ False               │ False        │ END            (max retries hit) │
    │ False               │ True         │ END  ✅  (perfect answer)        │
    └─────────────────────┴──────────────┴──────────────────────────────────┘

    Parameters
    ----------
    state : GraphState
        Current pipeline state — must contain ``is_hallucination``,
        ``is_relevant``, and ``retry_count``.

    Returns
    -------
    str
        LangGraph node name to route to next.
    """
    is_hallucination: bool = state.get("is_hallucination", True)
    is_relevant: bool = state.get("is_relevant", False)
    retry_count: int = state.get("retry_count", 0)

    # ---- Case 1: Answer is grounded AND relevant → done ------------------
    if not is_hallucination and is_relevant:
        logger.info(
            "✅ [router:critic] Generation is grounded and relevant — routing to END"
        )
        return END

    # ---- Case 2: Failure (hallucinated OR irrelevant) --------------------
    failure_reason = (
        "hallucinated answer" if is_hallucination else "grounded but irrelevant answer"
    )

    if retry_count < MAX_RETRIES:
        logger.warning(
            "🔄 [router:critic] %s detected — "
            "routing to REWRITE_QUERY for self-healing (attempt %d/%d)",
            failure_reason,
            retry_count + 1,
            MAX_RETRIES,
        )
        return "rewrite_query"

    # ---- Case 3: Max retries exhausted — best-effort termination ---------
    logger.error(
        "⛔ [router:critic] %s detected AND max retries (%d) reached — "
        "terminating with best-effort answer",
        failure_reason,
        MAX_RETRIES,
    )
    return END


# ===========================================================================
# Graph Construction
# ===========================================================================

def build_graph() -> StateGraph:
    """
    Assemble and compile the Self-Healing RAG ``StateGraph``.

    Node registration uses short, human-readable names (``retrieve``,
    ``grade_documents``, ``generate``, ``critic``, ``rewrite_query``) that
    appear in LangGraph traces and logs, keeping console output readable.

    Returns
    -------
    StateGraph (compiled)
        A compiled LangGraph graph ready to be invoked.  Thread-safe and
        pickleable — suitable for serving behind a FastAPI endpoint.
    """
    workflow = StateGraph(GraphState)

    # ------------------------------------------------------------------
    # 1. Register nodes
    # ------------------------------------------------------------------
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("rewrite_query", rewrite_query_node)

    # ------------------------------------------------------------------
    # 2. Set entry point
    # ------------------------------------------------------------------
    workflow.add_edge(START, "retrieve")

    # ------------------------------------------------------------------
    # 3. Static edges
    # ------------------------------------------------------------------

    # retrieve  →  grade_documents  (always)
    workflow.add_edge("retrieve", "grade_documents")

    # generate  →  critic  (always — critic evaluates every generation)
    workflow.add_edge("generate", "critic")

    # rewrite_query  →  retrieve  (self-healing loop-back)
    workflow.add_edge("rewrite_query", "retrieve")

    # ------------------------------------------------------------------
    # 4. Conditional edges
    # ------------------------------------------------------------------

    # After grading: go to generate if docs exist, else rewrite or stop
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            END: END,
        },
    )

    # After critic: end if grounded+relevant, else rewrite or stop
    workflow.add_conditional_edges(
        "critic",
        grade_generation_v_documents_and_question,
        {
            END: END,
            "rewrite_query": "rewrite_query",
        },
    )

    # ------------------------------------------------------------------
    # 5. Compile
    # ------------------------------------------------------------------
    logger.info("🔧 [build_graph] Compiling Self-Healing RAG state machine…")
    compiled = workflow.compile()
    logger.info("✅ [build_graph] Graph compiled successfully")
    return compiled


# ===========================================================================
# Module-level singleton
# ===========================================================================

# The compiled graph — import this directly in main.py or FastAPI:
#
#   from src.graph import graph
#   result = graph.invoke({
#       "question": "Your question here",
#       "search_query": "Your question here",
#       "retry_count": 0,
#   })
graph = build_graph()
