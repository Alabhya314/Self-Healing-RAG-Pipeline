"""
src/state.py
============
Canonical state schema for the Self-Healing RAG LangGraph pipeline.

Every node in the graph reads from and writes to a ``GraphState`` instance.
Keeping the schema in a dedicated module prevents circular imports and makes
the contract between nodes explicit.
"""

from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    Shared mutable state threaded through every node of the RAG graph.

    Fields
    ------
    question : str
        The current (possibly rewritten) user question fed into the retrieval
        step. Updated by the ``transform_query`` node on each self-healing loop.

    original_question : str
        The raw question exactly as submitted by the user. Never mutated after
        the entry node — used for final logging and RAGAS evaluation.

    documents : List[Document]
        Retrieved context chunks. Overwritten on each retrieval attempt.

    generation : str
        The LLM's answer derived from the graded documents. Empty until the
        ``generate`` node runs successfully.

    hallucination_score : Optional[float]
        Fraction of the generation that is grounded in the retrieved context
        (0.0 = fully hallucinated, 1.0 = fully grounded). Populated by the
        ``grade_generation`` node.

    answer_relevance_score : Optional[float]
        How well the generation answers the original question (RAGAS metric).
        Populated by the ``grade_generation`` node.

    retry_count : int
        Number of self-healing iterations executed so far. Guards against
        infinite loops — the graph terminates if this exceeds ``MAX_RETRIES``.

    web_search_needed : bool
        Flag set by the ``grade_documents`` node when retrieved context is
        deemed insufficient and a fallback web search should be attempted.

    error : Optional[str]
        Human-readable description of any unrecoverable error. When set, the
        graph short-circuits to the ``handle_error`` node.
    """

    # ---- Core RAG fields ------------------------------------------------
    question: str
    original_question: str
    documents: List[Document]
    generation: str

    # ---- Quality signals ------------------------------------------------
    hallucination_score: Optional[float]
    answer_relevance_score: Optional[float]

    # ---- Self-healing control fields ------------------------------------
    retry_count: int
    web_search_needed: bool

    # ---- Error handling -------------------------------------------------
    error: Optional[str]
