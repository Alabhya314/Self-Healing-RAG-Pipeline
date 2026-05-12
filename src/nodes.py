"""
src/nodes.py
============
Phase 2 — Agentic Logic & State Definition for the Self-Healing RAG Pipeline.

This module is the single source of truth for:
  - GraphState  : the shared mutable state that flows through every graph node.
  - Pydantic models (GradeDocuments, HallucinationCritic) for structured LLM output.
  - All node functions that LangGraph will wire together into the pipeline graph.

Node execution order (happy path):
  retrieve_node → grade_documents_node → generate_node → critic_node → END

Self-healing detour (poor retrieval or hallucination detected):
  … → rewrite_query_node → retrieve_node → … (up to MAX_RETRIES times)
"""

from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.config import get_langfuse_handler, get_llm
from src.retriever import get_retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety guard — maximum self-healing iterations before giving up
# ---------------------------------------------------------------------------

MAX_RETRIES: int = 3


# ===========================================================================
# 1. Graph State
# ===========================================================================

class GraphState(TypedDict, total=False):
    """
    Shared state object threaded through every node in the LangGraph pipeline.

    LangGraph reads and merges dicts returned by each node back into this state.
    Declaring ``total=False`` makes every key optional at construction time,
    allowing nodes to initialise only the keys they own.

    Fields
    ------
    question : str
        The original, unmodified user query. Set at graph entry and never
        mutated — used as the authoritative reference for hallucination checking
        and final answer relevance evaluation.

    documents : List[Document]
        Context chunks retrieved from Pinecone and surviving the relevance
        grader. Overwritten on each retrieval attempt.

    generation : str
        The LLM's drafted answer produced by ``generate_node``. Empty string
        until that node runs.

    search_query : str
        The query actually sent to the vector store. Starts as a copy of
        ``question`` and is overwritten by ``rewrite_query_node`` on each
        self-healing loop iteration.

    retry_count : int
        Number of self-healing iterations executed so far. Incremented by
        ``rewrite_query_node``. The graph terminates with an error when this
        value reaches ``MAX_RETRIES``.

    is_hallucination : bool
        Result of the critic node. ``True`` means the generation contains
        claims not grounded in the retrieved documents (i.e., hallucinated).
        ``False`` means the answer is fully supported by context.

    is_relevant : bool
        Result of the critic node's answer-relevance check. ``True`` means the
        generation directly addresses the original ``question``. ``False`` means
        the answer is grounded but off-topic, triggering a self-healing retry.
    """

    question: str
    documents: List[Document]
    generation: str
    search_query: str
    retry_count: int
    is_hallucination: bool
    is_relevant: bool


# ===========================================================================
# 2. Pydantic Models for Structured LLM Output
# ===========================================================================

class GradeDocuments(BaseModel):
    """
    Structured relevance verdict returned by the document grader LLM call.

    Using ``with_structured_output(GradeDocuments)`` forces the LLM to emit
    a clean boolean rather than free-form text, eliminating fragile string
    parsing and ensuring deterministic downstream routing.
    """

    binary_score: bool = Field(
        description=(
            "True if the document is relevant to the question and contains "
            "information useful for answering it. False otherwise."
        )
    )


class HallucinationCritic(BaseModel):
    """
    Structured groundedness verdict returned by the critic LLM call.

    ``is_grounded=True`` means every factual claim in the generation is
    directly supported by the provided context documents.
    ``is_grounded=False`` means the generation contains fabricated or
    unsupported information and must be regenerated.
    """

    is_grounded: bool = Field(
        description=(
            "True if every claim in the AI-generated answer is directly "
            "supported by the provided context documents. "
            "False if the answer contains any hallucinated or unsupported facts."
        )
    )


class AnswerRelevance(BaseModel):
    """
    Structured answer-relevance verdict returned by the critic LLM call.

    Separate from hallucination: an answer can be fully grounded in the
    documents yet still fail to address what the user actually asked.
    ``is_relevant=True`` means the generation directly and completely answers
    the original question.
    """

    is_relevant: bool = Field(
        description=(
            "True if the AI-generated answer directly and completely addresses "
            "the user's original question. "
            "False if the answer is off-topic, partial, or evasive."
        )
    )


# ===========================================================================
# 3. Shared Prompt Templates
# ===========================================================================

_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a strict document relevance grader for a RAG system. "
                "Your sole task is to decide whether a retrieved document "
                "contains information that is useful for answering the user's question. "
                "Grade conservatively: only return True if the document clearly "
                "contributes relevant context."
            ),
        ),
        (
            "human",
            "User question:\n{question}\n\nRetrieved document:\n{document}",
        ),
    ]
)

_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful and precise AI assistant. "
                "Answer the user's question using ONLY the information present "
                "in the provided context. "
                "If the context is insufficient to answer the question, say so explicitly "
                "rather than fabricating an answer."
            ),
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)

_CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a hallucination detection critic for a RAG system. "
                "Your task is to determine whether an AI-generated answer is fully "
                "grounded in the provided source documents. "
                "Return is_grounded=True ONLY if every factual claim in the answer "
                "can be traced back to the documents. "
                "Return is_grounded=False if ANY part of the answer is fabricated "
                "or goes beyond what the documents say."
            ),
        ),
        (
            "human",
            "Source documents:\n{context}\n\nAI-generated answer:\n{generation}",
        ),
    ]
)

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an answer relevance evaluator for a RAG system. "
                "Your task is to decide whether an AI-generated answer actually "
                "addresses the user's original question. "
                "An answer can be factually correct yet still be off-topic or evasive. "
                "Return is_relevant=True ONLY if the answer directly and completely "
                "responds to what the user asked. "
                "Return is_relevant=False if the answer avoids, partially addresses, "
                "or is unrelated to the question."
            ),
        ),
        (
            "human",
            "User question:\n{question}\n\nAI-generated answer:\n{generation}",
        ),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a search query optimisation expert for RAG systems. "
                "Your task is to rewrite a user question into a focused, keyword-rich "
                "search query that will maximise semantic retrieval recall from a "
                "vector database. "
                "Rules:\n"
                "  - Make the query more specific and information-dense.\n"
                "  - Retain the core intent of the original question.\n"
                "  - Output ONLY the rewritten query — no preamble, no explanation."
            ),
        ),
        (
            "human",
            "Original question: {question}\n\nRewritten search query:",
        ),
    ]
)


# ===========================================================================
# 4. Node Functions
# ===========================================================================

def retrieve_node(state: GraphState) -> dict:
    """
    Retrieve context documents from the Pinecone vector store.

    Uses ``search_query`` (which may have been rewritten by ``rewrite_query_node``
    on a previous iteration) as the retrieval query. Falls back to ``question``
    on the first pass when ``search_query`` has not yet been set.

    State reads:  ``search_query``, ``question``
    State writes: ``documents``

    Returns
    -------
    dict
        ``{"documents": List[Document]}`` — the top-k retrieved chunks.
    """
    query = state.get("search_query") or state["question"]
    logger.info("[retrieve_node] querying vector store with: %r", query)

    retriever = get_retriever(top_k=5)
    docs: List[Document] = retriever.invoke(
        query,
        config={"callbacks": [get_langfuse_handler()]},
    )

    logger.info("[retrieve_node] retrieved %d documents", len(docs))
    return {"documents": docs}


def grade_documents_node(state: GraphState) -> dict:
    """
    Filter retrieved documents by relevance using an LLM-as-judge.

    Iterates over every document in ``state["documents"]`` and asks the LLM
    to produce a ``GradeDocuments`` verdict via ``with_structured_output()``.
    Only documents scored ``binary_score=True`` are retained.

    This step eliminates noise and prevents irrelevant context from polluting
    the generation step. If zero documents survive, the router will trigger
    ``rewrite_query_node`` for a self-healing retry.

    State reads:  ``question``, ``documents``
    State writes: ``documents`` (filtered)

    Returns
    -------
    dict
        ``{"documents": List[Document]}`` — only the relevant subset.
    """
    question = state["question"]
    raw_docs: List[Document] = state.get("documents", [])
    logger.info(
        "[grade_documents_node] grading %d documents for relevance", len(raw_docs)
    )

    grader_llm = get_llm().with_structured_output(GradeDocuments)
    grading_chain = _GRADE_PROMPT | grader_llm

    relevant_docs: List[Document] = []
    for doc in raw_docs:
        try:
            verdict: GradeDocuments = grading_chain.invoke(
                {"question": question, "document": doc.page_content},
                config={"callbacks": [get_langfuse_handler()]},
            )
            if verdict.binary_score:
                relevant_docs.append(doc)
                logger.debug("[grade_documents_node] RELEVANT: %s…", doc.page_content[:60])
            else:
                logger.debug("[grade_documents_node] IRRELEVANT: %s…", doc.page_content[:60])
        except Exception as exc:  # noqa: BLE001
            # Fail open — keep the document if the grader errors to avoid losing context
            logger.warning("[grade_documents_node] grading error, keeping doc: %s", exc)
            relevant_docs.append(doc)

    logger.info(
        "[grade_documents_node] %d/%d documents passed relevance grading",
        len(relevant_docs),
        len(raw_docs),
    )
    return {"documents": relevant_docs}


def generate_node(state: GraphState) -> dict:
    """
    Generate an answer grounded in the filtered context documents.

    Joins all surviving document chunks into a single context block, then
    invokes the Gemini LLM with the RAG prompt to produce a draft answer.
    The generation is stored in ``state["generation"]`` for the critic to
    evaluate in the next step.

    State reads:  ``question``, ``documents``
    State writes: ``generation``

    Returns
    -------
    dict
        ``{"generation": str}`` — the LLM's drafted answer.
    """
    question = state["question"]
    docs: List[Document] = state.get("documents", [])
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    logger.info(
        "[generate_node] generating answer from %d docs (%d context chars)",
        len(docs),
        len(context),
    )

    generation_chain = _GENERATE_PROMPT | get_llm() | StrOutputParser()
    generation: str = generation_chain.invoke(
        {"context": context, "question": question},
        config={"callbacks": [get_langfuse_handler()]},
    )

    logger.info("[generate_node] generation produced (%d chars)", len(generation))
    return {"generation": generation}


def critic_node(state: GraphState) -> dict:
    """
    Evaluate the generation on two axes: hallucination and answer relevance.

    Makes two independent ``with_structured_output()`` calls — one for
    groundedness (``HallucinationCritic``) and one for answer relevance
    (``AnswerRelevance``) — and writes both boolean results to the state.
    Keeping the calls separate avoids conflating two orthogonal quality signals.

    Decision matrix for the downstream router:

      is_hallucination=True,  is_relevant=*     → rewrite_query (if retries remain)
      is_hallucination=False, is_relevant=False → rewrite_query (grounded but off-topic)
      is_hallucination=False, is_relevant=True  → END  (perfect answer)

    State reads:  ``documents``, ``generation``, ``question``
    State writes: ``is_hallucination``, ``is_relevant``

    Returns
    -------
    dict
        ``{"is_hallucination": bool, "is_relevant": bool}``
    """
    question: str = state.get("question", "")
    docs: List[Document] = state.get("documents", [])
    generation: str = state.get("generation", "")
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    logger.info("[critic_node] evaluating generation — hallucination check")

    # ---- 1. Hallucination / groundedness check ---------------------------
    hallucination_chain = _CRITIC_PROMPT | get_llm().with_structured_output(HallucinationCritic)
    try:
        h_verdict: HallucinationCritic = hallucination_chain.invoke(
            {"context": context, "generation": generation},
            config={"callbacks": [get_langfuse_handler()]},
        )
        is_hallucination = not h_verdict.is_grounded
    except Exception as exc:  # noqa: BLE001
        logger.warning("[critic_node] hallucination check failed, defaulting to True: %s", exc)
        is_hallucination = True

    logger.info("[critic_node] is_hallucination=%s", is_hallucination)

    # ---- 2. Answer relevance check ---------------------------------------
    logger.info("[critic_node] evaluating generation — answer relevance check")
    relevance_chain = _RELEVANCE_PROMPT | get_llm().with_structured_output(AnswerRelevance)
    try:
        r_verdict: AnswerRelevance = relevance_chain.invoke(
            {"question": question, "generation": generation},
            config={"callbacks": [get_langfuse_handler()]},
        )
        is_relevant = r_verdict.is_relevant
    except Exception as exc:  # noqa: BLE001
        logger.warning("[critic_node] relevance check failed, defaulting to False: %s", exc)
        is_relevant = False

    logger.info("[critic_node] is_relevant=%s", is_relevant)
    return {"is_hallucination": is_hallucination, "is_relevant": is_relevant}


def rewrite_query_node(state: GraphState) -> dict:
    """
    Rewrite the user's question into a better vector-store search query.

    Called whenever the pipeline enters a self-healing loop — either because
    the relevance grader filtered out all documents, or because the critic
    detected a hallucination in the generation.

    Behaviour:
      1. Increments ``retry_count`` to track how many healing attempts have
         been made (the graph router checks this against ``MAX_RETRIES``).
      2. Asks the LLM to produce a more focused, keyword-rich version of the
         original ``question`` and stores it as ``search_query``.

    The rewritten ``search_query`` is what ``retrieve_node`` will use on the
    next iteration — ``question`` itself is never mutated.

    State reads:  ``question``, ``retry_count``
    State writes: ``search_query``, ``retry_count``

    Returns
    -------
    dict
        ``{"search_query": str, "retry_count": int}``
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1

    logger.info(
        "[rewrite_query_node] rewriting query (attempt %d / %d)",
        retry_count,
        MAX_RETRIES,
    )

    rewrite_chain = _REWRITE_PROMPT | get_llm() | StrOutputParser()
    search_query: str = rewrite_chain.invoke(
        {"question": question},
        config={"callbacks": [get_langfuse_handler()]},
    )
    search_query = search_query.strip()

    logger.info("[rewrite_query_node] new search_query=%r", search_query)
    return {"search_query": search_query, "retry_count": retry_count}
