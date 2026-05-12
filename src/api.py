"""
src/api.py
==========
Phase 5 — FastAPI backend for the Self-Healing RAG demo.

Exposes ``POST /ask`` which runs the compiled LangGraph and returns the final
answer plus explicit retry / quality signals for the Streamlit UI.
"""

from __future__ import annotations

from typing import Any, List

from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.graph import graph

app = FastAPI(
    title="Self-Healing RAG API",
    description="Corporate-grade RAG with retrieval grading, critic loops, and observability.",
    version="5.0.0",
)


class AskRequest(BaseModel):
    """Inbound question payload."""

    question: str = Field(..., min_length=1, description="End-user question.")


class AskResponse(BaseModel):
    """Structured response for UI trace visualisation and reliability messaging."""

    generation: str = Field(..., description="Final model answer.")
    retry_count: int = Field(
        ...,
        description="Number of self-healing iterations (query rewrites) executed.",
    )
    healing_occurred: bool = Field(
        ...,
        description="True when retry_count > 0 (healing path was taken).",
    )
    healing_summary: str = Field(
        ...,
        description="Human-readable summary of whether self-healing was used.",
    )
    hallucination_detected: bool = Field(
        ...,
        description="From the critic: True if the final answer was judged ungrounded.",
    )
    answer_relevant: bool = Field(
        ...,
        description="From the critic: True if the answer addresses the original question.",
    )
    answer_verified: bool = Field(
        ...,
        description="True when grounded (not hallucinated) and relevant.",
    )
    documents_retrieved_count: int = Field(
        ...,
        description="Count of documents that survived grading and were used for generation.",
    )


def _build_healing_summary(retry_count: int) -> str:
    if retry_count <= 0:
        return (
            "Completed on the first pass: retrieval and generation succeeded without "
            "self-healing query refinement."
        )
    return (
        f"Self-healing was active: {retry_count} refinement "
        f"{'cycle' if retry_count == 1 else 'cycles'} "
        "(query rewrite and re-retrieval) before the final answer."
    )


def _count_documents(state: dict[str, Any]) -> int:
    docs: List[Document] = state.get("documents") or []
    return len(docs)


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight readiness probe for orchestrators and the Streamlit sidebar."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """
    Run the self-healing RAG graph and return generation plus retry metadata.

    Initial state mirrors production usage: ``search_query`` starts equal to
    ``question``; ``rewrite_query`` updates ``search_query`` on healing loops.
    """
    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    initial: dict[str, Any] = {
        "question": q,
        "search_query": q,
        "retry_count": 0,
    }

    try:
        result = graph.invoke(initial)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Graph invocation failed: {exc}") from exc

    result = result or {}
    generation = str(result.get("generation") or "").strip()
    retry_count = int(result.get("retry_count") or 0)

    is_hallucination = bool(result.get("is_hallucination", True))
    is_relevant = bool(result.get("is_relevant", False))
    verified = (not is_hallucination) and is_relevant

    return AskResponse(
        generation=generation,
        retry_count=retry_count,
        healing_occurred=retry_count > 0,
        healing_summary=_build_healing_summary(retry_count),
        hallucination_detected=is_hallucination,
        answer_relevant=is_relevant,
        answer_verified=verified,
        documents_retrieved_count=_count_documents(result),
    )
