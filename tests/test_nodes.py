from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

import src.nodes as nodes
from src.graph import decide_to_generate
from src.nodes import (
    AnswerRelevance,
    GradeDocuments,
    HallucinationCritic,
    critic_node,
    grade_documents_node,
)


@dataclass(frozen=True)
class _PassthroughPrompt:
    """
    Minimal stub to short-circuit LCEL composition in unit tests.

    The production code builds chains like:
        grading_chain = _GRADE_PROMPT | get_llm().with_structured_output(...)

    In tests, we replace the prompt object so that:
        _PassthroughPrompt | runnable  -> runnable
    """

    def __or__(self, other: Any) -> Any:  # noqa: ANN401 - intentionally generic
        return other


class _StructuredLLM:
    """Mock object returned by get_llm().with_structured_output(Model)."""

    def __init__(self, invoke_impl: Callable[[dict[str, Any]], Any]):
        self._invoke_impl = invoke_impl

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
        return self._invoke_impl(payload)


class _FakeLLM:
    """
    Minimal LLM stub that supports the subset used by src.nodes:
      - with_structured_output(Model) -> object with .invoke(...)
    """

    def __init__(self, structured_outputs: Dict[Type[Any], Any]):
        self._structured_outputs = structured_outputs

    def with_structured_output(self, model: Type[Any]) -> _StructuredLLM:
        if model not in self._structured_outputs:
            raise AssertionError(f"Test did not configure structured output for: {model}")

        value = self._structured_outputs[model]

        def _impl(_: dict[str, Any]) -> Any:
            return value

        return _StructuredLLM(_impl)


@pytest.fixture()
def graph_state() -> dict[str, Any]:
    return {
        "question": "What triggers query rewriting in this self-healing RAG pipeline?",
        "documents": [],
        "generation": "",
        "search_query": "What triggers query rewriting in this self-healing RAG pipeline?",
        "retry_count": 0,
    }


@pytest.fixture(autouse=True)
def _patch_prompts_and_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevent LCEL prompt composition from requiring real Runnable objects,
    and prevent Langfuse callback handler creation during tests.
    """
    monkeypatch.setattr(nodes, "_GRADE_PROMPT", _PassthroughPrompt())
    monkeypatch.setattr(nodes, "_CRITIC_PROMPT", _PassthroughPrompt())
    monkeypatch.setattr(nodes, "_RELEVANCE_PROMPT", _PassthroughPrompt())
    monkeypatch.setattr(nodes, "get_langfuse_handler", Mock(return_value=None))


def test_grader_positive_keeps_relevant_doc_and_does_not_trigger_healing(
    monkeypatch: pytest.MonkeyPatch,
    graph_state: dict[str, Any],
) -> None:
    # Arrange: one relevant doc + grader returns binary_score=True
    graph_state["documents"] = [Document(page_content="Self-healing RAG rewrites the query when docs are irrelevant.")]

    fake_llm = _FakeLLM({GradeDocuments: GradeDocuments(binary_score=True)})
    monkeypatch.setattr(nodes, "get_llm", Mock(return_value=fake_llm))

    # Act
    out = grade_documents_node(graph_state)

    # Assert: document survives grading
    assert "documents" in out
    assert len(out["documents"]) == 1

    # And: given non-empty docs, router would proceed to generation (no healing loop)
    next_step = decide_to_generate({**graph_state, **out})
    assert next_step == "generate"


def test_grader_negative_filters_unrelated_doc_and_enables_healing_loop(
    monkeypatch: pytest.MonkeyPatch,
    graph_state: dict[str, Any],
) -> None:
    # Arrange: one unrelated doc + grader returns binary_score=False
    graph_state["documents"] = [Document(page_content="This document is about cricket statistics and stadiums.")]

    fake_llm = _FakeLLM({GradeDocuments: GradeDocuments(binary_score=False)})
    monkeypatch.setattr(nodes, "get_llm", Mock(return_value=fake_llm))

    # Act
    out = grade_documents_node(graph_state)

    # Assert: all docs filtered out
    assert "documents" in out
    assert out["documents"] == []

    # And: empty docs with retries left routes to rewrite_query (healing loop)
    next_step = decide_to_generate({**graph_state, **out})
    assert next_step == "rewrite_query"


def test_critic_hallucination_detection_sets_is_hallucination_true(
    monkeypatch: pytest.MonkeyPatch,
    graph_state: dict[str, Any],
) -> None:
    # Arrange: context says one thing, generation claims unsupported facts.
    graph_state["documents"] = [Document(page_content="The max retries is 3.")]
    graph_state["generation"] = "The max retries is 10 and the system uses Redis caching."

    fake_llm = _FakeLLM(
        {
            HallucinationCritic: HallucinationCritic(is_grounded=False),
            AnswerRelevance: AnswerRelevance(is_relevant=True),
        }
    )
    monkeypatch.setattr(nodes, "get_llm", Mock(return_value=fake_llm))

    # Act
    out = critic_node(graph_state)

    # Assert
    assert out["is_hallucination"] is True


def test_critic_grounding_check_sets_is_hallucination_false(
    monkeypatch: pytest.MonkeyPatch,
    graph_state: dict[str, Any],
) -> None:
    # Arrange: context fully supports generation.
    graph_state["documents"] = [Document(page_content="The pipeline retries up to 3 times.")]
    graph_state["generation"] = "The pipeline retries up to 3 times."

    fake_llm = _FakeLLM(
        {
            HallucinationCritic: HallucinationCritic(is_grounded=True),
            AnswerRelevance: AnswerRelevance(is_relevant=True),
        }
    )
    monkeypatch.setattr(nodes, "get_llm", Mock(return_value=fake_llm))

    # Act
    out = critic_node(graph_state)

    # Assert
    assert out["is_hallucination"] is False

