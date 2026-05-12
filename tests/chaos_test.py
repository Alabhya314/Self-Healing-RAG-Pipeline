"""
tests/chaos_test.py
===================
Step 2 — Integration & Loop Validation (Self-Healing RAG)

Goal:
  Run the full compiled LangGraph and *force* a self-healing event by:
    - Mocking retrieval to return empty documents (simulated Pinecone failure)
    - Streaming graph execution and printing node names as they run
    - Asserting the loop order and retry_count increments

Run:
  python -m tests.chaos_test
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type
from unittest.mock import Mock, patch

# Allow running as either:
#   python -m tests.chaos_test   (recommended)
#   python tests/chaos_test.py   (direct execution)
#
# Direct execution does not include the repo root on sys.path, so `import src.*` fails.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graph import graph  # noqa: E402


OBSCURE_QUESTION = "What is the secret recipe for Martian oxygen?"


class _EmptyRetriever:
    """Fake retriever that always returns no documents."""

    def invoke(self, query: str, config: dict | None = None) -> list:  # noqa: ANN401
        return []


@dataclass(frozen=True)
class _PassthroughPrompt:
    """
    LCEL short-circuit for rewrite_query_node.

    rewrite_query_node uses:
        _REWRITE_PROMPT | get_llm() | StrOutputParser()

    We replace the prompt so:
        _PassthroughPrompt | llm  -> llm
    """

    def __or__(self, other: Any) -> Any:  # noqa: ANN401
        return other


class _TextLLM:
    """
    Minimal LLM stub used for this chaos test.

    It must satisfy two node behaviors:
    - rewrite_query_node: `_REWRITE_PROMPT | get_llm() | StrOutputParser()`
    - grade_documents_node: `get_llm().with_structured_output(GradeDocuments)`

    For the grader path, it returns a structured LLM stub whose `.invoke(...)`
    always returns `GradeDocuments(binary_score=False)`; this keeps documents empty
    and guarantees the healing loop is exercised.
    """

    def __or__(self, other: Any) -> Any:  # noqa: ANN401
        # We ignore the parser and return self; our invoke already returns str.
        return self

    def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> str:
        q = str(payload.get("question", "")).strip()
        # Deterministic, keyword-ish rewrite to prove the node ran.
        return f"martian oxygen secret recipe {q}".strip()

    def with_structured_output(self, model: Type[Any]) -> Any:  # noqa: ANN401
        # Import here to keep the top-of-file bootstrap minimal.
        from src.nodes import GradeDocuments

        if model is not GradeDocuments:
            raise AssertionError(f"Chaos test only mocks structured output for GradeDocuments, got: {model}")

        class _Structured:
            def invoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> GradeDocuments:
                return GradeDocuments(binary_score=False)

        return _Structured()


def _extract_step_name(step: Any) -> str | None:
    """
    LangGraph stream yields dict-like updates such as:
      {"retrieve": {...}}  or  {"grade_documents": {...}}
    We treat the single key as the executed node.
    """
    if not isinstance(step, dict) or not step:
        return None
    if len(step.keys()) != 1:
        return None
    return next(iter(step.keys()))


def _stream_and_collect(initial_state: Dict[str, Any]) -> Tuple[List[str], List[dict]]:
    executed: List[str] = []
    raw_steps: List[dict] = []

    for step in graph.stream(initial_state):
        raw_steps.append(step)
        name = _extract_step_name(step)
        if name:
            executed.append(name)
            print(f"-> node: {name}")
    return executed, raw_steps


def main() -> None:
    # We patch at the src.nodes level because retrieve_node calls a symbol imported there.
    # This preserves the compiled graph while changing runtime dependencies.
    with patch("src.nodes.get_retriever", Mock(return_value=_EmptyRetriever())), patch(
        "src.nodes._REWRITE_PROMPT", _PassthroughPrompt()
    ), patch("src.nodes._GRADE_PROMPT", _PassthroughPrompt()), patch(
        "src.nodes.get_llm", Mock(return_value=_TextLLM())
    ), patch("src.nodes.get_langfuse_handler", Mock(return_value=None)):
        initial_state = {
            "question": OBSCURE_QUESTION,
            "search_query": OBSCURE_QUESTION,
            "retry_count": 0,
        }

        print("\n=== CHAOS TEST: Forcing retrieval failure (empty docs) ===")
        print(f"Question: {OBSCURE_QUESTION!r}\n")

        executed, raw_steps = _stream_and_collect(initial_state)

        # -------------------------------
        # Assertions (loop validation)
        # -------------------------------
        # We want to see the loop:
        # retrieve -> grade_documents -> rewrite_query -> retrieve -> ...
        # (until max retries triggers END)
        assert "grade_documents" in executed, "Expected grade_documents to run"

        # Verify next node after first grade_documents is rewrite_query
        first_grade_idx = executed.index("grade_documents")
        assert executed[first_grade_idx + 1] == "rewrite_query", (
            "Expected rewrite_query immediately after grade_documents when no docs remain"
        )

        # Verify retrieve is attempted again after rewrite_query
        assert executed[first_grade_idx + 2] == "retrieve", (
            "Expected a second retrieve attempt after rewrite_query"
        )

        # Verify retry_count increments to 1 at first rewrite
        # The stream step for rewrite_query contains the node's state delta.
        rewrite_step = raw_steps[first_grade_idx + 1].get("rewrite_query")  # type: ignore[union-attr]
        assert isinstance(rewrite_step, dict), "Expected rewrite_query step payload to be a dict"
        assert rewrite_step.get("retry_count") == 1, "Expected retry_count to increment to 1"

        print("\n=== Assertions: PASS ===")
        print("Observed loop: grade_documents → rewrite_query → retrieve")
        print("Observed retry_count increment on rewrite_query: 0 → 1")

        print("\nHow to read the output:")
        print("- Each '-> node: X' line is one executed node in the graph.")
        print("- A successful heal is confirmed when you see:")
        print("  1) '-> node: grade_documents'")
        print("  2) followed immediately by '-> node: rewrite_query'")
        print("  3) followed by a new '-> node: retrieve' attempt")
        print("- The script also asserts retry_count increments on rewrite_query.")


if __name__ == "__main__":
    main()
