"""
tests/evaluator.py
==================
RAGAS evaluation suite for the Self-Healing RAG pipeline.

Modes
-----
MOCK_MODE=true
    No Google / Pinecone / graph API calls. Validates Langfuse score wiring,
    efficiency math, and console reporting using synthetic metrics.

MOCK_MODE=false (default)
    Sequential graph runs (no parallel ``asyncio.gather``) + 10s throttle
    between cases to reduce free-tier 429s. RAGAS runs once on the batch.

Environment
-----------
Set ``MOCK_MODE`` in ``.env`` or the shell. ``LANGFUSE_HOST`` defaults to
``https://cloud.langfuse.com`` if unset (evaluator-only bootstrap).
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: load .env from project root before any other imports that read env
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
LIVE_THROTTLE_SECONDS = float(os.getenv("EVALUATOR_THROTTLE_SECONDS", "10"))

if not os.environ.get("LANGFUSE_HOST"):
    os.environ.setdefault("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))


# ---------------------------------------------------------------------------
# Test dataset (question + gold answer for RAGAS)
# ---------------------------------------------------------------------------
TEST_DATASET: list[dict[str, str]] = [
    {
        "question": "What triggers query rewriting in this self-healing RAG pipeline?",
        "ground_truth": (
            "The pipeline rewrites the query when no relevant documents remain after "
            "grading, or when the critic flags hallucination/irrelevance and retries remain."
        ),
    },
    {
        "question": "What is the maximum number of self-healing retries?",
        "ground_truth": "The maximum retry count is 3.",
    },
    {
        "question": "When does the graph terminate successfully?",
        "ground_truth": (
            "The graph ends successfully when the answer is both grounded "
            "(not hallucinated) and relevant to the original question."
        ),
    },
]


def calculate_efficiency_from_retries(retries: int) -> float:
    """Efficiency = 1 / (1 + retries). 1.0 = no healing; 0.5 = one retry."""
    r = max(int(retries), 0)
    return 1.0 / (1.0 + r)


def calculate_efficiency(state_history: dict[str, Any]) -> float:
    """Same formula from graph final state."""
    return calculate_efficiency_from_retries(int(state_history.get("retry_count", 0)))


def _bootstrap_env() -> None:
    """Align env with ``src.config`` expectations."""
    if not os.environ.get("LANGFUSE_HOST") and os.environ.get("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]


def _extract_retry_delay_seconds(error_text: str) -> float | None:
    patterns = [
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]([0-9]+)s['\"]",
    ]
    lowered = error_text.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# MOCK_MODE: zero API tokens for graph / Gemini / RAGAS
# ---------------------------------------------------------------------------
async def mock_evaluate_all() -> None:
    print("\n[MOCK_MODE] Running synthetic evaluation (no graph / Gemini / RAGAS API calls).\n")

    # Synthetic results aligned with TEST_DATASET length (extend if you add cases)
    synthetic = [
        {"faithfulness": 0.95, "answer_relevancy": 0.88, "retries": 1},
        {"faithfulness": 1.0, "answer_relevancy": 0.92, "retries": 0},
        {"faithfulness": 0.91, "answer_relevancy": 0.90, "retries": 2},
    ]
    while len(synthetic) < len(TEST_DATASET):
        synthetic.append({"faithfulness": 0.9, "answer_relevancy": 0.85, "retries": 0})
    synthetic = synthetic[: len(TEST_DATASET)]

    try:
        from src.config import get_settings
        from langfuse import Langfuse

        _bootstrap_env()
        cfg = get_settings()
        langfuse = Langfuse(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key.get_secret_value(),
            host=cfg.langfuse_host,
        )
    except Exception as exc:  # noqa: BLE001
        langfuse = None
        print(f"[MOCK_MODE] Langfuse disabled ({exc}). Scores only printed locally.\n")

    rows_for_print: list[dict[str, Any]] = []
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    efficiency_scores: list[float] = []

    for idx, (sample, syn) in enumerate(zip(TEST_DATASET, synthetic)):
        eff = calculate_efficiency_from_retries(syn["retries"])
        faith = float(syn["faithfulness"])
        rel = float(syn["answer_relevancy"])
        faithfulness_scores.append(faith)
        relevancy_scores.append(rel)
        efficiency_scores.append(eff)

        trace_id = None
        if langfuse is not None:
            try:
                trace_id = langfuse.create_trace_id(seed=f"mock-eval-{idx}-{uuid.uuid4()}")
                langfuse.create_score(
                    trace_id=trace_id,
                    name="faithfulness",
                    value=faith,
                    comment="MOCK_MODE synthetic",
                    metadata={"question": sample["question"], "mode": "mock"},
                )
                langfuse.create_score(
                    trace_id=trace_id,
                    name="answer_relevancy",
                    value=rel,
                    comment="MOCK_MODE synthetic",
                    metadata={"question": sample["question"], "mode": "mock"},
                )
                langfuse.create_score(
                    trace_id=trace_id,
                    name="efficiency",
                    value=eff,
                    comment="1 / (1 + retries) MOCK",
                    metadata={"question": sample["question"], "retries": syn["retries"], "mode": "mock"},
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[MOCK_MODE] Langfuse score skip for case {idx + 1}: {exc}")

        rows_for_print.append(
            {
                "question": sample["question"],
                "faithfulness": faith,
                "answer_relevancy": rel,
                "efficiency": eff,
                "retry_count": syn["retries"],
                "status": "MOCK",
            }
        )
        print(
            f"  Case {idx + 1} | retries={syn['retries']} | faithfulness={faith:.3f} | "
            f"relevancy={rel:.3f} | efficiency={eff:.3f}"
        )

    if langfuse is not None:
        try:
            langfuse.flush()
        except Exception:
            pass

    avg_f = mean(faithfulness_scores)
    avg_r = mean(relevancy_scores)
    avg_e = mean(efficiency_scores)

    print("\n====================== RAGAS Quality Report (MOCK) ======================")
    print(
        f"{'Case':<4} {'Status':<7} {'Retries':<7} {'Faithfulness':<13} "
        f"{'Relevancy':<10} {'Efficiency':<10} Question"
    )
    print("-" * 78)
    for idx, row in enumerate(rows_for_print, start=1):
        print(
            f"{idx:<4} {row['status']:<7} {row['retry_count']:<7} {row['faithfulness']:<13.3f} "
            f"{row['answer_relevancy']:<10.3f} {row['efficiency']:<10.3f} {row['question']}"
        )
    print("-" * 78)
    print(
        f"{'AVG':<4} {'-':<7} {'-':<7} {avg_f:<13.3f} {avg_r:<10.3f} {avg_e:<10.3f} "
        f"({len(TEST_DATASET)} mock cases)"
    )
    print("==========================================================================")
    if avg_f < 0.8:
        print("FAIL: Average Faithfulness is below 0.8 (synthetic run).")
    else:
        print("PASS: Average Faithfulness meets the 0.8 threshold (synthetic run).")
    print("\nArchitecture check complete. Set MOCK_MODE=false for live RAGAS + graph.")


# ---------------------------------------------------------------------------
# LIVE: sequential graph + throttle + batch RAGAS
# ---------------------------------------------------------------------------
def _build_langfuse_client() -> Any:
    from src.config import get_settings
    from langfuse import Langfuse

    cfg = get_settings()
    return Langfuse(
        public_key=cfg.langfuse_public_key,
        secret_key=cfg.langfuse_secret_key.get_secret_value(),
        host=cfg.langfuse_host,
    )


def _build_ragas_judges() -> tuple[Any, Any]:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from src.config import get_settings

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except Exception:  # pragma: no cover
        from ragas.embeddings import LangchainEmbeddings as LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLM as LangchainLLMWrapper

    cfg = get_settings()
    api_key = cfg.google_api_key.get_secret_value()

    embedding_candidates = (
        (os.environ.get("GOOGLE_EMBEDDING_MODEL", "").strip(),)
        if os.environ.get("GOOGLE_EMBEDDING_MODEL", "").strip()
        else ()
    ) + (
        "models/gemini-embedding-001",
        "models/gemini-embedding-2",
        "models/gemini-embedding-2-preview",
        "text-embedding-004",
        "models/text-embedding-004",
    )

    judge_candidates = (
        (os.environ.get("RAGAS_JUDGE_MODEL", "").strip(),)
        if os.environ.get("RAGAS_JUDGE_MODEL", "").strip()
        else ()
    ) + (
        "gemini-3-flash",
        "models/gemini-3-flash-preview",
        "gemini-2.5-flash",
        "models/gemini-2.5-flash",
    )

    judge_llm = None
    last_judge_exc: Exception | None = None
    for name in judge_candidates:
        if not name:
            continue
        try:
            llm = ChatGoogleGenerativeAI(model=name, temperature=0.0, google_api_key=api_key)
            llm.invoke("ragas judge probe")
            judge_llm = llm
            break
        except Exception as exc:  # noqa: BLE001
            last_judge_exc = exc
    if judge_llm is None:
        raise RuntimeError("No compatible RAGAS judge LLM. Set RAGAS_JUDGE_MODEL.") from last_judge_exc

    judge_embeddings = None
    last_emb_exc: Exception | None = None
    for name in embedding_candidates:
        if not name:
            continue
        try:
            emb = GoogleGenerativeAIEmbeddings(model=name, google_api_key=api_key)
            emb.embed_query("ragas embedding probe")
            judge_embeddings = emb
            break
        except Exception as exc:  # noqa: BLE001
            last_emb_exc = exc
    if judge_embeddings is None:
        raise RuntimeError("No compatible embedding model. Set GOOGLE_EMBEDDING_MODEL.") from last_emb_exc

    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(judge_embeddings)


async def live_evaluate_all() -> None:
    """Sequential graph runs, throttle between cases, batch RAGAS, Langfuse scores."""
    from datasets import Dataset
    from ragas import evaluate

    try:
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness
    except Exception:  # pragma: no cover
        from ragas.metrics import answer_relevancy, faithfulness

        AnswerRelevancy = None  # type: ignore[misc, assignment]
        Faithfulness = None  # type: ignore[misc, assignment]

    from src.graph import graph

    _bootstrap_env()
    if not os.environ.get("GOOGLE_API_KEY"):
        from src.config import get_settings

        get_settings()

    print("\n[LIVE] Sequential evaluation + throttle between graph calls.\n")

    langfuse_client = _build_langfuse_client()
    test_outputs: list[dict[str, Any]] = []

    for idx, sample in enumerate(TEST_DATASET):
        print(f"  Processing case {idx + 1}/{len(TEST_DATASET)}: {sample['question']!r} ...")

        trace_id = langfuse_client.create_trace_id(seed=str(uuid.uuid4()))
        initial_state = {
            "question": sample["question"],
            "search_query": sample["question"],
            "retry_count": 0,
        }

        case_error: str | None = None
        try:
            result_state = await asyncio.to_thread(graph.invoke, initial_state)
        except Exception as exc:  # noqa: BLE001
            case_error = str(exc)
            result_state = {"generation": "", "documents": [], "retry_count": 0}

        generation = (result_state or {}).get("generation", "") or ""
        docs = (result_state or {}).get("documents", []) or []
        contexts = [getattr(doc, "page_content", str(doc)) for doc in docs]
        efficiency = calculate_efficiency(result_state or {})

        test_outputs.append(
            {
                "trace_id": trace_id,
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "answer": generation,
                "contexts": contexts,
                "efficiency": efficiency,
                "retry_count": int((result_state or {}).get("retry_count", 0)),
                "error": case_error,
            }
        )

        print(
            f"    Done | retries={test_outputs[-1]['retry_count']} | "
            f"efficiency={efficiency:.3f}"
            + (f" | ERROR: {case_error}" if case_error else "")
        )

        if idx < len(TEST_DATASET) - 1:
            print(f"    Sleeping {LIVE_THROTTLE_SECONDS:.0f}s (quota / rate-limit cooldown) ...")
            await asyncio.sleep(LIVE_THROTTLE_SECONDS)

    successful_cases = [row for row in test_outputs if not row.get("error")]
    ragas_scores_by_trace: dict[str, dict[str, float]] = {}

    if successful_cases:
        print("\n  Running RAGAS (batch) on successful cases ...")
        ragas_llm, ragas_embeddings = _build_ragas_judges()
        eval_dataset = Dataset.from_list(
            [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "contexts": row["contexts"],
                    "ground_truth": row["ground_truth"],
                }
                for row in successful_cases
            ]
        )
        if AnswerRelevancy is not None and Faithfulness is not None:
            metrics = [Faithfulness(), AnswerRelevancy()]
        else:
            metrics = [faithfulness, answer_relevancy]

        ragas_result = evaluate(
            eval_dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        ragas_rows = ragas_result.to_pandas().to_dict(orient="records")
        for source_row, ragas_row in zip(successful_cases, ragas_rows):
            ragas_scores_by_trace[source_row["trace_id"]] = {
                "faithfulness": float(ragas_row.get("faithfulness", 0.0) or 0.0),
                "answer_relevancy": float(ragas_row.get("answer_relevancy", 0.0) or 0.0),
            }

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    efficiency_scores: list[float] = []
    rows_for_print: list[dict[str, Any]] = []

    for row in test_outputs:
        score_row = ragas_scores_by_trace.get(row["trace_id"], {})
        faith = float(score_row.get("faithfulness", 0.0) or 0.0)
        relevancy = float(score_row.get("answer_relevancy", 0.0) or 0.0)
        efficiency = float(row["efficiency"])

        faithfulness_scores.append(faith)
        relevancy_scores.append(relevancy)
        efficiency_scores.append(efficiency)

        rows_for_print.append(
            {
                "question": row["question"],
                "faithfulness": faith,
                "answer_relevancy": relevancy,
                "efficiency": efficiency,
                "retry_count": int(row.get("retry_count", 0)),
                "status": "OK" if not row.get("error") else "ERROR",
            }
        )

        langfuse_client.create_score(
            trace_id=row["trace_id"],
            name="faithfulness",
            value=faith,
            comment="RAGAS faithfulness",
            metadata={"question": row["question"], "case_type": "ragas_eval"},
        )
        langfuse_client.create_score(
            trace_id=row["trace_id"],
            name="answer_relevancy",
            value=relevancy,
            comment="RAGAS answer relevancy",
            metadata={"question": row["question"], "case_type": "ragas_eval"},
        )
        langfuse_client.create_score(
            trace_id=row["trace_id"],
            name="efficiency",
            value=efficiency,
            comment="1 / (1 + retries)",
            metadata={
                "question": row["question"],
                "retry_count": row.get("retry_count", 0),
                "case_type": "ragas_eval",
            },
        )

    avg_faithfulness = mean(faithfulness_scores) if faithfulness_scores else 0.0
    avg_relevancy = mean(relevancy_scores) if relevancy_scores else 0.0
    avg_efficiency = mean(efficiency_scores) if efficiency_scores else 0.0

    print("\n====================== RAGAS Quality Report (LIVE) ======================")
    print(
        f"{'Case':<4} {'Status':<7} {'Retries':<7} {'Faithfulness':<13} "
        f"{'Relevancy':<10} {'Efficiency':<10} Question"
    )
    print("-" * 78)
    for idx, row in enumerate(rows_for_print, start=1):
        print(
            f"{idx:<4} {row['status']:<7} {row['retry_count']:<7} {row['faithfulness']:<13.3f} "
            f"{row['answer_relevancy']:<10.3f} {row['efficiency']:<10.3f} {row['question']}"
        )
    print("-" * 78)
    print(
        f"{'AVG':<4} {'-':<7} {'-':<7} {avg_faithfulness:<13.3f} {avg_relevancy:<10.3f} "
        f"{avg_efficiency:<10.3f} ({len(test_outputs)} cases, {len(successful_cases)} RAGAS-scored)"
    )
    print("==========================================================================")

    if avg_faithfulness < 0.8:
        print("FAIL: Average Faithfulness is below 0.8.")
    else:
        print("PASS: Average Faithfulness meets the 0.8 threshold.")

    langfuse_client.flush()


def main() -> None:
    if MOCK_MODE:
        asyncio.run(mock_evaluate_all())
    else:
        asyncio.run(live_evaluate_all())


if __name__ == "__main__":
    main()
