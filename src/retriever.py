"""
src/retriever.py
================
Vector-database connection and retriever factory for the Self-Healing RAG pipeline.

Responsibilities
----------------
* Bootstrap the Pinecone client from environment-sourced credentials.
* Bind the ``self-healing-rag`` index to a LangChain-compatible vector store
  backed by a Google embedding model compatible with the active API version.
* Expose ``get_retriever()`` as the single, stable interface for downstream
  graph nodes that need to perform similarity search.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec  # noqa: F401 – ServerlessSpec kept for type hinting if needed

from src.config import get_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PINECONE_INDEX_NAME: str = "self-healing-rag"
EMBEDDING_MODEL_CANDIDATES: tuple[str, ...] = (
    "models/gemini-embedding-001",
    "gemini-embedding-001",
    "models/gemini-embedding-2",
    "gemini-embedding-2",
    "models/gemini-embedding-2-preview",
    "gemini-embedding-2-preview",
    "text-embedding-004",
    "models/text-embedding-004",
    "embedding-001",
    "models/embedding-001",
)

# Default retriever search parameters — override via get_retriever() kwargs.
DEFAULT_TOP_K: int = 5
DEFAULT_SEARCH_TYPE: str = "similarity"  # alternatives: "mmr", "similarity_score_threshold"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_pinecone_client() -> Pinecone:
    """
    Initialise and cache the Pinecone client.

    The API key is fetched from Settings (which reads it from the .env file),
    so this function contains zero hardcoded credentials.
    """
    cfg = get_settings()
    return Pinecone(api_key=cfg.pinecone_api_key.get_secret_value())


@lru_cache(maxsize=1)
def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialise and cache the Google Generative AI Embeddings client.

    Selects the first embedding model that is accepted by the current
    Google GenAI endpoint in this environment.
    """
    cfg = get_settings()
    api_key = cfg.google_api_key.get_secret_value()

    # Optional override for environments with a known supported model.
    configured_model = os.environ.get("GOOGLE_EMBEDDING_MODEL", "").strip()
    candidates = ((configured_model,) if configured_model else ()) + EMBEDDING_MODEL_CANDIDATES

    last_exc: Exception | None = None
    for model_name in candidates:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            # Probe once so failures happen here (clearer error) instead of deep inside retrieval.
            embeddings.embed_query("self-healing rag probe")
            return embeddings
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    raise RuntimeError(
        "No compatible Google embedding model found. Set GOOGLE_EMBEDDING_MODEL "
        "to a supported model for your API endpoint."
    ) from last_exc


@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """
    Connect to the existing Pinecone index and wrap it in a LangChain
    ``PineconeVectorStore`` using the shared embeddings client.

    The index is assumed to already exist in Pinecone (it will NOT be
    created here). Connection is validated lazily on first query.
    """
    pc = _get_pinecone_client()
    pinecone_index_host = get_settings().pinecone_index_host.strip()

    # Support both deployment styles:
    # 1) explicit host in env (faster/direct)
    # 2) name-only lookup via Pinecone client metadata resolution
    if pinecone_index_host:
        index = pc.Index(
            name=PINECONE_INDEX_NAME,
            host=pinecone_index_host,
        )
    else:
        index = pc.Index(name=PINECONE_INDEX_NAME)

    return PineconeVectorStore(
        index=index,
        embedding=_get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_retriever(
    top_k: int = DEFAULT_TOP_K,
    search_type: str = DEFAULT_SEARCH_TYPE,
    search_kwargs: Optional[dict] = None,
) -> VectorStoreRetriever:
    """
    Return a configured LangChain ``VectorStoreRetriever`` backed by the
    ``self-healing-rag`` Pinecone index.

    Parameters
    ----------
    top_k : int
        Number of documents to retrieve per query (``k`` in nearest-neighbour
        search). Defaults to ``5``.
    search_type : str
        LangChain search strategy: ``"similarity"`` (default), ``"mmr"``, or
        ``"similarity_score_threshold"``.
    search_kwargs : dict, optional
        Additional keyword arguments forwarded to the underlying search call
        (e.g. ``{"score_threshold": 0.75}`` when using the threshold variant).

    Returns
    -------
    VectorStoreRetriever
        A fully configured retriever ready for use inside LangGraph nodes or
        any LangChain LCEL chain.

    Example
    -------
    >>> retriever = get_retriever(top_k=6, search_type="mmr")
    >>> docs = retriever.invoke("What is self-healing RAG?")
    """
    kwargs: dict = search_kwargs or {}
    kwargs.setdefault("k", top_k)

    return _get_vector_store().as_retriever(
        search_type=search_type,
        search_kwargs=kwargs,
    )
