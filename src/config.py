"""
src/config.py
=============
Production-grade configuration for the Self-Healing RAG pipeline.
Merged for Railway deployment and robust model fallback.
"""

import os
from functools import lru_cache
from typing import Iterable

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.langchain import CallbackHandler
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Typed Settings — Uses Pydantic v2 Settings for Build-Time Safety[cite: 3, 4]
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Resolved from environment variables. 
    Defaults are provided as empty strings to prevent crashes during Docker build[cite: 1].
    """
    # Google / Gemini
    google_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("GOOGLE_API_KEY", "")))
    llm_model: str = Field(default="gemini-3-flash-preview")
    llm_temperature: float = Field(default=0.0)

    # Pinecone
    pinecone_api_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("PINECONE_API_KEY", "")))
    pinecone_index_host: str = Field(default_factory=lambda: os.getenv("PINECONE_INDEX_HOST", ""))

    # Langfuse
    langfuse_public_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    langfuse_secret_key: SecretStr = Field(default_factory=lambda: SecretStr(os.getenv("LANGFUSE_SECRET_KEY", "")))
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # Resilience
    max_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        arbitrary_types_allowed=True
    )

# ---------------------------------------------------------------------------
# Singleton Accessors & Logic
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated Settings singleton[cite: 1]."""
    return Settings()

@lru_cache(maxsize=1)
def get_langfuse_handler() -> CallbackHandler:
    """Instantiate Langfuse with explicit environment propagation for OTEL[cite: 3]."""
    cfg = get_settings()
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", cfg.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", cfg.langfuse_secret_key.get_secret_value())
    os.environ.setdefault("LANGFUSE_HOST", cfg.langfuse_host)
    return CallbackHandler()

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Instantiate Gemini with a 'Self-Healing' model selection strategy.
    It attempts current 2026 models before falling back to defaults[cite: 1].
    """
    cfg = get_settings()
    api_key = cfg.google_api_key.get_secret_value()
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Check your Railway environment variables.")

    candidates: list[str] = _dedupe_models([
        os.environ.get("GOOGLE_LLM_MODEL", ""),
        cfg.llm_model,
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash-latest"
    ])

    last_exc = None
    for model_name in candidates:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=cfg.llm_temperature,
                google_api_key=api_key,
            )
            # Connectivity probe
            llm.invoke("ping")
            return llm
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError("No compatible Gemini model found.") from last_exc

def _dedupe_models(models: Iterable[str]) -> list[str]:
    """Clean and prioritize model list."""
    out = []
    seen = set()
    for m in models:
        val = m.strip() if m else ""
        if val and val not in seen:
            out.append(val)
            seen.add(val)
    return out