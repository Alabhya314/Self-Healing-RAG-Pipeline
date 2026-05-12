"""
app.py
======
Phase 5 — Streamlit front-end for the Self-Healing RAG demo.

Runs against the FastAPI service (``src.api``). Sidebar shows project metrics
from Langfuse (when available) with a local JSON fallback cache.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
METRICS_CACHE_PATH = CACHE_DIR / "project_metrics.json"

DEFAULT_API_BASE = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000").rstrip("/")


@dataclass
class ProjectMetrics:
    faithfulness: Optional[float]
    cost_efficiency: Optional[float]
    source: str
    updated: Optional[str]


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv_settings() -> None:
    """Reuse project .env via config loader (GOOGLE_API_KEY, Langfuse, etc.)."""
    try:
        from src.config import get_settings

        get_settings()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Langfuse + local metrics
# ---------------------------------------------------------------------------

def _basic_auth_header(public_key: str, secret_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _fetch_scores_from_langfuse(host: str, public_key: str, secret_key: str, name: str) -> list[float]:
    """
    Pull recent numeric scores by name from Langfuse public API.
    Returns empty list on any failure (caller falls back to cache).
    """
    url = f"{host.rstrip('/')}/api/public/scores"
    headers = _basic_auth_header(public_key, secret_key)
    values: list[float] = []
    page = 1
    limit = 50
    max_pages = 5
    while page <= max_pages:
        try:
            resp = requests.get(
                url,
                headers=headers,
                params={"name": name, "limit": limit, "page": page},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:  # noqa: BLE001
            break
        rows = payload.get("data") or []
        for row in rows:
            v = row.get("value")
            if isinstance(v, (int, float)):
                values.append(float(v))
        meta = payload.get("meta") or {}
        total_pages = int(meta.get("totalPages") or 1)
        if page >= total_pages or not rows:
            break
        page += 1
    return values


def _average_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _read_local_metrics_cache() -> dict[str, Any]:
    if not METRICS_CACHE_PATH.is_file():
        return {}
    try:
        with open(METRICS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _write_local_metrics_cache(data: dict[str, Any]) -> None:
    _ensure_cache_dir()
    payload = {**data, "updated": datetime.now(timezone.utc).isoformat()}
    with open(METRICS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_project_metrics() -> ProjectMetrics:
    """
    Prefer Langfuse aggregates (faithfulness + efficiency scores from eval runs);
    fall back to ``.cache/project_metrics.json``.
    """
    _load_dotenv_settings()
    try:
        from src.config import get_settings

        cfg = get_settings()
    except Exception:  # noqa: BLE001
        cached = _read_local_metrics_cache()
        return ProjectMetrics(
            faithfulness=cached.get("faithfulness"),
            cost_efficiency=cached.get("cost_efficiency"),
            source="local_cache",
            updated=cached.get("updated"),
        )

    faith_vals = _fetch_scores_from_langfuse(
        cfg.langfuse_host,
        cfg.langfuse_public_key,
        cfg.langfuse_secret_key.get_secret_value(),
        "faithfulness",
    )
    eff_vals = _fetch_scores_from_langfuse(
        cfg.langfuse_host,
        cfg.langfuse_public_key,
        cfg.langfuse_secret_key.get_secret_value(),
        "efficiency",
    )

    lf_faith = _average_or_none(faith_vals)
    lf_eff = _average_or_none(eff_vals)
    faith = lf_faith
    eff = lf_eff
    cached = _read_local_metrics_cache()
    used_cache_fill = False

    if faith is None:
        v = cached.get("faithfulness")
        if isinstance(v, (int, float)):
            faith = float(v)
            used_cache_fill = True
    if eff is None:
        v = cached.get("cost_efficiency")
        if isinstance(v, (int, float)):
            eff = float(v)
            used_cache_fill = True

    if lf_faith is not None or lf_eff is not None:
        merged_source = "langfuse+cache" if used_cache_fill else "langfuse"
    elif faith is not None or eff is not None:
        merged_source = "local_cache"
    else:
        merged_source = "unavailable"

    if faith is not None or eff is not None:
        merged_old = _read_local_metrics_cache()
        _write_local_metrics_cache(
            {
                "faithfulness": faith if faith is not None else merged_old.get("faithfulness"),
                "cost_efficiency": eff if eff is not None else merged_old.get("cost_efficiency"),
            }
        )

    ts = None
    if METRICS_CACHE_PATH.is_file():
        try:
            with open(METRICS_CACHE_PATH, encoding="utf-8") as f:
                ts = json.load(f).get("updated")
        except Exception:  # noqa: BLE001
            ts = None

    return ProjectMetrics(
        faithfulness=faith,
        cost_efficiency=eff,
        source=merged_source,
        updated=ts,
    )


def record_session_efficiency(efficiency: float) -> None:
    """Merge last-ask efficiency into cache for sidebar when Langfuse has no efficiency scores."""
    data = _read_local_metrics_cache()
    prev = data.get("cost_efficiency")
    if isinstance(prev, (int, float)):
        data["cost_efficiency"] = round((float(prev) * 0.7 + efficiency * 0.3), 4)
    else:
        data["cost_efficiency"] = round(efficiency, 4)
    _write_local_metrics_cache(data)


# ---------------------------------------------------------------------------
# Gemini 3 Flash — executive trace summary (corporate tone)
# ---------------------------------------------------------------------------

def summarize_trace_geminiflash(trace_context: str) -> str:
    """One short reliability-oriented summary for the Process Trace expander."""
    _load_dotenv_settings()
    try:
        from src.config import get_settings

        cfg = get_settings()
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash",
            temperature=0.2,
            google_api_key=cfg.google_api_key.get_secret_value(),
        )
        msg = (
            "You are writing for a corporate India enterprise audience. "
            "In two sentences maximum, summarise the RAG run outcome below: "
            "emphasise retrieval, grounding, and whether self-healing was needed. "
            "Neutral, precise, no marketing superlatives.\n\n"
            f"{trace_context}"
        )
        out = llm.invoke(msg)
        text = getattr(out, "content", str(out))
        return str(text).strip()
    except Exception as exc:  # noqa: BLE001
        return f"Summary unavailable ({exc})."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _corporate_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; max-width: 960px; }
            h1 { font-weight: 600; letter-spacing: -0.02em; color: #0f172a; }
            div[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
            .metric-panel { font-size: 0.9rem; color: #334155; }
            .trace-ok { color: #15803d; }
            .trace-warn { color: #a16207; }
            .trace-muted { color: #64748b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_metrics() -> None:
    st.sidebar.markdown("### Project metrics")
    st.sidebar.caption("Quality signals aligned with Phase 4 evaluation runs.")

    if st.sidebar.button("Refresh metrics", use_container_width=True):
        st.session_state.metrics_cache = None

    if st.session_state.metrics_cache is None:
        st.session_state.metrics_cache = get_project_metrics()

    m = st.session_state.metrics_cache

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "—"
        return f"{v:.3f}"

    st.sidebar.metric(label="Faithfulness (avg.)", value=_fmt(m.faithfulness))
    st.sidebar.metric(label="Cost efficiency (avg.)", value=_fmt(m.cost_efficiency))
    st.sidebar.markdown(
        f'<p class="metric-panel">Source: {m.source}<br/>'
        f'Updated: {m.updated or "—"}</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Service**\n\n"
        f"API: `{DEFAULT_API_BASE}`\n\n"
        "Start backend: `uvicorn src.api:app --host 127.0.0.1 --port 8000`"
    )


def call_ask_api(question: str) -> dict[str, Any]:
    url = f"{DEFAULT_API_BASE}/ask"
    r = requests.post(url, json={"question": question}, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> None:
    st.set_page_config(
        page_title="Self-Healing RAG",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _corporate_css()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "metrics_cache" not in st.session_state:
        st.session_state.metrics_cache = None

    render_sidebar_metrics()

    st.title("Self-Healing RAG")
    st.caption(
        "Reliable answers with retrieval grading, hallucination checks, and automated query refinement."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                _render_process_trace(msg["trace"], msg.get("exec_summary"))

    if prompt := st.chat_input("Ask a question about your knowledge base…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Processing request…", state="running")
            status.write("Connecting to inference service…")

            try:
                data = call_ask_api(prompt)
            except requests.RequestException as exc:
                status.update(label="Request failed", state="error")
                err = f"Could not reach the API at `{DEFAULT_API_BASE}`: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                return

            status.write("Running LangGraph (retrieve → grade → generate → critic)…")
            status.update(label="Completed", state="complete")

            answer = data.get("generation") or "_No generation returned._"
            st.markdown(answer)

            retries = int(data.get("retry_count") or 0)
            eff = 1.0 / (1.0 + max(retries, 0))
            record_session_efficiency(eff)

            trace = {
                "documents_retrieved": int(data.get("documents_retrieved_count") or 0),
                "hallucination": bool(data.get("hallucination_detected")),
                "retries": retries,
                "healing_occurred": bool(data.get("healing_occurred")),
                "verified": bool(data.get("answer_verified")),
                "healing_summary": str(data.get("healing_summary") or ""),
            }

            trace_blob = (
                f"Question: {prompt}\n"
                f"Retries: {retries}\n"
                f"Hallucination flagged: {trace['hallucination']}\n"
                f"Answer verified (grounded + relevant): {trace['verified']}\n"
                f"{trace['healing_summary']}"
            )
            exec_summary = summarize_trace_geminiflash(trace_blob)

            _render_process_trace(trace, exec_summary)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "trace": trace,
                    "exec_summary": exec_summary,
                }
            )


def _render_process_trace(trace: dict[str, Any], exec_summary: Optional[str]) -> None:
    with st.expander("Process trace — self-healing pipeline", expanded=False):
        if exec_summary:
            st.markdown(f"**Reliability brief:** {exec_summary}")

        doc_n = trace.get("documents_retrieved", 0)
        st.markdown(
            f"🟢 **Documents retrieved** — {doc_n} graded chunk(s) available for generation."
        )

        hal = trace.get("hallucination", True)
        hal_label = "Yes (critic flagged ungrounded content)" if hal else "No"
        hal_icon = "🟡" if hal else "🟢"
        st.markdown(f"{hal_icon} **Hallucination detected?** — {hal_label}")

        retries = int(trace.get("retries") or 0)
        if retries > 0:
            st.markdown(f"🔄 **Healing attempt(s):** #{retries}")
        else:
            st.markdown("🔄 **Healing attempt(s):** none (first-pass path)")

        ver = trace.get("verified", False)
        if ver:
            st.markdown("✅ **Final answer verified** — grounded in context and relevant to the question.")
        else:
            st.markdown(
                "⚠️ **Final answer not fully verified** — review recommended "
                "(grounding and/or relevance checks did not pass)."
            )

        if trace.get("healing_summary"):
            st.caption(trace["healing_summary"])


if __name__ == "__main__":
    main()
