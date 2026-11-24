"""Streamlit UI for RockyBot. It only talks to the FastAPI backend (make backend).

Run with `make frontend` (http://localhost:8501).
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("NEWS_RESEARCH_API_URL", "http://127.0.0.1:8000").rstrip("/")
URL_BOXES = 3
LOGO = Path(__file__).resolve().parent.parent / "rockybot.jpg"


# ---------- API calls ----------


def _detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    if isinstance(detail, dict):
        lines = [detail.get("message", "")]
        lines += [f"- {f['url']}: {f['reason']}" for f in detail.get("failures", [])]
        return "\n".join(lines)
    return str(detail)


def api(method: str, path: str, timeout: int = 300, **kwargs) -> dict | None:
    """Call the backend; show a readable error and return None on failure."""
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with `make backend`.")
        return None
    except requests.Timeout:
        st.error("The API took too long to answer. Try again.")
        return None
    if not response.ok:
        st.error(_detail(response))
        return None
    return response.json()


# ---------- page ----------

st.set_page_config(page_title="RockyBot: News Research", page_icon="📈", layout="wide")
st.title("RockyBot: News Research Agent 📈")
st.caption(
    "Index news articles, then ask questions. A LangGraph agent plans searches over the "
    "articles and answers with [A#] citations."
)

if "chat" not in st.session_state:
    st.session_state.chat = []

config = api("GET", "/api/config", timeout=10)
index = api("GET", "/api/index", timeout=10)

# ---------- sidebar: URLs and index ----------

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width="stretch")
    st.header("News article URLs")
    if config and not config["api_key_set"]:
        st.warning("OPENAI_API_KEY is not set in .env, so indexing and answers will fail.")

    typed = [st.text_input(f"URL {i + 1}", key=f"url_{i}") for i in range(URL_BOXES)]
    extra = st.text_area("More URLs (one per line)", height=80)
    uploaded = st.file_uploader("Or upload a .txt file of URLs", type=["txt"])
    file_urls = uploaded.getvalue().decode("utf-8", errors="ignore") if uploaded else ""

    if st.button("Process URLs", type="primary", width="stretch"):
        urls = [*typed, extra, file_urls]
        with st.status("Fetching, chunking and embedding articles…", expanded=True) as status:
            result = api("POST", "/api/ingest", json={"urls": urls})
            if result:
                manifest = result["manifest"]
                st.write(f"✅ {len(manifest['articles'])} articles, {result['chunks']} chunks indexed.")
                for failure in result["failures"]:
                    st.write(f"⚠️ Skipped {failure['url']}: {failure['reason']}")
                for url in result["invalid"]:
                    st.write(f"⚠️ Not a valid URL: {url}")
                status.update(label="Index ready", state="complete")
                st.session_state.chat = []
                index = api("GET", "/api/index", timeout=10)
            else:
                status.update(label="Indexing failed", state="error")

    st.divider()
    st.subheader("Indexed articles")
    if index and index.get("stale"):
        st.warning("The saved index uses another embedding model. Process the URLs again.")
    elif index and index.get("indexed"):
        for article in index["articles"]:
            st.markdown(f"**[{article['article_id']}]** [{article['title']}]({article['source']})")
            st.caption(f"{article['chunks']} chunks")
        st.caption(f"Built {index['built_at']} · {index['embedding_model']}")
        if st.button("Clear index", width="stretch"):
            if api("DELETE", "/api/index", timeout=10):
                st.session_state.chat = []
                st.rerun()
    else:
        st.caption("Nothing indexed yet.")

    if config:
        st.divider()
        st.caption(
            f"Model: {config['chat_model']} · top K {config['top_k']} · "
            f"max {config['max_tool_rounds']} search rounds · chunks "
            f"{config['chunk_size']}/{config['chunk_overlap']}"
        )

# ---------- chat ----------


def show_answer(result: dict) -> None:
    st.markdown(result["answer"])
    if result["sources"]:
        st.markdown("**Sources**")
        for source in result["sources"]:
            st.markdown(f"- [{source['article_id']}] [{source['title']}]({source['source']})")
    else:
        st.caption("No article was cited in this answer.")
    usage = result.get("usage", {})
    with st.expander("How the agent worked"):
        st.caption(
            f"{result['model']} · {result['latency_seconds']}s · "
            f"{usage.get('total_tokens', 0)} tokens (in {usage.get('input_tokens', 0)}, "
            f"out {usage.get('output_tokens', 0)}) · prompt v{result['prompt_version']}"
        )
        for number, step in enumerate(result["steps"], start=1):
            found = f" → {step['results']} passages" if step["results"] is not None else ""
            st.markdown(f"{number}. `{step['tool']}` {step['args']}{found}")
        if not result["steps"]:
            st.markdown("The agent answered without searching.")
        if result.get("unknown_citations"):
            st.warning(f"Cited ids not in the index: {', '.join(result['unknown_citations'])}")


for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        show_answer(turn)

ready = bool(index and index.get("indexed"))
question = st.chat_input(
    "Ask about the indexed articles" if ready else "Process some URLs first", disabled=not ready
)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Researching…"):
            result = api("POST", "/api/ask", json={"question": question})
        if result:
            show_answer(result)
            st.session_state.chat.append(result)
