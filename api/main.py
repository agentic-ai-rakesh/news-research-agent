"""FastAPI backend: index news URLs and ask the research agent.

Run with `make backend` (http://127.0.0.1:8000, docs at /docs).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import asdict
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from news_research.agent.research_agent import ask
from news_research.config import ConfigError, Settings, load_settings, require_api_key
from news_research.ingestion import ingest_urls, parse_urls
from news_research.logging_setup import setup_logging
from news_research.models import create_chat_model, create_embeddings
from news_research.vector_store import IndexNotReadyError, build_index, clear_index, read_manifest

load_dotenv()
logger = logging.getLogger("news_research.api")

app = FastAPI(
    title="RockyBot News Research API",
    version="2.0.0",
    description="Index news article URLs and ask a LangGraph research agent about them.",
)

# One index on disk: rebuilding and clearing must not overlap with each other.
_index_lock = threading.Lock()
_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{8,})")


# ---------- dependencies (overridden in tests) ----------


@lru_cache
def get_settings() -> Settings:
    settings = load_settings()
    setup_logging(settings.log_dir)
    return settings


def get_llm(settings: Settings = Depends(get_settings)) -> BaseChatModel:
    return create_chat_model(settings)


def get_embeddings(settings: Settings = Depends(get_settings)) -> Embeddings:
    return create_embeddings(settings)


# ---------- request / response models ----------


class IngestRequest(BaseModel):
    urls: list[str] = Field(..., description="News article URLs (one per item; blank items ignored).")


class AskRequest(BaseModel):
    question: str = Field(..., description="A question about the indexed articles.")


def _safe(error: Exception) -> str:
    """Error text for the client, with anything that looks like an API key removed."""
    return _SECRET.sub("sk-***", str(error)) or type(error).__name__


# ---------- endpoints ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config(settings: Settings = Depends(get_settings)) -> dict:
    """The non-secret settings, so the UI can show what the agent runs with."""
    try:
        require_api_key()
        key_set = True
    except ConfigError:
        key_set = False
    return {
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "max_tool_rounds": settings.max_tool_rounds,
        "max_urls": settings.max_urls,
        "api_key_set": key_set,
    }


@app.get("/api/index")
def get_index(settings: Settings = Depends(get_settings)) -> dict:
    manifest = read_manifest(settings)
    if manifest is None:
        return {"indexed": False, "articles": []}
    stale = manifest.embedding_model != settings.embedding_model
    return {"indexed": not stale, "stale": stale, **asdict(manifest)}


@app.post("/api/ingest")
def ingest(
    request: IngestRequest,
    settings: Settings = Depends(get_settings),
    embeddings: Embeddings = Depends(get_embeddings),
) -> dict:
    try:
        parsed = parse_urls(request.urls, settings.max_urls)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if not parsed.valid:
        raise HTTPException(
            status_code=422,
            detail="Enter at least one valid http(s) URL."
            + (f" Not valid: {', '.join(parsed.invalid)}" if parsed.invalid else ""),
        )
    result = ingest_urls(parsed.valid, settings)
    failures = [asdict(failure) for failure in result.failures]
    if not result.chunks:
        raise HTTPException(
            status_code=422,
            detail={"message": "No article text could be loaded from these URLs.", "failures": failures},
        )
    try:
        with _index_lock:
            manifest = build_index(result.chunks, embeddings, settings)
    except Exception as error:  # embedding API errors, disk errors
        logger.exception("Index build failed")
        raise HTTPException(status_code=502, detail=f"Building the index failed: {_safe(error)}") from error
    return {
        "manifest": asdict(manifest),
        "chunks": len(result.chunks),
        "failures": failures,
        "invalid": parsed.invalid,
        "duplicates": parsed.duplicates,
    }


@app.delete("/api/index")
def delete_index(settings: Settings = Depends(get_settings)) -> dict[str, bool]:
    with _index_lock:
        clear_index(settings)
    return {"cleared": True}


@app.post("/api/ask")
def ask_question(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    llm: BaseChatModel = Depends(get_llm),
    embeddings: Embeddings = Depends(get_embeddings),
) -> dict:
    try:
        answer = ask(request.question, llm, embeddings, settings)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except IndexNotReadyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:  # model / network errors
        logger.exception("Agent run failed")
        raise HTTPException(status_code=502, detail=f"The research agent failed: {_safe(error)}") from error
    return asdict(answer)


@app.exception_handler(ConfigError)
def config_error_handler(_request, error: ConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(error)})
