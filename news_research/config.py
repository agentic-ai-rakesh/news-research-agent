"""Settings read from environment variables (and .env), with validated defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """A setting is missing or invalid."""


def _int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}.") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _optional_float(name: str, minimum: float, maximum: float) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _path(name: str, default: str) -> Path:
    path = Path(os.getenv(name) or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Every tunable value in one place. See .env.example for the variables."""

    chat_model: str = "gpt-5.5"
    embedding_model: str = "text-embedding-3-small"
    temperature: float | None = None  # None = model default (GPT-5.x reasoning models need it)
    max_answer_tokens: int = 1500
    request_timeout: int = 60
    max_retries: int = 2
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 4
    max_tool_rounds: int = 4
    max_urls: int = 10
    max_question_chars: int = 1000
    fetch_timeout: int = 20
    min_article_chars: int = 200
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "faiss_index")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    user_agent: str = "RockyBot-NewsResearch/2.0 (+https://github.com/agentic-ai-rakesh)"

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")


def load_settings() -> Settings:
    """Build Settings from the environment; raises ConfigError with a readable message."""
    return Settings(
        chat_model=os.getenv("OPENAI_CHAT_MODEL") or Settings.chat_model,
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL") or Settings.embedding_model,
        temperature=_optional_float("LLM_TEMPERATURE", 0.0, 2.0),
        max_answer_tokens=_int("LLM_MAX_TOKENS", Settings.max_answer_tokens, 100, 32000),
        request_timeout=_int("LLM_TIMEOUT_SECONDS", Settings.request_timeout, 5, 600),
        max_retries=_int("LLM_MAX_RETRIES", Settings.max_retries, 0, 10),
        chunk_size=_int("CHUNK_SIZE", Settings.chunk_size, 200, 8000),
        chunk_overlap=_int("CHUNK_OVERLAP", Settings.chunk_overlap, 0, 4000),
        top_k=_int("TOP_K", Settings.top_k, 1, 20),
        max_tool_rounds=_int("AGENT_MAX_TOOL_ROUNDS", Settings.max_tool_rounds, 1, 10),
        max_urls=_int("MAX_URLS", Settings.max_urls, 1, 50),
        fetch_timeout=_int("FETCH_TIMEOUT_SECONDS", Settings.fetch_timeout, 3, 120),
        index_dir=_path("INDEX_DIR", "data/faiss_index"),
        log_dir=_path("LOG_DIR", "logs"),
    )


def require_api_key() -> None:
    """Fail early, with the fix in the message, when the OpenAI key is missing."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("your_"):
        raise ConfigError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your OpenAI API key.")
