"""Creates the OpenAI chat model and embeddings from Settings (the only provider code)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from news_research.config import Settings, require_api_key


def create_chat_model(settings: Settings) -> BaseChatModel:
    require_api_key()
    kwargs = {
        "model": settings.chat_model,
        "max_tokens": settings.max_answer_tokens,
        "timeout": settings.request_timeout,
        "max_retries": settings.max_retries,
    }
    if settings.temperature is not None:  # GPT-5.x reasoning models reject temperature
        kwargs["temperature"] = settings.temperature
    return ChatOpenAI(**kwargs)


def create_embeddings(settings: Settings) -> Embeddings:
    require_api_key()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
