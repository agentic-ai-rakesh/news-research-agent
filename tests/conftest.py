from __future__ import annotations

import pytest

from news_research.config import Settings
from news_research.ingestion import ingest_urls
from news_research.vector_store import build_index
from tests.fakes import PUNCH_HTML, PUNCH_URL, TIAGO_HTML, TIAGO_URL, FakeSession, KeywordEmbeddings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chunk_size=200,
        chunk_overlap=40,
        index_dir=tmp_path / "index",
        log_dir=tmp_path / "logs",
        embedding_model="fake-keyword-embeddings",
        min_article_chars=100,
    )


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def pages() -> dict:
    return {TIAGO_URL: TIAGO_HTML, PUNCH_URL: PUNCH_HTML}


@pytest.fixture
def indexed(settings, embeddings, pages):
    """An index of the two sample articles (A1 = Tiago, A2 = Punch)."""
    result = ingest_urls([TIAGO_URL, PUNCH_URL], settings, FakeSession(pages))
    return build_index(result.chunks, embeddings, settings)
