"""The FAISS index on disk, plus a manifest that says what is in it.

The manifest records the embedding model and the articles. An index built with a
different embedding model (for example an index from an older version) is refused instead of
silently returning wrong matches.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from news_research.config import Settings
from news_research.faiss_store import CHUNKS_FILE, INDEX_FILE, FaissStore

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"


class IndexNotReadyError(RuntimeError):
    """No usable index: nothing processed yet, or it was built with other settings."""


@dataclass
class ArticleInfo:
    article_id: str
    title: str
    source: str
    chunks: int


@dataclass
class Manifest:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    built_at: str
    articles: list[ArticleInfo]

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        return cls(
            embedding_model=data["embedding_model"],
            chunk_size=data["chunk_size"],
            chunk_overlap=data["chunk_overlap"],
            built_at=data["built_at"],
            articles=[ArticleInfo(**article) for article in data["articles"]],
        )


def build_index(chunks: list[Document], embeddings: Embeddings, settings: Settings) -> Manifest:
    """Embed the chunks, replace the index on disk, and write the manifest."""
    if not chunks:
        raise ValueError("There are no chunks to index.")
    store = FaissStore.from_documents(chunks, embeddings)
    counts: dict[str, ArticleInfo] = {}
    for chunk in chunks:
        meta = chunk.metadata
        info = counts.setdefault(
            meta["article_id"], ArticleInfo(meta["article_id"], meta["title"], meta["source"], 0)
        )
        info.chunks += 1
    manifest = Manifest(
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        articles=list(counts.values()),
    )
    # Build in a temporary folder, then swap, so a failure never leaves half an index.
    index_dir = settings.index_dir
    staging = index_dir.with_name(index_dir.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    store.save(staging)
    (staging / MANIFEST_FILE).write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(index_dir)
    logger.info("Index built: %d chunks from %d articles", len(chunks), len(manifest.articles))
    return manifest


def read_manifest(settings: Settings) -> Manifest | None:
    path = settings.index_dir / MANIFEST_FILE
    folder = settings.index_dir
    if not path.exists() or not (folder / INDEX_FILE).exists() or not (folder / CHUNKS_FILE).exists():
        return None
    try:
        return Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError) as error:
        logger.warning("Unreadable manifest %s: %s", path, error)
        return None


def load_index(embeddings: Embeddings, settings: Settings) -> tuple[FaissStore, Manifest]:
    """Load the saved index if it exists and matches the configured embedding model."""
    manifest = read_manifest(settings)
    if manifest is None:
        raise IndexNotReadyError("No articles are indexed yet. Add URLs and click Process URLs.")
    if manifest.embedding_model != settings.embedding_model:
        raise IndexNotReadyError(
            f"The index was built with {manifest.embedding_model}, but the app now uses "
            f"{settings.embedding_model}. Process the URLs again to rebuild it."
        )
    return FaissStore.load(settings.index_dir, embeddings), manifest


def clear_index(settings: Settings) -> None:
    shutil.rmtree(settings.index_dir, ignore_errors=True)
