"""Tools the research agent can call. Built per request around the loaded index."""

from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.tools import BaseTool, tool

from news_research.faiss_store import FaissStore
from news_research.vector_store import Manifest

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 300


def make_tools(store: FaissStore, manifest: Manifest, top_k: int) -> list[BaseTool]:
    known_ids = {article.article_id for article in manifest.articles}

    @tool
    def list_articles() -> str:
        """List the indexed news articles with their ids, titles and URLs."""
        return "\n".join(
            f"[{a.article_id}] {a.title} ({a.source}, {a.chunks} chunks)" for a in manifest.articles
        )

    @tool(response_format="content_and_artifact")
    def search_articles(
        query: Annotated[str, "Short, focused search text, e.g. 'Tiago iCNG price'."],
        article_id: Annotated[
            str | None, "Optional article id (A1, A2, ...) to search only that article."
        ] = None,
    ) -> tuple[str, list[dict]]:
        """Semantic search over the indexed article text. Returns passages tagged [A#]."""
        query = (query or "").strip()[:MAX_QUERY_CHARS]
        if not query:
            return "Empty query: give a few words to search for.", []
        if article_id and article_id not in known_ids:
            return f"Unknown article id {article_id}. Known ids: {', '.join(sorted(known_ids))}.", []
        search_filter = {"article_id": article_id} if article_id else None
        hits = store.similarity_search_with_score(query, k=top_k, filter=search_filter)
        if not hits:
            return "No matching passages.", []
        passages, artifact = [], []
        for document, score in hits:
            meta = document.metadata
            passages.append(f"[{meta['article_id']}] {meta['title']}\n{document.page_content}")
            artifact.append(
                {
                    "article_id": meta["article_id"],
                    "title": meta["title"],
                    "source": meta["source"],
                    "chunk_id": meta.get("chunk_id"),
                    "score": round(score, 4),
                }
            )
        logger.info("search_articles(%r, %s) -> %d passages", query, article_id, len(hits))
        return "\n\n---\n\n".join(passages), artifact

    return [list_articles, search_articles]
