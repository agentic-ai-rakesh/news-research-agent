"""A small FAISS vector store: cosine similarity, JSON metadata, no pickle.

Replaces langchain_community's FAISS wrapper (that package is being sunset, and its
save format needs pickle deserialization to load).
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


def _normalized(vectors: list[list[float]]) -> np.ndarray:
    array = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


class FaissStore:
    """Inner-product index over L2-normalized vectors, i.e. cosine similarity in [-1, 1]."""

    def __init__(self, index: faiss.Index, documents: list[Document], embeddings: Embeddings):
        if index.ntotal != len(documents):
            raise ValueError("The index and the stored chunks do not match.")
        self.index, self.documents, self.embeddings = index, documents, embeddings

    @classmethod
    def from_documents(cls, documents: list[Document], embeddings: Embeddings) -> FaissStore:
        if not documents:
            raise ValueError("There are no chunks to index.")
        vectors = _normalized(embeddings.embed_documents([d.page_content for d in documents]))
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, documents, embeddings)

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Top k chunks by cosine similarity (highest first), optionally filtered by metadata."""
        vector = _normalized([self.embeddings.embed_query(query)])
        # With a filter, rank everything and keep the matches (indexes here are small).
        limit = self.index.ntotal if filter else min(k, self.index.ntotal)
        scores, ids = self.index.search(vector, limit)
        hits: list[tuple[Document, float]] = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0:
                continue
            document = self.documents[idx]
            if filter and any(document.metadata.get(key) != value for key, value in filter.items()):
                continue
            hits.append((document, float(score)))
            if len(hits) == k:
                break
        return hits

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [document for document, _ in self.similarity_search_with_score(query, k)]

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / INDEX_FILE))
        chunks = [{"text": d.page_content, "metadata": d.metadata} for d in self.documents]
        (folder / CHUNKS_FILE).write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, embeddings: Embeddings) -> FaissStore:
        index = faiss.read_index(str(folder / INDEX_FILE))
        chunks = json.loads((folder / CHUNKS_FILE).read_text(encoding="utf-8"))
        documents = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
        return cls(index, documents, embeddings)
