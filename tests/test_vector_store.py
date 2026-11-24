import dataclasses

import pytest

from news_research.vector_store import IndexNotReadyError, clear_index, load_index, read_manifest


def test_build_writes_index_and_manifest(settings, indexed):
    assert (settings.index_dir / "index.faiss").exists()
    assert not settings.index_dir.with_name("index.tmp").exists()
    manifest = read_manifest(settings)
    assert [a.article_id for a in manifest.articles] == ["A1", "A2"]
    assert manifest.embedding_model == "fake-keyword-embeddings"
    assert sum(a.chunks for a in manifest.articles) > 2


def test_load_returns_a_searchable_store(settings, embeddings, indexed):
    store, _ = load_index(embeddings, settings)
    top = store.similarity_search("Punch sunroof touchscreen", k=1)[0]
    assert top.metadata["article_id"] == "A2"


def test_missing_index_is_reported(settings, embeddings):
    with pytest.raises(IndexNotReadyError, match="No articles are indexed"):
        load_index(embeddings, settings)


def test_index_from_another_embedding_model_is_refused(settings, embeddings, indexed):
    other = dataclasses.replace(settings, embedding_model="text-embedding-3-small")
    with pytest.raises(IndexNotReadyError, match="Process the URLs again"):
        load_index(embeddings, other)


def test_clear_index(settings, indexed):
    clear_index(settings)
    assert read_manifest(settings) is None


def test_faiss_store_round_trip_scores_and_filter(tmp_path, embeddings):
    from langchain_core.documents import Document

    from news_research.faiss_store import CHUNKS_FILE, FaissStore

    docs = [
        Document(page_content="Tiago iCNG price Rs 6.55 lakh", metadata={"article_id": "A1"}),
        Document(page_content="Punch iCNG sunroof and touchscreen", metadata={"article_id": "A2"}),
        Document(page_content="Punch price Rs 7.1 lakh", metadata={"article_id": "A2"}),
    ]
    FaissStore.from_documents(docs, embeddings).save(tmp_path)
    assert (tmp_path / CHUNKS_FILE).read_text().startswith("[")  # JSON, not pickle

    store = FaissStore.load(tmp_path, embeddings)
    hits = store.similarity_search_with_score("Tiago price", k=3)
    assert hits[0][0].metadata["article_id"] == "A1"
    assert [score for _, score in hits] == sorted((score for _, score in hits), reverse=True)
    assert -1.0 <= hits[-1][1] <= hits[0][1] <= 1.0

    filtered = store.similarity_search_with_score("Tiago price", k=5, filter={"article_id": "A2"})
    assert len(filtered) == 2 and {d.metadata["article_id"] for d, _ in filtered} == {"A2"}


def test_faiss_store_rejects_mismatched_files(tmp_path, embeddings):
    import json

    from langchain_core.documents import Document

    from news_research.faiss_store import CHUNKS_FILE, FaissStore

    docs = [Document(page_content="a b"), Document(page_content="c d")]
    FaissStore.from_documents(docs, embeddings).save(tmp_path)
    (tmp_path / CHUNKS_FILE).write_text(json.dumps([{"text": "a b", "metadata": {}}]))
    with pytest.raises(ValueError, match="do not match"):
        FaissStore.load(tmp_path, embeddings)
