import itertools

import pytest
import requests

from news_research.ingestion import extract_article, fetch_articles, ingest_urls, parse_urls
from tests.fakes import PUNCH_URL, TIAGO_HTML, TIAGO_URL, FakeSession


def test_parse_urls_skips_blanks_and_reports_invalid_and_duplicates():
    result = parse_urls(
        ["", f"  {TIAGO_URL}  ", "not a url", f"{PUNCH_URL}\n# comment\n{TIAGO_URL}", "ftp://x.com/a"],
        max_urls=10,
    )
    assert result.valid == [TIAGO_URL, PUNCH_URL]  # empty boxes no longer reach the loader
    assert result.invalid == ["not a url", "ftp://x.com/a"]
    assert result.duplicates == [TIAGO_URL]


def test_parse_urls_enforces_the_limit():
    with pytest.raises(ValueError, match="At most 1"):
        parse_urls([TIAGO_URL, PUNCH_URL], max_urls=1)


def test_extract_article_keeps_text_and_drops_navigation_scripts_and_footer():
    title, text = extract_article(TIAGO_HTML)
    assert title == "Tata Motors launches Tiago iCNG"
    assert "Rs 6.55 lakh" in text
    for noise in ("Home | Markets", "var x", "Copyright"):
        assert noise not in text


def test_fetch_reports_each_failure_and_numbers_only_loaded_articles(settings):
    session = FakeSession(
        {
            "https://a.example/down": requests.ConnectionError("connection refused"),
            "https://a.example/empty": "<html><body><p>Too short.</p></body></html>",
            TIAGO_URL: TIAGO_HTML,
        }
    )
    urls = ["https://a.example/down", "https://a.example/missing", "https://a.example/empty", TIAGO_URL]
    articles, failures = fetch_articles(urls, settings, session)
    assert [a.metadata["article_id"] for a in articles] == ["A1"]
    assert articles[0].metadata["source"] == TIAGO_URL
    reasons = {f.url: f.reason for f in failures}
    assert "could not download" in reasons["https://a.example/down"]
    assert "404" in reasons["https://a.example/missing"]
    assert "no article text" in reasons["https://a.example/empty"]


def test_chunks_overlap_and_keep_article_metadata(settings, pages):
    result = ingest_urls([TIAGO_URL, PUNCH_URL], settings, FakeSession(pages))
    assert len(result.articles) == 2 and not result.failures
    assert len(result.chunks) > 2
    assert all(len(c.page_content) <= settings.chunk_size for c in result.chunks)
    assert [c.metadata["chunk_id"] for c in result.chunks] == list(range(1, len(result.chunks) + 1))
    assert {c.metadata["article_id"] for c in result.chunks} == {"A1", "A2"}


def test_long_paragraphs_are_split_with_overlap(settings):
    from langchain_core.documents import Document

    from news_research.ingestion import split_articles

    sentences = " ".join(f"Sentence number {i} talks about Tata Motors sales." for i in range(30))
    article = Document(page_content=sentences, metadata={"article_id": "A1", "title": "T", "source": "u"})
    chunks = [c.page_content for c in split_articles([article], settings)]
    assert len(chunks) > 3
    # consecutive chunks share text: the 40-character overlap (it was 0 in the old app)
    assert all(a[-25:] in b for a, b in itertools.pairwise(chunks))
