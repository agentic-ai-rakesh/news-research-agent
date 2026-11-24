"""Turn news URLs into article Documents and chunks.

Deterministic code, no LLM: validate URLs, fetch each page, extract the article text,
split it into overlapping chunks. Every failure is reported per URL instead of stopping
the whole batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from news_research.config import Settings

logger = logging.getLogger(__name__)

# Tags that hold article text; navigation, scripts and footers are dropped.
_TEXT_TAGS = ("h1", "h2", "h3", "p", "li")
_NOISE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")


@dataclass
class UrlParseResult:
    valid: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)


@dataclass
class FetchFailure:
    url: str
    reason: str


@dataclass
class IngestionResult:
    articles: list[Document]
    chunks: list[Document]
    failures: list[FetchFailure]


def parse_urls(raw_values: list[str] | str, max_urls: int) -> UrlParseResult:
    """Clean user input: one URL per entry or per line; blanks skipped, duplicates removed.

    Only http(s) URLs with a host are accepted. Raises ValueError when more than
    max_urls valid URLs are given.
    """
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    result = UrlParseResult()
    seen: set[str] = set()
    for value in raw_values:
        for line in (value or "").splitlines():
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                result.invalid.append(url)
                continue
            if url in seen:
                result.duplicates.append(url)
                continue
            seen.add(url)
            result.valid.append(url)
    if len(result.valid) > max_urls:
        raise ValueError(f"At most {max_urls} URLs can be processed at once; got {len(result.valid)}.")
    return result


def extract_article(html: str) -> tuple[str, str]:
    """Return (title, text) from an HTML page, keeping headings, paragraphs and list items."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "") or ""
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    container = soup.find("article") or soup.body or soup
    lines: list[str] = []
    for element in container.find_all(_TEXT_TAGS):
        text = element.get_text(" ", strip=True)
        if text and (not lines or lines[-1] != text):
            lines.append(text)
    if not title:
        heading = soup.find("h1")
        title = heading.get_text(" ", strip=True) if heading else ""
    return title, "\n\n".join(lines)


def fetch_articles(
    urls: list[str], settings: Settings, session: requests.Session | None = None
) -> tuple[list[Document], list[FetchFailure]]:
    """Download and extract each URL. Article ids A1, A2, ... follow the input order."""
    http = session or requests.Session()
    headers = {"User-Agent": settings.user_agent}
    articles: list[Document] = []
    failures: list[FetchFailure] = []
    for url in urls:
        try:
            response = http.get(url, headers=headers, timeout=settings.fetch_timeout)
            response.raise_for_status()
        except requests.RequestException as error:
            logger.warning("Fetch failed for %s: %s", url, error)
            failures.append(FetchFailure(url, f"could not download the page ({error})"))
            continue
        title, text = extract_article(response.text)
        if len(text) < settings.min_article_chars:
            failures.append(
                FetchFailure(url, "no article text found (the page may need JavaScript or block bots)")
            )
            continue
        article_id = f"A{len(articles) + 1}"
        articles.append(
            Document(
                page_content=text,
                metadata={"article_id": article_id, "source": url, "title": title or url},
            )
        )
        logger.info("Fetched %s (%s, %d chars)", url, article_id, len(text))
    return articles, failures


def split_articles(articles: list[Document], settings: Settings) -> list[Document]:
    """Split into overlapping chunks; each chunk keeps its article's id, title and URL."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(articles)
    for number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = number
    return chunks


def ingest_urls(
    urls: list[str], settings: Settings, session: requests.Session | None = None
) -> IngestionResult:
    articles, failures = fetch_articles(urls, settings, session)
    chunks = split_articles(articles, settings) if articles else []
    return IngestionResult(articles=articles, chunks=chunks, failures=failures)
