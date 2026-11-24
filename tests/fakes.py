"""Test doubles: no network, no API key."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DIM = 256


class KeywordEmbeddings(Embeddings):
    """Bag-of-words hashing: texts sharing words get similar vectors, so ranking is real."""

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted AIMessages in order and records what it was sent."""

    responses: list[AIMessage]
    calls: list[dict[str, Any]] = []
    bound_tools: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound_tools": True})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        self.calls.append({"messages": messages, "tools": self.bound_tools})
        if not self.responses:
            raise AssertionError("ScriptedChatModel ran out of responses")
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message.model_copy())])


def tool_call(name: str, args: dict, call_id: str, tokens: int = 10) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": tokens, "output_tokens": 2, "total_tokens": tokens + 2},
    )


def final(text: str, tokens: int = 20) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": tokens, "output_tokens": 5, "total_tokens": tokens + 5},
    )


def scripted(*responses: AIMessage) -> ScriptedChatModel:
    # shared list objects so copies made by bind_tools still pop/record on the same script
    return ScriptedChatModel(responses=list(responses), calls=[])


class FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text, self.status_code = text, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    """requests.Session stand-in: url -> html, or an exception to raise."""

    def __init__(self, pages: dict[str, Any]):
        self.pages = pages
        self.requested: list[str] = []

    def get(self, url: str, headers=None, timeout=None):
        self.requested.append(url)
        page = self.pages.get(url)
        if isinstance(page, Exception):
            raise page
        if page is None:
            return FakeResponse("not found", 404)
        return FakeResponse(page)


def article_html(title: str, paragraphs: list[str]) -> str:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        f"<html><head><title>{title}</title><script>var x=1;</script></head><body>"
        f"<nav>Home | Markets | Login</nav><article><h1>{title}</h1>{body}</article>"
        f"<footer>Copyright footer text</footer></body></html>"
    )


TIAGO_URL = "https://news.example.com/tata-tiago-icng"
PUNCH_URL = "https://news.example.com/tata-punch-icng"

TIAGO_HTML = article_html(
    "Tata Motors launches Tiago iCNG",
    [
        "Tata Motors launched the Tiago iCNG on Monday with prices starting at Rs 6.55 lakh.",
        "The Tiago iCNG uses a 1.2 litre engine and comes with twin cylinder technology for boot space.",
        "Dealers report strong early bookings for the CNG hatchback across northern India.",
    ],
)
PUNCH_HTML = article_html(
    "Tata Punch iCNG price starts at Rs 7.1 lakh",
    [
        "The Punch iCNG micro SUV is priced from Rs 7.1 lakh, ex-showroom Delhi.",
        "Punch iCNG features a sunroof, voice assisted controls and a 7 inch touchscreen.",
        "Tata says the Punch is its fastest selling SUV, crossing one lakh units this year.",
    ],
)
