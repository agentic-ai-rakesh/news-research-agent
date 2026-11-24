import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from tests.fakes import PUNCH_URL, TIAGO_URL, FakeSession, final, scripted, tool_call


@pytest.fixture
def client(settings, embeddings, pages, monkeypatch):
    monkeypatch.setattr("news_research.ingestion.requests.Session", lambda: FakeSession(pages))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    state = {"llm": scripted()}
    app = api_main.app
    app.dependency_overrides[api_main.get_settings] = lambda: settings
    app.dependency_overrides[api_main.get_embeddings] = lambda: embeddings
    app.dependency_overrides[api_main.get_llm] = lambda: state["llm"]
    test_client = TestClient(app)
    test_client.state = state
    yield test_client
    app.dependency_overrides.clear()


def ingest(client, urls=(TIAGO_URL, PUNCH_URL)):
    return client.post("/api/ingest", json={"urls": list(urls)})


def test_health_and_config(client):
    assert client.get("/health").json() == {"status": "ok"}
    config = client.get("/api/config").json()
    assert config["api_key_set"] is True and config["top_k"] == 4


def test_full_flow_ingest_ask_clear(client):
    assert client.get("/api/index").json() == {"indexed": False, "articles": []}
    assert client.post("/api/ask", json={"question": "Price?"}).status_code == 409

    response = ingest(client, ["", TIAGO_URL, "https://news.example.com/gone", PUNCH_URL, "nope"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert [a["article_id"] for a in body["manifest"]["articles"]] == ["A1", "A2"]
    assert body["invalid"] == ["nope"]
    assert body["failures"][0]["url"] == "https://news.example.com/gone"
    assert client.get("/api/index").json()["indexed"] is True

    client.state["llm"] = scripted(
        tool_call("search_articles", {"query": "Punch iCNG price"}, "c1"),
        final("The Punch iCNG starts at Rs 7.1 lakh [A2]."),
    )
    answer = client.post("/api/ask", json={"question": "Punch iCNG price?"})
    assert answer.status_code == 200, answer.text
    data = answer.json()
    assert data["sources"][0]["source"] == PUNCH_URL
    assert data["steps"][0]["tool"] == "search_articles"
    assert data["prompt_version"] == "1.0"

    assert client.delete("/api/index").json() == {"cleared": True}
    assert client.get("/api/index").json()["indexed"] is False


def test_ingest_rejects_input_without_valid_urls(client):
    response = ingest(client, ["", "  ", "not-a-url"])
    assert response.status_code == 422
    assert "not-a-url" in response.json()["detail"]


def test_ingest_reports_when_no_page_has_article_text(client):
    response = ingest(client, ["https://news.example.com/gone"])
    assert response.status_code == 422
    assert response.json()["detail"]["failures"][0]["url"] == "https://news.example.com/gone"


def test_empty_question_is_422(client):
    ingest(client)
    assert client.post("/api/ask", json={"question": " "}).status_code == 422


def test_model_errors_are_502_with_keys_redacted(client):
    ingest(client)

    class Broken:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("Incorrect API key provided: sk-abcdefghijklmnop")

    client.state["llm"] = Broken()
    response = client.post("/api/ask", json={"question": "Price?"})
    assert response.status_code == 502
    assert "sk-abcdefghijklmnop" not in response.json()["detail"]
    assert "sk-***" in response.json()["detail"]


def test_missing_api_key_is_503_with_the_fix(settings, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = api_main.app
    app.dependency_overrides[api_main.get_settings] = lambda: settings
    try:
        response = TestClient(app).post("/api/ingest", json={"urls": [TIAGO_URL]})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
