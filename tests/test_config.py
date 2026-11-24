import pytest

from news_research.config import ConfigError, Settings, load_settings, require_api_key

ENV_VARS = ["OPENAI_CHAT_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "LLM_TEMPERATURE", "INDEX_DIR"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_fix_the_old_settings():
    s = load_settings()
    assert s.chunk_overlap == 200 and s.chunk_size == 1000  # overlap was 0 before
    assert s.temperature is None  # was 0.9: too high for factual answers
    assert s.chat_model == "gpt-5.5" and s.embedding_model == "text-embedding-3-small"
    assert s.index_dir.name == "faiss_index"


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("TOP_K", "6")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "idx"))
    s = load_settings()
    assert (s.chat_model, s.top_k, s.temperature, s.index_dir) == ("gpt-4.1-mini", 6, 0.1, tmp_path / "idx")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [("TOP_K", "many", "whole number"), ("TOP_K", "0", "between"), ("LLM_TEMPERATURE", "3", "between")],
)
def test_invalid_values_are_rejected_with_the_variable_name(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=name):
        load_settings()


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=500, chunk_overlap=500)


@pytest.mark.parametrize("value", ["", "   ", "your_openai_api_key_here"])
def test_missing_or_placeholder_api_key_is_refused(monkeypatch, value):
    monkeypatch.setenv("OPENAI_API_KEY", value)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        require_api_key()
