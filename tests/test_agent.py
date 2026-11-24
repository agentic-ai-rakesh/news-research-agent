import dataclasses

import pytest

from news_research.agent.research_agent import ask
from news_research.agent.tools import make_tools
from news_research.vector_store import IndexNotReadyError, load_index
from tests.fakes import PUNCH_URL, TIAGO_URL, final, scripted, tool_call


def test_agent_searches_then_answers_with_cited_sources(settings, embeddings, indexed):
    llm = scripted(
        tool_call("search_articles", {"query": "Tiago iCNG price"}, "c1"),
        final("The Tiago iCNG starts at Rs 6.55 lakh [A1]."),
    )
    result = ask("What is the price of Tiago iCNG?", llm, embeddings, settings)

    assert result.answer == "The Tiago iCNG starts at Rs 6.55 lakh [A1]."
    assert result.sources == [
        {"article_id": "A1", "title": "Tata Motors launches Tiago iCNG", "source": TIAGO_URL}
    ]
    assert result.grounded
    assert result.steps == [
        {"tool": "search_articles", "args": {"query": "Tiago iCNG price"}, "results": 4, "status": "success"}
    ]
    assert result.retrieved[0]["article_id"] == "A1"  # most relevant passage first
    assert result.usage == {"input_tokens": 30, "output_tokens": 7, "total_tokens": 37}
    assert result.latency_seconds >= 0
    # the passages the tool returned were sent back to the model
    assert "Rs 6.55 lakh" in llm.calls[1]["messages"][-1].content


def test_comparison_question_runs_several_searches(settings, embeddings, indexed):
    llm = scripted(
        tool_call("search_articles", {"query": "Tiago iCNG price"}, "c1"),
        tool_call("search_articles", {"query": "Punch iCNG price", "article_id": "A2"}, "c2"),
        final("Tiago starts at Rs 6.55 lakh [A1]; Punch at Rs 7.1 lakh [A2]."),
    )
    result = ask("Compare the Tiago and Punch iCNG prices", llm, embeddings, settings)
    assert [s["article_id"] for s in result.sources] == ["A1", "A2"]
    assert len(result.steps) == 2
    assert {r["article_id"] for r in result.retrieved if r["source"] == PUNCH_URL} == {"A2"}


def test_tool_rounds_are_capped_and_the_model_must_answer(settings, embeddings, indexed):
    capped = dataclasses.replace(settings, max_tool_rounds=2)
    llm = scripted(
        tool_call("search_articles", {"query": "a"}, "c1"),
        tool_call("search_articles", {"query": "b"}, "c2"),
        final("The indexed articles do not say."),
    )
    result = ask("Unanswerable question?", llm, embeddings, capped)
    assert [call["tools"] for call in llm.calls] == [True, True, False]  # tools removed at the cap
    assert "You have used all your searches" in llm.calls[-1]["messages"][1].content
    assert result.sources == [] and not result.grounded


def test_unknown_citations_are_reported_not_shown_as_sources(settings, embeddings, indexed):
    llm = scripted(final("Prices rose [A9] and [A2]."))
    result = ask("Anything?", llm, embeddings, settings)
    assert [s["article_id"] for s in result.sources] == ["A2"]
    assert result.unknown_citations == ["A9"]
    assert result.steps == []


@pytest.mark.parametrize(("question", "message"), [("   ", "Type a question"), ("x" * 1001, "too long")])
def test_invalid_questions_never_reach_the_model(settings, embeddings, indexed, question, message):
    llm = scripted()
    with pytest.raises(ValueError, match=message):
        ask(question, llm, embeddings, settings)
    assert llm.calls == []


def test_asking_before_indexing_explains_what_to_do(settings, embeddings):
    with pytest.raises(IndexNotReadyError, match="Process URLs"):
        ask("Any news?", scripted(), embeddings, settings)


def test_tools_list_articles_and_filter_by_article(settings, embeddings, indexed):
    store, manifest = load_index(embeddings, settings)
    list_articles, search = make_tools(store, manifest, top_k=3)
    assert "[A1] Tata Motors launches Tiago iCNG" in list_articles.invoke({})

    content, artifact = search.func("price", article_id="A2")
    assert artifact and {a["article_id"] for a in artifact} == {"A2"}
    assert content.startswith("[A2]")

    assert "Unknown article id A7" in search.func("price", article_id="A7")[0]
    assert "Empty query" in search.func("   ")[0]
