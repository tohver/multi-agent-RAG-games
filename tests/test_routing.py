"""Routing tests: which nodes run, and why.

These use stub tools, so the whole suite runs offline in milliseconds and costs
nothing. That matters because routing is the part most likely to break during a
refactor, and the end-to-end evaluation needs network, keys and about a minute.

Only `_answer` needs a model, so a subclass overrides it. It has to be a
subclass rather than a patched attribute: `_create_state_machine` binds the
methods when the agent is constructed, so assigning `agent._answer` afterwards
would leave the already-wired node pointing at the original.
"""

import pytest

from research_agent.config import Settings
from research_agent.tools import EvaluationReport, ToolSet
from research_agent.workflow import ResearchAgent


class OfflineAgent(ResearchAgent):
    """A `ResearchAgent` whose answer node does not call a model."""

    def _answer(self, state):
        return {"answer": "stub answer"}


class StubTool:
    """Stands in for a `Tool`, recording that it was called."""

    def __init__(self, name, result):
        self.name = name
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def build_agent(*, docs_useful, cached, web_answer="A web answer"):
    """Build an agent whose tools return exactly what a test needs.

    Args:
        docs_useful: What the judge should say about the retrieved documents.
        cached: Cache entries the lookup should return; empty means a miss.
        web_answer: What a web search would return.

    Returns:
        A tuple of the agent and its stub tools.
    """
    tools = ToolSet(
        retrieve_game=StubTool("retrieve_game", ["[PS5] [Racing] Some Game (2020) - x"]),
        evaluate_retrieval=StubTool(
            "evaluate_retrieval",
            EvaluationReport(useful=docs_useful, description="stub verdict"),
        ),
        game_web_search=StubTool(
            "game_web_search",
            {"answer": web_answer, "sources": [{"title": "T", "url": "http://u"}]},
        ),
        search_memory=StubTool("search_memory", cached),
        register_memory=StubTool("register_memory", "Cached: 'x'"),
    )

    settings = Settings(openai_api_key="test", tavily_api_key="test")
    return OfflineAgent(tools, settings), tools


def path_for(**kwargs):
    """Run one question and return the nodes it visited."""
    agent, tools = build_agent(**kwargs)
    return ResearchAgent.path_of(agent.invoke("a question")), tools


def test_good_documents_answer_without_touching_the_cache():
    path, tools = path_for(docs_useful=True, cached=[])
    assert path == ["retrieve", "evaluate", "answer"]
    assert tools.search_memory.calls == []
    assert tools.game_web_search.calls == []


def test_weak_documents_fall_through_to_the_cache():
    path, tools = path_for(docs_useful=False, cached=["q -> a"])
    assert path == ["retrieve", "evaluate", "recall", "answer"]
    assert tools.game_web_search.calls == []


def test_cache_miss_goes_to_the_web_and_stores_the_result():
    path, tools = path_for(docs_useful=False, cached=[])
    assert path == ["retrieve", "evaluate", "recall", "web_search", "answer", "remember"]
    assert len(tools.register_memory.calls) == 1


def test_cache_hit_is_not_written_back():
    _, tools = path_for(docs_useful=False, cached=["q -> a"])
    assert tools.register_memory.calls == []


def test_a_failed_web_search_is_not_cached():
    agent, tools = build_agent(docs_useful=False, cached=[], web_answer="")
    path = ResearchAgent.path_of(agent.invoke("a question"))
    assert path == ["retrieve", "evaluate", "recall", "web_search", "answer"]
    assert tools.register_memory.calls == []


@pytest.mark.parametrize(
    "state, expected_origin",
    [
        ({"useful": True, "documents": ["d"]}, "the internal game database"),
        ({"useful": False, "cached": ["c"]}, "a previously cached web answer"),
        ({"useful": False, "cached": [], "web_answer": "w", "sources": []}, "a web search"),
    ],
)
def test_context_names_the_source_it_used(state, expected_origin):
    agent, _ = build_agent(docs_useful=True, cached=[])
    origin, _context = agent._context(state)
    assert origin == expected_origin
