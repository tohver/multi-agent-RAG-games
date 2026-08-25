"""The five tools the agent is built from.

Each tool is created by `build_tools`, which closes over the settings and the
already-open clients. Nothing here runs at import time, so importing the module
never opens a database or needs an API key.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List

from chromadb.api import ClientAPI
from pydantic import BaseModel, Field
from tavily import TavilyClient

from .framework.llm import LLM
from .framework.memory import LongTermMemory, MemoryFragment, TimestampFilter
from .framework.messages import SystemMessage, UserMessage
from .framework.parsers import PydanticOutputParser
from .framework.tooling import Tool
from .framework.vector_db import VectorStore

from .config import Settings

JUDGE_INSTRUCTIONS = (
    "Your task is to evaluate if the documents are enough to respond the query. "
    "Give a detailed explanation, so it's possible to take an action to accept "
    "it or not."
)


class EvaluationReport(BaseModel):
    """Structured verdict of the LLM judge on a set of retrieved documents."""

    useful: bool = Field(
        description="Whether the retrieved documents are sufficient to answer the question"
    )
    description: str = Field(
        description="Detailed explanation of the verdict, so a caller can act on it"
    )


@dataclass(frozen=True)
class ToolSet:
    """The five tools, named so the workflow can wire them to nodes."""

    retrieve_game: Tool
    evaluate_retrieval: Tool
    game_web_search: Tool
    search_memory: Tool
    register_memory: Tool

    def as_list(self) -> List[Tool]:
        '''
        In plain English: hands back the five tools as a plain list.

        The agent refers to each tool by name, but anything that just wants to look at
        all of them - a log line, a test, a check that nothing is missing - wants them
        in a row. This keeps that order in one place so it is always the same.

        Output: the five tools in pipeline order. Used mainly by `names` below.
        '''
        """Return the tools in pipeline order."""
        return [
            self.retrieve_game,
            self.evaluate_retrieval,
            self.game_web_search,
            self.search_memory,
            self.register_memory,
        ]

    @property
    def names(self) -> List[str]:
        '''
        In plain English: the names of the five tools, as text.

        Useful when you want to see what the agent is equipped with without printing the
        tools themselves, which would be a wall of object addresses.

        Output: a list like `['retrieve_game', 'evaluate_retrieval', ...]`. Printed at
        startup and used in tests to confirm the full set was built.
        '''
        """Return the tool names, handy for logging and tests."""
        return [tool.name for tool in self.as_list()]


def build_tools(
    settings: Settings,
    chroma_client: ClientAPI,
    memory: LongTermMemory,
) -> ToolSet:
    '''
    In plain English: this builds the five tools the agent works with, and connects
    each one to the things it needs - the database, the API clients, your settings.

    Tools are created here rather than at the top of the file for one reason: a tool
    needs an open database and a live API client, and those should not spring into
    existence merely because someone imported a file. Building them inside a
    function means nothing happens until you ask for it, and it means a test can
    build a second set pointed somewhere else.

    Each tool below is defined as an ordinary Python function and then wrapped, so
    the model can call it. The wrapper reads the function's own description and
    argument types to work out how to describe it to the model.

    Output: a `ToolSet` - the five finished tools. It goes straight to
    `ResearchAgent`, which turns each one into a step in the pipeline.
    '''
    """Create the five tools bound to one configuration and one set of clients.

    Args:
        settings: Runtime configuration.
        chroma_client: Client holding the game collection.
        memory: The web-answer cache.

    Returns:
        A `ToolSet` ready to hand to `ResearchAgent`.
    """
    game_store = VectorStore(chroma_client.get_collection(settings.collection_name))

    # One LLM per role, built once. Each LLM owns an HTTP connection pool, so
    # rebuilding them per call would throw away connection reuse for nothing.
    judge_llm = LLM(model=settings.model, temperature=0.0, api_key=settings.openai_api_key)
    tavily_client = TavilyClient(api_key=settings.tavily_api_key)

    def retrieve_game(query: str) -> List[str]:
        '''
        In plain English: searches the local game database for whatever the question is
        about.

        It compares the meaning of the question against the meaning of every stored
        game, so it can find "the first 3D Mario game" even though those exact words
        appear nowhere in the data. This is the first thing the agent tries for every
        question, because a local lookup is faster and cheaper than a web search.

        Output: the closest game records as text - platform, genre, name, year and
        description. They are not an answer yet; they are raw material. The next step
        judges whether they are good enough, and the answer step turns them into a
        sentence.
        '''
        """
        Semantic search over the video game vector database.

        Args:
            query: A natural-language question about the game industry.

        Returns:
            A list of matching game records (platform, name, year, description),
            ordered by relevance.
        """
        # A plain similarity search, not a RAG run: the answer node composes the
        # prose later, so generating one here would be a discarded LLM call.
        results = game_store.query(
            query_texts=[query], n_results=settings.retrieval_results
        )
        documents = results.get("documents") or []
        return documents[0] if documents else []

    def evaluate_retrieval(question: str, retrieved_docs: List[str]) -> EvaluationReport:
        '''
        In plain English: a second opinion on whether the search actually found
        anything useful.

        This exists because a similarity search always returns its closest matches, even
        when none of them is relevant. Ask about a game that is not in the database and
        you still get three games back - just wrong ones. Without this check the agent
        would confidently answer from them.

        So the question and the documents go to the model with one job: are these enough
        to answer this? The reply comes back as a structured yes/no plus a reason,
        rather than a paragraph, because the pipeline has to branch on it.

        Output: a report with `useful` (true/false) and `description` (why). The
        true/false is the single most important value in the whole pipeline - it decides
        whether the agent answers now or starts looking elsewhere. If the reply cannot
        be read, it deliberately returns false, so a garbled verdict sends the question
        onward rather than being mistaken for approval.
        '''
        """
        Based on the user's question and on the list of retrieved documents,
        it will analyze the usability of the documents to respond to that question.

        Args:
            question: original question from user
            retrieved_docs: retrieved documents most similar to the user query
                in the Vector Database

        Returns:
            useful: whether the documents are useful to answer the question
            description: description about the evaluation result
        """
        context = "\n\n".join(f"- {doc}" for doc in retrieved_docs) or (
            "(no documents retrieved)"
        )

        ai_message = judge_llm.invoke(
            [
                SystemMessage(content=JUDGE_INSTRUCTIONS),
                UserMessage(
                    content=(
                        f"# Question:\n{question}\n\n"
                        f"# Retrieved documents:\n{context}\n\n"
                        "# Evaluation:"
                    )
                ),
            ],
            response_format=EvaluationReport,
        )

        try:
            return PydanticOutputParser(model_class=EvaluationReport).parse(ai_message)
        except Exception as error:
            # Fail closed: an unreadable verdict must not be taken as "good
            # enough", it should push the pipeline on to the web instead.
            return EvaluationReport(
                useful=False,
                description=(
                    f"Could not parse the judge's answer ({error}); "
                    f"treating documents as unusable."
                ),
            )

    def game_web_search(question: str) -> Dict:
        '''
        In plain English: looks the question up on the internet, as a last resort.

        Reached only when the local database came up short and no past answer was
        cached. It uses a search service that reads the results and writes a short
        answer, so the agent gets a usable summary rather than ten links.

        Most of the page text is thrown away on purpose. Keeping it would add thousands
        of words to the conversation with the model - slow, expensive, and mostly
        irrelevant. A short excerpt per source is enough to check the summary is not
        invented.

        Output: the summary, the sources it came from, and the time of the search. If
        the search itself fails, it returns an error message instead of crashing, so one
        bad request cannot take down the whole run.
        '''
        """
        Search the web for information about the video game industry.

        Use this when the internal vector database does not contain the answer.

        Args:
            question: a question about game industry.

        Returns:
            answer: a short synthesized answer built from the search results
            sources: list of {title, url, snippet} backing that answer
            retrieved_at: when the search ran, for time-sensitive questions
        """
        try:
            response = tavily_client.search(
                query=question,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                max_results=settings.max_web_results,
            )
        except Exception as error:
            # A failing search must not crash the run - report it instead.
            return {"answer": "", "sources": [], "error": f"Web search failed: {error}"}

        return {
            "answer": response.get("answer", ""),
            "sources": [
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    # The full page text is ~2.6k tokens per search; a snippet
                    # is enough to sanity-check the synthesized answer.
                    "snippet": (result.get("content") or "")[: settings.snippet_chars],
                }
                for result in response.get("results", [])
            ],
            "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        }

    def search_memory(query: str) -> List[str]:
        '''
        In plain English: checks whether this question was already answered from the
        internet before, so the search does not have to be paid for twice.

        It only counts as a match if the stored question means very nearly the same
        thing - a merely related question is not good enough, or the agent would answer
        a question about one game using the answer about another. Entries older than the
        cache lifetime are ignored, because facts like "is it on PS5" can change.

        Output: the stored `question -> answer` text, or an empty list if there is no
        usable match. An empty list is what sends the agent to the web.
        '''
        """
        Look for a web answer to this question that was already found and cached.

        Only entries newer than the cache TTL are considered, and only ones
        close enough to be the same question rather than merely a related one.

        Args:
            query: the user's question

        Returns:
            The cached "question -> answer" entry, or an empty list on a miss.
        """
        cutoff = int(
            (datetime.now() - timedelta(days=settings.cache_ttl_days)).timestamp()
        )
        result = memory.search(
            query_text=query,
            owner=settings.memory_owner,
            namespace=settings.memory_namespace,
            limit=1,
            timestamp_filter=TimestampFilter(greater_than_value=cutoff),
        )
        return [
            fragment.content
            for fragment, distance in zip(result.fragments, result.metadata["distances"])
            if distance < settings.cache_hit_distance
        ]

    def register_memory(question: str, answer: str, sources: List[Dict]) -> str:
        '''
        In plain English: saves an answer that cost a web search, so next time it is
        free.

        Only the finished answer is stored, never the articles it came from. The answer
        step has already condensed them, so keeping the sources as well would waste
        space and be re-read into the conversation on every future match.

        The question is stored alongside the answer, in the same searchable text. That
        is deliberate: the next lookup compares against a question, so having the
        original wording in there makes the match far more reliable.

        Output: a short confirmation line, mostly for logs. The real result is the entry
        now sitting in the cache, waiting for the next time someone asks the same thing.
        '''
        """
        Cache an answer that required a web search, so the same question does
        not have to hit the web again.

        The embedded text is "question -> answer" - keeping the question in it
        makes a later lookup match on the question's wording. The source URLs go
        to metadata instead, so they stay available without polluting the
        embedding.

        Args:
            question: the user's original question
            answer: the final answer produced from the web results
            sources: the {title, url, snippet} entries the answer came from

        Returns:
            A short description of what was stored.
        """
        content = f"{question} -> {answer}"
        memory.register(
            MemoryFragment(
                content=content,
                owner=settings.memory_owner,
                namespace=settings.memory_namespace,
            ),
            # Chroma metadata takes scalars only, so URLs are joined into one string.
            metadata={
                "sources": ", ".join(source.get("url", "") for source in sources),
                "retrieved_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return f"Cached: {content[:70]!r}"

    return ToolSet(
        retrieve_game=Tool(retrieve_game),
        evaluate_retrieval=Tool(evaluate_retrieval),
        game_web_search=Tool(game_web_search),
        search_memory=Tool(search_memory),
        register_memory=Tool(register_memory),
    )
