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

from .lib.llm import LLM
from .lib.memory import LongTermMemory, MemoryFragment, TimestampFilter
from .lib.messages import SystemMessage, UserMessage
from .lib.parsers import PydanticOutputParser
from .lib.tooling import Tool
from .lib.vector_db import VectorStore

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
    reason: str = Field(
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
        """Return the tool names, handy for logging and tests."""
        return [tool.name for tool in self.as_list()]


def build_tools(
    settings: Settings,
    chroma_client: ClientAPI,
    memory: LongTermMemory,
) -> ToolSet:
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
    judge_llm = LLM(
        model=settings.chat_model, temperature=0.0, api_key=settings.openai_api_key
    )
    tavily_client = TavilyClient(api_key=settings.tavily_api_key)

    def retrieve_game(query: str) -> List[str]:
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
                reason=(
                    f"Could not parse the judge's answer ({error}); "
                    f"treating documents as unusable."
                ),
            )

    def game_web_search(question: str) -> Dict:
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
            timestamp_filter=TimestampFilter(newer_than=cutoff),
        )
        return [
            fragment.content
            for fragment, distance in zip(result.fragments, result.distances)
            if distance < settings.cache_hit_distance
        ]

    def register_memory(question: str, answer: str, sources: List[Dict]) -> str:
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
        memory.store(
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
