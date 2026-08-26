"""Configuration for the research agent.

Everything tunable lives here, resolved once from the environment. Importing
this module has no side effects beyond reading `.env`; nothing connects to a
database or an API until `Settings.from_env()` is used to build the app.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class MissingCredentialsError(RuntimeError):
    """Raised when a required API key is absent from the environment."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration.

    Attributes:
        openai_api_key: Key for chat completions and embeddings.
        tavily_api_key: Key for the web search tool.
        chat_model: Chat model used by every LLM role in the pipeline. The
            embedding model is a separate one, owned by ChromaDB.
        chroma_path: Directory holding the persistent Chroma database.
        games_path: Directory of source JSON files, read by the indexer.
        skip_report_path: Where the indexer lists source files it had to
            skip. Deleted again on a clean run, so a stale report can never
            be mistaken for a current one.
        collection_name: Collection with the indexed game documents. Created by
            the indexer, so this name is fixed rather than free choice.
        memory_collection: Collection holding the web-answer cache.
        memory_owner: Owner label every cache entry is stored under.
        memory_namespace: Namespace label, keeps these entries separate from
            anything else stored in the same collection.
        cache_ttl_days: How long a cached web answer stays trustworthy.
        cache_hit_distance: Maximum embedding distance still counted as "the
            same question". Above it, the cache is treated as a miss.
        retrieval_results: How many documents a collection search returns.
        max_web_results: Number of web results requested per search.
        snippet_chars: How much of each web result's text is kept.
        answer_temperature: Temperature for the answer step. The judge always
            runs at 0.0, because a verdict should be reproducible.
    """

    openai_api_key: str
    tavily_api_key: str
    chat_model: str = "gpt-4o-mini"
    chroma_path: Path = Path("chromadb")
    games_path: Path = Path("data/games")
    skip_report_path: Path = Path("index-skipped.md")
    collection_name: str = "udaplay"
    memory_collection: str = "long_term_memory"
    memory_owner: str = "player"
    memory_namespace: str = "research"
    cache_ttl_days: int = 30
    cache_hit_distance: float = 0.12
    retrieval_results: int = 3
    max_web_results: int = 5
    snippet_chars: int = 300
    answer_temperature: float = 0.2

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        """Build settings from `.env` / the process environment.

        Args:
            **overrides: Any field above, to override the resolved value.

        Returns:
            A populated, immutable `Settings`.

        Raises:
            MissingCredentialsError: If a required key is not set.
        """
        load_dotenv()

        keys = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        }
        missing = [name.upper() for name, value in keys.items() if not value]
        if missing:
            raise MissingCredentialsError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                f"Add them to a .env file next to the project root."
            )

        return cls(**keys, **overrides)
