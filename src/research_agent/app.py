"""Composition root: turn settings into a ready-to-use agent."""

from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI

from .cache import build_cache
from .config import Settings
from .lib.memory import LongTermMemory
from .http_patch import install_html_404_retry
from .tools import ToolSet, build_tools
from .workflow import ResearchAgent


@dataclass(frozen=True)
class Application:
    """A fully wired agent together with the resources it depends on.

    Opens a database connection or an API client. 
    Collecting them in one immutable object keeps that wiring explicit and 
    avoids module-level globals, so two applications can coexist in the same process 
    against different databases and different configurations.

    One instance serves any number of questions and is intended to be built
    once at startup and reused. Answering a question may write to `memory`,
    when the answer had to be researched on the web; every other field is
    read-only for the lifetime of the application.

    Attributes:
        settings: The resolved configuration every other field was built from.
            Read values from here rather than re-reading the environment.
        chroma_client: Open connection to the persistent ChromaDB instance,
            shared by the game collection and the answer cache.
        memory: Long-term cache of answers that previously required a web
            search. Use it to inspect or clear cached entries.
        tools: The five tools, already bound to their clients and settings.
        agent: The state machine that routes a question through those tools.
            The usual entry point is `app.agent.invoke(question)`.

    Example:
        >>> app = build_application()
        >>> run = app.agent.invoke("What is the genre of Halo Infinite?")
        >>> print(run.get_final_state()["answer"])
    """

    settings: Settings
    chroma_client: ClientAPI
    memory: LongTermMemory
    tools: ToolSet
    agent: ResearchAgent


def build_application(
    settings: Settings = None,
    *,
    patch_http: bool = True,
) -> Application:
    """Open the clients, build the tools, and wire up the agent.

    Args:
        settings: Configuration to use; resolved from the environment if absent.
        patch_http: Install the HTML-404 retry. Leave on unless you are talking
            to a proxy that does not have that fault.

    Returns:
        The assembled `Application`.
    """
    settings = settings or Settings.from_env()

    if patch_http:
        install_html_404_retry()

    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))
    memory = build_cache(
        chroma_client, settings.openai_api_key, settings.memory_collection
    )
    tools = build_tools(settings, chroma_client, memory)

    return Application(
        settings=settings,
        chroma_client=chroma_client,
        memory=memory,
        tools=tools,
        agent=ResearchAgent(tools, settings),
    )
