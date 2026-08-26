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

    Built once at startup by `build_application` and reused for any number of
    questions. Answering may write to `memory`; every other field is read-only.

    Attributes:
        settings: The configuration every other field was built from.
        chroma_client: Connection shared by the game collection and the cache.
        memory: Cache of answers that previously required a web search.
        tools: The five tools, bound to their clients.
        agent: The state machine; call `app.agent.invoke(question)`.
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
