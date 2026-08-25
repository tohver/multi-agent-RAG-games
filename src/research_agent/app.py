"""Composition root: turn settings into a ready-to-use agent."""

from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI

from .cache import build_cache
from .config import Settings
from .framework.memory import LongTermMemory
from .http_patch import install_html_404_retry
from .tools import ToolSet, build_tools
from .workflow import ResearchAgent


@dataclass(frozen=True)
class Application:
    """Every long-lived object of one run, wired together.

    Handed around instead of module-level globals, so tests can build a second
    application against a different database without touching the first.

    Attributes:
        settings: The configuration everything was built from.
        chroma_client: Open connection to the on-disk database.
        memory: The web-answer cache.
        tools: The five tools.
        agent: The state machine that uses them.
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
    '''
    In plain English: this is the assembly line. It builds every piece the
    program needs, in the right order, and connects them.

    Nothing in this project creates a database connection or an API client on
    its own - they are all created here and handed down. That is deliberate. It
    means importing any file is free and safe, errors about missing keys happen
    in one predictable place, and a test can build a second, completely separate
    copy of the program pointing at a different database without disturbing the
    first.

    The order matters: the flaky-server workaround goes in before anything
    touches the network, the database opens before the tools that read from it,
    and the agent is built last because it needs the finished tools.

    Output: an `Application` holding all five pieces. In practice you use
    `app.agent` to ask questions and `app.memory` to inspect or clear the cache.
    '''
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
