"""An AI research agent for the video game industry.

Answers from a local vector database first, judges whether that retrieval was
good enough, and only then falls back to a web search - caching the result so
the same question never costs a second search.

Typical use::

    from research_agent import build_application

    app = build_application()
    run = app.agent.invoke("What is the genre of Halo Infinite?")
    print(run.get_final_state()["answer"])
"""

from .app import Application, build_application
from .cache import build_cache
from .config import MissingCredentialsError, Settings
from .evaluation import WorkflowCase, default_cases, run_case, run_suite
from .http_patch import install_html_404_retry
from .indexer import LoadReport, SkippedFile, build_index, load_games, write_skip_report
from .tools import EvaluationReport, ToolSet, build_tools
from .workflow import ResearchAgent, ResearchState

__version__ = "1.0.0"

__all__ = [
    "Application",
    "EvaluationReport",
    "LoadReport",
    "MissingCredentialsError",
    "ResearchAgent",
    "ResearchState",
    "Settings",
    "SkippedFile",
    "ToolSet",
    "WorkflowCase",
    "build_application",
    "build_cache",
    "build_index",
    "build_tools",
    "default_cases",
    "install_html_404_retry",
    "load_games",
    "run_case",
    "run_suite",
    "write_skip_report",
]
