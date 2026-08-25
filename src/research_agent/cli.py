"""Command line entry point.

    research-agent                      ask questions interactively
    research-agent "when was X made?"   answer one question and exit
    research-agent --build-index        create or refresh the game collection
    research-agent --evaluate           run the evaluation suite
"""

import argparse
import sys
from pathlib import Path

from .app import Application, build_application
from .config import MissingCredentialsError, Settings
from .evaluation import run_suite
from .indexer import build_index
from .workflow import ResearchAgent

# What each terminal node means, for people who should not have to read code.
SOURCE_LABELS = {
    "evaluate": "the local game database",
    "recall": "a previously cached answer",
    "web_search": "a web search",
}


def _describe_source(path: list) -> str:
    """Describe where the answer came from (local data or internet).
    Shown after every answer in interactive mode. 

    Args:
        path: The node ids the 'run' visited.

    Returns:
        A short phrase naming the source.
    """
    for node in ("web_search", "recall", "evaluate"):
        if node in path:
            return SOURCE_LABELS[node]
    return "an unknown source"


def _answer_once(app: Application, question: str, show_path: bool) -> None:
    """
    Answer one question and print the result.
    The route is shown above the answer unless asked for quiet output, 
    which is handy when feeding the answer into a downstream app.
    """

    run = app.agent.invoke(question)
    path = ResearchAgent.path_of(run)

    if show_path:
        print(f"path  : {' -> '.join(path)}")
    print(run.get_final_state()["answer"])


def _interactive(app: Application) -> int:
    '''
    

    Output: an exit code of 0. Leaving the loop is a normal end, not an error.
    '''
    """The conversational mode, ask questions in a loop until the user stops.
    The default mode, command without arguments.
    Typing `quit`, `exit`, `q`, or pressing Ctrl-C, ends it.

    Returns:
        Always 0; leaving the loop is a normal exit.
    """
    print("Ask about video games. Type  quit  to finish.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if question.lower() in {"quit", "exit", "q"}:
            return 0
        if not question:
            continue

        run = app.agent.invoke(question)
        path = ResearchAgent.path_of(run)
        print(f"Answer  : {run.get_final_state()['answer']}")
        print(f"          (from {_describe_source(path)})\n")


def _parse_args(argv=None) -> argparse.Namespace:
    """
    Parse the command line arguments.
    Argument --chroma-path is for the case, the user wants to pick a different Chroma database.
    """

    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Ask a research agent about video games.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="A question to answer. Omit it to ask questions interactively.",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Create or refresh the game collection from data/games, then exit.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the evaluation suite instead of answering a question.",
    )
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Directory holding the Chroma database (default: chromadb).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the answer, without the path through the pipeline.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Run the CLI.

    Returns:
        exit code 0 on success; 1 on a credentials problem, a missing index, or an
        evaluation case that took the wrong path.
    """
    args = _parse_args(argv)

    overrides = {}
    if args.chroma_path:
        overrides["chroma_path"] = Path(args.chroma_path)

    try:
        settings = Settings.from_env(**overrides)
    except MissingCredentialsError as error:
        print(error, file=sys.stderr)
        return 1

    # Indexing runs before the agent exists
    if args.build_index:
        try:
            count = build_index(settings)
        except (FileNotFoundError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
        print(f"Indexed {count} games into '{settings.collection_name}'.")
        return 0

    try:
        app = build_application(settings)
    except Exception as error:
        # The usual cause is an index that was never built.
        print(f"{error}\n\nIf the collection is missing, run: research-agent --build-index",
              file=sys.stderr)
        return 1

    if args.evaluate:
        results = run_suite(app)
        return 0 if all(row["path_ok"] for row in results) else 1

    if args.question:
        _answer_once(app, args.question, show_path=not args.quiet)
        return 0

    return _interactive(app)


if __name__ == "__main__":
    raise SystemExit(main())
