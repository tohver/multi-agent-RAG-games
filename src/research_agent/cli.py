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
    '''
    In plain English: translates the internal step names into something a person
    would say.

    The route through the pipeline is written in the vocabulary of the code -
    `recall`, `web_search`. Useful when debugging, meaningless to someone who just
    wants an answer. This turns it into a phrase like "a web search".

    Output: one short phrase naming where the answer came from. Shown after every
    answer in interactive mode, so it is always clear whether you are reading
    something from the local data or from the internet.
    '''
    """Describe, in plain words, where an answer came from.

    Args:
        path: The node ids the run visited.

    Returns:
        A short phrase naming the source.
    """
    for node in ("web_search", "recall", "evaluate"):
        if node in path:
            return SOURCE_LABELS[node]
    return "an unknown source"


def _answer_once(app: Application, question: str, show_path: bool) -> None:
    '''
    In plain English: answers a single question and prints it, for when the command
    was run with a question already typed in.

    Output: nothing returned - it prints. The route is shown above the answer unless
    you asked for quiet output, which is handy when feeding the answer into another
    program.
    '''
    """Answer one question and print the result."""
    run = app.agent.invoke(question)
    path = ResearchAgent.path_of(run)

    if show_path:
        print(f"path  : {' -> '.join(path)}")
    print(run.get_final_state()["answer"])


def _interactive(app: Application) -> int:
    '''
    In plain English: the conversational mode - keeps asking for questions until you
    stop.

    This is what runs when the command is given no arguments at all, and it is
    deliberately the default. Typing a question into a prompt avoids the two things
    that trip people up on a command line: remembering flags, and getting quotation
    marks around a sentence right.

    Typing `quit`, or pressing Ctrl-C, ends it.

    Output: an exit code of 0. Leaving the loop is a normal end, not an error.
    '''
    """Ask questions in a loop until the user stops.

    Chosen as the no-argument default because it needs no quoting and no flags -
    the two things that trip people up on a command line.

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
    '''
    In plain English: reads what the user typed after the command name.

    Output: an object holding the question and any flags. Everything downstream
    reads its instructions from it, so this is the single place where command line
    text turns into decisions.
    '''
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
    '''
    In plain English: the front door of the whole program - everything starts here.

    It works out what you asked for and routes accordingly: build the database,
    answer one question, run the tests, or open the interactive prompt. The order is
    deliberate. Settings are read first, because nothing works without keys.
    Index-building comes before the agent is assembled, because that step is what
    creates the database the agent needs.

    The one failure worth expecting is a missing database, so the error message for
    it says exactly which command to run.

    Output: an exit code - 0 when things went well, 1 for a missing key, a missing
    database, or a test that took the wrong route. That number is what lets this be
    used in a script or an automated check.
    '''
    """Run the CLI.

    Returns:
        0 on success; 1 on a credentials problem, a missing index, or an
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

    # Indexing runs before the agent exists, because the agent needs the
    # collection it creates.
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
