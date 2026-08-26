"""Build the game collection from the JSON files. 
For this project the documents are located in folder `data/games/`.
This step has to run before anything else.
The incomplete/damaged source files are ignored and listed in the report.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

import chromadb
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions

from .config import Settings
from .lib.documents import Document
from .lib.vector_db import VectorStore

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ("Name", "Platform", "Genre", "YearOfRelease", "Description")


@dataclass(frozen=True)
class SkippedFile:
    """Not indexed source file and the reason why is wasn't indexed.

    Attributes:
        name: The file name.
        problem: What was wrong with it, phrased for the report.
    """

    name: str
    problem: str


@dataclass(frozen=True)
class LoadReport:
    """The outcome of one pass over the games directory.

    Attributes:
        documents: Files that were read successfully and are ready to index.
        skipped: Files that were set aside, with the reason for each.
    """

    documents: List[Document] = field(default_factory=list)
    skipped: List[SkippedFile] = field(default_factory=list)

    @property
    def files_examined(self) -> int:
        """How many JSON files were looked at, usable or not."""
        return len(self.documents) + len(self.skipped)


def _missing_fields(game: dict) -> List[str]:
    """Return the required fields that are absent, null, or blank.
    """
    missing = []
    for name in REQUIRED_FIELDS:
        value = game.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(name)
    return missing


def load_games(games_dir: Path) -> LoadReport:
    """Read every JSON file in `games_dir` into a `Document`.

    Platform, genre, name, year and description are indexed as one line; the
    full record is kept as metadata. Unusable files are collected, not raised.

    Returns:
        A `LoadReport`: one `Document` per usable file, ordered by filename and
        keyed on its stem so re-indexing updates rather than duplicates, plus a
        `SkippedFile` for each file that could not be used.

    Raises:
        FileNotFoundError: If `games_dir` does not exist.
    """
    if not games_dir.is_dir():
        raise FileNotFoundError(f"No games directory at {games_dir}")

    documents: List[Document] = []
    skipped: List[SkippedFile] = []

    for path in sorted(games_dir.glob("*.json")):
        try:
            game = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
            skipped.append(SkippedFile(path.name, f"could not be read: {error}"))
            continue

        if not isinstance(game, dict):
            problem = f"top-level value is {type(game).__name__}, not an object"
            skipped.append(SkippedFile(path.name, problem))
            continue

        missing = _missing_fields(game)
        if missing:
            problem = f"missing or empty field(s): {', '.join(missing)}"
            skipped.append(SkippedFile(path.name, problem))
            continue

        content = (
            f"[{game['Platform']}] [{game['Genre']}] {game['Name']} "
            f"({game['YearOfRelease']}) - {game['Description']}"
        )
        documents.append(Document(id=path.stem, content=content, metadata=game))

    return LoadReport(documents=documents, skipped=skipped)


def write_skip_report(report: LoadReport, path: Path, games_dir: Path) -> bool:
    """Write the list of unusable files, or clear a stale report.

    A report left over from an earlier run is worse than no report at all, so
    a clean pass deletes the file rather than leaving it to be misread.

    Args:
        report: The outcome of `load_games`.
        path: Where to write the report.
        games_dir: The origin directory the files in the report.

    Returns:
        True if a report was written, False if there was nothing to report.
    """
    if not report.skipped:
        path.unlink(missing_ok=True)
        return False

    lines = [
        "# Skipped source files",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} from `{games_dir}`.",
        "",
        f"{len(report.skipped)} of {report.files_examined} file(s) were not indexed. "
        "The remaining files were indexed normally; fix these and re-run "
        "`research-agent --build-index`.",
        "",
        "| File | Problem |",
        "|------|---------|",
    ]
    lines += [f"| `{item.name}` | {item.problem} |" for item in report.skipped]
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def build_index(settings: Settings) -> int:
    """Create or refresh the game collection vector database.
    Must be run once before the agent can answer anything.

    Args:
        settings: the games directory, database path, collection name
            and the API key used to embed.

    Returns:
        How many documents the collection holds afterwards, or 0 if no file in
        the directory was usable and nothing was indexed.
    """
    report = load_games(settings.games_path)

    if write_skip_report(report, settings.skip_report_path, settings.games_path):
        logger.warning(
            "skipped %d of %d source file(s); details in %s",
            len(report.skipped),
            report.files_examined,
            settings.skip_report_path,
        )

    if not report.documents:
        logger.error(
            "no usable game files in %s - the collection was left untouched",
            settings.games_path,
        )
        return 0

    logger.info("indexing %d games from %s", len(report.documents), settings.games_path)

    client: ClientAPI = chromadb.PersistentClient(path=str(settings.chroma_path))
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key
        ),
    )

    store = VectorStore(collection)
    store.upsert(report.documents)
    return store.count()
