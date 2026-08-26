"""The web-answer cache: a `LongTermMemory` kept on disk.

Only answers that once cost a live web search. The answers are stored as
`question -> answer` in the memory, 
so a later lookup matches on how the question was phrased.
"""

from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions

from .lib.memory import LongTermMemory
from .lib.vector_db import VectorStore


def build_cache(
    client: ClientAPI, api_key: str, collection_name: str
) -> LongTermMemory:
    """Open (or create) the cache collection on a persistent Chroma client.

    Keeping the cache on the same on-disk client as the game collection is what
    makes it long term: an in-memory client would lose every entry on restart.

    Args:
        client: A Chroma client, normally the persistent one.
        api_key: OpenAI key used by the embedding function.
        collection_name: Collection the entries live in.

    Returns:
        A `LongTermMemory` ready to search and register.
    """
    return LongTermMemory(
        VectorStore(
            client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key
                ),
            )
        )
    )
