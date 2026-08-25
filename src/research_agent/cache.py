"""The web-answer cache: a `LongTermMemory` kept on disk.

Only answers that once cost a web search live here. They are stored as
`question -> answer`, so a later lookup matches on how the question was phrased.
"""

from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions

from .framework.memory import LongTermMemory
from .framework.vector_db import VectorStore


def build_cache(
    client: ClientAPI, api_key: str, collection_name: str
) -> LongTermMemory:
    '''
    In plain English: this opens the drawer where past web answers are kept.

    The agent stores every answer it had to go to the internet for, so the same
    question never costs a second search. This function opens that storage -
    creating it the first time - on the same on-disk database that holds the
    game data. Using the on-disk one is the whole point: an in-memory store
    would forget everything the moment the program closed, which would make a
    "long term" cache useless.

    Output: a `LongTermMemory` object with two things the agent needs - a way to
    look up a past answer, and a way to save a new one. It is created once at
    startup and handed to the tools that read and write the cache.
    '''
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
