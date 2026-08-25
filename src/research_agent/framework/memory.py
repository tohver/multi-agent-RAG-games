"""Persistent memory: text fragments retrievable by meaning rather than by key."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .documents import Document
from .vector_db import VectorStore


@dataclass
class MemoryFragment:
    """One stored piece of information.

    Attributes:
        content: The text that gets embedded and searched.
        owner: Who this fragment belongs to; searches are always scoped by it.
        namespace: Logical grouping, so unrelated kinds of memory can share a
            collection without colliding.
        timestamp: Unix time of creation, used for age filtering.
    """

    content: str
    owner: str
    namespace: str = "default"
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class MemorySearchResult:
    """Fragments returned by a search, with the distances that ranked them.

    Attributes:
        fragments: Matches, nearest first.
        metadata: Search metadata; `distances` holds one float per fragment.
    """

    fragments: List[MemoryFragment]
    metadata: Dict


@dataclass
class TimestampFilter:
    """Age bounds for a search, as Unix timestamps.

    Attributes:
        greater_than_value: Only fragments created after this moment.
        lower_than_value: Only fragments created before this moment.
    """

    greater_than_value: Optional[int] = None
    lower_than_value: Optional[int] = None


class LongTermMemory:
    """Stores and retrieves fragments using vector similarity.

    Every operation is scoped by owner and namespace, so one caller's memory
    can never surface in another's search.
    """

    def __init__(self, vector_store: VectorStore):
        '''
        In plain English: connects this memory to the place its entries are stored.

        Output: nothing returned. The memory is ready to save and search from here on.
        '''
        """Bind the memory to a vector store.

        Args:
            vector_store: Where fragments are embedded and kept.
        """
        self.vector_store = vector_store

    def register(
        self,
        memory_fragment: MemoryFragment,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        '''
        In plain English: saves one piece of information so it can be found later.

        Along with the text itself it records who it belongs to, which group it is part
        of, and when it was saved. Those three labels are what make it possible to search
        only one user's entries, and to ignore entries that have gone stale.

        Output: nothing returned. The effect is a new entry on disk, searchable
        immediately.
        '''
        """Store one fragment.

        Args:
            memory_fragment: What to remember.
            metadata: Extra metadata to attach. ChromaDB accepts scalars only,
                so collections must be flattened to a string by the caller.
        """
        complete_metadata = {
            "owner": memory_fragment.owner,
            "namespace": memory_fragment.namespace,
            "timestamp": memory_fragment.timestamp,
        }
        if metadata:
            complete_metadata.update(metadata)

        self.vector_store.add(
            Document(content=memory_fragment.content, metadata=complete_metadata)
        )

    def search(
        self,
        query_text: str,
        owner: str,
        limit: int = 3,
        timestamp_filter: Optional[TimestampFilter] = None,
        namespace: str = "default",
    ) -> MemorySearchResult:
        '''
        In plain English: finds saved entries that mean roughly the same as what you
        asked, rather than ones containing the same words.

        That distinction is the point: ask "which console do I use" and it can find "the
        user plays on a Nintendo Switch", despite no shared wording. Results are always
        limited to one owner and one group, so entries can never leak between users.

        Output: the matching entries, plus a distance for each - a number saying how far
        apart the meanings are, smaller being closer. Those numbers matter: the caller
        uses them to throw away weak matches, which is how the cache avoids answering
        one question with another question's answer.
        '''
        """Find fragments semantically close to the query.

        Args:
            query_text: What to look for.
            owner: Restrict the search to this owner.
            limit: Maximum number of fragments to return.
            timestamp_filter: Optional age bounds.
            namespace: Restrict the search to this namespace.

        Returns:
            The matching fragments and their distances. Distances are what make
            a relevance cutoff possible, so they are always returned.
        """
        conditions = [
            {"namespace": {"$eq": namespace}},
            {"owner": {"$eq": owner}},
        ]
        if timestamp_filter:
            if timestamp_filter.greater_than_value:
                conditions.append(
                    {"timestamp": {"$gt": timestamp_filter.greater_than_value}}
                )
            if timestamp_filter.lower_than_value:
                conditions.append(
                    {"timestamp": {"$lt": timestamp_filter.lower_than_value}}
                )

        result = self.vector_store.query(
            query_texts=[query_text], n_results=limit, where={"$and": conditions}
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        fragments = [
            MemoryFragment(
                content=content,
                owner=meta.get("owner"),
                namespace=meta.get("namespace", "default"),
                timestamp=meta.get("timestamp"),
            )
            for content, meta in zip(documents, metadatas)
        ]

        return MemorySearchResult(
            fragments=fragments,
            metadata={"distances": result.get("distances", [[]])[0]},
        )

    def clear(self, owner: str, namespace: str) -> None:
        '''
        In plain English: deletes every entry belonging to one owner and group.

        Used to start from a clean slate - mainly before a test run, so results do not
        depend on what an earlier run happened to leave behind.

        Output: nothing returned. The entries are gone from disk.
        '''
        """Delete every fragment for one owner and namespace.

        Args:
            owner: Owner whose fragments to drop.
            namespace: Namespace to drop them from.
        """
        self.vector_store.delete(
            where={
                "$and": [
                    {"owner": {"$eq": owner}},
                    {"namespace": {"$eq": namespace}},
                ]
            }
        )
