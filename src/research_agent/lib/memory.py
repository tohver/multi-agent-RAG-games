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
        distances: One distance per fragment, in the same order. Smaller means
            closer in meaning. Callers need these to reject weak matches.
    """

    fragments: List[MemoryFragment]
    distances: List[float]


@dataclass
class TimestampFilter:
    """Age bounds for a search, as Unix timestamps.

    Attributes:
        newer_than: Only fragments created after this moment.
        older_than: Only fragments created before this moment.
    """

    newer_than: Optional[int] = None
    older_than: Optional[int] = None


class LongTermMemory:
    """Stores and retrieves fragments using vector similarity.

    Every operation is scoped by owner and namespace, so one caller's memory
    can never surface in another's search.
    """

    def __init__(self, vector_store: VectorStore):
        """Bind the memory to a vector store.

        Args:
            vector_store: Where fragments are embedded and kept.
        """
        self.vector_store = vector_store

    def store(
        self,
        memory_fragment: MemoryFragment,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
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
            if timestamp_filter.newer_than:
                conditions.append({"timestamp": {"$gt": timestamp_filter.newer_than}})
            if timestamp_filter.older_than:
                conditions.append({"timestamp": {"$lt": timestamp_filter.older_than}})

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
            distances=result.get("distances", [[]])[0],
        )

    def clear(self, owner: str, namespace: str) -> None:
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
