"""A thin, typed wrapper over a ChromaDB collection."""

from typing import Any, Dict, List, Optional, Union

from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.api.types import GetResult, QueryResult

from .documents import Corpus, Document


class VectorStore:
    """High-level interface to one ChromaDB collection.

    Converts between the `Document` / `Corpus` abstractions and the parallel
    lists ChromaDB expects, so callers never assemble those by hand. Embeddings
    are produced by the embedding function the collection was created with.
    """

    def __init__(self, chroma_collection: ChromaCollection):
        self._collection = chroma_collection

    @property
    def name(self) -> str:
        """Name of the underlying collection."""
        return self._collection.name

    def count(self) -> int:
        """Number of documents currently stored."""
        return self._collection.count()

    def add(self, item: Union[Document, Corpus, List[Document]]) -> None:
        """Add documents, embedding them with the collection's function.

        Args:
            item: A single `Document`, a list of them, or a `Corpus`.

        Raises:
            TypeError: If the input is not one of those, or a list holds
                something other than `Document`.

        Example:
            >>> store.add(Document(content="AI is transforming healthcare"))
            >>> store.add([doc1, doc2, doc3])
        """
        if isinstance(item, Document):
            item = Corpus([item])
        elif isinstance(item, list):
            if not all(isinstance(doc, Document) for doc in item):
                raise TypeError("List must contain Document objects only.")
            item = Corpus(item)
        elif not isinstance(item, Corpus):
            raise TypeError("item must be Document, Corpus, or List[Document].")

        batch = item.to_columns()
        self._collection.add(
            documents=batch["contents"],
            ids=batch["ids"],
            metadatas=batch["metadatas"],
        )

    def upsert(self, item: Union[Document, Corpus, List[Document]]) -> None:
        """Add documents, replacing any that already exist under the same id.

        Unlike `add`, this is safe to run twice - useful when an index is
        rebuilt from source files.

        Args:
            item: A single `Document`, a list of them, or a `Corpus`.
        """
        if isinstance(item, Document):
            item = Corpus([item])
        elif isinstance(item, list):
            item = Corpus(item)

        batch = item.to_columns()
        self._collection.upsert(
            documents=batch["contents"],
            ids=batch["ids"],
            metadatas=batch["metadatas"],
        )

    def query(
        self,
        query_texts: Union[str, List[str]],
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Find the documents most similar to the query text.

        Args:
            query_texts: One query string, or several.
            n_results: Maximum matches per query.
            where: Metadata filter, in ChromaDB's query syntax.
            where_document: Document-content filter.

        Returns:
            The ChromaDB result, carrying documents, distances, metadatas and ids.

        Example:
            >>> hits = store.query(["racing games"], n_results=5)
            >>> for doc, dist in zip(hits["documents"][0], hits["distances"][0]):
            ...     print(f"{dist:.3f}  {doc[:60]}")
        """
        return self._collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "distances", "metadatas"],
        )

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> GetResult:
        """Fetch documents by id or metadata filter, without a similarity search.

        Args:
            ids: Specific document ids to fetch.
            where: Metadata filter.
            limit: Maximum number of documents to return.

        Returns:
            The ChromaDB result with the requested documents and metadata.

        Note:
            `distances` is deliberately absent from `include` - ChromaDB only
            computes distances for `query`, and asking for them here raises.
        """
        return self._collection.get(
            ids=ids,
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )

    def delete(self, ids: Optional[List[str]] = None,
               where: Optional[Dict[str, Any]] = None) -> None:
        """Delete documents by id or metadata filter.

        Args:
            ids: Specific document ids to delete.
            where: Metadata filter selecting what to delete.
        """
        self._collection.delete(ids=ids, where=where)
