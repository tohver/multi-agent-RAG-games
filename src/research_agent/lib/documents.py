from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
from collections.abc import MutableSequence


@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = field(default_factory=str)
    metadata: Dict[str, Any] = None

class Corpus(MutableSequence):
    def __init__(self, documents: Optional[List[Document]] = None):
        self._documents = documents or []

    def __getitem__(self, index):
        return self._documents[index]

    def __setitem__(self, index, value: Document):
        if not isinstance(value, Document):
            raise TypeError("Collection only supports Document items")
        self._documents[index] = value

    def __delitem__(self, index):
        del self._documents[index]

    def __len__(self):
        return len(self._documents)

    def insert(self, index, value: Document):
        """Insert a document at the given position.

        Raises:
            TypeError: If `value` is not a `Document`.
        """
        if not isinstance(value, Document):
            raise TypeError("Collection only supports Document items")
        self._documents.insert(index, value)

    def to_columns(self) -> Dict[str, List[Any]]:
        """Split the corpus into the parallel lists batch operations expect.

        Returns:
            Keys `contents`, `metadatas` and `ids`, each a list in corpus order.
        """
        
        # Use zip with unpacking to efficiently extract all fields
        # Handle empty corpus case by providing empty defaults
        contents, metadatas, ids = zip(*(
            (doc.content, doc.metadata, doc.id) for doc in self._documents
        )) if self._documents else ([], [], [])

        return {
            'contents': list(contents),
            'metadatas': list(metadatas),
            'ids': list(ids)
        }
