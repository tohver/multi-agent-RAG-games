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
        '''
        In plain English: starts a collection of documents, empty unless you hand it
        some.

        Output: nothing returned. The collection behaves like a normal Python list from
        here on, except it refuses anything that is not a document.
        '''
        self._documents = documents or []

    def __getitem__(self, index):
        '''
        In plain English: lets you read one document by position, as with a list.

        Output: the document at that position.
        '''
        return self._documents[index]

    def __setitem__(self, index, value: Document):
        '''
        In plain English: lets you replace one document by position, as with a list.

        It refuses anything that is not a document. That check is the reason this class
        exists at all rather than using a plain list: it stops the wrong kind of object
        reaching the database, where the failure would be far harder to trace.

        Output: nothing returned; the collection is changed in place.
        '''
        if not isinstance(value, Document):
            raise TypeError("Collection only supports Document items")
        self._documents[index] = value

    def __delitem__(self, index):
        '''
        In plain English: removes one document by position.

        Output: nothing returned; the collection is changed in place.
        '''
        del self._documents[index]

    def __len__(self):
        '''
        In plain English: how many documents are in the collection.

        Output: the count, so `len(corpus)` works as expected.
        '''
        return len(self._documents)

    def insert(self, index, value: Document):
        '''
        In plain English: puts a document in at a given position, pushing the rest along.

        Refuses anything that is not a document, for the same reason as above.

        Output: nothing returned; the collection is changed in place.
        '''
        if not isinstance(value, Document):
            raise TypeError("Collection only supports Document items")
        self._documents.insert(index, value)

    def to_columns(self) -> Dict[str, List[Any]]:
        """
        Convert the corpus to the parallel lists batch operations expect.
        
        This method extracts all document contents, metadata, and IDs into
        separate lists, which is the format typically expected by vector
        databases and other batch processing systems. This allows for efficient
        bulk operations on the entire corpus.
        
        Returns:
            Dict[str, List[Any]]: Dictionary containing:
                - 'contents': List of all document content strings
                - 'metadatas': List of all document metadata dictionaries
                - 'ids': List of all document ID strings
                
        Example:
            >>> corpus = Corpus([doc1, doc2])
            >>> batch_data = corpus.to_columns()
            >>> chroma_collection.add(
            ...     documents=batch_data['contents'],
            ...     metadatas=batch_data['metadatas'],
            ...     ids=batch_data['ids']
            ... )
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
