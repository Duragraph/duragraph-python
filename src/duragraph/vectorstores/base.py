"""Base classes for vector store integrations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document with content and metadata.

    Attributes:
        content: The text content of the document.
        metadata: Optional metadata associated with the document.
        id: Optional unique identifier.
        embedding: Optional pre-computed embedding vector.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    """A search result from a vector store.

    Attributes:
        document: The matched document.
        score: Similarity score (higher is more similar).
        id: Document ID in the vector store.
    """

    document: Document
    score: float
    id: str


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Vector stores provide similarity search over document embeddings.
    All implementations must support the core operations: add, search, delete.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the vector store.

        Args:
            **kwargs: Provider-specific configuration.
        """
        self._config = kwargs

    @abstractmethod
    async def add(
        self,
        documents: Sequence[Document],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: Documents to add.
            embeddings: Optional pre-computed embeddings. If not provided,
                the vector store should compute them.
            ids: Optional IDs for the documents. If not provided,
                IDs will be auto-generated.

        Returns:
            List of document IDs.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str | list[float],
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: Query string or embedding vector.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of search results ordered by similarity.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete documents from the vector store.

        Args:
            ids: Document IDs to delete.
            filter: Optional metadata filter for bulk deletion.

        Returns:
            Number of documents deleted.
        """
        ...

    async def get(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by ID.

        Args:
            ids: Document IDs to retrieve.

        Returns:
            List of documents.

        Raises:
            NotImplementedError: If the store doesn't support direct retrieval.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support direct document retrieval"
        )

    async def update(
        self,
        documents: Sequence[Document],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[str]:
        """Update existing documents.

        Default implementation: delete then add.

        Args:
            documents: Documents to update (must have IDs).
            embeddings: Optional new embeddings.

        Returns:
            List of document IDs.
        """
        ids = [doc.id for doc in documents if doc.id]
        if ids:
            await self.delete(ids)
        return await self.add(documents, embeddings=embeddings)

    async def clear(self) -> int:
        """Clear all documents from the store.

        Returns:
            Number of documents deleted.

        Raises:
            NotImplementedError: If the store doesn't support clear.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support clearing all documents"
        )

    async def count(self) -> int:
        """Get the number of documents in the store.

        Returns:
            Document count.

        Raises:
            NotImplementedError: If the store doesn't support counting.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support counting documents")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Embedding providers convert text to dense vector representations.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the embedding provider.

        Args:
            **kwargs: Provider-specific configuration.
        """
        self._config = kwargs

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

        Some providers optimize query embeddings differently.

        Args:
            text: Query text.

        Returns:
            Embedding vector.
        """
        embeddings = await self.embed([text])
        return embeddings[0]

    @property
    def dimension(self) -> int | None:
        """Get the embedding dimension.

        Returns:
            Embedding dimension, or None if unknown.
        """
        return None
