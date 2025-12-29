"""Chroma vector store integration."""

import uuid
from collections.abc import Sequence
from typing import Any

from duragraph.vectorstores.base import Document, EmbeddingProvider, SearchResult, VectorStore

try:
    import chromadb
    from chromadb.config import Settings

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    chromadb = None  # type: ignore
    Settings = None  # type: ignore


class ChromaVectorStore(VectorStore):
    """Chroma vector store.

    Chroma is an open-source embedding database. It can run:
    - In-memory (for development/testing)
    - Persistent (local directory)
    - Client-server (production)

    Example:
        ```python
        from duragraph.vectorstores import ChromaVectorStore

        # In-memory (ephemeral)
        store = ChromaVectorStore(collection="my_docs")

        # Persistent
        store = ChromaVectorStore(
            collection="my_docs",
            persist_directory="./chroma_data"
        )

        # Client-server
        store = ChromaVectorStore(
            collection="my_docs",
            host="localhost",
            port=8000
        )
        ```
    """

    def __init__(
        self,
        collection: str,
        *,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_function: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Chroma vector store.

        Args:
            collection: Name of the collection.
            persist_directory: Directory for persistent storage.
            host: Chroma server host (for client-server mode).
            port: Chroma server port (for client-server mode).
            embedding_provider: Optional embedding provider for automatic embedding.
            embedding_function: Optional Chroma embedding function.
            **kwargs: Additional Chroma client settings.
        """
        if not HAS_CHROMA:
            raise ImportError(
                "chromadb package not installed. Install with: pip install duragraph-python[chroma]"
            )

        super().__init__(**kwargs)

        self._collection_name = collection
        self._embedding_provider = embedding_provider

        # Create client based on configuration
        if host and port:
            # Client-server mode
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory:
            # Persistent mode
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            # In-memory mode
            self._client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Get or create collection
        collection_kwargs: dict[str, Any] = {}
        if embedding_function:
            collection_kwargs["embedding_function"] = embedding_function

        self._collection = self._client.get_or_create_collection(
            name=collection,
            **collection_kwargs,
        )

    async def add(
        self,
        documents: Sequence[Document],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """Add documents to Chroma."""
        if not documents:
            return []

        # Generate IDs if not provided
        doc_ids = list(ids) if ids else [doc.id or str(uuid.uuid4()) for doc in documents]

        # Extract content and metadata
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings if needed
        if embeddings is None and self._embedding_provider:
            embedding_list = await self._embedding_provider.embed(contents)
        elif embeddings:
            embedding_list = [list(e) for e in embeddings]
        else:
            embedding_list = None

        # Add to collection
        add_kwargs: dict[str, Any] = {
            "ids": doc_ids,
            "documents": contents,
            "metadatas": metadatas,
        }
        if embedding_list:
            add_kwargs["embeddings"] = embedding_list

        self._collection.add(**add_kwargs)

        return doc_ids

    async def search(
        self,
        query: str | list[float],
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents in Chroma."""
        query_kwargs: dict[str, Any] = {
            "n_results": k,
        }

        if filter:
            query_kwargs["where"] = filter

        if isinstance(query, str):
            # Query by text
            if self._embedding_provider:
                # Use our embedding provider
                embedding = await self._embedding_provider.embed_query(query)
                query_kwargs["query_embeddings"] = [embedding]
            else:
                # Let Chroma handle embedding
                query_kwargs["query_texts"] = [query]
        else:
            # Query by embedding
            query_kwargs["query_embeddings"] = [query]

        results = self._collection.query(**query_kwargs)

        # Convert to SearchResult
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Chroma returns distances, convert to similarity scores
                # For L2 distance: similarity = 1 / (1 + distance)
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 / (1.0 + distance)

                content = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                search_results.append(
                    SearchResult(
                        document=Document(
                            content=content,
                            metadata=metadata or {},
                            id=doc_id,
                        ),
                        score=score,
                        id=doc_id,
                    )
                )

        return search_results

    async def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete documents from Chroma."""
        if not ids and not filter:
            return 0

        delete_kwargs: dict[str, Any] = {}
        if ids:
            delete_kwargs["ids"] = list(ids)
        if filter:
            delete_kwargs["where"] = filter

        # Chroma doesn't return count, so we estimate
        before_count = self._collection.count()
        self._collection.delete(**delete_kwargs)
        after_count = self._collection.count()

        return before_count - after_count

    async def get(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by ID from Chroma."""
        if not ids:
            return []

        results = self._collection.get(ids=list(ids))

        documents = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                embedding = results["embeddings"][i] if results.get("embeddings") else None

                documents.append(
                    Document(
                        content=content,
                        metadata=metadata or {},
                        id=doc_id,
                        embedding=embedding,
                    )
                )

        return documents

    async def clear(self) -> int:
        """Clear all documents from Chroma collection."""
        count = self._collection.count()
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(name=self._collection_name)
        return count

    async def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()
