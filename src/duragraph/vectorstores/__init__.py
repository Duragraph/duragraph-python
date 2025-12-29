"""Vector store integrations for DuraGraph.

This module provides abstractions for working with vector databases,
enabling semantic search and retrieval-augmented generation (RAG) workflows.

Supported vector stores:
- Chroma (local/self-hosted)
- Pinecone (managed cloud)
- Qdrant (open-source, self-hosted/cloud)
- Weaviate (open-source, self-hosted/cloud)
- pgvector (PostgreSQL extension - works with Neon, Supabase)
- More coming soon (Milvus, Elasticsearch)

Example:
    ```python
    from duragraph.vectorstores import ChromaVectorStore, Document

    # Create a vector store
    store = ChromaVectorStore(collection="my_docs")

    # Add documents
    docs = [
        Document(content="Hello world", metadata={"type": "greeting"}),
        Document(content="Goodbye world", metadata={"type": "farewell"}),
    ]
    await store.add(docs)

    # Search for similar documents
    results = await store.search("hello", k=5)
    for result in results:
        print(f"{result.score:.3f}: {result.document.content}")
    ```
"""

from duragraph.vectorstores.base import (
    Document,
    EmbeddingProvider,
    SearchResult,
    VectorStore,
)

__all__ = [
    # Base classes
    "Document",
    "EmbeddingProvider",
    "SearchResult",
    "VectorStore",
    # Implementations (lazy imports for optional deps)
    "ChromaVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "WeaviateVectorStore",
    "PgVectorStore",
]


def __getattr__(name: str):
    """Lazy import vector store implementations."""
    if name == "ChromaVectorStore":
        from duragraph.vectorstores.chroma import ChromaVectorStore

        return ChromaVectorStore
    elif name == "PineconeVectorStore":
        from duragraph.vectorstores.pinecone import PineconeVectorStore

        return PineconeVectorStore
    elif name == "QdrantVectorStore":
        from duragraph.vectorstores.qdrant import QdrantVectorStore

        return QdrantVectorStore
    elif name == "WeaviateVectorStore":
        from duragraph.vectorstores.weaviate import WeaviateVectorStore

        return WeaviateVectorStore
    elif name == "PgVectorStore":
        from duragraph.vectorstores.pgvector import PgVectorStore

        return PgVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
