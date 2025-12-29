"""Embedding provider integrations for DuraGraph.

This module provides abstractions for generating text embeddings,
which are essential for vector stores and semantic search.

Supported embedding providers:
- OpenAI (text-embedding-3-small, text-embedding-3-large, ada-002)
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)
- Ollama (local models like nomic-embed-text, mxbai-embed-large)
- More coming soon (Voyage AI, Google, AWS Bedrock)

Example:
    ```python
    from duragraph.embeddings import OpenAIEmbeddings

    # Create embeddings provider
    embeddings = OpenAIEmbeddings()  # Uses OPENAI_API_KEY env var

    # Embed multiple texts
    vectors = await embeddings.embed([
        "Hello world",
        "Goodbye world",
    ])

    # Embed a single query
    query_vector = await embeddings.embed_query("What is the meaning of life?")
    ```
"""

from duragraph.vectorstores.base import EmbeddingProvider

__all__ = [
    # Base class
    "EmbeddingProvider",
    # Implementations (lazy imports for optional deps)
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "OllamaEmbeddings",
]


def __getattr__(name: str):
    """Lazy import embedding implementations."""
    if name == "OpenAIEmbeddings":
        from duragraph.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings
    elif name == "CohereEmbeddings":
        from duragraph.embeddings.cohere import CohereEmbeddings

        return CohereEmbeddings
    elif name == "OllamaEmbeddings":
        from duragraph.embeddings.ollama import OllamaEmbeddings

        return OllamaEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
