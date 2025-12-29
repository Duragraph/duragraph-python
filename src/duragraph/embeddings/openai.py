"""OpenAI embedding provider integration."""

import os
from collections.abc import Sequence
from typing import Any

from duragraph.vectorstores.base import EmbeddingProvider

try:
    from openai import AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    AsyncOpenAI = None  # type: ignore


# Known dimensions for OpenAI embedding models
OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider.

    Uses OpenAI's embedding API to generate dense vector representations of text.

    Example:
        ```python
        from duragraph.embeddings import OpenAIEmbeddings

        # Using default model (text-embedding-3-small)
        embeddings = OpenAIEmbeddings()

        # Using a specific model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # With custom dimensions (only for v3 models)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=512,  # Reduce for faster search
        )

        # Embed texts
        vectors = await embeddings.embed(["Hello world", "Goodbye world"])
        ```
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI embeddings.

        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            dimensions: Optional output dimensions (only for v3 models).
            base_url: Optional base URL for API (for Azure or proxies).
            **kwargs: Additional configuration.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install duragraph-python[openai]"
            )

        super().__init__(**kwargs)

        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._dimensions = dimensions
        self._base_url = base_url

        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

        # Initialize client
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        # OpenAI API call
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": list(texts),
        }

        # Only pass dimensions for v3 models
        if self._dimensions and "text-embedding-3" in self._model:
            kwargs["dimensions"] = self._dimensions

        response = await self._client.embeddings.create(**kwargs)

        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)

        return [item.embedding for item in sorted_data]

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.

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
            Embedding dimension.
        """
        if self._dimensions:
            return self._dimensions
        return OPENAI_EMBEDDING_DIMENSIONS.get(self._model)

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model
