"""Tests for vector store integrations."""

import pytest

from duragraph.vectorstores.base import Document, EmbeddingProvider, SearchResult, VectorStore


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    async def embed(self, texts):
        # Simple mock: return fixed-length vectors based on text length
        return [[float(len(t) % 10) / 10] * 3 for t in texts]

    @property
    def dimension(self):
        return 3


class MockVectorStore(VectorStore):
    """Mock vector store for testing base class functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._id_counter = 0

    async def add(self, documents, *, embeddings=None, ids=None):
        doc_ids = []
        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids else doc.id or f"doc_{self._id_counter}"
            self._id_counter += 1
            self._documents[doc_id] = Document(
                content=doc.content,
                metadata=doc.metadata,
                id=doc_id,
            )
            if embeddings:
                self._embeddings[doc_id] = list(embeddings[i])
            doc_ids.append(doc_id)
        return doc_ids

    async def search(self, query, *, k=10, filter=None):
        # Simple mock: return all documents with fake scores
        results = []
        for doc_id, doc in list(self._documents.items())[:k]:
            if filter:
                # Simple filter matching
                match = all(doc.metadata.get(key) == val for key, val in filter.items())
                if not match:
                    continue
            results.append(
                SearchResult(
                    document=doc,
                    score=0.9,
                    id=doc_id,
                )
            )
        return results

    async def delete(self, ids=None, *, filter=None):
        count = 0
        if ids:
            for doc_id in ids:
                if doc_id in self._documents:
                    del self._documents[doc_id]
                    self._embeddings.pop(doc_id, None)
                    count += 1
        return count

    async def get(self, ids):
        return [self._documents[doc_id] for doc_id in ids if doc_id in self._documents]

    async def count(self):
        return len(self._documents)

    async def clear(self):
        count = len(self._documents)
        self._documents.clear()
        self._embeddings.clear()
        return count


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self):
        """Test creating a document."""
        doc = Document(content="Hello world")
        assert doc.content == "Hello world"
        assert doc.metadata == {}
        assert doc.id is None
        assert doc.embedding is None

    def test_document_with_metadata(self):
        """Test document with metadata."""
        doc = Document(
            content="Test content",
            metadata={"source": "test", "page": 1},
            id="doc_123",
        )
        assert doc.metadata["source"] == "test"
        assert doc.metadata["page"] == 1
        assert doc.id == "doc_123"

    def test_document_with_embedding(self):
        """Test document with pre-computed embedding."""
        embedding = [0.1, 0.2, 0.3]
        doc = Document(content="Test", embedding=embedding)
        assert doc.embedding == embedding


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a search result."""
        doc = Document(content="Test")
        result = SearchResult(document=doc, score=0.95, id="doc_1")
        assert result.document.content == "Test"
        assert result.score == 0.95
        assert result.id == "doc_1"


@pytest.mark.asyncio
class TestMockVectorStore:
    """Tests for mock vector store functionality."""

    async def test_add_documents(self):
        """Test adding documents."""
        store = MockVectorStore()
        docs = [
            Document(content="Hello world", metadata={"type": "greeting"}),
            Document(content="Goodbye world", metadata={"type": "farewell"}),
        ]

        ids = await store.add(docs)

        assert len(ids) == 2
        assert await store.count() == 2

    async def test_add_with_ids(self):
        """Test adding documents with custom IDs."""
        store = MockVectorStore()
        docs = [Document(content="Test")]

        ids = await store.add(docs, ids=["custom_id"])

        assert ids == ["custom_id"]

    async def test_add_with_embeddings(self):
        """Test adding documents with pre-computed embeddings."""
        store = MockVectorStore()
        docs = [Document(content="Test")]
        embeddings = [[0.1, 0.2, 0.3]]

        ids = await store.add(docs, embeddings=embeddings)

        assert len(ids) == 1
        assert store._embeddings[ids[0]] == [0.1, 0.2, 0.3]

    async def test_search(self):
        """Test searching for documents."""
        store = MockVectorStore()
        docs = [
            Document(content="Hello world", metadata={"type": "greeting"}),
            Document(content="Goodbye world", metadata={"type": "farewell"}),
        ]
        await store.add(docs)

        results = await store.search("hello", k=10)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    async def test_search_with_filter(self):
        """Test searching with metadata filter."""
        store = MockVectorStore()
        docs = [
            Document(content="Hello", metadata={"type": "greeting"}),
            Document(content="Goodbye", metadata={"type": "farewell"}),
        ]
        await store.add(docs)

        results = await store.search("test", filter={"type": "greeting"})

        assert len(results) == 1
        assert results[0].document.metadata["type"] == "greeting"

    async def test_delete_by_ids(self):
        """Test deleting documents by ID."""
        store = MockVectorStore()
        docs = [Document(content="Test1"), Document(content="Test2")]
        ids = await store.add(docs)

        deleted = await store.delete([ids[0]])

        assert deleted == 1
        assert await store.count() == 1

    async def test_get_documents(self):
        """Test retrieving documents by ID."""
        store = MockVectorStore()
        docs = [Document(content="Test content", metadata={"key": "value"})]
        ids = await store.add(docs)

        retrieved = await store.get(ids)

        assert len(retrieved) == 1
        assert retrieved[0].content == "Test content"
        assert retrieved[0].metadata["key"] == "value"

    async def test_update_documents(self):
        """Test updating documents."""
        store = MockVectorStore()
        docs = [Document(content="Original", id="doc_1")]
        await store.add(docs)

        updated_docs = [Document(content="Updated", id="doc_1")]
        await store.update(updated_docs)

        retrieved = await store.get(["doc_1"])
        assert retrieved[0].content == "Updated"

    async def test_clear(self):
        """Test clearing all documents."""
        store = MockVectorStore()
        docs = [Document(content="Test1"), Document(content="Test2")]
        await store.add(docs)

        cleared = await store.clear()

        assert cleared == 2
        assert await store.count() == 0


@pytest.mark.asyncio
class TestMockEmbeddingProvider:
    """Tests for mock embedding provider."""

    async def test_embed_texts(self):
        """Test embedding multiple texts."""
        provider = MockEmbeddingProvider()

        embeddings = await provider.embed(["hello", "world"])

        assert len(embeddings) == 2
        assert all(len(e) == 3 for e in embeddings)

    async def test_embed_query(self):
        """Test embedding a single query."""
        provider = MockEmbeddingProvider()

        embedding = await provider.embed_query("hello")

        assert len(embedding) == 3

    def test_dimension(self):
        """Test dimension property."""
        provider = MockEmbeddingProvider()
        assert provider.dimension == 3


class TestVectorStoreNotImplemented:
    """Tests for NotImplementedError cases."""

    def test_get_not_implemented(self):
        """Test that base VectorStore.get raises NotImplementedError."""

        class MinimalStore(VectorStore):
            async def add(self, documents, *, embeddings=None, ids=None):
                return []

            async def search(self, query, *, k=10, filter=None):
                return []

            async def delete(self, ids=None, *, filter=None):
                return 0

        store = MinimalStore()

        with pytest.raises(NotImplementedError):
            import asyncio

            asyncio.get_event_loop().run_until_complete(store.get(["id"]))

    def test_clear_not_implemented(self):
        """Test that base VectorStore.clear raises NotImplementedError."""

        class MinimalStore(VectorStore):
            async def add(self, documents, *, embeddings=None, ids=None):
                return []

            async def search(self, query, *, k=10, filter=None):
                return []

            async def delete(self, ids=None, *, filter=None):
                return 0

        store = MinimalStore()

        with pytest.raises(NotImplementedError):
            import asyncio

            asyncio.get_event_loop().run_until_complete(store.clear())
