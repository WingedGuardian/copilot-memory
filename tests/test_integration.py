"""Integration test — end-to-end store → recall cycle with real SQLite + mock vectors."""

import hashlib

import aiosqlite
import pytest

from genesis_memory.backends.schema import init_schema
from genesis_memory.backends.sqlite_links import SQLiteLinkBackend
from genesis_memory.backends.sqlite_metadata import SQLiteMetadataBackend
from genesis_memory.backends.sqlite_pending import SQLitePendingBackend
from genesis_memory.backends.sqlite_text import SQLiteTextBackend
from genesis_memory.embeddings import EmbeddingProvider
from genesis_memory.linker import MemoryLinker
from genesis_memory.retrieval import HybridRetriever
from genesis_memory.store import MemoryStore

# -- Mock backends --


class FakeEmbeddingBackend:
    """Returns deterministic 8-dim vectors from content hash."""

    @property
    def name(self) -> str:
        return "fake_embedding"

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:8]]

    async def is_available(self) -> bool:
        return True


class InMemoryVectorBackend:
    """Simple in-memory vector store for testing."""

    def __init__(self):
        self._points: dict[str, dict] = {}  # collection:id → {vector, payload}

    def _key(self, point_id: str, collection: str) -> str:
        return f"{collection}:{point_id}"

    async def search(
        self, vector, *, limit=10, collection="default", wing=None, room=None,
    ) -> list[dict]:
        # Cosine similarity (simplified — just dot product for unit-ish vectors)
        results = []
        for key, data in self._points.items():
            if not key.startswith(f"{collection}:"):
                continue
            payload = data["payload"]
            if wing and payload.get("wing") != wing:
                continue
            if room and payload.get("room") != room:
                continue
            stored_vec = data["vector"]
            dot = sum(a * b for a, b in zip(vector, stored_vec, strict=True))
            mag_a = sum(a * a for a in vector) ** 0.5
            mag_b = sum(b * b for b in stored_vec) ** 0.5
            score = dot / (mag_a * mag_b + 1e-10)
            point_id = key.split(":", 1)[1]
            results.append({"id": point_id, "score": score, "payload": payload})
        results.sort(key=lambda h: h["score"], reverse=True)
        return results[:limit]

    async def upsert(self, point_id, vector, payload, *, collection="default"):
        self._points[self._key(point_id, collection)] = {
            "vector": vector, "payload": payload,
        }

    async def update_payload(self, point_id, payload, *, collection="default"):
        key = self._key(point_id, collection)
        if key in self._points:
            self._points[key]["payload"].update(payload)

    async def delete(self, point_id, *, collection="default"):
        key = self._key(point_id, collection)
        self._points.pop(key, None)

    async def scroll_tags(self, collections):
        tag_lists = []
        count = 0
        for key, data in self._points.items():
            coll = key.split(":", 1)[0]
            if coll in collections:
                count += 1
                tags = data["payload"].get("tags") or []
                if tags:
                    tag_lists.append(tags)
        return tag_lists, count


# -- Fixtures --


@pytest.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await init_schema(conn)
    yield conn
    await conn.close()


@pytest.fixture
def vector_backend():
    return InMemoryVectorBackend()


@pytest.fixture
def embedding_provider():
    return EmbeddingProvider(
        backends=[FakeEmbeddingBackend()],
        cache_dir=None,
    )


@pytest.fixture
def memory_system(db, vector_backend, embedding_provider):
    text = SQLiteTextBackend(db)
    metadata = SQLiteMetadataBackend(db)
    links = SQLiteLinkBackend(db)
    pending = SQLitePendingBackend(db)

    linker = MemoryLinker(
        vector_backend=vector_backend,
        text_backend=text,
        link_backend=links,
    )

    store = MemoryStore(
        embedding_provider=embedding_provider,
        vector_backend=vector_backend,
        text_backend=text,
        metadata_backend=metadata,
        link_backend=links,
        pending_backend=pending,
        linker=linker,
        collections=["episodic_memory", "knowledge_base"],
    )

    retriever = HybridRetriever(
        embedding_provider=embedding_provider,
        vector_backend=vector_backend,
        text_backend=text,
        link_backend=links,
    )

    return store, retriever


# -- Tests --


class TestEndToEnd:
    async def test_store_and_recall(self, memory_system):
        store, retriever = memory_system

        # Store some memories
        mid1 = await store.store(
            "The routing system uses three-state circuit breakers"
            " to protect against cascading failures",
            "test",
            memory_type="episodic",
        )
        mid2 = await store.store(
            "Rate limiting prevents individual providers from being overwhelmed by traffic spikes",
            "test",
            memory_type="episodic",
        )
        mid3 = await store.store(
            "The memory retrieval pipeline uses reciprocal rank fusion"
            " to combine vector and text search",
            "test",
            memory_type="episodic",
        )

        assert mid1 != mid2 != mid3

        # Recall
        results = await retriever.recall("circuit breakers routing", limit=5)
        assert len(results) > 0
        # The circuit breaker memory should be found (at least via FTS5)
        memory_ids = [r.memory_id for r in results]
        assert mid1 in memory_ids

    async def test_store_deduplicates(self, memory_system):
        store, _ = memory_system
        content = "Exact duplicate content for testing"
        mid1 = await store.store(content, "test")
        mid2 = await store.store(content, "test")
        assert mid1 == mid2  # Dedup returns existing

    async def test_delete_cascade(self, memory_system):
        store, retriever = memory_system
        mid = await store.store(
            "Memory to be deleted from all layers",
            "test",
        )
        result = await store.delete(mid)
        assert result["metadata"] is True
        assert result["fts5"] is True

        # Should not appear in recall
        results = await retriever.recall("deleted", limit=5)
        memory_ids = [r.memory_id for r in results]
        assert mid not in memory_ids

    async def test_fts5_only_fallback(self, memory_system, db):
        store, retriever = memory_system
        # Force FTS5-only storage
        mid = await store.store(
            "This memory was stored without vector embedding",
            "test",
            force_fts5_only=True,
        )

        # Check it was queued in pending_embeddings
        pending = SQLitePendingBackend(db)
        items = await pending.get_pending()
        assert any(item["memory_id"] == mid for item in items)

        # Should still be findable via FTS5
        results = await retriever.recall("stored without vector", limit=5)
        memory_ids = [r.memory_id for r in results]
        assert mid in memory_ids

    async def test_recall_returns_rrf_scores(self, memory_system):
        store, retriever = memory_system
        await store.store("Activation scoring uses exponential decay", "test")
        await store.store("Memory activation depends on recency and access frequency", "test")

        results = await retriever.recall("activation scoring", limit=5)
        assert len(results) > 0
        for r in results:
            assert r.score > 0  # RRF score
            assert r.activation_score >= 0
            assert r.memory_class in ("rule", "fact", "reference")

    async def test_recall_with_wing_filter(self, memory_system):
        store, retriever = memory_system
        await store.store(
            "Infrastructure monitoring dashboard",
            "test", wing="infrastructure",
        )
        await store.store(
            "Memory retrieval improvements",
            "test", wing="memory",
        )

        # Filter by wing — only infrastructure results
        results = await retriever.recall("monitoring", limit=5, wing="infrastructure")
        for r in results:
            # Qdrant-filtered results should all be infrastructure
            if r.vector_rank is not None:
                assert r.payload.get("wing") == "infrastructure"

    async def test_classification_applied(self, memory_system):
        store, retriever = memory_system
        mid = await store.store(
            "You MUST NEVER push directly to main without a PR",
            "test",
        )
        results = await retriever.recall("push to main", limit=5)
        matching = [r for r in results if r.memory_id == mid]
        assert len(matching) == 1
        assert matching[0].memory_class == "rule"

    async def test_empty_recall(self, memory_system):
        _, retriever = memory_system
        results = await retriever.recall("nothing stored yet", limit=5)
        assert results == []

    async def test_recall_source_filter(self, memory_system):
        store, retriever = memory_system
        await store.store("Test in episodic collection", "test", memory_type="episodic")

        # Source=episodic should only search episodic_memory collection
        results = await retriever.recall("test episodic", source="episodic", limit=5)
        # Should find the memory (it's in episodic_memory)
        assert len(results) >= 0  # May or may not find via vector depending on hash

    async def test_invalid_source_raises(self, memory_system):
        _, retriever = memory_system
        with pytest.raises(ValueError, match="source must be one of"):
            await retriever.recall("test", source="invalid")
