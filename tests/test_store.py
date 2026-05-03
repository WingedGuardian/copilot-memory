"""MemoryStore pipeline tests with mock backends."""

from __future__ import annotations

import pytest

from genesis_memory.embeddings import EmbeddingUnavailableError
from genesis_memory.store import _COLLECTION_MAP, MemoryStore

# ---------------------------------------------------------------------------
# Test doubles — concrete classes implementing the protocol interfaces
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Minimal embedding provider for store tests.

    Can be configured to raise EmbeddingUnavailableError.
    """

    def __init__(self, *, should_fail: bool = False):
        self._should_fail = should_fail
        self.embed_calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if self._should_fail:
            raise EmbeddingUnavailableError("all backends down")
        # Deterministic 8-dim vector
        return [float(b % 256) / 255.0 for b in text.encode()[:8]]

    @staticmethod
    def enrich(content: str, memory_type: str, tags: list[str]) -> str:
        if tags:
            return f"{memory_type}: {' '.join(tags)}: {content}"
        return f"{memory_type}: {content}"


class FakeVectorBackend:
    """In-memory vector backend implementing VectorBackend protocol."""

    def __init__(self):
        self.points: dict[str, dict] = {}  # "collection:id" -> {vector, payload}
        self.deleted: list[tuple[str, str]] = []  # (point_id, collection)
        self.payload_updates: list[tuple[str, dict, str]] = []

    async def search(
        self, vector, *, limit=10, collection="default", wing=None, room=None,
    ) -> list[dict]:
        return []

    async def upsert(self, point_id, vector, payload, *, collection="default"):
        self.points[f"{collection}:{point_id}"] = {"vector": vector, "payload": payload}

    async def update_payload(self, point_id, payload, *, collection="default"):
        key = f"{collection}:{point_id}"
        if key in self.points:
            self.points[key]["payload"].update(payload)
        self.payload_updates.append((point_id, payload, collection))

    async def delete(self, point_id, *, collection="default"):
        key = f"{collection}:{point_id}"
        self.points.pop(key, None)
        self.deleted.append((point_id, collection))

    async def scroll_tags(self, collections):
        return [], 0


class FakeTextBackend:
    """In-memory FTS5 backend implementing TextBackend protocol."""

    def __init__(self):
        self.entries: dict[str, dict] = {}  # memory_id -> {content, source_type, tags, collection}
        self._duplicate_map: dict[str, str] = {}  # content_hash -> memory_id

    async def search_ranked(
        self, query, *, collection=None, limit=30, boolean=False,
    ) -> list[dict]:
        results = []
        query_lower = query.lower()
        for mid, entry in self.entries.items():
            if collection and entry["collection"] != collection:
                continue
            if query_lower in entry["content"].lower():
                results.append({
                    "memory_id": mid,
                    "content": entry["content"],
                    "source_type": entry["source_type"],
                    "collection": entry["collection"],
                    "rank": -1.0,
                })
        return results[:limit]

    async def search(self, query, *, limit=10) -> list[dict]:
        results = []
        query_lower = query.lower()
        for mid, entry in self.entries.items():
            if query_lower in entry["content"].lower():
                results.append({
                    "memory_id": mid,
                    "content": entry["content"],
                    "source_type": entry["source_type"],
                    "collection": entry["collection"],
                })
        return results[:limit]

    async def upsert(
        self, memory_id, content, *, source_type="memory", tags="", collection="default",
    ) -> str:
        self.entries[memory_id] = {
            "content": content,
            "source_type": source_type,
            "tags": tags,
            "collection": collection,
        }
        # Track for dedup
        self._duplicate_map[content] = memory_id
        return memory_id

    async def find_exact_duplicate(self, content: str) -> str | None:
        return self._duplicate_map.get(content)

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self.entries:
            # Remove from duplicate map too
            content = self.entries[memory_id]["content"]
            self._duplicate_map.pop(content, None)
            del self.entries[memory_id]
            return True
        return False


class FakeMetadataBackend:
    """In-memory metadata backend implementing MetadataBackend protocol."""

    def __init__(self):
        self.records: dict[str, dict] = {}

    async def create(
        self,
        memory_id,
        *,
        created_at,
        collection="default",
        confidence=None,
        embedding_status="embedded",
        memory_class="fact",
        wing=None,
        room=None,
    ) -> str:
        self.records[memory_id] = {
            "memory_id": memory_id,
            "created_at": created_at,
            "collection": collection,
            "confidence": confidence,
            "embedding_status": embedding_status,
            "memory_class": memory_class,
            "wing": wing,
            "room": room,
        }
        return memory_id

    async def get(self, memory_id: str) -> dict | None:
        return self.records.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self.records:
            del self.records[memory_id]
            return True
        return False


class FakeLinkBackend:
    """In-memory link backend implementing LinkBackend protocol."""

    def __init__(self):
        self.links: list[dict] = []

    async def create(self, source_id, target_id, link_type, strength, created_at):
        self.links.append({
            "source_id": source_id,
            "target_id": target_id,
            "link_type": link_type,
            "strength": strength,
            "created_at": created_at,
        })
        return (source_id, target_id)

    async def count_links(self, memory_id: str) -> int:
        return sum(
            1 for link in self.links
            if link["source_id"] == memory_id or link["target_id"] == memory_id
        )

    async def delete_by_memory(self, memory_id: str) -> int:
        before = len(self.links)
        self.links = [
            link for link in self.links
            if link["source_id"] != memory_id and link["target_id"] != memory_id
        ]
        return before - len(self.links)


class FakePendingBackend:
    """In-memory pending queue implementing PendingBackend protocol."""

    def __init__(self):
        self.queue: list[dict] = []

    async def enqueue(
        self,
        memory_id,
        content,
        memory_type,
        tags,
        collection,
        *,
        source="",
        confidence=None,
        source_session_id=None,
        transcript_path=None,
        source_line_range=None,
        extraction_timestamp=None,
        source_pipeline=None,
    ) -> None:
        self.queue.append({
            "memory_id": memory_id,
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
            "collection": collection,
            "source": source,
            "confidence": confidence,
            "source_session_id": source_session_id,
            "transcript_path": transcript_path,
            "source_line_range": source_line_range,
            "extraction_timestamp": extraction_timestamp,
            "source_pipeline": source_pipeline,
        })

    async def get_pending(self, *, limit=50) -> list[dict]:
        return self.queue[:limit]

    async def delete(self, memory_id: str) -> bool:
        before = len(self.queue)
        self.queue = [item for item in self.queue if item["memory_id"] != memory_id]
        return len(self.queue) < before


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backends():
    """Create a full set of fake backends."""
    return {
        "embedding_provider": FakeEmbeddingProvider(),
        "vector_backend": FakeVectorBackend(),
        "text_backend": FakeTextBackend(),
        "metadata_backend": FakeMetadataBackend(),
        "link_backend": FakeLinkBackend(),
        "pending_backend": FakePendingBackend(),
    }


@pytest.fixture
def store(backends):
    """Create a MemoryStore with all fake backends."""
    return MemoryStore(**backends)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStoreFullPipeline:
    """Test the full store pipeline with mock backends."""

    async def test_store_returns_memory_id(self, store):
        mid = await store.store("test content", "test_source")
        assert mid  # non-empty string
        assert isinstance(mid, str)

    async def test_store_writes_to_all_backends(self, backends, store):
        mid = await store.store(
            "The system uses circuit breakers for fault tolerance",
            "test_source",
            memory_type="episodic",
            tags=["reliability"],
        )

        # Vector backend should have the point
        vec = backends["vector_backend"]
        assert any(mid in key for key in vec.points)

        # Text backend should have the entry
        text = backends["text_backend"]
        assert mid in text.entries
        assert "circuit breakers" in text.entries[mid]["content"]

        # Metadata backend should have the record
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record is not None
        assert record["embedding_status"] == "embedded"

    async def test_store_resolves_collection_from_type(self, backends, store):
        mid = await store.store("test", "src", memory_type="knowledge")
        text = backends["text_backend"]
        assert text.entries[mid]["collection"] == "knowledge_base"

    async def test_store_uses_explicit_collection(self, backends, store):
        mid = await store.store("test", "src", collection="custom_collection")
        text = backends["text_backend"]
        assert text.entries[mid]["collection"] == "custom_collection"


class TestClassificationIntegration:
    """Test that content gets classified during store."""

    async def test_rule_content_classified_as_rule(self, backends, store):
        mid = await store.store(
            "You MUST NEVER push directly to main without a PR",
            "test",
        )
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["memory_class"] == "rule"

        # class:rule tag should appear in FTS5 entry
        text = backends["text_backend"]
        tags_str = text.entries[mid]["tags"]
        assert "class:rule" in tags_str

    async def test_fact_content_classified_as_fact(self, backends, store):
        mid = await store.store(
            "The system uses SQLite for metadata storage",
            "test",
        )
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["memory_class"] == "fact"

    async def test_reference_content_classified_as_reference(self, backends, store):
        mid = await store.store(
            "See also https://docs.example.com/long-path/to/docs/page",
            "test",
        )
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["memory_class"] == "reference"

    async def test_explicit_memory_class_overrides(self, backends, store):
        mid = await store.store(
            "Just some content",
            "test",
            memory_class="rule",
        )
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["memory_class"] == "rule"


class TestConfidenceGate:
    """Test min_confidence parameter — low confidence forces FTS5-only."""

    async def test_low_confidence_forces_fts5_only(self, backends):
        store = MemoryStore(**backends, min_confidence=0.5)
        mid = await store.store(
            "Low confidence memory",
            "test",
            confidence=0.3,
        )

        # Vector backend should NOT have the point (confidence < min_confidence)
        vec = backends["vector_backend"]
        assert not any(mid in key for key in vec.points)

        # Text backend should still have it
        text = backends["text_backend"]
        assert mid in text.entries

        # Metadata should show pending
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["embedding_status"] == "pending"

        # Should be enqueued in pending
        pending = backends["pending_backend"]
        assert any(item["memory_id"] == mid for item in pending.queue)

    async def test_high_confidence_stores_normally(self, backends):
        store = MemoryStore(**backends, min_confidence=0.5)
        mid = await store.store(
            "High confidence memory",
            "test",
            confidence=0.8,
        )

        # Vector backend should have the point
        vec = backends["vector_backend"]
        assert any(mid in key for key in vec.points)

        # Metadata should show embedded
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["embedding_status"] == "embedded"

    async def test_no_confidence_skips_gate(self, backends):
        """When confidence is None, the gate should not trigger."""
        store = MemoryStore(**backends, min_confidence=0.5)
        mid = await store.store(
            "No confidence specified",
            "test",
            confidence=None,
        )

        # Should store normally (vector embedded)
        vec = backends["vector_backend"]
        assert any(mid in key for key in vec.points)

    async def test_no_min_confidence_skips_gate(self, backends):
        """When min_confidence is None, the gate should not trigger."""
        store = MemoryStore(**backends, min_confidence=None)
        mid = await store.store(
            "Low confidence but no gate",
            "test",
            confidence=0.1,
        )

        # Should store normally
        vec = backends["vector_backend"]
        assert any(mid in key for key in vec.points)


class TestEmbeddingFallback:
    """Test FTS5 fallback when embedding raises EmbeddingUnavailableError."""

    async def test_fts5_fallback_on_embedding_failure(self, backends):
        backends["embedding_provider"] = FakeEmbeddingProvider(should_fail=True)
        store = MemoryStore(**backends)

        mid = await store.store("Memory with failed embedding", "test")

        # Vector backend should NOT have the point
        vec = backends["vector_backend"]
        assert not any(mid in key for key in vec.points)

        # Text backend should still have it
        text = backends["text_backend"]
        assert mid in text.entries

        # Metadata should show pending
        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["embedding_status"] == "pending"

    async def test_successful_embed_stores_vector(self, backends, store):
        mid = await store.store("Successfully embedded memory", "test")

        vec = backends["vector_backend"]
        assert any(mid in key for key in vec.points)

        meta = backends["metadata_backend"]
        record = await meta.get(mid)
        assert record["embedding_status"] == "embedded"


class TestPendingQueue:
    """Test pending queue: failed embedding enqueues via PendingBackend."""

    async def test_embedding_failure_enqueues(self, backends):
        backends["embedding_provider"] = FakeEmbeddingProvider(should_fail=True)
        store = MemoryStore(**backends)

        mid = await store.store(
            "Queue this memory",
            "test",
            tags=["tag1", "tag2"],
            memory_type="episodic",
        )

        pending = backends["pending_backend"]
        assert len(pending.queue) == 1
        item = pending.queue[0]
        assert item["memory_id"] == mid
        assert item["content"] == "Queue this memory"
        assert item["memory_type"] == "episodic"
        assert item["collection"] == "episodic_memory"

    async def test_no_pending_backend_is_fine(self, backends):
        """If pending_backend is None, embedding failure should not crash."""
        backends["embedding_provider"] = FakeEmbeddingProvider(should_fail=True)
        backends["pending_backend"] = None
        store = MemoryStore(**backends)

        # Should not raise even though embedding fails and no pending backend
        mid = await store.store("No pending queue available", "test")
        assert mid  # returns a valid id

    async def test_successful_embed_does_not_enqueue(self, backends, store):
        await store.store("Good embedding", "test")
        pending = backends["pending_backend"]
        assert len(pending.queue) == 0

    async def test_force_fts5_enqueues(self, backends, store):
        mid = await store.store(
            "Forced FTS5 storage",
            "test",
            force_fts5_only=True,
        )

        pending = backends["pending_backend"]
        assert any(item["memory_id"] == mid for item in pending.queue)


class TestDedup:
    """Test deduplication: find_exact_duplicate returns existing ID."""

    async def test_duplicate_returns_existing_id(self, store):
        content = "Exact duplicate content for testing"
        mid1 = await store.store(content, "test")
        mid2 = await store.store(content, "test")
        assert mid1 == mid2

    async def test_different_content_gets_new_id(self, store):
        mid1 = await store.store("First content", "test")
        mid2 = await store.store("Second content", "test")
        assert mid1 != mid2

    async def test_dedup_check_failure_proceeds(self, backends):
        """If dedup check raises, store should continue (not crash)."""

        class BrokenTextBackend(FakeTextBackend):
            async def find_exact_duplicate(self, content: str) -> str | None:
                raise RuntimeError("DB connection lost")

        backends["text_backend"] = BrokenTextBackend()
        store = MemoryStore(**backends)
        mid = await store.store("Content with broken dedup", "test")
        assert mid  # should still return a valid id


class TestDelete:
    """Test delete: should try all collections in self._collections."""

    async def test_delete_returns_per_layer_status(self, backends, store):
        mid = await store.store("Memory to delete", "test")
        result = await store.delete(mid)

        assert result["metadata"] is True
        assert result["fts5"] is True

    async def test_delete_tries_all_collections(self, backends, store):
        mid = await store.store("Memory to delete", "test")
        await store.delete(mid)

        vec = backends["vector_backend"]
        # Should have attempted delete on both collections from _COLLECTION_MAP
        deleted_collections = {coll for _, coll in vec.deleted}
        expected = set(_COLLECTION_MAP.values())
        assert deleted_collections == expected

    async def test_delete_removes_from_all_backends(self, backends, store):
        mid = await store.store("Delete me from everywhere", "test")

        # Verify it exists before delete
        assert mid in backends["text_backend"].entries
        assert (await backends["metadata_backend"].get(mid)) is not None

        result = await store.delete(mid)

        # Verify removal
        assert mid not in backends["text_backend"].entries
        assert (await backends["metadata_backend"].get(mid)) is None
        assert result["metadata"] is True
        assert result["fts5"] is True

    async def test_delete_cleans_links(self, backends, store):
        mid = await store.store("Memory with links", "test")
        # Create a link manually
        await backends["link_backend"].create(mid, "other_id", "supports", 0.8, "2024-01-01")

        result = await store.delete(mid)
        assert result["links_deleted"] == 1

    async def test_delete_cleans_pending(self, backends):
        backends["embedding_provider"] = FakeEmbeddingProvider(should_fail=True)
        store = MemoryStore(**backends)

        mid = await store.store("Pending memory to delete", "test")
        assert len(backends["pending_backend"].queue) == 1

        result = await store.delete(mid)
        assert result["pending_deleted"] is True
        assert len(backends["pending_backend"].queue) == 0

    async def test_delete_nonexistent_memory(self, backends, store):
        result = await store.delete("nonexistent-id")
        assert result["metadata"] is False
        assert result["fts5"] is False


class TestTagSerialization:
    """Test that tags are comma-separated in both FTS5 upsert and pending enqueue."""

    async def test_tags_comma_separated_in_fts5(self, backends, store):
        mid = await store.store(
            "Tagged memory",
            "test",
            tags=["alpha", "beta", "gamma"],
        )

        text = backends["text_backend"]
        tags_str = text.entries[mid]["tags"]
        # Tags should be comma-separated (includes the class tag appended by store)
        parts = tags_str.split(",")
        assert "alpha" in parts
        assert "beta" in parts
        assert "gamma" in parts

    async def test_tags_comma_separated_in_pending(self, backends):
        backends["embedding_provider"] = FakeEmbeddingProvider(should_fail=True)
        store = MemoryStore(**backends)

        await store.store(
            "Pending tagged memory",
            "test",
            tags=["x", "y"],
        )

        pending = backends["pending_backend"]
        assert len(pending.queue) == 1
        tags_str = pending.queue[0]["tags"]
        parts = tags_str.split(",")
        assert "x" in parts
        assert "y" in parts

    async def test_empty_tags_produce_empty_string(self, backends, store):
        mid = await store.store("No explicit tags", "test")
        text = backends["text_backend"]
        tags_str = text.entries[mid]["tags"]
        # Should still have the class tag at minimum
        assert "class:" in tags_str

    async def test_wing_tag_appended(self, backends, store):
        mid = await store.store(
            "Infrastructure memory",
            "test",
            wing="infrastructure",
            tags=["monitoring"],
        )

        text = backends["text_backend"]
        tags_str = text.entries[mid]["tags"]
        assert "wing:infrastructure" in tags_str
        assert "monitoring" in tags_str


class TestDefaultCollections:
    """Test that _collections defaults to _COLLECTION_MAP values."""

    async def test_default_collections(self, backends):
        store = MemoryStore(**backends)
        expected = sorted(set(_COLLECTION_MAP.values()))
        assert sorted(store._collections) == expected

    async def test_explicit_collections_override(self, backends):
        store = MemoryStore(**backends, collections=["custom"])
        assert store._collections == ["custom"]
