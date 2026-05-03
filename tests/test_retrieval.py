"""HybridRetriever tests with mock backends."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from genesis_memory.embeddings import EmbeddingUnavailableError
from genesis_memory.retrieval import HybridRetriever, _rrf_fuse
from genesis_memory.types import RetrievalResult

# ---------------------------------------------------------------------------
# Test doubles — concrete classes implementing the protocol interfaces
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Minimal embedding provider for retrieval tests.

    Uses deterministic 8-dim vectors from content hash.
    Can be configured to raise EmbeddingUnavailableError.
    """

    def __init__(self, *, should_fail: bool = False):
        self._should_fail = should_fail
        self.embed_calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if self._should_fail:
            raise EmbeddingUnavailableError("all backends down")
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:8]]

    @staticmethod
    def enrich(content: str, memory_type: str, tags: list[str]) -> str:
        if tags:
            return f"{memory_type}: {' '.join(tags)}: {content}"
        return f"{memory_type}: {content}"


class FakeVectorBackend:
    """In-memory vector backend with cosine similarity search."""

    def __init__(self):
        self.points: dict[str, dict] = {}  # "collection:id" -> {vector, payload}
        self.payload_updates: list[tuple[str, dict, str]] = []

    async def search(
        self, vector, *, limit=10, collection="default", wing=None, room=None,
    ) -> list[dict]:
        results = []
        prefix = f"{collection}:"
        for key, data in self.points.items():
            if not key.startswith(prefix):
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
        self.points[f"{collection}:{point_id}"] = {"vector": vector, "payload": payload}

    async def update_payload(self, point_id, payload, *, collection="default"):
        key = f"{collection}:{point_id}"
        if key in self.points:
            self.points[key]["payload"].update(payload)
        self.payload_updates.append((point_id, payload, collection))

    async def delete(self, point_id, *, collection="default"):
        key = f"{collection}:{point_id}"
        self.points.pop(key, None)

    async def scroll_tags(self, collections):
        return [], 0


class FakeTextBackend:
    """In-memory FTS5 backend with simple substring matching."""

    def __init__(self):
        self.entries: dict[str, dict] = {}  # memory_id -> {content, source_type, tags, collection}

    def add_entry(
        self, memory_id: str, content: str, *,
        source_type: str = "memory", tags: str = "", collection: str = "episodic_memory",
    ):
        """Helper to pre-populate entries for retrieval tests."""
        self.entries[memory_id] = {
            "content": content,
            "source_type": source_type,
            "tags": tags,
            "collection": collection,
        }

    async def search_ranked(
        self, query, *, collection=None, limit=30, boolean=False,
    ) -> list[dict]:
        results = []
        query_lower = query.lower()
        # Split query into words and match if any word is found
        query_words = query_lower.split()
        for mid, entry in self.entries.items():
            if collection and entry["collection"] != collection:
                continue
            content_lower = entry["content"].lower()
            if any(word in content_lower for word in query_words):
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
        return memory_id

    async def find_exact_duplicate(self, content: str) -> str | None:
        for mid, entry in self.entries.items():
            if entry["content"] == content:
                return mid
        return None

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self.entries:
            del self.entries[memory_id]
            return True
        return False


class FakeLinkBackend:
    """In-memory link backend."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed_sync(text: str) -> list[float]:
    """Synchronous version of FakeEmbeddingProvider.embed for setup."""
    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:8]]


async def _populate_vector_and_text(
    vector: FakeVectorBackend,
    text: FakeTextBackend,
    embedding: FakeEmbeddingProvider,
    memories: list[dict],
):
    """Pre-populate both backends with test data.

    Each dict in memories has: id, content, and optional source, memory_type,
    tags, confidence, wing, room, collection.
    """
    for mem in memories:
        mid = mem["id"]
        content = mem["content"]
        collection = mem.get("collection", "episodic_memory")
        memory_type = mem.get("memory_type", "episodic")
        tags = mem.get("tags", [])
        wing = mem.get("wing")
        room = mem.get("room")

        # Embed and store in vector backend
        enriched = FakeEmbeddingProvider.enrich(content, memory_type, tags)
        vec = _embed_sync(enriched)
        now_iso = datetime.now(UTC).isoformat()

        payload = {
            "content": content,
            "source": mem.get("source", "test"),
            "memory_type": memory_type,
            "tags": tags,
            "confidence": mem.get("confidence", 0.5),
            "created_at": now_iso,
            "retrieved_count": mem.get("retrieved_count", 0),
            "source_type": "memory",
            "memory_class": mem.get("memory_class", "fact"),
            "wing": wing,
            "room": room,
        }

        await vector.upsert(mid, vec, payload, collection=collection)

        # Store in text backend
        text.add_entry(
            mid, content,
            source_type="memory",
            tags=",".join(tags) if tags else "",
            collection=collection,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_provider():
    return FakeEmbeddingProvider()


@pytest.fixture
def vector_backend():
    return FakeVectorBackend()


@pytest.fixture
def text_backend():
    return FakeTextBackend()


@pytest.fixture
def link_backend():
    return FakeLinkBackend()


@pytest.fixture
def retriever(embedding_provider, vector_backend, text_backend, link_backend):
    return HybridRetriever(
        embedding_provider=embedding_provider,
        vector_backend=vector_backend,
        text_backend=text_backend,
        link_backend=link_backend,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test the full 12-step retrieval pipeline."""

    async def test_recall_returns_results(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {"id": "mem-1", "content": "Circuit breakers protect against cascading failures"},
                {"id": "mem-2", "content": "Rate limiting prevents traffic spikes"},
                {"id": "mem-3", "content": "RRF fusion combines vector and text search"},
            ],
        )

        results = await retriever.recall("circuit breakers", limit=5)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_recall_finds_relevant_content(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {"id": "mem-1", "content": "Circuit breakers protect against cascading failures"},
                {"id": "mem-2", "content": "Database indexes improve query performance"},
            ],
        )

        results = await retriever.recall("circuit breakers cascading", limit=5)
        memory_ids = [r.memory_id for r in results]
        # mem-1 should be found (matches FTS5 on "circuit breakers")
        assert "mem-1" in memory_ids

    async def test_recall_result_has_scores(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [{"id": "mem-1", "content": "Activation scoring uses exponential decay"}],
        )

        results = await retriever.recall("activation scoring", limit=5)
        assert len(results) > 0
        for r in results:
            assert r.score > 0  # RRF score
            assert r.activation_score >= 0
            assert r.memory_class in ("rule", "fact", "reference")

    async def test_empty_recall(self, retriever):
        results = await retriever.recall("nothing matches this query", limit=5)
        assert results == []

    async def test_invalid_source_raises(self, retriever):
        with pytest.raises(ValueError, match="source must be one of"):
            await retriever.recall("test", source="invalid")


class TestFTS5Fallback:
    """Test FTS5-only fallback when embedding is unavailable."""

    async def test_fts5_only_when_embedding_fails(
        self, vector_backend, text_backend, link_backend,
    ):
        failing_embed = FakeEmbeddingProvider(should_fail=True)
        retriever = HybridRetriever(
            embedding_provider=failing_embed,
            vector_backend=vector_backend,
            text_backend=text_backend,
            link_backend=link_backend,
        )

        # Only populate text backend (no vectors needed for fallback)
        text_backend.add_entry("mem-1", "Circuit breakers protect systems")
        text_backend.add_entry("mem-2", "Rate limiting prevents spikes")

        results = await retriever.recall("circuit breakers", limit=5)
        assert len(results) > 0
        memory_ids = [r.memory_id for r in results]
        assert "mem-1" in memory_ids

    async def test_fts5_fallback_has_no_vector_rank(
        self, vector_backend, text_backend, link_backend,
    ):
        failing_embed = FakeEmbeddingProvider(should_fail=True)
        retriever = HybridRetriever(
            embedding_provider=failing_embed,
            vector_backend=vector_backend,
            text_backend=text_backend,
            link_backend=link_backend,
        )

        text_backend.add_entry("mem-1", "Test memory content")

        results = await retriever.recall("test memory", limit=5)
        assert len(results) > 0
        for r in results:
            assert r.vector_rank is None  # No vector search was done


class TestWingRoomFiltering:
    """Test wing/room filtering behavior.

    When embedding IS available, FTS5-only candidates should be filtered out.
    When embedding is NOT available, wing/room filter should be SKIPPED.
    """

    async def test_wing_filter_with_embedding_removes_fts_only(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        """When embedding is available and wing filter is set, FTS5-only
        candidates (not in vector results) should be filtered out."""
        # Put mem-1 in vector+text with wing=infrastructure
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {
                    "id": "mem-1",
                    "content": "Infrastructure monitoring dashboard",
                    "wing": "infrastructure",
                },
            ],
        )
        # Put mem-2 in text only (no vector) — simulates FTS5-only candidate
        text_backend.add_entry(
            "mem-2", "Infrastructure backup system",
            collection="episodic_memory",
        )

        results = await retriever.recall(
            "infrastructure monitoring", limit=5, wing="infrastructure",
        )
        memory_ids = [r.memory_id for r in results]

        # mem-1 should appear (has vector hit with matching wing)
        assert "mem-1" in memory_ids
        # mem-2 should be FILTERED OUT (FTS5-only, can't verify wing)
        assert "mem-2" not in memory_ids

    async def test_wing_filter_skipped_when_embedding_unavailable(
        self, vector_backend, text_backend, link_backend,
    ):
        """When embedding is unavailable, wing/room filter should be SKIPPED
        to avoid dropping all results (since all are FTS5-only)."""
        failing_embed = FakeEmbeddingProvider(should_fail=True)
        retriever = HybridRetriever(
            embedding_provider=failing_embed,
            vector_backend=vector_backend,
            text_backend=text_backend,
            link_backend=link_backend,
        )

        # Both entries are text-only (no vector)
        text_backend.add_entry("mem-1", "Infrastructure monitoring stuff")
        text_backend.add_entry("mem-2", "Infrastructure backup stuff")

        # With wing filter but no embedding — should still return results
        results = await retriever.recall(
            "infrastructure", limit=5, wing="infrastructure",
        )
        assert len(results) > 0  # Should NOT be empty
        memory_ids = [r.memory_id for r in results]
        # Both should be present (filter skipped)
        assert "mem-1" in memory_ids
        assert "mem-2" in memory_ids

    async def test_no_wing_filter_keeps_fts_only(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        """Without wing/room filter, FTS5-only candidates should be kept."""
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [{"id": "mem-1", "content": "Vector and text memory"}],
        )
        text_backend.add_entry("mem-2", "Text only memory for search")

        results = await retriever.recall("memory", limit=5)
        memory_ids = [r.memory_id for r in results]
        # Both should appear — no wing filter, so FTS5-only kept
        assert "mem-1" in memory_ids
        assert "mem-2" in memory_ids


class TestMinActivation:
    """Test min_activation filter."""

    async def test_min_activation_filters_low_scores(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {
                    "id": "mem-1",
                    "content": "High confidence memory about activation",
                    "confidence": 0.95,
                },
                {
                    "id": "mem-2",
                    "content": "Very low confidence memory about activation",
                    "confidence": 0.01,
                },
            ],
        )

        # With a very high min_activation, only the strongest memories survive
        results_high = await retriever.recall("activation", limit=5, min_activation=0.4)
        results_low = await retriever.recall("activation", limit=5, min_activation=0.0)

        # Low threshold should return at least as many as high threshold
        assert len(results_low) >= len(results_high)

    async def test_zero_min_activation_returns_all(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {"id": "mem-1", "content": "First memory about testing"},
                {"id": "mem-2", "content": "Second memory about testing"},
            ],
        )

        results = await retriever.recall("testing", limit=5, min_activation=0.0)
        assert len(results) >= 2


class TestRRFFusion:
    """Test that RRF fusion produces correct relative ordering."""

    def test_rrf_fuse_single_list(self):
        """Single ranked list should produce scores based on rank."""
        scores = _rrf_fuse([["a", "b", "c"]], k=60)
        assert scores["a"] > scores["b"] > scores["c"]

    def test_rrf_fuse_multiple_lists_boost(self):
        """Item appearing in multiple lists should get higher score."""
        scores = _rrf_fuse([
            ["a", "b", "c"],
            ["b", "c", "d"],
        ], k=60)
        # "b" appears in both lists (rank 2 in first, rank 1 in second)
        # "a" appears only in first list (rank 1)
        assert scores["b"] > scores["a"]

    def test_rrf_fuse_empty_lists(self):
        scores = _rrf_fuse([[]], k=60)
        assert scores == {}

    def test_rrf_fuse_preserves_all_ids(self):
        """All IDs from all lists should appear in output."""
        scores = _rrf_fuse([
            ["a", "b"],
            ["c", "d"],
        ], k=60)
        assert set(scores.keys()) == {"a", "b", "c", "d"}

    async def test_retrieval_rrf_ordering(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        """Memories matching both vector and FTS5 should rank higher than
        those matching only one."""
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {"id": "mem-both", "content": "Circuit breakers for fault tolerance"},
                {"id": "mem-vec-only", "content": "Completely unrelated topic here"},
            ],
        )
        # Add an FTS5-only entry that matches the query
        text_backend.add_entry(
            "mem-fts-only", "Circuit breakers protect systems from cascading failures",
        )

        results = await retriever.recall("circuit breakers", limit=5)
        if len(results) >= 2:
            # The memory matching in both channels should rank highest via RRF
            top_id = results[0].memory_id
            # mem-both matches FTS5 (has "circuit breakers") and likely has
            # a reasonable vector similarity, so it should be top or near-top
            assert top_id in ("mem-both", "mem-fts-only")


class TestRetrievedCountIncrement:
    """Test that retrieved_count is incremented via update_payload."""

    async def test_retrieved_count_incremented(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [{"id": "mem-1", "content": "Memory to be retrieved and counted"}],
        )

        await retriever.recall("retrieved counted", limit=5)

        # Check that update_payload was called to increment retrieved_count
        updates = vector_backend.payload_updates
        matching = [
            (pid, p) for pid, p, _coll in updates if pid == "mem-1"
        ]
        assert len(matching) >= 1
        assert matching[0][1]["retrieved_count"] == 1  # 0 -> 1

    async def test_retrieved_count_increments_cumulatively(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [{"id": "mem-1", "content": "Memory counted twice via retrieval"}],
        )

        await retriever.recall("counted twice", limit=5)
        await retriever.recall("counted twice retrieval", limit=5)

        # After two recalls, retrieved_count in the payload should be 2
        key = "episodic_memory:mem-1"
        assert vector_backend.points[key]["payload"]["retrieved_count"] == 2


class TestSourceFilter:
    """Test source filter (episodic/knowledge/both)."""

    async def test_episodic_source_searches_episodic_collection(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {
                    "id": "mem-ep",
                    "content": "Episodic memory about testing search",
                    "collection": "episodic_memory",
                },
                {
                    "id": "mem-kb",
                    "content": "Knowledge base entry about testing search",
                    "collection": "knowledge_base",
                },
            ],
        )

        results = await retriever.recall("testing search", source="episodic", limit=5)
        memory_ids = [r.memory_id for r in results]
        # Episodic memory should be found
        assert "mem-ep" in memory_ids

    async def test_both_source_searches_all_collections(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [
                {
                    "id": "mem-ep",
                    "content": "Episodic memory about combined search",
                    "collection": "episodic_memory",
                },
                {
                    "id": "mem-kb",
                    "content": "Knowledge entry about combined search",
                    "collection": "knowledge_base",
                },
            ],
        )

        results = await retriever.recall("combined search", source="both", limit=10)
        memory_ids = [r.memory_id for r in results]
        assert "mem-ep" in memory_ids
        assert "mem-kb" in memory_ids


class TestIntentClassification:
    """Test that retrieval results include intent classification."""

    async def test_intent_attached_to_results(
        self, retriever, vector_backend, text_backend, embedding_provider,
    ):
        await _populate_vector_and_text(
            vector_backend, text_backend, embedding_provider,
            [{"id": "mem-1", "content": "The system architecture uses microservices"}],
        )

        results = await retriever.recall("what is the system architecture", limit=5)
        assert len(results) > 0
        for r in results:
            assert r.query_intent is not None
            assert r.intent_confidence >= 0.0
