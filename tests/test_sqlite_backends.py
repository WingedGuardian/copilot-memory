"""Tests for all SQLite backends: text, metadata, links, pending."""

import aiosqlite
import pytest

from genesis_memory.backends.schema import init_schema
from genesis_memory.backends.sqlite_links import SQLiteLinkBackend
from genesis_memory.backends.sqlite_metadata import SQLiteMetadataBackend
from genesis_memory.backends.sqlite_pending import SQLitePendingBackend
from genesis_memory.backends.sqlite_text import SQLiteTextBackend


@pytest.fixture
async def db():
    """In-memory SQLite database with schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await init_schema(conn)
    yield conn
    await conn.close()


# ── TextBackend ──


class TestSQLiteTextBackend:
    @pytest.fixture
    def text(self, db):
        return SQLiteTextBackend(db)

    async def test_upsert_and_search(self, text):
        await text.upsert("mem-1", "The routing system uses circuit breakers", tags="routing")
        results = await text.search("routing")
        assert len(results) == 1
        assert results[0]["memory_id"] == "mem-1"

    async def test_search_ranked(self, text):
        await text.upsert("mem-1", "Circuit breakers protect against cascading failures")
        await text.upsert("mem-2", "Rate limiting prevents overload on the circuit")
        results = await text.search_ranked("circuit breakers")
        assert len(results) >= 1
        assert all("rank" in r for r in results)

    async def test_search_ranked_with_collection_filter(self, text):
        await text.upsert("mem-1", "Memory in default", collection="default")
        await text.upsert("mem-2", "Memory in knowledge", collection="knowledge")
        results = await text.search_ranked("memory", collection="knowledge")
        assert len(results) == 1
        assert results[0]["memory_id"] == "mem-2"

    async def test_search_ranked_boolean(self, text):
        await text.upsert("mem-1", "routing circuit breakers")
        await text.upsert("mem-2", "memory retrieval fusion")
        results = await text.search_ranked("routing OR memory", boolean=True)
        assert len(results) == 2

    async def test_upsert_replaces_existing(self, text):
        await text.upsert("mem-1", "original content")
        await text.upsert("mem-1", "updated content")
        results = await text.search("updated")
        assert len(results) == 1
        assert results[0]["content"] == "updated content"

    async def test_find_exact_duplicate(self, text):
        content = "This is a specific piece of content for dedup testing"
        await text.upsert("mem-1", content)
        dup = await text.find_exact_duplicate(content)
        assert dup == "mem-1"

    async def test_find_exact_duplicate_no_match(self, text):
        await text.upsert("mem-1", "some content")
        dup = await text.find_exact_duplicate("different content entirely")
        assert dup is None

    async def test_find_exact_duplicate_empty(self, text):
        assert await text.find_exact_duplicate("") is None

    async def test_delete(self, text):
        await text.upsert("mem-1", "to be deleted")
        assert await text.delete("mem-1") is True
        results = await text.search("deleted")
        assert len(results) == 0

    async def test_delete_nonexistent(self, text):
        assert await text.delete("nonexistent") is False

    async def test_empty_query_returns_empty(self, text):
        await text.upsert("mem-1", "some content")
        assert await text.search("") == []
        assert await text.search_ranked("") == []

    async def test_special_chars_sanitized(self, text):
        await text.upsert("mem-1", "routing system test")
        # Special chars should be stripped, not crash FTS5
        results = await text.search("routing! @system#")
        assert len(results) == 1

    async def test_boolean_unbalanced_parens(self, text):
        await text.upsert("mem-1", "test content")
        # Unbalanced parens should be stripped
        results = await text.search_ranked("(test", boolean=True)
        assert len(results) == 1


# ── MetadataBackend ──


class TestSQLiteMetadataBackend:
    @pytest.fixture
    def meta(self, db):
        return SQLiteMetadataBackend(db)

    async def test_create_and_get(self, meta):
        await meta.create(
            "mem-1",
            created_at="2026-05-01T00:00:00Z",
            collection="default",
            confidence=0.9,
            memory_class="rule",
            wing="learning",
            room="procedures",
        )
        result = await meta.get("mem-1")
        assert result is not None
        assert result["memory_id"] == "mem-1"
        assert result["confidence"] == 0.9
        assert result["memory_class"] == "rule"
        assert result["wing"] == "learning"
        assert result["room"] == "procedures"

    async def test_get_nonexistent(self, meta):
        assert await meta.get("nonexistent") is None

    async def test_delete(self, meta):
        await meta.create("mem-1", created_at="2026-05-01T00:00:00Z")
        assert await meta.delete("mem-1") is True
        assert await meta.get("mem-1") is None

    async def test_delete_nonexistent(self, meta):
        assert await meta.delete("nonexistent") is False

    async def test_create_ignores_duplicate(self, meta):
        await meta.create("mem-1", created_at="2026-05-01T00:00:00Z", confidence=0.5)
        await meta.create("mem-1", created_at="2026-05-02T00:00:00Z", confidence=0.9)
        result = await meta.get("mem-1")
        # INSERT OR IGNORE keeps the first one
        assert result["confidence"] == 0.5


# ── LinkBackend ──


class TestSQLiteLinkBackend:
    @pytest.fixture
    def links(self, db):
        return SQLiteLinkBackend(db)

    async def test_create_and_count(self, links):
        await links.create("a", "b", "supports", 0.8, "2026-05-01T00:00:00Z")
        assert await links.count_links("a") == 1
        assert await links.count_links("b") == 1  # bidirectional count

    async def test_multiple_links(self, links):
        await links.create("a", "b", "supports", 0.8, "2026-05-01T00:00:00Z")
        await links.create("a", "c", "extends", 0.9, "2026-05-01T00:00:00Z")
        await links.create("d", "a", "contradicts", 0.6, "2026-05-01T00:00:00Z")
        assert await links.count_links("a") == 3

    async def test_delete_by_memory(self, links):
        await links.create("a", "b", "supports", 0.8, "2026-05-01T00:00:00Z")
        await links.create("a", "c", "extends", 0.9, "2026-05-01T00:00:00Z")
        await links.create("d", "a", "related_to", 0.5, "2026-05-01T00:00:00Z")
        deleted = await links.delete_by_memory("a")
        assert deleted == 3
        assert await links.count_links("a") == 0

    async def test_count_nonexistent(self, links):
        assert await links.count_links("nonexistent") == 0


# ── PendingBackend ──


class TestSQLitePendingBackend:
    @pytest.fixture
    def pending(self, db):
        return SQLitePendingBackend(db)

    async def test_enqueue_and_get(self, pending):
        await pending.enqueue(
            "mem-1", "test content", "episodic", "tag1,tag2", "default",
            source="test", confidence=0.8,
        )
        items = await pending.get_pending()
        assert len(items) == 1
        assert items[0]["memory_id"] == "mem-1"
        assert items[0]["content"] == "test content"
        assert items[0]["source"] == "test"

    async def test_get_pending_respects_limit(self, pending):
        for i in range(5):
            await pending.enqueue(f"mem-{i}", f"content {i}", "episodic", "", "default")
        items = await pending.get_pending(limit=3)
        assert len(items) == 3

    async def test_delete(self, pending):
        await pending.enqueue("mem-1", "content", "episodic", "", "default")
        assert await pending.delete("mem-1") is True
        items = await pending.get_pending()
        assert len(items) == 0

    async def test_delete_nonexistent(self, pending):
        assert await pending.delete("nonexistent") is False

    async def test_empty_queue(self, pending):
        items = await pending.get_pending()
        assert items == []
