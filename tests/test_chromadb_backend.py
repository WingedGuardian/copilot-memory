"""Tests for ChromaDB vector backend."""

import pytest

from genesis_memory.backends.chromadb import ChromaDBVectorBackend


@pytest.fixture
def backend(tmp_path):
    """Isolated ChromaDB backend per test (unique path)."""
    return ChromaDBVectorBackend(path=tmp_path / "chroma", collections=["test"], vector_dim=8)


def _make_vector(seed: int) -> list[float]:
    """Create a simple deterministic 8-dim vector."""
    import hashlib

    h = hashlib.sha256(str(seed).encode()).digest()
    return [b / 255.0 for b in h[:8]]


class TestChromaDBVectorBackend:
    async def test_upsert_and_search(self, backend):
        vec = _make_vector(1)
        await backend.upsert(
            "mem-1", vec,
            {"content": "test content", "wing": "infra"},
            collection="test",
        )
        results = await backend.search(vec, limit=5, collection="test")
        assert len(results) == 1
        assert results[0]["id"] == "mem-1"
        assert results[0]["score"] > 0.99  # Same vector — near-perfect similarity

    async def test_score_convention(self, backend):
        """Verify higher score = more similar."""
        v1 = _make_vector(1)
        v2 = _make_vector(2)
        v3 = _make_vector(3)

        await backend.upsert("mem-1", v1, {"content": "a"}, collection="test")
        await backend.upsert("mem-2", v2, {"content": "b"}, collection="test")
        await backend.upsert("mem-3", v3, {"content": "c"}, collection="test")

        results = await backend.search(v1, limit=3, collection="test")
        assert results[0]["id"] == "mem-1"  # Most similar to itself
        assert results[0]["score"] >= results[1]["score"]

    async def test_wing_filter(self, backend):
        vec1 = _make_vector(1)
        vec2 = _make_vector(2)
        await backend.upsert(
            "mem-1", vec1, {"content": "a", "wing": "infra"}, collection="test",
        )
        await backend.upsert(
            "mem-2", vec2, {"content": "b", "wing": "memory"}, collection="test",
        )

        results = await backend.search(
            vec1, limit=5, collection="test", wing="infra",
        )
        assert len(results) == 1
        assert results[0]["id"] == "mem-1"

    async def test_update_payload(self, backend):
        vec = _make_vector(1)
        await backend.upsert(
            "mem-1", vec,
            {"content": "test", "retrieved_count": 0},
            collection="test",
        )
        await backend.update_payload(
            "mem-1", {"retrieved_count": 5}, collection="test",
        )
        results = await backend.search(vec, limit=1, collection="test")
        assert results[0]["payload"]["retrieved_count"] == 5

    async def test_delete(self, backend):
        vec = _make_vector(1)
        await backend.upsert("mem-1", vec, {"content": "test"}, collection="test")
        await backend.delete("mem-1", collection="test")
        results = await backend.search(vec, limit=5, collection="test")
        assert len(results) == 0

    async def test_scroll_tags(self, backend):
        for i in range(5):
            vec = _make_vector(i)
            tags = f"tag_{i},common"
            await backend.upsert(
                f"mem-{i}", vec, {"content": f"content {i}", "tags": tags},
                collection="test",
            )

        tag_lists, total = await backend.scroll_tags(["test"])
        assert total == 5
        assert len(tag_lists) == 5
        assert all("common" in tags for tags in tag_lists)

    async def test_empty_collection(self, backend):
        results = await backend.search(_make_vector(1), limit=5, collection="test")
        assert results == []

    async def test_multiple_collections(self, backend):
        vec = _make_vector(1)
        await backend.upsert(
            "mem-1", vec, {"content": "in test"}, collection="test",
        )
        # Search different collection — should find nothing
        other = ChromaDBVectorBackend(path=None, collections=["other"])
        results = await other.search(vec, limit=5, collection="other")
        assert len(results) == 0
