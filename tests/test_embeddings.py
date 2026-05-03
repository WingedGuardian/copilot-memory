"""Tests for EmbeddingProvider — cache and chain fallback."""

import pytest

from genesis_memory.embeddings import EmbeddingProvider, EmbeddingUnavailableError


class FakeBackend:
    """Test embedding backend with configurable behavior."""

    def __init__(self, name: str, fail: bool = False):
        self._name = name
        self._fail = fail
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if self._fail:
            raise ConnectionError(f"{self._name} unavailable")
        # Deterministic vector based on text hash
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:8]]

    async def is_available(self) -> bool:
        return not self._fail


class TestEmbeddingProvider:
    async def test_basic_embed(self):
        backend = FakeBackend("test")
        provider = EmbeddingProvider(backends=[backend], cache_dir=None)
        vec = await provider.embed("hello")
        assert len(vec) == 8
        assert backend.call_count == 1

    async def test_cache_hit(self):
        backend = FakeBackend("test")
        provider = EmbeddingProvider(backends=[backend], cache_dir=None)
        vec1 = await provider.embed("hello")
        vec2 = await provider.embed("hello")
        assert vec1 == vec2
        assert backend.call_count == 1  # cached

    async def test_cache_miss_different_text(self):
        backend = FakeBackend("test")
        provider = EmbeddingProvider(backends=[backend], cache_dir=None)
        await provider.embed("hello")
        await provider.embed("world")
        assert backend.call_count == 2

    async def test_chain_fallback(self):
        primary = FakeBackend("primary", fail=True)
        fallback = FakeBackend("fallback")
        provider = EmbeddingProvider(backends=[primary, fallback], cache_dir=None)
        vec = await provider.embed("hello")
        assert len(vec) == 8
        assert primary.call_count == 1
        assert fallback.call_count == 1

    async def test_all_backends_fail(self):
        b1 = FakeBackend("b1", fail=True)
        b2 = FakeBackend("b2", fail=True)
        provider = EmbeddingProvider(backends=[b1, b2], cache_dir=None)
        with pytest.raises(EmbeddingUnavailableError):
            await provider.embed("hello")

    async def test_is_available(self):
        available = FakeBackend("ok")
        provider = EmbeddingProvider(backends=[available], cache_dir=None)
        assert await provider.is_available() is True

    async def test_is_available_none(self):
        unavailable = FakeBackend("fail", fail=True)
        provider = EmbeddingProvider(backends=[unavailable], cache_dir=None)
        assert await provider.is_available() is False

    async def test_embed_batch(self):
        backend = FakeBackend("test")
        provider = EmbeddingProvider(backends=[backend], cache_dir=None)
        vecs = await provider.embed_batch(["hello", "world"])
        assert len(vecs) == 2
        assert backend.call_count == 2

    async def test_cache_stats(self):
        backend = FakeBackend("test")
        provider = EmbeddingProvider(backends=[backend], cache_dir=None)
        await provider.embed("hello")
        await provider.embed("hello")  # cache hit
        stats = provider.cache_stats()
        assert stats["l1_hits"] == 1
        assert stats["misses"] == 1
        assert stats["remote_calls"] == 1

    async def test_enrich(self):
        enriched = EmbeddingProvider.enrich("test content", "episodic", ["routing", "config"])
        assert enriched == "episodic: routing config: test content"

    async def test_enrich_no_tags(self):
        enriched = EmbeddingProvider.enrich("test content", "episodic", [])
        assert enriched == "episodic: test content"

    async def test_on_event_callback(self):
        events = []
        primary = FakeBackend("primary", fail=True)
        fallback = FakeBackend("fallback")
        provider = EmbeddingProvider(
            backends=[primary, fallback],
            cache_dir=None,
            on_event=lambda t, m: events.append((t, m)),
        )
        await provider.embed("hello")
        assert len(events) == 1
        assert events[0][0] == "embedding.fallback"

    async def test_cache_eviction(self):
        backend = FakeBackend("test")
        provider = EmbeddingProvider(backends=[backend], cache_dir=None)
        provider._cache_max = 3  # small cache
        for i in range(5):
            await provider.embed(f"text-{i}")
        assert len(provider._cache) == 3
