"""Tests for MemoryLinker — auto-linking and typed links."""


from genesis_memory.linker import MemoryLinker
from genesis_memory.types import LinkRecord

# -- Test doubles --


class StubVectorBackend:
    """Returns configurable search results for auto_link testing."""

    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    async def search(self, vector, *, limit=10, collection="default",
                     wing=None, room=None) -> list[dict]:
        return self._results[:limit]

    async def upsert(self, point_id, vector, payload, *, collection="default"):
        pass

    async def update_payload(self, point_id, payload, *, collection="default"):
        pass

    async def delete(self, point_id, *, collection="default"):
        pass

    async def scroll_tags(self, collections):
        return [], 0


class StubTextBackend:
    """Returns configurable search results for entity name lookups."""

    def __init__(self, results: list[dict] | None = None):
        self._results = results or []

    async def search(self, query, *, limit=10) -> list[dict]:
        return self._results[:limit]

    async def search_ranked(self, query, *, collection=None, limit=30,
                            boolean=False) -> list[dict]:
        return self._results[:limit]

    async def upsert(self, memory_id, content, *, source_type="memory",
                     tags="", collection="default") -> str:
        return memory_id

    async def find_exact_duplicate(self, content) -> str | None:
        return None

    async def delete(self, memory_id) -> bool:
        return True


class RecordingLinkBackend:
    """Records link creation calls for verification."""

    def __init__(self):
        self.created: list[tuple] = []
        self._link_count = 0

    async def create(self, source_id, target_id, link_type, strength,
                     created_at) -> tuple[str, str]:
        self.created.append((source_id, target_id, link_type, strength))
        return (source_id, target_id)

    async def count_links(self, memory_id) -> int:
        return self._link_count

    async def delete_by_memory(self, memory_id) -> int:
        return 0


class FailingLinkBackend(RecordingLinkBackend):
    """Raises on create to test error handling in create_typed_links."""

    async def create(self, source_id, target_id, link_type, strength,
                     created_at) -> tuple[str, str]:
        raise RuntimeError("Link creation failed")


# -- auto_link tests --


class TestAutoLink:
    async def test_creates_links_for_similar_memories(self):
        vector_backend = StubVectorBackend([
            {"id": "target_1", "score": 0.92, "payload": {}},
            {"id": "target_2", "score": 0.80, "payload": {}},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("source_mem", [0.1, 0.2, 0.3])
        assert len(links) == 2
        assert links[0].source_id == "source_mem"
        assert links[0].target_id == "target_1"
        assert links[0].link_type == "extends"   # score >= 0.90
        assert links[1].link_type == "supports"   # score < 0.90

    async def test_self_links_avoided(self):
        """auto_link should skip results where target_id == memory_id."""
        vector_backend = StubVectorBackend([
            {"id": "source_mem", "score": 1.0, "payload": {}},  # self
            {"id": "other_mem", "score": 0.85, "payload": {}},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("source_mem", [0.1, 0.2])
        assert len(links) == 1
        assert links[0].target_id == "other_mem"
        # Only one link created (self was skipped)
        assert len(link_backend.created) == 1

    async def test_similarity_threshold_default(self):
        """Results below the default threshold (0.75) should be excluded."""
        vector_backend = StubVectorBackend([
            {"id": "high", "score": 0.80, "payload": {}},
            {"id": "low", "score": 0.70, "payload": {}},  # below 0.75
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("src", [0.1])
        assert len(links) == 1
        assert links[0].target_id == "high"

    async def test_custom_similarity_threshold(self):
        vector_backend = StubVectorBackend([
            {"id": "m1", "score": 0.95, "payload": {}},
            {"id": "m2", "score": 0.85, "payload": {}},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("src", [0.1], similarity_threshold=0.90)
        assert len(links) == 1
        assert links[0].target_id == "m1"

    async def test_max_links_respected(self):
        vector_backend = StubVectorBackend([
            {"id": f"m{i}", "score": 0.95 - i * 0.01, "payload": {}}
            for i in range(10)
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("src", [0.1], max_links=3)
        assert len(links) == 3

    async def test_no_similar_memories(self):
        vector_backend = StubVectorBackend([])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("src", [0.1])
        assert links == []

    async def test_extends_vs_supports_threshold(self):
        """Score >= 0.90 gives 'extends', below gives 'supports'."""
        vector_backend = StubVectorBackend([
            {"id": "ext", "score": 0.91, "payload": {}},
            {"id": "sup", "score": 0.89, "payload": {}},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("src", [0.1])
        link_types = {lnk.target_id: lnk.link_type for lnk in links}
        assert link_types["ext"] == "extends"
        assert link_types["sup"] == "supports"

    async def test_link_record_fields(self):
        vector_backend = StubVectorBackend([
            {"id": "tgt", "score": 0.85, "payload": {}},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=vector_backend,
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.auto_link("src", [0.1])
        assert len(links) == 1
        link = links[0]
        assert isinstance(link, LinkRecord)
        assert link.source_id == "src"
        assert link.target_id == "tgt"
        assert link.strength == 0.85
        assert link.created_at  # non-empty ISO string


# -- create_typed_links tests --


class TestCreateTypedLinks:
    async def test_creates_links_from_relationships(self):
        text_backend = StubTextBackend([
            {"memory_id": "entity_1", "content": "AgentMail service description"},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=link_backend,
        )

        relationships = [
            {"type": "supports", "to": "AgentMail"},
        ]
        links = await linker.create_typed_links("src_mem", relationships)
        assert len(links) == 1
        assert links[0].link_type == "supports"
        assert links[0].target_id == "entity_1"
        assert links[0].strength == 0.7  # fixed strength for typed links

    async def test_skips_invalid_link_types(self):
        text_backend = StubTextBackend([
            {"memory_id": "ent", "content": "some entity"},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=link_backend,
        )

        relationships = [
            {"type": "invalid_type", "to": "some entity"},
        ]
        links = await linker.create_typed_links("src", relationships)
        assert links == []

    async def test_skips_self_links(self):
        text_backend = StubTextBackend([
            {"memory_id": "src_mem", "content": "self reference"},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=link_backend,
        )

        relationships = [
            {"type": "supports", "to": "self reference"},
        ]
        links = await linker.create_typed_links("src_mem", relationships)
        assert links == []

    async def test_skips_missing_target(self):
        text_backend = StubTextBackend([])  # no results
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=link_backend,
        )

        relationships = [
            {"type": "supports", "to": "nonexistent entity"},
        ]
        links = await linker.create_typed_links("src", relationships)
        assert links == []

    async def test_empty_relationships(self):
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        links = await linker.create_typed_links("src", [])
        assert links == []

    async def test_link_creation_error_handled(self):
        """If link creation raises, the link is skipped but others proceed."""
        text_backend = StubTextBackend([
            {"memory_id": "ent_1", "content": "first entity"},
        ])
        link_backend = FailingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=link_backend,
        )

        relationships = [
            {"type": "supports", "to": "first entity"},
        ]
        links = await linker.create_typed_links("src", relationships)
        assert links == []  # failed, so no links returned

    async def test_skips_empty_type_or_target(self):
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=StubTextBackend(),
            link_backend=link_backend,
        )

        relationships = [
            {"type": "", "to": "something"},   # empty type
            {"type": "supports", "to": ""},     # empty target
            {"to": "something"},                # missing type
            {"type": "supports"},               # missing target
        ]
        links = await linker.create_typed_links("src", relationships)
        assert links == []


# -- _find_entity_by_name tests --


class TestFindEntityByName:
    async def test_exact_substring_match_preferred(self):
        """When FTS5 returns results, prefer the one containing the entity name."""
        text_backend = StubTextBackend([
            {"memory_id": "m1", "content": "routing configuration details"},
            {"memory_id": "m2", "content": "AgentMail service is an email tool"},
        ])
        link_backend = RecordingLinkBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=link_backend,
        )

        result = await linker._find_entity_by_name("AgentMail")
        assert result == "m2"

    async def test_falls_back_to_first_result(self):
        """If no result contains the entity name, return the first result."""
        text_backend = StubTextBackend([
            {"memory_id": "m1", "content": "some related content"},
        ])
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=RecordingLinkBackend(),
        )

        result = await linker._find_entity_by_name("nonexistent entity")
        assert result == "m1"

    async def test_fuzzy_match_threshold_0_6(self):
        """Fuzzy matching threshold is 0.6 (not 0.3)."""

        class WordSearchTextBackend(StubTextBackend):
            """Returns different results for different queries."""
            def __init__(self):
                self._call_count = 0

            async def search(self, query, *, limit=10):
                self._call_count += 1
                if self._call_count == 1:
                    # First call (full entity name) returns empty
                    return []
                # Subsequent calls (individual words) return candidates
                return [
                    {"memory_id": "fuzzy_m", "content": "agent mail service setup"},
                ]

        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=WordSearchTextBackend(),
            link_backend=RecordingLinkBackend(),
        )

        # "AgentMail service" vs "agent mail service setup" should have
        # a decent SequenceMatcher ratio (well above 0.6)
        result = await linker._find_entity_by_name("AgentMail service")
        assert result == "fuzzy_m"

    async def test_fuzzy_match_below_threshold_returns_none(self):
        """If fuzzy ratio < 0.6, return None."""

        class WordSearchTextBackend(StubTextBackend):
            def __init__(self):
                self._call_count = 0

            async def search(self, query, *, limit=10):
                self._call_count += 1
                if self._call_count == 1:
                    return []
                # Very different content — low SequenceMatcher ratio
                return [
                    {"memory_id": "bad_m",
                     "content": "completely unrelated topic about quantum computing"},
                ]

        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=WordSearchTextBackend(),
            link_backend=RecordingLinkBackend(),
        )

        result = await linker._find_entity_by_name("AgentMail")
        assert result is None

    async def test_no_results_returns_none(self):
        text_backend = StubTextBackend([])
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=RecordingLinkBackend(),
        )

        result = await linker._find_entity_by_name("nonexistent")
        assert result is None

    async def test_short_words_skipped_in_fallback(self):
        """Words shorter than 3 characters should be skipped in fallback."""

        class TrackingTextBackend(StubTextBackend):
            def __init__(self):
                self.queries: list[str] = []

            async def search(self, query, *, limit=10):
                self.queries.append(query)
                return []

        text_backend = TrackingTextBackend()
        linker = MemoryLinker(
            vector_backend=StubVectorBackend(),
            text_backend=text_backend,
            link_backend=RecordingLinkBackend(),
        )

        await linker._find_entity_by_name("is a test")
        # "is" and "a" (< 3 chars) should be skipped
        # Only the full query and "test" should be searched
        assert text_backend.queries[0] == "is a test"  # first call
        assert "test" in text_backend.queries  # word fallback
        assert "is" not in text_backend.queries[1:]  # skipped
        assert "a" not in text_backend.queries[1:]   # skipped
