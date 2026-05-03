"""Tests for intent classification, query expansion, and tag co-occurrence."""

import pytest

from genesis_memory.intent import (
    TagCooccurrenceIndex,
    classify_intent,
    compute_intent_affinity,
    expand_query,
    rank_by_intent,
)

# -- Test doubles --


class StubVectorBackend:
    """Minimal VectorBackend that returns canned tag lists for scroll_tags."""

    def __init__(self, tag_lists: list[list[str]], total_count: int):
        self._tag_lists = tag_lists
        self._total_count = total_count

    async def scroll_tags(self, collections: list[str]) -> tuple[list[list[str]], int]:
        return self._tag_lists, self._total_count

    async def search(self, vector, *, limit=10, collection="default",
                     wing=None, room=None) -> list[dict]:
        return []

    async def upsert(self, point_id, vector, payload, *, collection="default"):
        pass

    async def update_payload(self, point_id, payload, *, collection="default"):
        pass

    async def delete(self, point_id, *, collection="default"):
        pass


# -- classify_intent --


class TestClassifyIntent:
    def test_what_query(self):
        result = classify_intent("what is the routing system")
        assert result.category == "WHAT"
        assert result.confidence == 0.80

    def test_why_query(self):
        result = classify_intent("why did we choose that approach")
        assert result.category == "WHY"
        assert result.confidence == 0.85

    def test_how_query(self):
        result = classify_intent("how to deploy the service")
        assert result.category == "HOW"
        assert result.confidence == 0.85

    def test_when_query(self):
        result = classify_intent("when was the last deployment")
        assert result.category == "WHEN"
        assert result.confidence == 0.85

    def test_where_query(self):
        result = classify_intent("where is the config file")
        assert result.category == "WHERE"
        assert result.confidence == 0.85

    def test_status_query(self):
        result = classify_intent("status of the migration")
        assert result.category == "STATUS"
        assert result.confidence == 0.80

    def test_general_query(self):
        result = classify_intent("circuit breaker patterns")
        assert result.category == "GENERAL"
        assert result.confidence == 0.0
        assert result.matched_pattern == ""

    def test_empty_query(self):
        result = classify_intent("")
        assert result.category == "GENERAL"
        assert result.confidence == 0.0

    def test_whitespace_only(self):
        result = classify_intent("   ")
        assert result.category == "GENERAL"
        assert result.confidence == 0.0

    def test_what_describe(self):
        result = classify_intent("describe the memory system")
        assert result.category == "WHAT"

    def test_what_define(self):
        result = classify_intent("define activation scoring")
        assert result.category == "WHAT"

    def test_what_tell_me_about(self):
        result = classify_intent("tell me about the graph traversal")
        assert result.category == "WHAT"

    def test_why_rationale(self):
        result = classify_intent("rationale for using RRF fusion")
        assert result.category == "WHY"

    def test_how_procedure(self):
        result = classify_intent("procedure for setting up the server")
        assert result.category == "HOW"

    def test_how_steps(self):
        result = classify_intent("steps to configure embedding")
        assert result.category == "HOW"

    def test_where_find_the(self):
        result = classify_intent("find the database schema")
        assert result.category == "WHERE"

    def test_where_which_file(self):
        result = classify_intent("which file has the activation logic")
        assert result.category == "WHERE"

    def test_status_progress(self):
        result = classify_intent("progress on the refactoring")
        assert result.category == "STATUS"

    def test_status_is_it_done(self):
        result = classify_intent("is it done yet")
        assert result.category == "STATUS"

    def test_case_insensitive(self):
        result = classify_intent("WHY did this happen")
        assert result.category == "WHY"

    def test_priority_why_over_what(self):
        """WHY patterns should match before WHAT since they're more specific."""
        result = classify_intent("why not what")
        assert result.category == "WHY"


# -- compute_intent_affinity --


class TestComputeIntentAffinity:
    def test_general_returns_zero(self):
        intent = classify_intent("random query")
        assert intent.category == "GENERAL"
        score = compute_intent_affinity(intent, "test", ["tag"], "content")
        assert score == 0.0

    def test_source_boost(self):
        intent = classify_intent("why did we decide that")
        score = compute_intent_affinity(
            intent, "deep_reflection", [], "some content",
        )
        assert score >= 2.0  # boosted source

    def test_tag_boost(self):
        intent = classify_intent("why this approach")
        score = compute_intent_affinity(
            intent, "unknown_source", ["decision"], "content",
        )
        assert score >= 1.5  # boosted tag

    def test_content_signal_boost(self):
        intent = classify_intent("why this choice")
        score = compute_intent_affinity(
            intent, "unknown_source", [], "because we decided to use RRF",
        )
        assert score >= 1.0  # content signal

    def test_all_boosts_combined(self):
        intent = classify_intent("why this choice")
        score = compute_intent_affinity(
            intent,
            "deep_reflection",       # +2.0
            ["decision"],            # +1.5
            "because we decided",    # +1.0
        )
        assert score == pytest.approx(4.5, abs=0.01)


# -- rank_by_intent --


class TestRankByIntent:
    def test_general_intent_returns_empty(self):
        intent = classify_intent("random stuff")
        result = rank_by_intent(intent, {"m1": {}, "m2": {}})
        assert result == []

    def test_ranking_order(self):
        intent = classify_intent("why this decision")
        candidates = {
            "m1": {"source": "deep_reflection", "tags": ["decision"],
                    "content": "because we chose RRF"},
            "m2": {"source": "test", "tags": [], "content": "nothing relevant"},
            "m3": {"source": "deep_reflection", "tags": [], "content": "some text"},
        }
        ranked = rank_by_intent(intent, candidates)
        assert len(ranked) == 3
        # m1 has all boosts, should be first
        assert ranked[0] == "m1"
        # m3 has source boost only (2.0), should be second
        assert ranked[1] == "m3"

    def test_empty_candidates(self):
        intent = classify_intent("why this")
        result = rank_by_intent(intent, {})
        assert result == []

    def test_deterministic_tiebreaking(self):
        """When scores are equal, IDs should be sorted alphabetically."""
        intent = classify_intent("how to do this")
        candidates = {
            "b_memory": {"source": "test", "tags": [], "content": ""},
            "a_memory": {"source": "test", "tags": [], "content": ""},
        }
        ranked = rank_by_intent(intent, candidates)
        # Both have score 0.0 → alphabetical tiebreak
        assert ranked == ["a_memory", "b_memory"]


# -- TagCooccurrenceIndex --


class TestTagCooccurrenceIndex:
    def test_build_and_expand(self):
        index = TagCooccurrenceIndex()
        tag_lists = [
            ["routing", "config", "infrastructure"],
            ["routing", "circuit-breaker", "infrastructure"],
            ["memory", "retrieval"],
        ]
        index.build(tag_lists, memory_count=3)

        # "routing" co-occurs with config, infrastructure, circuit-breaker
        expansions = index.expand(["routing"], max_expansions=5)
        assert "infrastructure" in expansions
        assert "config" in expansions or "circuit-breaker" in expansions

    def test_expand_no_results(self):
        index = TagCooccurrenceIndex()
        index.build([["a", "b"]], memory_count=1)
        result = index.expand(["nonexistent"], max_expansions=5)
        assert result == []

    def test_expand_excludes_input_keywords(self):
        index = TagCooccurrenceIndex()
        index.build([["alpha", "beta", "gamma"]], memory_count=1)
        expansions = index.expand(["alpha"], max_expansions=5)
        assert "alpha" not in expansions

    def test_expand_empty_index(self):
        index = TagCooccurrenceIndex()
        result = index.expand(["any"], max_expansions=5)
        assert result == []

    def test_expand_empty_keywords(self):
        index = TagCooccurrenceIndex()
        index.build([["a", "b"]], memory_count=1)
        result = index.expand([], max_expansions=5)
        assert result == []

    def test_is_stale_initially(self):
        index = TagCooccurrenceIndex()
        assert index.is_stale(10) is True

    def test_is_stale_after_build(self):
        index = TagCooccurrenceIndex()
        index.build([["a", "b"]], memory_count=100)
        # Same count — not stale
        assert index.is_stale(100) is False
        # 5% change — not stale (threshold is 10%)
        assert index.is_stale(105) is False
        # 15% change — stale
        assert index.is_stale(115) is True

    def test_max_expansions_respected(self):
        index = TagCooccurrenceIndex()
        tag_lists = [
            ["a", "b", "c", "d", "e", "f", "g"],
        ]
        index.build(tag_lists, memory_count=1)
        expansions = index.expand(["a"], max_expansions=3)
        assert len(expansions) <= 3

    def test_obs_prefix_tags_excluded(self):
        """Tags starting with obs: should be excluded from co-occurrence."""
        index = TagCooccurrenceIndex()
        tag_lists = [
            ["routing", "obs:timestamp123", "config"],
        ]
        index.build(tag_lists, memory_count=1)
        expansions = index.expand(["routing"], max_expansions=5)
        assert "obs:timestamp123" not in expansions

    def test_single_tag_lists_skipped(self):
        """Lists with fewer than 2 tags should be skipped."""
        index = TagCooccurrenceIndex()
        tag_lists = [
            ["lonely"],
            ["routing", "config"],
        ]
        index.build(tag_lists, memory_count=2)
        # "lonely" has no co-occurrences because its list was skipped
        expansions = index.expand(["lonely"], max_expansions=5)
        assert expansions == []

    def test_case_normalization(self):
        index = TagCooccurrenceIndex()
        tag_lists = [
            ["Routing", "Config"],
            ["routing", "config"],
        ]
        index.build(tag_lists, memory_count=2)
        expansions = index.expand(["routing"], max_expansions=5)
        assert "config" in expansions


# -- expand_query (async) --


class TestExpandQuery:
    async def test_expands_with_cooccurrences(self):
        tag_lists = [
            ["routing", "config", "infrastructure"],
            ["routing", "circuit-breaker", "infrastructure"],
        ]
        backend = StubVectorBackend(tag_lists, total_count=2)

        # Reset the module-level singleton to force rebuild
        import genesis_memory.intent as intent_mod
        intent_mod._tag_index = TagCooccurrenceIndex()

        result = await expand_query(
            "routing patterns",
            backend,
            ["episodic_memory"],
            max_expansions=3,
        )
        # Should contain the original query terms and expansions
        assert "routing" in result
        assert "OR" in result or result == "routing patterns"

    async def test_returns_original_on_empty_query(self):
        backend = StubVectorBackend([], total_count=0)
        result = await expand_query("", backend, ["default"])
        assert result == ""

    async def test_returns_original_on_no_tags(self):
        backend = StubVectorBackend([], total_count=0)
        result = await expand_query("test query", backend, ["default"])
        assert result == "test query"

    async def test_returns_original_on_error(self):
        """If scroll_tags raises, should fall back to original query."""

        class FailingBackend(StubVectorBackend):
            async def scroll_tags(self, collections):
                raise RuntimeError("connection lost")

        backend = FailingBackend([], 0)
        result = await expand_query("test query", backend, ["default"])
        assert result == "test query"
