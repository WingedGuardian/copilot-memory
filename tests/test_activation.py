"""Tests for activation scoring."""

import pytest

from genesis_memory.activation import compute_activation


class TestComputeActivation:
    def test_basic_score(self):
        score = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            now="2026-05-01T00:00:00+00:00",
        )
        # Fresh memory, no access, no links: confidence * 1.0 * 0.5 * 1.0
        assert score.final_score == pytest.approx(0.5, abs=0.01)
        assert score.recency_factor == pytest.approx(1.0, abs=0.01)
        assert score.base_score == 1.0

    def test_recency_decay(self):
        # 30 days old with default 30-day half-life → recency ≈ 0.5
        score = compute_activation(
            confidence=1.0,
            created_at="2026-04-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            now="2026-05-01T00:00:00+00:00",
        )
        assert score.recency_factor == pytest.approx(0.5, abs=0.01)

    def test_access_frequency_boost(self):
        no_access = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            now="2026-05-01T00:00:00+00:00",
        )
        high_access = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=20,
            link_count=0,
            now="2026-05-01T00:00:00+00:00",
        )
        assert high_access.final_score > no_access.final_score
        assert high_access.access_frequency > no_access.access_frequency

    def test_connectivity_boost(self):
        no_links = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            now="2026-05-01T00:00:00+00:00",
        )
        high_links = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=10,
            now="2026-05-01T00:00:00+00:00",
        )
        assert high_links.final_score > no_links.final_score
        assert high_links.connectivity_factor > no_links.connectivity_factor

    def test_rule_class_weight(self):
        fact = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            memory_class="fact",
            now="2026-05-01T00:00:00+00:00",
        )
        rule = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            memory_class="rule",
            now="2026-05-01T00:00:00+00:00",
        )
        assert rule.final_score == pytest.approx(fact.final_score * 1.3, abs=0.01)

    def test_reference_class_weight(self):
        fact = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            memory_class="fact",
            now="2026-05-01T00:00:00+00:00",
        )
        ref = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            memory_class="reference",
            now="2026-05-01T00:00:00+00:00",
        )
        assert ref.final_score == pytest.approx(fact.final_score * 0.7, abs=0.01)

    def test_entity_tag_doubles_half_life(self):
        # Entity tag should make memory decay slower
        normal = compute_activation(
            confidence=1.0,
            created_at="2026-04-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            tags=["routing", "config"],
            now="2026-05-01T00:00:00+00:00",
        )
        entity = compute_activation(
            confidence=1.0,
            created_at="2026-04-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            tags=["AgentMail", "config"],
            now="2026-05-01T00:00:00+00:00",
        )
        assert entity.recency_factor > normal.recency_factor

    def test_custom_half_life(self):
        # 30 days old with 60-day half-life → less decay than default
        default = compute_activation(
            confidence=1.0,
            created_at="2026-04-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            now="2026-05-01T00:00:00+00:00",
        )
        long_hl = compute_activation(
            confidence=1.0,
            created_at="2026-04-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            half_life_days=60.0,
            now="2026-05-01T00:00:00+00:00",
        )
        assert long_hl.recency_factor > default.recency_factor

    def test_session_extraction_source_has_longer_half_life(self):
        default = compute_activation(
            confidence=1.0,
            created_at="2026-04-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            source="unknown_source",
            now="2026-05-01T00:00:00+00:00",
        )
        session = compute_activation(
            confidence=1.0,
            created_at="2026-04-01T00:00:00+00:00",
            retrieved_count=0,
            link_count=0,
            source="session_extraction",
            now="2026-05-01T00:00:00+00:00",
        )
        assert session.recency_factor > default.recency_factor

    def test_zero_confidence(self):
        score = compute_activation(
            confidence=0.0,
            created_at="2026-05-01T00:00:00+00:00",
            retrieved_count=10,
            link_count=5,
            now="2026-05-01T00:00:00+00:00",
        )
        assert score.final_score == 0.0

    def test_naive_datetime_gets_utc(self):
        # Naive datetime (no tz) should be treated as UTC
        score = compute_activation(
            confidence=1.0,
            created_at="2026-05-01T00:00:00",
            retrieved_count=0,
            link_count=0,
            now="2026-05-01T00:00:00",
        )
        assert score.recency_factor == pytest.approx(1.0, abs=0.01)
